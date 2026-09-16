"""Kill switches. The part that decides the system is wrong and stops it.

A self-improving trader has an unusual failure mode: the thing deciding what
to trade is itself changing, so "it behaved normally yesterday" carries much
less information than usual. These checks are deliberately outside the
learning loop and cannot be adjusted by it. Evolution proposes; risk disposes.

Every breach is latched. A breached system does not resume because the number
came back -- it resumes when a person clears the latch, having looked at why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .config import resolve

STATE = "state/risk_state.json"


@dataclass
class RiskState:
    equity_high_water: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    orders_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    halted_at: str = ""
    breaches: list = field(default_factory=list)

    @classmethod
    def load(cls) -> "RiskState":
        p = resolve(STATE)
        if p.exists():
            try:
                return cls(**json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls()

    def save(self):
        p = resolve(STATE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


@dataclass
class RiskDecision:
    allow: bool
    halt: bool = False
    reason: str = ""
    checks: list = field(default_factory=list)

    def __bool__(self):
        return self.allow


class RiskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.r = cfg.trade.risk
        self.state = RiskState.load()

    # ------------------------------------------------------------ bookkeeping
    def roll_day(self, equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.day != today:
            self.state.day = today
            self.state.day_start_equity = equity
            self.state.orders_today = 0
            self.state.save()
        if equity > self.state.equity_high_water:
            self.state.equity_high_water = equity
            self.state.save()

    def note_order(self, n: int = 1):
        self.state.orders_today += n
        self.state.save()

    # ----------------------------------------------------------------- checks
    def check(self, equity: float, last_data_utc: datetime | None = None,
              account=None, last_bar_date=None) -> RiskDecision:
        self.roll_day(equity)
        checks, breach = [], None

        if self.state.halted:
            return RiskDecision(
                allow=False, halt=True,
                reason="latched halt: %s (at %s). Clear with "
                       "scripts/clear_halt.py after reviewing."
                       % (self.state.halt_reason, self.state.halted_at),
                checks=checks)

        # --- daily loss ---
        start = self.state.day_start_equity or equity
        day_pl = (equity - start) / max(start, 1e-9)
        lim = -abs(float(self.r.max_daily_loss_pct))
        checks.append(("daily_pl", day_pl, lim, day_pl > lim))
        if day_pl <= lim:
            breach = "daily loss %.2f%% breached limit %.2f%%" % (
                100 * day_pl, 100 * lim)

        # --- peak-to-trough drawdown ---
        hw = self.state.equity_high_water or equity
        dd = (equity - hw) / max(hw, 1e-9)
        ddlim = -abs(float(self.r.max_drawdown_pct))
        checks.append(("drawdown", dd, ddlim, dd > ddlim))
        if breach is None and dd <= ddlim:
            breach = "drawdown %.2f%% breached limit %.2f%%" % (
                100 * dd, 100 * ddlim)

        # --- runaway order count ---
        omax = int(self.r.max_orders_per_day)
        checks.append(("orders_today", self.state.orders_today, omax,
                       self.state.orders_today < omax))
        if breach is None and self.state.orders_today >= omax:
            breach = "order count %d reached daily cap %d" % (
                self.state.orders_today, omax)

        # --- stale data: never trade on a signal you cannot date ---
        # Daily bars are stale by construction overnight, so age is measured in
        # calendar days against the last bar, not minutes against the clock. A
        # cache that quietly stopped updating is the most likely way this system
        # trades on a week-old view of the world while looking perfectly healthy.
        if last_bar_date is not None:
            age_d = (datetime.now(timezone.utc).date()
                     - pd.Timestamp(last_bar_date).date()).days
            dmax = float(self.r.get("max_data_age_days", 5))
            checks.append(("data_age_days", age_d, dmax, age_d <= dmax))
            if breach is None and age_d > dmax:
                breach = ("most recent bar is %d days old (limit %.0f) -- the "
                          "data cache is not updating" % (age_d, dmax))

        if last_data_utc is not None:
            age = (datetime.now(timezone.utc) - last_data_utc).total_seconds() / 60
            amax = float(self.r.halt_on_data_staleness_min)
            checks.append(("quote_age_min", age, amax, age <= amax))
            if breach is None and age > amax:
                breach = "quotes are %.0f min old (limit %.0f)" % (age, amax)

        # --- broker-side block ---
        if account is not None and getattr(account, "blocked", False):
            checks.append(("broker_block", 1, 0, False))
            if breach is None:
                breach = "broker reports the account is blocked"

        if breach:
            return self.halt(breach, checks)
        return RiskDecision(allow=True, checks=checks)

    def check_order(self, symbol: str, notional: float, equity: float,
                    account=None) -> RiskDecision:
        """Per-order sizing limits."""
        maxpos = float(self.cfg.trade.max_position_pct) * max(equity, 1e-9)
        if notional > maxpos * 1.001:
            return RiskDecision(
                allow=False,
                reason="%s notional $%.0f exceeds max_position_pct ($%.0f)"
                       % (symbol, notional, maxpos))
        if notional < float(self.cfg.trade.min_order_notional):
            return RiskDecision(allow=False, reason="%s below min notional" % symbol)
        if account is not None:
            gross = sum(abs(p.market_value) for p in account.positions.values())
            cap = float(self.cfg.trade.max_gross_exposure) * max(equity, 1e-9)
            if gross + notional > cap * 1.001:
                return RiskDecision(
                    allow=False,
                    reason="gross exposure would reach $%.0f, cap $%.0f"
                           % (gross + notional, cap))
        return RiskDecision(allow=True)

    # ------------------------------------------------------------------ halt
    def halt(self, reason: str, checks=None) -> RiskDecision:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state.breaches.append({"at": self.state.halted_at, "reason": reason})
        self.state.save()
        return RiskDecision(allow=False, halt=True, reason=reason,
                            checks=checks or [])

    def clear(self, note: str = "") -> None:
        self.state.breaches.append({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": "CLEARED: %s" % note})
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halted_at = ""
        self.state.save()
