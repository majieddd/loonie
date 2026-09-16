"""Broker interface and the safety interlocks around live trading.

Live trading is behind three independent locks that must ALL be open:

    1. config.yaml          trade.allow_live: true
    2. environment          ALPACA_MODE=live
    3. command line         --i-understand-this-is-real-money

They are deliberately in three different places, owned by three different
actions, so that no single edit -- and no automated process -- can arm real
money on its own. The defaults ship closed and nothing in this repository
opens them for you.

Paper mode is not a lesser mode. It is where a strategy earns the right to be
considered, and the allocator in allocator.py learns from paper fills exactly
as it would from live ones.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


class LiveTradingLocked(RuntimeError):
    pass


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.market_price

    @property
    def unrealized_pl(self) -> float:
        return self.qty * (self.market_price - self.avg_price)


@dataclass
class Order:
    symbol: str
    side: str            # buy | sell
    qty: float
    type: str = "market"
    tif: str = "day"
    notional: float | None = None
    client_id: str = ""
    status: str = "new"
    filled_qty: float = 0.0
    filled_price: float = 0.0
    submitted_at: str = field(default_factory=lambda: _now())
    note: str = ""

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Account:
    cash: float
    equity: float
    buying_power: float
    positions: dict = field(default_factory=dict)
    blocked: bool = False

    @property
    def gross_exposure(self) -> float:
        gross = sum(abs(p.market_value) for p in self.positions.values())
        return gross / max(self.equity, 1e-9)


class Broker:
    name = "base"
    is_live = False

    def account(self) -> Account:
        raise NotImplementedError

    def positions(self) -> dict:
        raise NotImplementedError

    def submit(self, order: Order) -> Order:
        raise NotImplementedError

    def cancel_all(self) -> int:
        raise NotImplementedError

    def close_all(self) -> int:
        raise NotImplementedError

    def last_price(self, symbol: str) -> float | None:
        raise NotImplementedError

    def is_market_open(self) -> bool:
        raise NotImplementedError


def assert_live_allowed(cfg, cli_flag: bool) -> None:
    """All three locks, checked together, with a readable failure."""
    mode = str(cfg.trade.mode).lower()
    env_mode = os.getenv("ALPACA_MODE", "paper").lower()
    allow = bool(cfg.trade.get("allow_live", False))
    if mode != "live":
        return
    missing = []
    if not allow:
        missing.append("config.yaml -> trade.allow_live: true")
    if env_mode != "live":
        missing.append("environment -> ALPACA_MODE=live")
    if not cli_flag:
        missing.append("command line -> --i-understand-this-is-real-money")
    if missing:
        raise LiveTradingLocked(
            "live trading requested but these locks are still closed:\n  - "
            + "\n  - ".join(missing)
            + "\n\nOpen them yourself, deliberately, one at a time. Run in paper "
              "for a meaningful period first; the allocator needs live fills to "
              "learn from, and paper fills are how it earns the right to ask."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
