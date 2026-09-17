"""A registry of trading strategies across markets, each measured the same way.

The point of a registry rather than a pile of scripts is comparability. Every
strategy here returns the same shape -- an equity curve, a win rate, a cost
charge and a note about what it is not -- so that a crypto trend follower and
a modelled iron condor can sit in one table without the table lying about
either of them.

Three rules the registry enforces, because they are the ways this kind of
table usually misleads:

EVERY STRATEGY DECLARES ITS OWN CAVEAT. Not a footnote on the page, a field on
the record. A modelled options P&L and a measured equity return are different
kinds of number, and the difference has to travel with them.

COSTS ARE CHARGED, NOT MENTIONED. Each market gets its own rate, because a
crypto round trip and an FX fixing are not the same thing, and a strategy that
only works gross is not a strategy.

WIN RATE IS PER PERIOD, AND IS NOT AN EDGE. A strategy winning 95% of days and
losing everything on the other 5% is the classic short-volatility profile.
The registry reports win rate beside the worst drawdown for exactly that
reason -- read alone it is the most misleading number in trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TD = {"stocks": 252.0, "crypto": 365.0, "forex": 252.0, "options": 252.0}

# One-way cost in basis points, by market. Equities measured at this book size
# in loonie/execution.py; crypto and FX are conventional retail estimates and
# deliberately pessimistic.
COST_BPS = {"stocks": 2.6, "crypto": 10.0, "forex": 2.0, "options": 50.0}

REGISTRY: dict = {}


@dataclass
class Result:
    """What every strategy returns, whatever market it trades."""
    id: str
    title: str
    market: str
    description: str
    caveat: str = ""
    dates: object = None
    equity: np.ndarray = None
    returns: np.ndarray = None
    benchmark: np.ndarray = None
    ok: bool = True
    reason: str = ""
    # How many of this strategy's periods make a year. Daily strategies leave
    # it None and inherit the market calendar; anything returning one row per
    # holding period MUST set it. A monthly options model whose 442 periods
    # are annualised as 442 trading days reports 266% a year for something
    # that made 6%, and every other column inherits the error.
    periods_per_year: float = None
    extra: dict = field(default_factory=dict)

    def stats(self) -> dict:
        if not self.ok or self.returns is None or len(self.returns) < 30:
            return {"id": self.id, "title": self.title, "market": self.market,
                    "ok": False, "reason": self.reason or "too little data"}
        r = np.asarray(self.returns, dtype=np.float64)
        td = self.periods_per_year or TD.get(self.market, 252.0)
        yrs = max(len(r) / td, 1e-9)
        eq = np.cumprod(1.0 + r)
        total = float(eq[-1] - 1.0)
        sd = float(np.std(r, ddof=1))

        active = r[r != 0.0]
        dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())

        out = {
            "id": self.id, "title": self.title, "market": self.market,
            "description": self.description, "caveat": self.caveat, "ok": True,
            "total_pl_pct": 100.0 * total,
            "cagr_pct": 100.0 * ((1.0 + total) ** (1.0 / yrs) - 1.0
                                 if total > -1 else -1.0),
            # "Consistency" is the share of ACTIVE periods that made money.
            # Counting flat days as wins would let a strategy that trades
            # twice a year report 99%.
            "win_rate_pct": 100.0 * float(np.mean(active > 0)) if len(active) else 0.0,
            "periods": int(len(r)), "active_periods": int(len(active)),
            "sharpe": (float(np.mean(r)) / sd * np.sqrt(td)) if sd > 0 else 0.0,
            "max_drawdown_pct": 100.0 * dd,
            "years": yrs,
            "t_stat": (float(np.mean(r)) / sd * np.sqrt(len(r))) if sd > 0 else 0.0,
        }
        if self.benchmark is not None and len(self.benchmark) == len(r):
            b = np.asarray(self.benchmark, dtype=np.float64)
            exc = r - b
            es = float(np.std(exc, ddof=1))
            beq = float(np.prod(1.0 + b) - 1.0)
            out["benchmark_total_pl_pct"] = 100.0 * beq
            out["excess_t"] = (float(np.mean(exc)) / es * np.sqrt(len(exc))
                               if es > 0 else 0.0)
        out.update(self.extra)
        return out

    def curve(self, points: int = 260) -> list:
        """Downsampled equity curve for the hover graph."""
        if self.equity is None or not len(self.equity):
            return []
        eq = np.asarray(self.equity, dtype=float)
        idx = np.unique(np.linspace(0, len(eq) - 1, min(points, len(eq))).astype(int))
        return [[str(self.dates[i].date()), round(float(eq[i]), 5)] for i in idx]


def register(id, title, market, description, caveat=""):
    def deco(fn):
        REGISTRY[id] = {"id": id, "title": title, "market": market,
                        "description": description, "caveat": caveat, "fn": fn}
        return fn
    return deco


# =============================================================================
#  Shared machinery
# =============================================================================
def _ranks(x, mask):
    import pandas as pd
    return pd.DataFrame(np.where(mask, x, np.nan)).rank(axis=1, pct=True).to_numpy()


def _roll(a, w, fn):
    """Trailing window statistic, causal, NaN until the window fills."""
    T, N = a.shape
    out = np.full((T, N), np.nan)
    for t in range(w, T):
        out[t] = fn(a[t - w:t], axis=0)          # strictly before t
    return out


def long_short_book(score, mask, rets, n_long, n_short, hold, cost_bps,
                    market_neutral=True):
    """Equal-weight top/bottom `n`, rebalanced every `hold` periods.

    Weights set at t earn the return of t+1. Turnover is charged when the book
    actually changes, not once per period, so a strategy holding the same
    names for a month is not billed for a month of trading.
    """
    T, N = score.shape
    w = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(0, T, max(1, hold)):
        s = np.where(mask[t], score[t], np.nan)
        ok = np.isfinite(s)
        n_ok = int(ok.sum())
        if n_ok >= max(4, n_long + n_short):
            idx = np.where(ok)[0]
            order = idx[np.argsort(-s[idx])]
            cur = np.zeros(N)
            nl = min(n_long, len(order) // 2)
            cur[order[:nl]] = 1.0 / max(nl, 1)
            if market_neutral and n_short:
                ns = min(n_short, len(order) // 2)
                cur[order[-ns:]] = -1.0 / max(ns, 1)
        w[t:min(t + hold, T)] = cur

    held = np.vstack([np.zeros((1, N)), w[:-1]])
    gross = (held * np.nan_to_num(rets)).sum(axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, N)), w]), axis=0)).sum(axis=1)
    return gross - turn * cost_bps / 1e4, turn
