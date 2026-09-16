"""Point-in-time index membership.

The single most important file in this project.

A backtest that picks stocks from *today's* S&P 500 constituent list and runs
it back to 2016 is not a backtest. It is a lookup of what already worked. The
706 companies that were dropped from the index since 1996 -- the bankruptcies,
the takeunders, the slow bleeds -- are exactly the losing trades your strategy
would have taken, and deleting them is how a mediocre model reports 26% CAGR.

This module answers exactly one question, honestly:
    "Which tickers was I *allowed* to buy on date D?"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import resolve


@dataclass
class Universe:
    """Point-in-time membership. `panel` is (n_change_dates, n_tickers) bool."""

    change_dates: pd.DatetimeIndex
    tickers: list[str]
    panel: np.ndarray  # bool (D, N)

    # ---------------------------------------------------------------- build
    @classmethod
    def from_pit_csv(cls, path: str | Path) -> "Universe":
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not {"date", "tickers"} <= set(df.columns):
            raise ValueError(f"unexpected PIT columns: {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        rows = [
            {t.strip() for t in str(s).split(",") if t.strip()}
            for s in df["tickers"]
        ]
        tickers = sorted(set().union(*rows))
        index = {t: i for i, t in enumerate(tickers)}

        panel = np.zeros((len(rows), len(tickers)), dtype=bool)
        for r, members in enumerate(rows):
            for t in members:
                panel[r, index[t]] = True

        return cls(pd.DatetimeIndex(df["date"]), tickers, panel)

    @classmethod
    def load(cls, cfg) -> "Universe":
        """Load from cache, downloading the PIT dataset on first use."""
        cache = resolve("data/universe/sp500_pit.csv")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            url = cfg.universe.pit_url
            print(f"[universe] downloading point-in-time membership\n           {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            cache.write_bytes(r.content)
            print(f"[universe] cached -> {cache} ({len(r.content)/1e6:.1f} MB)")
        return cls.from_pit_csv(cache)

    # ----------------------------------------------------------- membership
    def members_on(self, date) -> set[str]:
        """Constituents as of `date` (the last change on or before it)."""
        date = pd.Timestamp(date)
        pos = int(self.change_dates.searchsorted(date, side="right")) - 1
        if pos < 0:
            return set()
        return {t for t, m in zip(self.tickers, self.panel[pos]) if m}

    def mask_for(self, dates: pd.DatetimeIndex, tickers: list[str]) -> np.ndarray:
        """(T, N) bool: was `tickers[j]` an index member on `dates[i]`?

        Forward-fills membership between change dates. This array is ANDed into
        every trading decision the backtester makes.
        """
        pos = self.change_dates.searchsorted(dates, side="right") - 1
        pos = np.clip(pos, 0, len(self.change_dates) - 1)
        cols = [self.tickers.index(t) if t in self.tickers else -1 for t in tickers]

        out = np.zeros((len(dates), len(tickers)), dtype=bool)
        known = [(j, c) for j, c in enumerate(cols) if c >= 0]
        if known:
            js = np.array([j for j, _ in known])
            cs = np.array([c for _, c in known])
            out[:, js] = self.panel[np.ix_(pos, cs)]
        # Tickers before the first change date were never members.
        out[np.asarray(dates) < self.change_dates[0]] = False
        return out

    def tickers_active_between(self, start, end) -> list[str]:
        """Every ticker that was a member at ANY point in [start, end].

        Includes the ones that went to zero. That is the point.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        lo = max(0, int(self.change_dates.searchsorted(start, side="right")) - 1)
        hi = int(self.change_dates.searchsorted(end, side="right"))
        if hi <= lo:
            hi = lo + 1
        ever = self.panel[lo:hi].any(axis=0)
        return [t for t, e in zip(self.tickers, ever) if e]

    # ----------------------------------------------------------------- audit
    def survivorship_report(self, start, end) -> dict:
        """Quantify how much of the historical universe is 'gone'."""
        hist = set(self.tickers_active_between(start, end))
        today = self.members_on(self.change_dates[-1])
        departed = hist - today
        return {
            "tickers_ever_in_window": len(hist),
            "still_in_index_today": len(hist & today),
            "departed": len(departed),
            "departed_frac": len(departed) / max(1, len(hist)),
            "departed_sample": sorted(departed)[:20],
        }
