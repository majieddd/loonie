"""Macro regime features: the economic state the stock picking happens inside.

Everything in features.py is cross-sectional -- it compares stocks to each
other on a given day. None of it knows whether that day was March 2020 or a
quiet Tuesday in 2017. So a strategy can only ever express one idea and apply
it identically in every environment.

These series supply the missing axis. They are market-traded proxies rather
than published statistics, chosen deliberately: a Treasury yield or the VIX is
priced continuously and revised never, whereas CPI and payrolls are published
with a lag and then *restated*. A backtest that reads the current value of a
revised series is reading a number nobody had on the day. Traded proxies have
no such hole, need no API key, and settle daily alongside the equity panel.

Two rules make them safe to hand to the search:

  CAUSALITY. Every series is forward-filled from the last known close and then
  trailing z-scored. No value is computed from data after its own date.

  CENTRING. The z-score is what makes these usable as `ite` conditions. The
  grammar's branch test is `> 0`, and raw VIX is always positive, so an
  un-centred VIX would send every branch the same way forever. Trailing
  z-scored, `ite(vix_z, A, B)` reads as "if volatility is above its own recent
  normal, do A, otherwise B" -- which is the regime switch the `ite` node
  existed for and never previously had anything to condition on.

Each feature is a (T,) vector broadcast across every ticker, so a genome using
one alone scores every stock identically and selects arbitrarily. That is
correct: on its own a macro series carries no cross-sectional information, and
fitness will say so. Their value is as conditions and as multipliers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import resolve

warnings.filterwarnings("ignore")

# Traded proxies only -- continuously priced, never restated.
SERIES = {
    "vix": "^VIX",            # equity volatility
    "move": "^MOVE",          # bond volatility
    "y10": "^TNX",            # 10-year yield
    "y05": "^FVX",            # 5-year yield
    "y03m": "^IRX",           # 13-week bill
    "dollar": "DX-Y.NYB",     # dollar index
    "gold": "GC=F",
    "oil": "CL=F",
    "copper": "HG=F",
    "hyg": "HYG",             # high-yield credit
    "lqd": "LQD",             # investment-grade credit
    "tlt": "TLT",             # long duration
    "xlu": "XLU",             # utilities  (defensive)
    "xly": "XLY",             # discretionary (cyclical)
    "xlp": "XLP",             # staples
    "spy": "SPY",
    "iwm": "IWM",             # small caps
}

MACRO_NAMES = [
    "m_vix", "m_vix_chg", "m_move", "m_term_spread", "m_curve_5_10",
    "m_credit", "m_credit_chg", "m_dollar", "m_copper_gold", "m_oil",
    "m_defensive", "m_breadth", "m_mkt_dd", "m_mkt_vol", "m_mkt_mom",
    "m_real_rate",
]
CACHE = "data/cache/macro.parquet"


# ---------------------------------------------------------------- fetching
def fetch(start: str = "2015-01-01", end: str | None = None,
          refresh: bool = False, min_refetch_hours: float = 6.0) -> pd.DataFrame:
    """Daily closes for every macro series, one column each, parquet-cached."""
    import time

    p = resolve(CACHE)
    p.parent.mkdir(parents=True, exist_ok=True)
    end = end or str((pd.Timestamp.now() + pd.Timedelta(days=1)).date())

    if p.exists() and not refresh:
        age_h = (time.time() - p.stat().st_mtime) / 3600.0
        if age_h < min_refetch_hours:
            try:
                return pd.read_parquet(p)
            except Exception:
                pass

    import yfinance as yf

    cols = {}
    for name, sym in SERIES.items():
        try:
            df = yf.download(sym, start=start, end=end, progress=False,
                             auto_adjust=True, threads=False)
            if df is None or len(df) == 0:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            if "close" not in df.columns:
                continue
            s = df["close"].astype("float64")
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            cols[name] = s
        except Exception:
            continue

    if not cols:
        raise RuntimeError("no macro series fetched -- check network")
    out = pd.DataFrame(cols).sort_index()
    try:
        out.to_parquet(p)
    except Exception:
        pass
    return out


# ------------------------------------------------------- causal transforms
def _z(a: np.ndarray, win: int = 252) -> np.ndarray:
    """Trailing z-score. Uses only rows <= t; NaN until the window fills."""
    a = np.asarray(a, dtype=np.float64)
    T = len(a)
    out = np.full(T, np.nan)
    for t in range(T):
        lo = max(0, t - win + 1)
        w = a[lo:t + 1]
        w = w[np.isfinite(w)]
        if len(w) < max(20, win // 8):
            continue
        sd = w.std(ddof=1)
        if sd > 1e-12:
            out[t] = (a[t] - w.mean()) / sd
    return out


def _mom(a: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    prev = np.concatenate([np.full(k, np.nan), a[:-k]]) if k < len(a) else np.full(len(a), np.nan)
    return (a - prev) / np.maximum(np.abs(prev), 1e-12)


def build(dates: pd.DatetimeIndex, refresh: bool = False) -> dict:
    """Macro regime features aligned to `dates`, each a causal (T,) z-score."""
    raw = fetch(refresh=refresh)
    # Forward-fill onto the equity calendar: holidays differ between the
    # futures, bond and equity sessions. ffill carries the last KNOWN close,
    # which is causal; bfill would not be, and is never used.
    df = raw.reindex(raw.index.union(dates)).sort_index().ffill().reindex(dates)

    g = lambda k: (df[k].to_numpy(dtype=np.float64)          # noqa: E731
                   if k in df.columns else np.full(len(dates), np.nan))
    f = {}

    f["m_vix"] = _z(np.log(np.maximum(g("vix"), 1e-9)))
    f["m_vix_chg"] = _z(_mom(g("vix"), 21))
    f["m_move"] = _z(np.log(np.maximum(g("move"), 1e-9)))

    # Curve shape. Inversion is the single most-watched recession proxy.
    f["m_term_spread"] = _z(g("y10") - g("y03m"))
    f["m_curve_5_10"] = _z(g("y10") - g("y05"))
    # Nominal 10y less its own trailing year: a crude real-rate direction.
    f["m_real_rate"] = _z(g("y10"))

    # Risk appetite: high-yield against investment-grade. Credit leads equity.
    credit = g("hyg") / np.maximum(g("lqd"), 1e-9)
    f["m_credit"] = _z(credit)
    f["m_credit_chg"] = _z(_mom(credit, 21))

    f["m_dollar"] = _z(_mom(g("dollar"), 63))
    f["m_copper_gold"] = _z(g("copper") / np.maximum(g("gold"), 1e-9))
    f["m_oil"] = _z(_mom(g("oil"), 63))

    # Defensive rotation: utilities over discretionary is a risk-off tell.
    f["m_defensive"] = _z(_mom(g("xlu") / np.maximum(g("xly"), 1e-9), 63))
    f["m_breadth"] = _z(_mom(g("iwm") / np.maximum(g("spy"), 1e-9), 63))

    spy = g("spy")
    peak = pd.Series(spy).rolling(252, min_periods=20).max().to_numpy()
    f["m_mkt_dd"] = _z(spy / np.maximum(peak, 1e-9) - 1.0)
    r1 = _mom(spy, 1)
    f["m_mkt_vol"] = _z(pd.Series(r1).rolling(21, min_periods=10).std().to_numpy())
    f["m_mkt_mom"] = _z(_mom(spy, 126))

    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def broadcast(macro: dict, n_tickers: int) -> dict:
    """Expand each (T,) macro series to (T, N) so it can join the panel."""
    return {k: np.repeat(v[:, None], n_tickers, axis=1).astype(np.float32)
            for k, v in macro.items()}


def coverage(macro: dict) -> dict:
    """How much of each series is actually usable, for the provenance report."""
    return {k: float(np.mean(np.isfinite(v))) for k, v in macro.items()}
