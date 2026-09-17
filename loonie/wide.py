"""A wider universe, with membership defined by a rule rather than a committee.

The S&P 500 gives roughly 500 names on any date. Cross-sectional statistics
scale with that number: an information coefficient measured across 500 names
has a standard error about sqrt(6) times larger than the same coefficient
across 3,000. The search has spent 300,000 trials failing to clear a bar it
could clear on the same underlying signal with a wider field, so breadth is
the cheapest remaining lever on statistical power.

The problem is that point-in-time membership for a wide index is not free.
Nobody publishes a survivorship-free Russell 3000 constituent history you can
download, and a current constituent list applied to history is the exact bias
this project spends most of its effort avoiding.

So membership here is a RULE, not a list: on any date, the universe is the N
most liquid names by trailing median dollar volume. That is point-in-time by
construction -- the rule reads only data from before the date it applies to --
and it needs no vendor. It is also close to what an index committee is
approximating anyway.

WHAT THIS DOES NOT FIX, and must not be allowed to obscure: the price data
still comes from a provider that drops delisted securities. A wider universe
reaches further down the liquidity curve, where companies fail more often, so
the survivorship gap is likely WORSE here than in the S&P 500 panel, not
better. Breadth buys statistical power; it buys nothing at all in honesty, and
the coverage audit is the only thing standing between the two.

The membership rule recomputes on a schedule rather than daily, because a
universe that churns every session generates turnover that is an artefact of
the definition rather than of any decision the strategy made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import resolve

DIRECTORY = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
SYMBOLS_CACHE = "data/universe/us_symbols.csv"

# Suffixes the directory uses for things that are not common stock. Units,
# warrants and rights trade thinly, have their own price dynamics, and would
# be selected FOR by a liquidity rule on some days purely because a SPAC was
# in the news that week.
NOT_COMMON = (".U", ".W", ".R", ".WS", ".RT", ".UN")


def fetch_symbols(refresh: bool = False) -> pd.DataFrame:
    """Every currently traded US symbol, minus ETFs, tests and non-common."""
    import urllib.request

    p = resolve(SYMBOLS_CACHE)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not refresh:
        return pd.read_csv(p)

    req = urllib.request.Request(
        DIRECTORY, headers={"User-Agent": "loonie-research (individual)"})
    with urllib.request.urlopen(req, timeout=90) as fh:
        text = fh.read().decode("utf-8", errors="replace")

    rows = [ln.split("|") for ln in text.splitlines()]
    hdr = rows[0]
    body = [r for r in rows[1:] if len(r) == len(hdr)]
    df = pd.DataFrame(body, columns=hdr)

    keep = (df["ETF"] == "N") & (df["Test Issue"] == "N")
    df = df[keep].reset_index(drop=True)
    # Recompute the symbol series after each filter. Reusing one built before
    # the previous filter silently reindexes, which pandas warns about and
    # which would drop the wrong rows.
    sym = df["Symbol"].astype(str)
    df = df[~sym.str.endswith(NOT_COMMON)].reset_index(drop=True)
    # A dollar sign marks a non-equity instrument in this file; a caret is
    # used for preferred series.
    sym = df["Symbol"].astype(str)
    df = df[~sym.str.contains(r"[\$\^]", regex=True, na=False)].reset_index(drop=True)

    out = df[["Symbol", "Security Name", "Listing Exchange"]].copy()
    out.columns = ["symbol", "name", "exchange"]
    out = out.drop_duplicates("symbol").sort_values("symbol")
    out.to_csv(p, index=False)
    return out


# =============================================================================
#  Rule-based point-in-time membership
# =============================================================================
def liquidity_rank(close: np.ndarray, volume: np.ndarray,
                   lookback: int = 60) -> np.ndarray:
    """Trailing median dollar volume, (T, N), using only data before each row.

    The shift by one is what makes it usable: a rule that reads today's volume
    to decide whether today's name is in the universe is reading the session
    it is about to trade.
    """
    dv = np.asarray(close, dtype=np.float64) * np.nan_to_num(
        np.asarray(volume, dtype=np.float64))
    T, N = dv.shape
    out = np.full((T, N), np.nan)
    prev = np.vstack([np.full((1, N), np.nan), dv[:-1]])
    for t in range(lookback, T):
        w = prev[t - lookback + 1:t + 1]
        with np.errstate(invalid="ignore"):
            out[t] = np.nanmedian(np.where(w > 0, w, np.nan), axis=0)
    return out


def membership(close, volume, n_names: int = 3000, lookback: int = 60,
               refresh_every: int = 21, min_dollar_vol: float = 1e6,
               min_price: float = 3.0) -> np.ndarray:
    """(T, N) bool: is this name in the universe on this date?

    Recomputed every `refresh_every` sessions and held flat in between. A
    universe that re-ranks daily would push names in and out on noise, and the
    strategy would be charged for turnover that no decision of its own caused.
    """
    liq = liquidity_rank(close, volume, lookback)
    T, N = liq.shape
    px = np.asarray(close, dtype=np.float64)
    out = np.zeros((T, N), dtype=bool)

    for start in range(lookback, T, refresh_every):
        row = liq[start]
        ok = np.isfinite(row) & (row >= min_dollar_vol) & (px[start] >= min_price)
        idx = np.where(ok)[0]
        if len(idx) > n_names:
            idx = idx[np.argsort(-row[idx])[:n_names]]
        stop = min(start + refresh_every, T)
        out[start:stop, idx] = True
    return out


def summarise(member: np.ndarray, dates) -> dict:
    """What the rule actually produced, for the coverage audit."""
    per_day = member.sum(axis=1)
    live = per_day[per_day > 0]
    ever = int(member.any(axis=0).sum())
    # How much of the membership turns over between refreshes. A number far
    # above a few percent means the rule itself is generating trading.
    churn = []
    prev = None
    for i in range(0, len(member), 21):
        cur = set(np.where(member[i])[0])
        if prev is not None and prev:
            churn.append(len(cur ^ prev) / max(len(prev), 1))
        prev = cur
    return {
        "names_ever": ever,
        "median_per_day": float(np.median(live)) if len(live) else 0.0,
        "min_per_day": int(live.min()) if len(live) else 0,
        "max_per_day": int(live.max()) if len(live) else 0,
        "median_churn_per_refresh": float(np.median(churn)) if churn else 0.0,
        "first_populated": str(pd.Timestamp(dates[int(np.argmax(per_day > 0))]).date())
        if (per_day > 0).any() else None,
    }
