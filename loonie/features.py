"""Vectorised, strictly causal feature library over an (T, N) price panel.

Every feature at row t is computed from data at rows <= t. The backtester then
applies the signal at t to the return from t to t+1. That one-bar gap is the
difference between a strategy and a time machine.

These are the primitive terminals the genetic program composes. They are
deliberately raw -- no cross-sectional normalisation is baked in, because the
expression grammar has `rank` and `zscore` operators and the search should get
to decide where normalisation belongs.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
EPS = 1e-12


# =============================================================================
#  Causal rolling primitives  (axis 0 = time)
# =============================================================================
def shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _nan_cumsum(a: np.ndarray):
    """Returns (cumsum of zero-filled a, cumulative count of finite cells)."""
    ok = np.isfinite(a)
    z = np.where(ok, a, 0.0).astype(np.float64)
    return np.cumsum(z, axis=0), np.cumsum(ok, axis=0)


def roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    cs, cn = _nan_cumsum(a)
    s = cs.copy()
    n = cn.copy().astype(np.float64)
    s[w:] = cs[w:] - cs[:-w]
    n[w:] = cn[w:] - cn[:-w]
    return np.where(n > 0, s / np.maximum(n, EPS), np.nan).astype(np.float32)


def roll_std(a: np.ndarray, w: int) -> np.ndarray:
    m = roll_mean(a, w)
    m2 = roll_mean(a.astype(np.float64) ** 2, w)
    v = m2 - m.astype(np.float64) ** 2
    return np.sqrt(np.maximum(v, 0.0)).astype(np.float32)


def roll_sum(a: np.ndarray, w: int) -> np.ndarray:
    cs, cn = _nan_cumsum(a)
    s = cs.copy()
    n = cn.copy()
    s[w:] = cs[w:] - cs[:-w]
    n[w:] = cn[w:] - cn[:-w]
    return np.where(n > 0, s, np.nan).astype(np.float32)


def _roll_reduce(a: np.ndarray, w: int, fn) -> np.ndarray:
    T = a.shape[0]
    out = np.full(a.shape, np.nan, np.float32)
    for t in range(T):
        lo = max(0, t - w + 1)
        out[t] = fn(a[lo:t + 1], axis=0)
    return out


def roll_max(a, w):
    return _roll_reduce(a, w, np.nanmax)


def roll_min(a, w):
    return _roll_reduce(a, w, np.nanmin)


def roll_rank_ts(a: np.ndarray, w: int) -> np.ndarray:
    """Percentile of the current value within its own trailing window."""
    T = a.shape[0]
    out = np.full(a.shape, np.nan, np.float32)
    for t in range(T):
        lo = max(0, t - w + 1)
        win = a[lo:t + 1]
        cur = a[t]
        n = np.sum(np.isfinite(win), axis=0)
        less = np.sum(np.where(np.isfinite(win), win < cur, False), axis=0)
        out[t] = np.where(n > 1, less / np.maximum(n - 1, 1), np.nan)
    return out


def pct_change(a: np.ndarray, k: int) -> np.ndarray:
    prev = shift(a, k)
    return ((a - prev) / np.maximum(np.abs(prev), EPS)).astype(np.float32)


# =============================================================================
#  Cross-sectional operators (axis 1 = ticker)  -- used by the grammar
# =============================================================================
def cs_rank(a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-row percentile rank in [0,1] over eligible names only."""
    x = np.where(mask, a, np.nan) if mask is not None else a.copy()
    order = np.argsort(np.where(np.isfinite(x), x, np.inf), axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(x.shape[1], dtype=np.float32)[None, :]
    n = np.sum(np.isfinite(x), axis=1, keepdims=True).astype(np.float32)
    out = np.where(np.isfinite(x), ranks / np.maximum(n - 1, 1), np.nan)
    return out.astype(np.float32)


def cs_zscore(a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    # float64 accumulation -- see cs_demean for why this is not fussiness.
    x = (np.where(mask, a, np.nan) if mask is not None else a).astype(np.float64)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return np.where(np.isfinite(x), (x - mu) / np.maximum(sd, EPS),
                    np.nan).astype(np.float32)


def cs_demean(a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Subtract the cross-sectional mean. Accumulated in float64, deliberately.

    Regression, and a genuinely nasty one. Summing 500 float32 values and
    subtracting the float32 mean does not give zero for a constant input -- it
    gives about -3e-08, with a *consistent sign*. The search found that: its
    fittest genome branched on `mul(mom_252, demean(-0.3359))`, which is
    mathematically identically zero and therefore takes one branch always, but
    in float32 is a tiny negative number, so the branch flipped on the sign of
    `mom_252`. A rounding error was being used as a regime signal.

    That is the degenerate-solution failure mode in its purest form: the
    optimiser is not wrong, the objective simply contained a channel nobody
    meant to offer it. Accumulating in float64 closes the channel -- a constant
    now demeans to exactly 0.0 and the branch is genuinely dead, where the
    simplifier can see and remove it.
    """
    x = (np.where(mask, a, np.nan) if mask is not None else a).astype(np.float64)
    mu = np.nanmean(x, axis=1, keepdims=True)
    return (x - mu).astype(np.float32)


# =============================================================================
#  The feature set
# =============================================================================
def build(panel, macro: bool = True, peers: bool = True,
          fundamentals: bool = True) -> dict:
    """Compute the terminal feature dictionary from a Panel. All causal.

    `macro=True` appends the regime series from macro.py -- (T,) vectors
    broadcast across tickers, so the search can condition a cross-sectional
    idea on the economic environment it is running in.
    """
    c = panel.bars["close"].astype(np.float32)
    h = panel.bars["high"].astype(np.float32)
    lo = panel.bars["low"].astype(np.float32)
    o = panel.bars["open"].astype(np.float32)
    v = np.nan_to_num(panel.bars["volume"]).astype(np.float32)

    r1 = pct_change(c, 1)
    f: dict = {}

    # ---- momentum ---------------------------------------------------------
    for w in (1, 5, 10, 21, 63, 126, 252):
        f["mom_%d" % w] = pct_change(c, w)
    # 12-1 momentum: the classic, skipping the short-term reversal month
    f["mom_12_1"] = (shift(c, 21) / np.maximum(shift(c, 252), EPS) - 1).astype(np.float32)

    # ---- mean reversion / trend position ---------------------------------
    for w in (10, 21, 50, 200):
        ma = roll_mean(c, w)
        f["ma_ratio_%d" % w] = (c / np.maximum(ma, EPS) - 1).astype(np.float32)
    f["rev_5"] = -f["mom_5"]
    f["rev_21"] = -f["mom_21"]

    hi252, lo252 = roll_max(c, 252), roll_min(c, 252)
    f["pct_52w_high"] = (c / np.maximum(hi252, EPS)).astype(np.float32)
    f["pct_52w_range"] = ((c - lo252) / np.maximum(hi252 - lo252, EPS)).astype(np.float32)

    # ---- volatility / risk ------------------------------------------------
    for w in (10, 21, 63, 126):
        f["vol_%d" % w] = (roll_std(r1, w) * np.sqrt(252.0)).astype(np.float32)
    f["vol_ratio"] = (f["vol_21"] / np.maximum(f["vol_126"], EPS)).astype(np.float32)
    down = np.where(r1 < 0, r1, np.nan)
    f["downside_vol_63"] = (roll_std(down, 63) * np.sqrt(252.0)).astype(np.float32)
    f["skew_63"] = _roll_moment(r1, 63, 3)
    f["kurt_63"] = _roll_moment(r1, 63, 4)

    tr = np.maximum(h - lo, np.maximum(np.abs(h - shift(c, 1)), np.abs(lo - shift(c, 1))))
    f["atr_14"] = (roll_mean(tr, 14) / np.maximum(c, EPS)).astype(np.float32)
    dd = c / np.maximum(roll_max(c, 63), EPS) - 1
    f["drawdown_63"] = dd.astype(np.float32)

    # ---- volume / liquidity ----------------------------------------------
    dv = c * v
    f["dollar_vol_21"] = np.log1p(roll_mean(dv, 21)).astype(np.float32)
    f["vol_surge"] = (roll_mean(v, 5) / np.maximum(roll_mean(v, 63), EPS)).astype(np.float32)
    # Amihud illiquidity: |return| per dollar traded
    f["amihud_21"] = roll_mean(np.abs(r1) / np.maximum(dv, EPS) * 1e9, 21).astype(np.float32)
    f["obv_slope_21"] = pct_change(np.cumsum(np.nan_to_num(np.sign(r1)) * v, axis=0), 21)

    # ---- oscillators ------------------------------------------------------
    f["rsi_14"] = _rsi(r1, 14)
    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    macd = ema12 - ema26
    f["macd_hist"] = ((macd - _ema(macd, 9)) / np.maximum(c, EPS)).astype(np.float32)
    ma20, sd20 = roll_mean(c, 20), roll_std(c, 20)
    f["bollinger_z"] = ((c - ma20) / np.maximum(sd20, EPS)).astype(np.float32)
    f["ts_rank_63"] = roll_rank_ts(c, 63)

    # ---- microstructure ---------------------------------------------------
    f["clv"] = ((2 * c - h - lo) / np.maximum(h - lo, EPS)).astype(np.float32)
    f["gap"] = ((o - shift(c, 1)) / np.maximum(shift(c, 1), EPS)).astype(np.float32)
    f["intraday_range"] = ((h - lo) / np.maximum(c, EPS)).astype(np.float32)
    f["close_strength"] = ((c - o) / np.maximum(o, EPS)).astype(np.float32)

    # ---- market-relative --------------------------------------------------
    mkt = np.nanmean(np.where(panel.member, r1, np.nan), axis=1, keepdims=True)
    cov = roll_mean(r1 * mkt, 126)
    mvar = roll_mean(mkt ** 2, 126)
    beta = cov / np.maximum(mvar, EPS)
    f["beta_126"] = beta.astype(np.float32)
    f["idio_vol_126"] = (roll_std(r1 - beta * mkt, 126) * np.sqrt(252.0)).astype(np.float32)
    f["rel_strength_63"] = (f["mom_63"] - roll_sum(mkt, 63)).astype(np.float32)

    for k in f:
        f[k] = np.where(np.isfinite(f[k]), f[k], np.nan).astype(np.float32)

    # Peer-relative: how this stock compares to the names that move like it.
    # Everything above is either self-referential or measured against the whole
    # universe; nothing asked the question an analyst asks first.
    if peers:
        try:
            from . import peers as peer_mod
            f.update(peer_mod.build(panel))
        except Exception as e:
            print("[features] peer-relative unavailable (%s: %s); continuing"
                  % (type(e).__name__, e))

    if macro:
        try:
            from . import macro as macro_mod
            m = macro_mod.build(panel.dates)
            f.update(macro_mod.broadcast(m, len(panel.tickers)))
        except Exception as e:
            print("[features] macro regime series unavailable (%s: %s); "
                  "continuing with cross-sectional features only"
                  % (type(e).__name__, e))

    # Everything above is derived from price and volume. Factor attribution
    # showed what that omission costs: the search loads -0.19 on RMW across
    # its best candidates -- a persistent bet on weak-profitability companies
    # placed by a process that cannot see profitability. These terminals are
    # keyed to SEC filing dates, never to the periods they describe.
    if fundamentals:
        try:
            from . import fundamentals as fund_mod
            f.update(fund_mod.build(panel))
        except Exception as e:
            print("[features] fundamentals unavailable (%s: %s); continuing"
                  % (type(e).__name__, e))
    return f


def _roll_moment(a: np.ndarray, w: int, order: int) -> np.ndarray:
    m = roll_mean(a, w)
    sd = roll_std(a, w)
    cen = roll_mean((a - m) ** order, w)
    return (cen / np.maximum(sd ** order, EPS)).astype(np.float32)


def _ema(a: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.full_like(a, np.nan, dtype=np.float32)
    prev = None
    for t in range(a.shape[0]):
        cur = a[t]
        if prev is None:
            prev = np.where(np.isfinite(cur), cur, np.nan)
        else:
            prev = np.where(np.isfinite(cur), alpha * cur + (1 - alpha) * np.where(
                np.isfinite(prev), prev, cur), prev)
        out[t] = prev
    return out


def _rsi(r1: np.ndarray, w: int) -> np.ndarray:
    up = roll_mean(np.where(r1 > 0, r1, 0.0), w)
    dn = roll_mean(np.where(r1 < 0, -r1, 0.0), w)
    rs = up / np.maximum(dn, EPS)
    return (100.0 - 100.0 / (1.0 + rs)).astype(np.float32)


FEATURE_NAMES = [
    "mom_1", "mom_5", "mom_10", "mom_21", "mom_63", "mom_126", "mom_252",
    "mom_12_1", "ma_ratio_10", "ma_ratio_21", "ma_ratio_50", "ma_ratio_200",
    "rev_5", "rev_21", "pct_52w_high", "pct_52w_range", "vol_10", "vol_21",
    "vol_63", "vol_126", "vol_ratio", "downside_vol_63", "skew_63", "kurt_63",
    "atr_14", "drawdown_63", "dollar_vol_21", "vol_surge", "amihud_21",
    "obv_slope_21", "rsi_14", "macd_hist", "bollinger_z", "ts_rank_63",
    "clv", "gap", "intraday_range", "close_strength", "beta_126",
    "idio_vol_126", "rel_strength_63",
]


# =============================================================================
#  Feature families
# =============================================================================
# The operator bandit in evolve.py learns which EDITS help. This grouping lets
# a second bandit learn which INPUTS generalise forward -- credited by whether
# strategies containing them survive the held-back validation tail. That is a
# level above operator credit: the search adapting what it looks at, not just
# how it mutates.
FAMILIES = {
    "momentum": ("mom_", "mom_12_1", "rel_strength"),
    "reversal": ("rev_", "ma_ratio_", "bollinger", "pct_52w"),
    "volatility": ("vol_", "downside_vol", "atr_", "drawdown_", "skew_", "kurt_"),
    "volume": ("dollar_vol", "vol_surge", "amihud", "obv_"),
    "oscillator": ("rsi_", "macd_", "ts_rank_"),
    "microstructure": ("clv", "gap", "intraday_range", "close_strength"),
    "market_relative": ("beta_", "idio_vol"),
    "macro": ("m_",),
    "peer": ("peer_",),
    "fundamental": ("f_",),
}


def family_of(name: str) -> str:
    """Which family a terminal belongs to. Unknown names land in 'other'."""
    for fam, prefixes in FAMILIES.items():
        for pre in prefixes:
            if name.startswith(pre) or name == pre.rstrip("_"):
                return fam
    return "other"


def family_map(names) -> dict:
    return {n: family_of(n) for n in names}
