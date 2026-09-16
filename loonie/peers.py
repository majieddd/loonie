"""Peer-relative features: how a stock compares to the stocks that move like it.

The 57 existing terminals are either self-referential — a stock's own momentum,
volatility, oscillators — or market-relative, measured against the whole
eligible universe. None of them asks the question an equity analyst asks first:
*compared to its peers*, is this cheap, strong, or unusual?

That gap matters because the effect is well documented. A semiconductor name up
8% in a week when every semiconductor is up 8% has done nothing; the same move
while its peers are flat is information. A feature set that cannot tell those
apart is throwing away the distinction, and every strategy built from it
inherits the blindness.

PEERS WITHOUT SECTOR DATA. Proper industry classifications (GICS, SIC) are
licensed. But the thing a sector code approximates — "these securities are
driven by common factors" — is directly measurable from returns. Each stock's
peers here are simply the M names whose trailing returns it correlates with
most. That is arguably closer to what we actually want than a sector label:
correlation reflects how the market is *currently* grouping names, and it
re-forms when the market re-forms, which a static classification cannot do.

CAUSALITY. Peer sets are recomputed on a block schedule, and the correlations
for a block are computed strictly from data BEFORE that block begins. A peer
set chosen using the returns it is about to be evaluated on would be a
beautifully disguised way of looking at the answer, and it would be invisible
in every other test.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
EPS = 1e-12

PEER_NAMES = [
    "peer_rel_mom_5", "peer_rel_mom_21", "peer_rel_mom_63",
    "peer_rel_vol", "peer_rank_mom_21", "peer_dispersion",
    "peer_beta", "peer_corr", "peer_rel_range",
]


def _corr_matrix(r: np.ndarray) -> np.ndarray:
    """Correlation across columns of a (T, N) return block, NaN-safe."""
    x = np.where(np.isfinite(r), r, 0.0).astype(np.float64)
    ok = np.isfinite(r).sum(axis=0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = np.sqrt((x ** 2).sum(axis=0))
    sd = np.where(sd > EPS, sd, np.inf)          # dead columns -> zero corr
    c = (x.T @ x) / np.outer(sd, sd)
    c[~np.isfinite(c)] = 0.0
    c[ok < max(20, r.shape[0] // 4), :] = 0.0    # too little data to trust
    c[:, ok < max(20, r.shape[0] // 4)] = 0.0
    np.fill_diagonal(c, -np.inf)                 # never a peer of itself
    return c


def build(panel, m: int = 20, lookback: int = 252, refresh: int = 63) -> dict:
    """Peer-relative features over the panel. All causal.

    `m`         peers per stock
    `lookback`  trailing sessions used to measure who the peers are
    `refresh`   how often the peer sets are recomputed (quarterly by default)
    """
    from . import features as F

    close = panel.bars["close"].astype(np.float32)
    high = panel.bars["high"].astype(np.float32)
    low = panel.bars["low"].astype(np.float32)
    T, N = close.shape
    r1 = F.pct_change(close, 1)

    # Precompute the per-stock inputs once; the peer step only averages them.
    mom = {w: F.pct_change(close, w) for w in (5, 21, 63)}
    vol21 = F.roll_std(r1, 21)
    rng = ((high - low) / np.maximum(close, EPS)).astype(np.float32)

    out = {k: np.full((T, N), np.nan, np.float32) for k in PEER_NAMES}

    starts = list(range(lookback, T, refresh))
    if not starts:
        return out                                # panel too short for peers

    for s in starts:
        stop = min(s + refresh, T)
        # Peers are chosen from data strictly BEFORE this block.
        c = _corr_matrix(r1[max(0, s - lookback):s])
        idx = np.argpartition(-c, min(m, N - 1) - 1, axis=1)[:, :m]   # (N, m)

        blk = slice(s, stop)
        rows = np.arange(stop - s)[:, None, None]

        def peer_mean(a):
            g = a[blk][rows, idx[None, :, :]]     # (L, N, m)
            return np.nanmean(g, axis=2)

        for w in (5, 21, 63):
            out["peer_rel_mom_%d" % w][blk] = mom[w][blk] - peer_mean(mom[w])

        pv = peer_mean(vol21)
        out["peer_rel_vol"][blk] = vol21[blk] / np.maximum(pv, EPS) - 1.0
        out["peer_rel_range"][blk] = rng[blk] - peer_mean(rng)

        # Rank within the peer group: 0 = worst of its cohort, 1 = best.
        g = mom[21][blk][rows, idx[None, :, :]]
        out["peer_rank_mom_21"][blk] = np.nanmean(
            (g < mom[21][blk][:, :, None]).astype(np.float32), axis=2)

        # How tightly the cohort is moving together. A dispersed peer group is
        # a different trading environment from a lockstep one.
        out["peer_dispersion"][blk] = np.nanstd(
            mom[21][blk][rows, idx[None, :, :]], axis=2)

        # Beta and correlation to the peer cohort rather than to the market:
        # the residual after removing the cohort is the stock-specific part.
        #
        # Measured on the PRE-BLOCK window, not the block itself. An earlier
        # version averaged across the whole block and wrote one value to every
        # row in it, so the number at the block's start was computed partly
        # from returns at its end. The peer SET was causal and the statistic
        # built on it was not -- which is the kind of leak that survives every
        # test except one that truncates the panel and re-runs.
        pre = slice(max(0, s - lookback), s)
        hist_rows = np.arange(pre.stop - pre.start)[:, None, None]
        pr = np.nanmean(r1[pre][hist_rows, idx[None, :, :]], axis=2)  # (W, N)
        own = r1[pre]
        pmn = np.nanmean(pr, axis=0, keepdims=True)
        omn = np.nanmean(own, axis=0, keepdims=True)
        cov = np.nanmean((own - omn) * (pr - pmn), axis=0)
        vp = np.nanmean((pr - pmn) ** 2, axis=0)
        vo = np.nanmean((own - omn) ** 2, axis=0)
        out["peer_beta"][blk] = (cov / np.maximum(vp, EPS))[None, :]
        out["peer_corr"][blk] = (
            cov / np.maximum(np.sqrt(vp * vo), EPS))[None, :]

    for k in out:
        out[k] = np.where(np.isfinite(out[k]), out[k], np.nan).astype(np.float32)
    return out
