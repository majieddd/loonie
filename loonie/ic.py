"""Cross-sectional information coefficient: the statistic that has power here.

The promotion gate used to turn on a portfolio alpha t-stat. Measuring the
power of that test showed it cannot resolve a realistic edge: 5.6 years at
14.8% tracking error needs +12.5% a year to reach t = 2, and a genuine 3%/yr
edge scores 0.48. Four honest walk-forwards returned "no edge", and every one
of them was equally consistent with a dead search and a good one.

The information coefficient does not have that problem, because it does not
throw the cross-section away. A portfolio return compresses 500 ranked names
into one number per day and then asks whether 1,400 of those numbers trend up.
The IC asks, on each of those days, whether the ranking of all 500 names
matched what actually happened next -- and there are roughly 2,700 such days.
Same data, far more of it used, and the noise of which 25 names happened to be
held never enters.

    IC_t = spearman( score[t] , forward return over [t+1, t+1+h] )

averaged over t, with a t-stat that accounts for the overlap h introduces.

WHY THE THRESHOLD MOVES. A fixed bar is the wrong shape for a search that has
evaluated two hundred thousand genomes. The largest |t| among k independent
draws from the null grows like sqrt(2 ln k), so the honest bar is not a
constant -- it is a function of how hard the search has looked. At 35,000
effective trials that is 4.57, and it rises as the search continues.

This is the multiple-testing correction applied where it belongs: to the
threshold, continuously, rather than as a footnote at the end. A candidate
clears it by being better than the best thing the search would have found in
pure noise having looked exactly as hard as it has.

CAUSALITY. Forward returns start at t+1. The score at t is built from features
that close at t. The one-bar gap is the same convention the backtest uses, and
it is what makes the number tradable rather than contemporaneous.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Return from t+1 to t+1+h, aligned so row t is knowable only after t.

    The score at t is acted on at t+1, so the return it should be judged
    against begins at t+1. Starting at t would score a signal against a bar it
    is already inside.
    """
    c = np.asarray(close, dtype=np.float64)
    T = len(c)
    out = np.full(c.shape, np.nan)
    h = max(1, int(horizon))
    if T <= h + 1:
        return out
    base = c[1:T - h]
    fut = c[1 + h:]
    denom = np.where(np.abs(base) > EPS, base, np.nan)
    out[:T - h - 1] = fut / denom - 1.0
    return out


def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise rank in [0,1], NaN preserved, ties averaged.

    Ties are averaged rather than broken. A score with many equal values --
    a flag, a clipped ratio, a fundamental that has not been refiled -- would
    otherwise be ordered by column index, and the IC would partly measure the
    alphabet.
    """
    import pandas as pd

    return pd.DataFrame(a).rank(axis=1, method="average", pct=True).to_numpy()


def ic_series(score: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
              min_names: int = 20) -> np.ndarray:
    """Daily cross-sectional Spearman IC. NaN on days too thin to score."""
    x = np.where(mask, score, np.nan).astype(np.float64)
    y = np.where(mask, fwd, np.nan).astype(np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(both, x, np.nan), np.where(both, y, np.nan)

    rx, ry = _rank_rows(x), _rank_rows(y)
    n = both.sum(axis=1)

    with np.errstate(invalid="ignore"):
        mx = np.nanmean(rx, axis=1, keepdims=True)
        my = np.nanmean(ry, axis=1, keepdims=True)
        dx = np.where(both, rx - mx, 0.0)
        dy = np.where(both, ry - my, 0.0)
        num = (dx * dy).sum(axis=1)
        den = np.sqrt((dx ** 2).sum(axis=1) * (dy ** 2).sum(axis=1))
        ic = np.where(den > EPS, num / np.maximum(den, EPS), np.nan)
    return np.where(n >= min_names, ic, np.nan)


def _nw_tstat(x: np.ndarray, lags: int) -> float:
    """t-stat of the mean, robust to the autocorrelation overlap induces.

    Consecutive ICs at horizon h share h-1 days of the same forward window, so
    they are mechanically correlated. Treating them as independent would
    inflate the t-stat by roughly sqrt(h) -- which at h=10 is a factor of
    three, and the difference between a gate that means something and one that
    passes everything.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 20:
        return 0.0
    d = x - x.mean()
    var = float((d ** 2).sum() / n)
    if var <= EPS:
        # A series with no spread has an undefined t-stat, not a zero one.
        # Returning 0.0 would score a signal that ranked the cross-section
        # correctly every single day as worthless. Degenerate either way, so
        # say which degenerate case it is rather than flattening both to the
        # same number.
        m = float(x.mean())
        return 0.0 if abs(m) <= EPS else math.copysign(float("inf"), m)
    s = var
    for l in range(1, min(lags, n - 1) + 1):
        cov = float((d[l:] * d[:-l]).sum() / n)
        s += 2.0 * (1.0 - l / (lags + 1.0)) * cov
    s = max(s, EPS)
    return float(x.mean() / math.sqrt(s / n))


def summarize(score: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
              horizon: int, min_names: int = 20) -> dict:
    """Everything the gate needs from one genome's score matrix."""
    s = ic_series(score, fwd, mask, min_names)
    ok = np.isfinite(s)
    n = int(ok.sum())
    if n < 60:
        return {"ic": float("nan"), "ic_t": float("nan"), "ic_n": n,
                "ic_ir": float("nan"), "ic_hit": float("nan"),
                "ok": False, "reason": "only %d scoreable days" % n}

    v = s[ok]
    # Overlap runs h-1 days, so that is the window the HAC estimator needs to
    # cover; a little extra costs precision, too little overstates certainty.
    lags = max(1, int(horizon))
    sd = float(v.std(ddof=1))
    return {
        "ic": float(v.mean()),
        "ic_t": _nw_tstat(v, lags),
        "ic_n": n,
        "ic_ir": float(v.mean() / sd) if sd > EPS else float("nan"),
        "ic_hit": float(np.mean(v > 0)),
        "ok": True,
        "reason": "",
    }


def null_bar(n_effective: float, floor: float = 2.0) -> float:
    """The largest |t| a search this size would find in pure noise.

    The maximum of k independent standard normals concentrates near
    sqrt(2 ln k). A candidate that does not beat it has not outperformed the
    null -- it has outperformed nothing, k times, and been graded on the best
    attempt.

    Held at `floor` from below so a young search still has to clear ordinary
    significance before the multiple-testing term takes over.
    """
    k = max(2.0, float(n_effective))
    return float(max(floor, math.sqrt(2.0 * math.log(k))))
