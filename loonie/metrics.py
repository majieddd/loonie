"""Multiple-testing corrections. The most important module in the project.

If you try one strategy and it has a Sharpe of 1.5, that is interesting.
If you try 60,000 strategies and the best has a Sharpe of 1.5, that is
arithmetic. Under the null that every strategy is worthless, the expected
maximum Sharpe over N independent trials grows like sqrt(2 log N) -- with
N = 60,000 you should *expect* a best-of-sample Sharpe near 1.4 from pure
noise. Reporting it as a discovery is the single most common way backtests lie.

The video's loop ran 14 LLM-proposed variants and kept the one that improved
the backtest. That is a 14-trial search with no correction applied, which is
why the sealed-window result came back indistinguishable from random. This
module is how that failure mode gets priced instead of discovered at the end.

References:
  Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
  Bailey, Borwein, Lopez de Prado & Zhu (2016), "The Probability of Backtest
    Overfitting" (CSCV)
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

EPS = 1e-12
EULER = 0.5772156649015329
TRADING_DAYS = 252.0


# =============================================================================
#  Expected maximum Sharpe under the null
# =============================================================================
def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] over `n_trials` worthless strategies whose SR estimates have
    variance `var_sr`. This is the bar a "discovery" has to clear.

    Both SR and the return are per-period (daily), not annualised.
    """
    n = max(int(n_trials), 2)
    sd = np.sqrt(max(var_sr, EPS))
    a = stats.norm.ppf(1.0 - 1.0 / n)
    b = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe_ratio(sr: float, n_obs: int, n_trials: int,
                          var_sr: float | None = None,
                          skew: float = 0.0, kurt: float = 3.0,
                          sr_benchmark: float | None = None) -> float:
    """Probability that the true Sharpe exceeds the selection threshold.

    `sr` is the per-period (daily) Sharpe of the selected strategy.
    Returns a probability in [0, 1]; treat < 0.95 as "not established" for a
    single decision, and use the configured gate for search-time promotion.
    """
    if n_obs < 20:
        return 0.0
    if var_sr is None:
        # Variance of the SR estimator itself under the null (Lo, 2002).
        var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / max(n_obs - 1, 1)
    sr_star = (expected_max_sharpe(n_trials, var_sr)
               if sr_benchmark is None else sr_benchmark)

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    denom = max(denom, EPS)
    z = (sr - sr_star) * np.sqrt(max(n_obs - 1, 1)) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def dsr_from_returns(r: np.ndarray, n_trials: int,
                     benchmark: np.ndarray | None = None) -> dict:
    """Convenience wrapper: deflate the Sharpe of a daily return series.

    When `benchmark` is supplied the deflation is applied to the EXCESS return
    series, which is the quantity that has to be non-zero for stock picking to
    have added anything.
    """
    x = np.asarray(r, dtype=np.float64)
    if benchmark is not None:
        x = x - np.asarray(benchmark, dtype=np.float64)
    n = len(x)
    if n < 20:
        return {"dsr": 0.0, "sr_daily": 0.0, "sr_ann": 0.0, "sr_star_ann": 0.0}
    sd = float(np.std(x, ddof=1))
    sr = float(np.mean(x)) / max(sd, EPS)
    sk = float(stats.skew(x))
    ku = float(stats.kurtosis(x, fisher=False))
    var_sr = (1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2) / max(n - 1, 1)
    return {
        "dsr": deflated_sharpe_ratio(sr, n, n_trials, var_sr, sk, ku),
        "sr_daily": sr,
        "sr_ann": sr * np.sqrt(TRADING_DAYS),
        "sr_star_ann": expected_max_sharpe(n_trials, var_sr) * np.sqrt(TRADING_DAYS),
        "skew": sk,
        "kurtosis": ku,
        "n_trials": int(n_trials),
    }


def min_track_record_length(sr: float, n_obs: int, skew: float = 0.0,
                            kurt: float = 3.0, target_sr: float = 0.0,
                            conf: float = 0.95) -> float:
    """How many observations you'd need for this SR to be significant."""
    if sr <= target_sr:
        return float("inf")
    z = stats.norm.ppf(conf)
    num = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return float(1.0 + num * (z / (sr - target_sr)) ** 2)


# =============================================================================
#  Probability of Backtest Overfitting (CSCV)
# =============================================================================
def pbo_cscv(returns: np.ndarray, n_partitions: int = 8,
             max_combos: int = 200) -> dict:
    """Probability that the in-sample best strategy underperforms out-of-sample.

    `returns` is (T, M): T periods, M candidate strategies.

    Method: chop the timeline into S even blocks; for every way of splitting
    those blocks into half-IS / half-OOS, pick the strategy that won IS and
    look up where it ranked OOS. If the IS winner lands below the OOS median
    more than half the time, your selection procedure is anti-predictive.

    PBO near 0.5 means your search is a coin flip. PBO above ~0.4 means the
    thing you are about to trade was selected by noise.
    """
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] < 2:
        return {"pbo": float("nan"), "n_combos": 0, "logits": []}

    S = int(n_partitions)
    if S % 2:
        S += 1
    T, M = R.shape
    if T < S * 10:
        S = max(2, (T // 10) // 2 * 2)
    if S < 2:
        return {"pbo": float("nan"), "n_combos": 0, "logits": []}

    blocks = np.array_split(np.arange(T), S)
    logits = []
    combos = list(itertools.combinations(range(S), S // 2))
    if len(combos) > max_combos:
        idx = np.linspace(0, len(combos) - 1, max_combos).astype(int)
        combos = [combos[i] for i in idx]

    for is_blocks in combos:
        oos_blocks = [b for b in range(S) if b not in is_blocks]
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_sr = _sharpe_cols(R[is_idx])
        oos_sr = _sharpe_cols(R[oos_idx])
        if not np.isfinite(is_sr).any():
            continue

        best = int(np.nanargmax(is_sr))
        finite = np.isfinite(oos_sr)
        if finite.sum() < 2 or not finite[best]:
            continue
        # relative rank of the IS winner within the OOS distribution
        rank = float((oos_sr[finite] <= oos_sr[best]).sum()) / float(finite.sum())
        w = min(max(rank, 1.0 / (M + 1.0)), 1.0 - 1.0 / (M + 1.0))
        logits.append(float(np.log(w / (1.0 - w))))

    if not logits:
        return {"pbo": float("nan"), "n_combos": 0, "logits": []}
    lg = np.array(logits)
    return {
        "pbo": float(np.mean(lg <= 0.0)),
        "n_combos": len(lg),
        "median_logit": float(np.median(lg)),
        "logits": lg.tolist(),
    }


def effective_trials(returns: np.ndarray, n_total: int) -> dict:
    """How many *independent* hypotheses were really tested?

    Deflation asks "how many chances did you give yourself to get lucky?", and
    naively that is the number of candidates evaluated. But a genetic program's
    children are near-copies of their parents: evaluating 1,200 genomes that
    are 95% the same expression is not 1,200 independent shots at the data, and
    charging for 1,200 makes the correction so punitive that nothing can ever
    clear it -- which is its own kind of dishonesty.

    So: build the correlation matrix of candidate return streams and take its
    participation ratio (sum of eigenvalues, squared, over sum of squared
    eigenvalues). That is the standard effective-number-of-tests estimator from
    genome-wide association studies, where the identical problem appears as
    correlated SNPs. M perfectly correlated candidates give 1; M independent
    ones give M. Scale the observed ratio up to the full trial count.

    Returns both numbers, because the gap between them is itself worth seeing.
    """
    R = np.asarray(returns, dtype=np.float64)
    n_total = max(int(n_total), 1)
    if R.ndim != 2 or R.shape[1] < 3:
        return {"n_total": n_total, "n_effective": float(n_total),
                "independence": 1.0}

    sd = R.std(axis=0, ddof=1)
    keep = sd > EPS
    if keep.sum() < 3:
        return {"n_total": n_total, "n_effective": float(n_total),
                "independence": 1.0}

    C = np.corrcoef(R[:, keep], rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    denom = float((ev ** 2).sum())
    if denom <= EPS:
        return {"n_total": n_total, "n_effective": float(n_total),
                "independence": 1.0}

    m = int(keep.sum())
    n_eff_sample = float(ev.sum() ** 2) / denom
    independence = min(1.0, max(1.0 / m, n_eff_sample / m))
    return {
        "n_total": n_total,
        "n_effective": float(max(1.0, n_total * independence)),
        "independence": independence,
        "sample_effective": n_eff_sample,
        "sample_size": m,
    }


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=1)
    return np.where(sd > EPS, mu / np.maximum(sd, EPS), np.nan)


# =============================================================================
#  Haircuts and helpers
# =============================================================================
def haircut_sharpe(sr_ann: float, n_trials: int, n_years: float) -> float:
    """Bonferroni-style haircut: what annual Sharpe survives the search?"""
    n_obs = max(int(n_years * TRADING_DAYS), 20)
    sr_d = sr_ann / np.sqrt(TRADING_DAYS)
    var_sr = (1.0 + 0.5 * sr_d ** 2) / max(n_obs - 1, 1)
    return float(max(0.0, sr_d - expected_max_sharpe(n_trials, var_sr))
                 * np.sqrt(TRADING_DAYS))


def summarize_gate(name: str, value: float, op: str, threshold: float) -> dict:
    ok = (value >= threshold) if op == ">=" else (value <= threshold)
    return {"gate": name, "value": float(value), "op": op,
            "threshold": float(threshold), "pass": bool(ok)}
