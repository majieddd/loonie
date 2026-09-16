"""Purged, embargoed, block cross-validation for path-dependent strategies.

Standard k-fold is invalid here for two reasons:

  1. Returns are serially dependent and a strategy holds positions across
     fold boundaries, so a naive split lets the test fold inherit positions
     that were chosen using test-fold information.
  2. Features have lookback windows up to 252 days. A training sample dated
     one day before the test fold was computed from data inside it.

So: contiguous time blocks, an embargo gap on both sides of every test block,
and each fold is backtested as its own independent path starting flat.

A strategy's fitness is the *distribution* of alpha across folds, not its
mean. Something that makes all its money in one fold and loses in five is not
an edge -- it is one lucky quarter, and `frac_positive` is what catches it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import backtest as bt

EPS = 1e-12


@dataclass
class Fold:
    index: int
    start: int
    stop: int          # exclusive

    def __len__(self):
        return self.stop - self.start


def block_folds(n_obs: int, n_splits: int, embargo: int = 10,
                min_len: int = 120) -> list:
    """Contiguous test blocks with an embargo carved out of each edge.

    The embargo is removed from the *test* block so that positions inherited
    across the boundary, and features whose lookback straddles it, cannot
    contaminate the measurement.
    """
    n_splits = max(2, int(n_splits))
    edges = np.linspace(0, n_obs, n_splits + 1).astype(int)
    folds = []
    for i in range(n_splits):
        lo, hi = int(edges[i]), int(edges[i + 1])
        lo2 = lo + (embargo if i > 0 else 0)
        hi2 = hi - (embargo if i < n_splits - 1 else 0)
        if hi2 - lo2 >= min_len:
            folds.append(Fold(i, lo2, hi2))
    return folds


@dataclass
class CVResult:
    folds: list = field(default_factory=list)   # per-fold stat dicts
    returns: np.ndarray = None                  # (T,) stitched OOS returns
    bench: np.ndarray = None
    frac_positive: float = 0.0
    mean_alpha: float = 0.0
    median_alpha: float = 0.0
    mean_ir: float = 0.0
    worst_fold_alpha: float = 0.0
    alpha_tstat: float = 0.0
    total_trades: int = 0
    ann_turnover: float = 0.0
    corr_bench: float = 0.0
    ok: bool = True
    reason: str = ""

    def as_dict(self):
        return {
            "frac_positive": self.frac_positive,
            "mean_alpha": self.mean_alpha,
            "median_alpha": self.median_alpha,
            "mean_ir": self.mean_ir,
            "worst_fold_alpha": self.worst_fold_alpha,
            "alpha_tstat": self.alpha_tstat,
            "total_trades": self.total_trades,
            "ann_turnover": self.ann_turnover,
            "corr_bench": self.corr_bench,
            "n_folds": len(self.folds),
            "ok": self.ok,
            "reason": self.reason,
        }


def evaluate(panel, score: np.ndarray, genome, cfg,
             bench_ret: np.ndarray | None = None,
             folds: list | None = None) -> CVResult:
    """Backtest a genome independently on every fold and aggregate."""
    T = panel.shape[0]
    if folds is None:
        folds = block_folds(T, int(cfg.cv.n_splits), int(cfg.cv.embargo_days))
    if not folds:
        return CVResult(ok=False, reason="no usable folds")

    if bench_ret is None:
        bench_ret = bt.equal_weight_benchmark(panel)

    stats, stitched_r, stitched_b = [], [], []
    trades = 0
    turns = []
    for f in folds:
        sub = _slice_panel(panel, f.start, f.stop)
        res = bt.run(sub, score[f.start:f.stop], genome, cfg,
                     bench_ret=bench_ret[f.start:f.stop])
        if not res.ok:
            continue
        stats.append(res.stats)
        stitched_r.append(res.ret)
        stitched_b.append(res.bench_ret)
        trades += res.n_trades
        turns.append(res.stats["ann_turnover"])

    if not stats:
        return CVResult(ok=False, reason="every fold degenerate")

    alphas = np.array([s["alpha_ann"] for s in stats], dtype=np.float64)
    irs = np.array([s["ir"] for s in stats], dtype=np.float64)
    r = np.concatenate(stitched_r)
    b = np.concatenate(stitched_b)

    bv = float(np.var(b, ddof=1))
    beta = float(np.cov(r, b, ddof=1)[0, 1] / bv) if bv > EPS else 0.0
    t_alpha = bt._newey_west_tstat(r - beta * b)

    return CVResult(
        folds=stats,
        returns=r, bench=b,
        frac_positive=float(np.mean(alphas > 0)),
        mean_alpha=float(np.mean(alphas)),
        median_alpha=float(np.median(alphas)),
        mean_ir=float(np.mean(irs)),
        worst_fold_alpha=float(np.min(alphas)),
        alpha_tstat=float(t_alpha),
        total_trades=int(trades),
        ann_turnover=float(np.mean(turns)) if turns else 0.0,
        corr_bench=float(np.corrcoef(r, b)[0, 1]) if bv > EPS else 0.0,
    )


def _slice_panel(panel, lo: int, hi: int):
    from .data import Panel

    return Panel(
        dates=panel.dates[lo:hi],
        tickers=panel.tickers,
        bars={k: v[lo:hi] for k, v in panel.bars.items()},
        member=panel.member[lo:hi],
        tradable=panel.tradable[lo:hi],
        coverage=panel.coverage,
    )
