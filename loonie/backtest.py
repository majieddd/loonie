"""Cross-sectional backtester.

Two conventions do all the work of keeping this honest:

  TIMING.  `score[t]` is computed from data up to and including the close of
  day t. The resulting target weights are executed at that same close (a
  market-on-close order, which is a real thing you can submit) and therefore
  earn `ret[t+1]`. There is no path by which a decision sees its own outcome.

  DELISTING.  When a held name stops printing prices its return is frozen at
  zero, which is arithmetically identical to liquidating at the last traded
  price and sitting in cash. If that last price was $0.30 on the way to zero,
  the strategy eats the whole loss -- which is the entire reason the
  point-in-time universe exists.

The headline number is NOT total return. It is alpha against an equal-weight
portfolio of the same eligible universe. A strategy that picks 25 S&P names
will track the S&P; reporting its 30% CAGR as a result tells you nothing about
whether the picking did anything. `stats["alpha_ann"]` and `stats["ir"]` are
the numbers that can actually be wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRADING_DAYS = 252.0
EPS = 1e-12


@dataclass
class BacktestResult:
    dates: object
    ret: np.ndarray            # (T,) daily strategy return, net of costs
    bench_ret: np.ndarray      # (T,) daily benchmark return
    equity: np.ndarray
    bench_equity: np.ndarray
    turnover: np.ndarray       # (T,) one-way turnover charged that day
    n_trades: int = 0
    n_rebalances: int = 0
    avg_positions: float = 0.0
    stats: dict = field(default_factory=dict)
    ok: bool = True
    reason: str = ""

    @property
    def excess(self) -> np.ndarray:
        return self.ret - self.bench_ret


def _to_returns(close: np.ndarray) -> np.ndarray:
    prev = np.vstack([np.full((1, close.shape[1]), np.nan, np.float32), close[:-1]])
    r = (close - prev) / np.where(np.abs(prev) < EPS, np.nan, prev)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def equal_weight_benchmark(panel) -> np.ndarray:
    """Equal-weight portfolio of everything tradable. The bar to clear."""
    rets = _to_returns(panel.close)
    w = panel.tradable.astype(np.float32)
    prev_w = np.vstack([np.zeros((1, w.shape[1]), np.float32), w[:-1]])
    n = prev_w.sum(axis=1, keepdims=True)
    wn = prev_w / np.maximum(n, 1.0)
    return (wn * rets).sum(axis=1).astype(np.float64)


def run(panel, score: np.ndarray, genome, cfg,
        bench_ret: np.ndarray | None = None) -> BacktestResult:
    """Run one genome over one panel."""
    T, N = panel.shape
    rets = _to_returns(panel.close)
    tradable = panel.tradable
    k = int(genome.n_positions)
    rb = max(1, int(genome.rebalance_days))

    bps = (float(cfg.backtest.commission_bps) + float(cfg.backtest.slippage_bps)) / 1e4
    if bench_ret is None:
        bench_ret = equal_weight_benchmark(panel)

    rebalance_idx = list(range(0, T - 1, rb))
    port_ret = np.zeros(T, np.float64)
    turnover = np.zeros(T, np.float64)
    w_cur = np.zeros(N, np.float64)
    n_trades = 0
    pos_counts = []

    for bi, t0 in enumerate(rebalance_idx):
        t1 = rebalance_idx[bi + 1] if bi + 1 < len(rebalance_idx) else T - 1
        if t1 <= t0:
            continue

        # ---- select on information available at the close of t0 -----------
        s = score[t0]
        elig = tradable[t0] & np.isfinite(s)
        n_elig = int(elig.sum())
        if n_elig == 0:
            w_new = np.zeros(N, np.float64)
        else:
            kk = min(k, n_elig)
            masked = np.where(elig, s, -np.inf)
            top = np.argpartition(-masked, kk - 1)[:kk]
            top = top[np.isfinite(masked[top])]
            w_new = np.zeros(N, np.float64)
            if len(top):
                if genome.weighting == "score":
                    sv = masked[top]
                    sv = sv - sv.min() + EPS
                    w_new[top] = sv / sv.sum()
                else:
                    w_new[top] = 1.0 / len(top)
            pos_counts.append(len(top))

        # ---- cost of getting from the drifted book to the new one ---------
        tno = float(np.abs(w_new - w_cur).sum())
        turnover[t0] = tno
        n_trades += int((np.abs(w_new - w_cur) > 1e-6).sum())
        cost = tno * bps

        # ---- hold: value drifts with prices until the next rebalance ------
        blk = rets[t0 + 1:t1 + 1]                     # (L, N)
        held = np.nonzero(w_new > 0)[0]
        if len(held) == 0:
            w_cur = np.zeros(N, np.float64)
            port_ret[t0 + 1] -= cost
            continue

        cp = np.cumprod(1.0 + blk[:, held].astype(np.float64), axis=0)  # (L, k)
        pv = (w_new[held] * cp).sum(axis=1)                             # (L,)
        prev = np.concatenate([[w_new[held].sum()], pv[:-1]])
        blk_ret = pv / np.maximum(prev, EPS) - 1.0
        port_ret[t0 + 1:t1 + 1] = blk_ret
        port_ret[t0 + 1] -= cost

        w_cur = np.zeros(N, np.float64)
        w_cur[held] = w_new[held] * cp[-1] / max(pv[-1], EPS)

    equity = np.cumprod(1.0 + port_ret)
    bench_equity = np.cumprod(1.0 + bench_ret)
    res = BacktestResult(
        dates=panel.dates, ret=port_ret, bench_ret=bench_ret,
        equity=equity, bench_equity=bench_equity, turnover=turnover,
        n_trades=n_trades, n_rebalances=len(rebalance_idx),
        avg_positions=float(np.mean(pos_counts)) if pos_counts else 0.0,
    )
    res.stats = summarize(res)
    if res.avg_positions < 1:
        res.ok, res.reason = False, "no positions taken"
    return res


# =============================================================================
#  Summary statistics
# =============================================================================
def summarize(res: BacktestResult) -> dict:
    r, b = res.ret, res.bench_ret
    T = len(r)
    yrs = max(T / TRADING_DAYS, EPS)

    def cagr(x):
        v = float(np.prod(1.0 + x))
        return v ** (1.0 / yrs) - 1.0 if v > 0 else -1.0

    def sharpe(x):
        sd = float(np.std(x, ddof=1))
        return float(np.mean(x)) / max(sd, EPS) * np.sqrt(TRADING_DAYS)

    def maxdd(x):
        eq = np.cumprod(1.0 + x)
        return float((eq / np.maximum.accumulate(eq) - 1.0).min())

    # ---- the honest part: regress strategy on benchmark ------------------
    bv = float(np.var(b, ddof=1))
    beta = float(np.cov(r, b, ddof=1)[0, 1] / bv) if bv > EPS else 0.0
    resid = r - beta * b
    alpha_d = float(np.mean(resid))
    t_alpha = _newey_west_tstat(resid)

    exc = r - b
    ex_sd = float(np.std(exc, ddof=1))

    return {
        "cagr": cagr(r),
        "bench_cagr": cagr(b),
        "excess_cagr": cagr(r) - cagr(b),
        "sharpe": sharpe(r),
        "bench_sharpe": sharpe(b),
        "vol_ann": float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_dd": maxdd(r),
        "bench_max_dd": maxdd(b),
        "beta": beta,
        "alpha_ann": alpha_d * TRADING_DAYS,
        "alpha_tstat": t_alpha,
        "ir": float(np.mean(exc)) / max(ex_sd, EPS) * np.sqrt(TRADING_DAYS),
        "corr_bench": float(np.corrcoef(r, b)[0, 1]) if bv > EPS else 0.0,
        "ann_turnover": float(res.turnover.sum()) / yrs,
        "n_trades": res.n_trades,
        "avg_positions": res.avg_positions,
        "n_days": T,
        "hit_rate_vs_bench": float(np.mean(exc > 0)),
    }


def _newey_west_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """t-stat of the mean, robust to autocorrelation and heteroskedasticity.

    Daily strategy returns are autocorrelated (overlapping holding periods),
    and an OLS t-stat will happily tell you a coin is loaded. Newey-West is
    the minimum honest correction here.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 20:
        return 0.0
    lags = lags or max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    e = x - x.mean()
    s = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * float(e[L:] @ e[:-L]) / n
    se = np.sqrt(max(s, EPS) / n)
    return float(x.mean() / max(se, EPS))
