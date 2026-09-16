"""Fama-French factor attribution: is the alpha real, or is it a tilt?

Every result this system reports measures the strategy against an equal-weight
universe benchmark. That benchmark answers one question -- did stock selection
beat holding everything -- and it is silent on a second question that matters
just as much: *how* did it beat it.

A strategy that systematically holds smaller, cheaper, higher-momentum names
will beat an equal-weight benchmark over most windows. It will show positive
alpha and a respectable t-stat. And it will be worth nothing, because those
three exposures are sold in ETF form for a few basis points. Paying a search
process to rediscover the size premium is not an edge; it is an expensive way
to buy SMB.

The only way to tell the two apart is to regress the strategy's returns on the
known factors and look at what survives. What survives is the part no published
factor explains. That residual is the only thing worth defending.

    strategy excess  =  a  +  b1*(Mkt-RF) + b2*SMB + b3*HML
                             + b4*RMW + b5*CMA + b6*Mom  +  e

`a` is the alpha that matters. If the benchmark-relative t-stat is 2.4 and the
factor-relative t-stat is 0.3, the search has found a factor portfolio and
nothing else -- and the first number was never evidence of skill.

WHY THIS DATA. The factors come from Ken French's data library, built on CRSP,
published for exactly this use. Two properties make them the right yardstick:
they are constructed from a survivorship-free universe that includes delisted
securities (which our own price panel is not), and they are fixed, public
series that we cannot tune. A benchmark you could accidentally fit to is not a
benchmark.

CAUSALITY. Attribution is a diagnostic applied to returns that already exist;
it never enters the search, sets no thresholds, and generates no features. It
answers a question about a result rather than helping produce one, which is why
it is safe for it to see the whole window at once.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .config import resolve

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
SOURCES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
CACHE = "data/cache/factors.parquet"
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# Ken French encodes absent observations as these sentinels. Reading one as a
# -99.99% daily return would not be a subtle error, but it would be silent.
MISSING = (-99.99, -999.0, -99.99e2)


def _download(fname: str) -> pd.DataFrame:
    req = urllib.request.Request(
        BASE + fname, headers={"User-Agent": "loonie-research (academic use)"})
    with urllib.request.urlopen(req, timeout=90) as fh:
        blob = fh.read()

    z = zipfile.ZipFile(io.BytesIO(blob))
    raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()

    # The files carry a prose header of varying length, then the column row,
    # then data, then sometimes an annual section appended below the daily one.
    # Anchoring on "first line whose first field is 8 digits" is robust to all
    # three, where a fixed skiprows is not.
    start = None
    for i, line in enumerate(raw):
        head = line.split(",")[0].strip()
        if len(head) == 8 and head.isdigit():
            start = i
            break
    if start is None:
        raise RuntimeError("no daily rows found in %s" % fname)

    cols = [c.strip() for c in raw[start - 1].split(",")]
    cols[0] = "date"

    rows, vals = [], []
    for line in raw[start:]:
        parts = [p.strip() for p in line.split(",")]
        head = parts[0]
        if len(head) != 8 or not head.isdigit():
            break                      # end of the daily block
        try:
            vals.append([float(p) for p in parts[1:len(cols)]])
            rows.append(head)
        except ValueError:
            break

    df = pd.DataFrame(vals, index=pd.to_datetime(rows, format="%Y%m%d"),
                      columns=cols[1:])
    for s in MISSING:
        df = df.mask(np.isclose(df, s))
    return df / 100.0             # published in percent; we work in fractions


def fetch(refresh: bool = False, max_age_days: float = 7.0) -> pd.DataFrame:
    """Daily factor returns, cached. Falls back to the cache on any failure."""
    p = resolve(CACHE)
    if p.exists() and not refresh:
        age_d = (time.time() - p.stat().st_mtime) / 86400.0
        if age_d < max_age_days:
            try:
                return pd.read_parquet(p)
            except Exception:
                pass

    try:
        parts = [_download(f) for f in SOURCES.values()]
        df = pd.concat(parts, axis=1).sort_index()
        df = df.loc[:, ~df.columns.duplicated()]
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        return df
    except Exception:
        # A network failure must not take down a search that is otherwise
        # fine. Stale factors still attribute; no factors just means the
        # diagnostic is skipped.
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        raise


# =============================================================================
#  Attribution
# =============================================================================
def _nw_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficients.

    Daily strategy returns are autocorrelated -- a 5-day rebalance holds the
    same book for a week -- and plain OLS standard errors would understate the
    uncertainty on exactly the coefficient we care about.
    """
    T, k = X.shape
    xu = X * u[:, None]
    S = (xu.T @ xu) / T
    for l in range(1, lags + 1):
        G = (xu[l:].T @ xu[:-l]) / T
        S += (1.0 - l / (lags + 1.0)) * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    return XtX_inv @ (T * S) @ XtX_inv


def attribute(ret: np.ndarray, dates, is_excess: bool = True,
              factors: pd.DataFrame | None = None,
              lags: int | None = None) -> dict:
    """Regress a daily return stream on the six published factors.

    `is_excess` says whether `ret` is already a spread over a benchmark. A
    spread is self-financing, so the risk-free rate is not subtracted from it;
    a raw long-only return is, or the market beta absorbs the whole T-bill
    yield and the alpha is wrong by that amount.
    """
    try:
        f = fetch() if factors is None else factors
    except Exception as e:
        return {"ok": False, "reason": "factors unavailable: %s" % e}

    idx = pd.DatetimeIndex(dates)
    y = pd.Series(np.asarray(ret, dtype=np.float64), index=idx)

    have = [c for c in FACTORS if c in f.columns]
    if len(have) < 3:
        return {"ok": False, "reason": "only %d factors present" % len(have)}

    join = f.reindex(idx)[have + (["RF"] if "RF" in f.columns else [])]
    ok = np.isfinite(y.to_numpy()) & np.isfinite(join[have].to_numpy()).all(1)
    n = int(ok.sum())
    # The factor library publishes on a lag of a few weeks, so the tail of a
    # live panel has no factors yet. That shortens the sample; it does not
    # invalidate it, as long as we say how much was usable.
    if n < 120:
        return {"ok": False, "reason": "only %d overlapping sessions" % n,
                "n": n}

    yv = y.to_numpy()[ok]
    if not is_excess and "RF" in join.columns:
        yv = yv - np.nan_to_num(join["RF"].to_numpy()[ok])

    F = join[have].to_numpy()[ok]
    X = np.column_stack([np.ones(n), F])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ yv)
    resid = yv - X @ beta

    L = lags if lags is not None else max(5, int(round(4 * (n / 100.0) ** 0.25)))
    se = np.sqrt(np.maximum(np.diag(_nw_cov(X, resid, L)), 0.0))

    tss = float(((yv - yv.mean()) ** 2).sum())
    rss = float((resid ** 2).sum())

    return {
        "ok": True,
        "n": n,
        "span": "%s..%s" % (idx[ok][0].date(), idx[ok][-1].date()),
        "alpha_daily": float(beta[0]),
        "alpha_ann": float(beta[0] * 252.0),
        "alpha_tstat": float(beta[0] / se[0]) if se[0] > 0 else 0.0,
        "betas": {name: float(b) for name, b in zip(have, beta[1:])},
        "beta_tstats": {name: (float(b / s) if s > 0 else 0.0)
                        for name, b, s in zip(have, beta[1:], se[1:])},
        "r2": 1.0 - rss / tss if tss > 0 else 0.0,
        "resid_vol_ann": float(np.std(resid, ddof=1) * np.sqrt(252.0)),
        "factors_used": have,
        "nw_lags": L,
    }


def explain(a: dict) -> str:
    """One line a human can read without decoding the dict."""
    if not a.get("ok"):
        return "factor attribution unavailable (%s)" % a.get("reason", "?")
    big = sorted(a["betas"].items(), key=lambda kv: -abs(kv[1]))[:3]
    tilt = ", ".join("%s %+.2f" % (k, v) for k, v in big)
    return ("factor alpha %+.2f%%/yr (t %+.2f) over %d sessions; "
            "R2 %.2f; largest tilts: %s"
            % (100 * a["alpha_ann"], a["alpha_tstat"], a["n"], a["r2"], tilt))


def verdict(bench_t: float, a: dict, floor: float = 0.5) -> str:
    """Compare a benchmark-relative t-stat against the factor-relative one.

    The interesting case is the third: a result that looked significant
    against the equal-weight benchmark and evaporates once the published
    factors are allowed to explain it.
    """
    if not a.get("ok"):
        return "unknown"
    ft = a["alpha_tstat"]
    if ft >= 2.0 and bench_t >= 2.0:
        return "survives factors"
    if ft >= 2.0 > bench_t:
        return "factor-hedged gain"
    if bench_t >= 2.0 and abs(ft) < 2.0:
        return "explained by factors"
    if abs(ft) < floor and abs(bench_t) < floor:
        return "no signal either way"
    return "inconclusive"
