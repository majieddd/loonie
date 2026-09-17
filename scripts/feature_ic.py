"""Rank every terminal by how much it predicts, one at a time.

    python scripts/feature_ic.py [--horizon 5] [--json state/feature_ic.json]

Before a feature earns a place in the hypothesis space it should be asked the
simplest possible question: cross-sectionally, does today's value have any
relationship to tomorrow's return? That is the information coefficient -- the
Spearman correlation between a feature's cross-section at t and forward returns
from t to t+h, averaged over time.

This is a diagnostic, not a gate. Two reasons it must not become one:

A feature with no standalone IC can still be valuable. The grammar composes --
`ite(m_vix, A, B)` uses a regime terminal that predicts nothing by itself to
switch between two that do. Filtering terminals on solo IC would delete exactly
the conditioning variables that make conditioning possible.

And the reverse: selecting features by IC on the same window the search then
runs on is selection on the test set, one level up. The number reported here
describes what is in the data; it is not permission to go looking for it.

What it is honestly good for is catching the opposite failure -- a whole family
that is empty, mis-scaled, or misaligned. A fundamentals block that reads 0.00
across fourteen terminals is not a finding about markets, it is a bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from loonie import backtest as bt  # noqa: E402
from loonie import features as F  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402
from loonie import seal  # noqa: E402


def _rank(a: np.ndarray) -> np.ndarray:
    """Row-wise rank in [0,1], NaN preserved. Ties share their mean rank.

    Ties are averaged rather than broken arbitrarily. A feature with many
    equal values -- a flag, a clipped ratio, a stale fundamental that has not
    been refiled -- would otherwise be ranked by column order, and the IC
    would be measuring the alphabet.
    """
    import pandas as pd

    return pd.DataFrame(a).rank(axis=1, method="average", pct=True).to_numpy()


def ic_series(feat: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
              min_names: int = 20) -> np.ndarray:
    """Daily cross-sectional Spearman IC between a feature and forward return.

    Ranks are recomputed on the pairwise-complete cross-section for each day,
    so a day where a feature covers 80 names is scored on those 80 -- not on
    ranks borrowed from a wider set the return vector does not cover.
    """
    x = np.where(mask, feat, np.nan).astype(np.float64)
    y = np.where(mask, fwd, np.nan).astype(np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(both, x, np.nan), np.where(both, y, np.nan)

    rx, ry = _rank(x), _rank(y)
    n = both.sum(axis=1).astype(np.float64)

    with np.errstate(invalid="ignore"):
        mx = np.nanmean(rx, axis=1, keepdims=True)
        my = np.nanmean(ry, axis=1, keepdims=True)
        dx, dy = np.where(both, rx - mx, 0.0), np.where(both, ry - my, 0.0)
        num = (dx * dy).sum(axis=1)
        den = np.sqrt((dx ** 2).sum(axis=1) * (dy ** 2).sum(axis=1))
        ic = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
    return np.where(n >= min_names, ic, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--json", default="state/feature_ic.json")
    ap.add_argument("--min-obs", type=int, default=200)
    ap.add_argument("--include-holdout", action="store_true",
                    help="read the sealed window too; logged to the ledger")
    args = ap.parse_args()

    load_env()
    cfg = load()
    panel = seal.training_panel(cfg, by="scripts/feature_ic.py",
                                include_holdout=args.include_holdout)
    feats = F.build(panel)
    print("[ic] panel %s..%s | %d names | %d terminals"
          % (panel.dates[0].date(), panel.dates[-1].date(),
             len(panel.tickers), len(feats)))

    r1 = bt._to_returns(panel.close)
    h = args.horizon
    # Forward return from t to t+h, aligned so row t holds a return that
    # starts at t+1. Anything else would score a feature against a window it
    # is already inside.
    fwd = np.full_like(r1, np.nan)
    cum = np.cumprod(1.0 + r1, axis=0)
    fwd[:-h - 1] = cum[h + 1:] / np.maximum(cum[1:-h], 1e-12) - 1.0
    mask = panel.tradable

    rows = []
    for name in sorted(feats):
        s = ic_series(feats[name], fwd, mask)
        ok = np.isfinite(s)
        n = int(ok.sum())
        if n < args.min_obs:
            # A macro series is one number a day broadcast across every name,
            # so its cross-section has no spread and a cross-sectional IC is
            # undefined -- not small, undefined. Saying "too few sessions"
            # would read as a data gap and send someone off to fix nothing.
            #
            # "Constant" has to be judged against float32 resolution, not
            # against zero. A macro row holding one value 616 times has a
            # measured spread around 6e-08 -- the same rounding dust that
            # once got promoted to a regime signal here, when a genome
            # branched on demean(Const) = -3e-08 and cost a third of the
            # alpha. Scale-relative, or this check silently never fires.
            vals = np.where(mask, feats[name], np.nan).astype(np.float64)
            alive = np.isfinite(vals).any(axis=1)
            sd = np.nanstd(vals, axis=1)[alive]
            scale = np.maximum(np.abs(np.nanmean(vals, axis=1))[alive], 1.0)
            constant = float(np.mean(sd / scale < 1e-6)) if alive.any() else 1.0
            rows.append({
                "feature": name, "family": F.family_of(name), "n": n,
                "ic": None,
                "reason": ("constant across the cross-section"
                           if constant > 0.9 else "too few sessions")})
            continue
        v = s[ok]
        sd = float(v.std(ddof=1))
        # Overlapping h-day windows make neighbouring ICs dependent; the
        # naive t-stat would be inflated by roughly sqrt(h).
        t = float(v.mean() / max(sd, 1e-12) * np.sqrt(n / max(h, 1)))
        rows.append({
            "feature": name, "family": F.family_of(name), "n": n,
            "ic": float(v.mean()), "ic_std": sd, "t": t,
            "hit": float(np.mean(v > 0)),
        })

    live = [r for r in rows if r.get("ic") is not None]
    live.sort(key=lambda r: -abs(r["ic"]))

    print("\n%-22s %-15s %8s %8s %7s %6s" %
          ("terminal", "family", "IC", "t", "hit", "n"))
    print("-" * 72)
    for r in live[:25]:
        print("%-22s %-15s %+8.4f %+8.2f %6.1f%% %6d"
              % (r["feature"], r["family"], r["ic"], r["t"],
                 100 * r["hit"], r["n"]))

    dead = [r for r in rows if r.get("ic") is None]
    for why in sorted({r["reason"] for r in dead}):
        names = [r["feature"] for r in dead if r["reason"] == why]
        print("\n%d terminals unscored -- %s:\n  %s"
              % (len(names), why, ", ".join(names[:14])
                 + (" ..." if len(names) > 14 else "")))

    # The number that matters more than any single row. Testing 80 terminals
    # and reporting the largest t-stat is a multiple-comparisons problem; the
    # expected maximum under the null is the bar a winner has to clear.
    if live:
        best = max(live, key=lambda r: abs(r["t"]))
        k = len(live)
        exp_max = float(np.sqrt(2.0 * np.log(max(k, 2))))
        print("\nlargest |t| across %d scored terminals : %.2f  (%s)"
              % (k, abs(best["t"]), best["feature"]))
        print("expected largest |t| under the null    : %.2f" % exp_max)
        print("-> %s" % ("clears the multiple-testing bar"
                         if abs(best["t"]) > exp_max else
                         "no terminal clears it; every IC here is "
                         "consistent with noise"))

    # Family roll-up. This is the part that catches a broken block.
    print("\n%-15s %6s %9s %9s" % ("family", "n", "mean |IC|", "max |IC|"))
    print("-" * 44)
    fams = {}
    for r in live:
        fams.setdefault(r["family"], []).append(abs(r["ic"]))
    for fam, xs in sorted(fams.items(), key=lambda kv: -np.mean(kv[1])):
        print("%-15s %6d %9.4f %9.4f"
              % (fam, len(xs), float(np.mean(xs)), float(np.max(xs))))

    out = resolve(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "horizon": h,
        "panel": {"start": str(panel.dates[0].date()),
                  "end": str(panel.dates[-1].date()),
                  "names": len(panel.tickers)},
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
