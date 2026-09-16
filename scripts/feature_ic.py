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
from loonie.data import load_panel  # noqa: E402


def _rank(a: np.ndarray) -> np.ndarray:
    """Row-wise rank in [0,1], NaN preserved. Ties share their mean rank."""
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        row = a[i]
        ok = np.isfinite(row)
        n = int(ok.sum())
        if n < 10:
            continue
        v = row[ok]
        order = np.argsort(v, kind="stable")
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n, dtype=np.float64)
        # Average ranks within tied groups, or a feature with many equal
        # values gets an ordering that is really just column order.
        s = v[order]
        i0 = 0
        for i1 in range(1, n + 1):
            if i1 == n or s[i1] != s[i0]:
                if i1 - i0 > 1:
                    r[order[i0:i1]] = np.mean(r[order[i0:i1]])
                i0 = i1
        out[i, ok] = r / max(n - 1, 1)
    return out


def ic_series(feat: np.ndarray, fwd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Daily cross-sectional Spearman IC between a feature and forward return."""
    x = np.where(mask, feat, np.nan).astype(np.float64)
    y = np.where(mask, fwd, np.nan).astype(np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    x = np.where(both, x, np.nan)
    y = np.where(both, y, np.nan)

    rx, ry = _rank(x), _rank(y)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        ok = np.isfinite(rx[i]) & np.isfinite(ry[i])
        if ok.sum() < 20:
            continue
        a, b = rx[i][ok], ry[i][ok]
        sa, sb = a.std(), b.std()
        if sa > 1e-12 and sb > 1e-12:
            out[i] = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--json", default="state/feature_ic.json")
    ap.add_argument("--min-obs", type=int, default=200)
    args = ap.parse_args()

    load_env()
    cfg = load()
    panel = load_panel(cfg, progress=False)
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
            rows.append({"feature": name, "family": F.family_of(name),
                         "n": n, "ic": None, "reason": "too few sessions"})
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
    if dead:
        print("\n%d terminals had too little data to score: %s"
              % (len(dead), ", ".join(r["feature"] for r in dead[:12])))

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
