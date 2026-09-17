"""Does combining signals beat picking one? Measured on a window that was not
used to pick them.

The theory is sound and standard: average several partly-independent signals
and their errors cancel faster than their signal does, so the combination
scores better than any component. It is the cheapest idea in quantitative
finance and it is usually right.

It is not right here, and the way it fails is instructive enough to keep the
script around.

Run in-sample -- rank candidates by IC on a window, average the best ten, score
the average on the same window -- the ensemble reads t = +4.49 against a best
single of +2.93. A 53% improvement, exactly the shape the theory predicts, and
completely false. Selecting the top ten BY IC and then measuring IC on the
window that did the selecting inflates the result twice over.

Split the window and the effect vanishes:

    best single       select +2.60   ->  test  -0.99
    ensemble top 3    select +3.70   ->  test  +0.08
    ensemble top 5    select +4.59   ->  test  -0.10
    ensemble top 10   select +3.57   ->  test  +0.04
    ensemble top 20   select +4.19   ->  test  +0.55

Out-of-sample ICs of +0.0005, -0.0006, +0.0004, +0.0051. Zero, four times.

So the split is not optional here and the script does not offer a flag to skip
it. A tool that can produce the flattering number on request will eventually
be asked to.

WHAT THE RESULT MEANS. Averaging cannot manufacture signal that is not in the
components. These candidates do not have small independent edges that add up;
on the evidence they have no edge, and averaging no-edge with no-edge gives
no-edge with tidier error bars -- which is precisely what a t-stat near zero
on a 663-session test window looks like.
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
import pandas as pd  # noqa: E402

from loonie import cv as cvmod  # noqa: E402
from loonie import features as F  # noqa: E402
from loonie import ic as icmod  # noqa: E402
from loonie import seal  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402
from loonie.genome import Genome  # noqa: E402


def ranks(score: np.ndarray, panel) -> np.ndarray:
    """Cross-sectional percentile ranks.

    Raw scores live on arbitrary and wildly different scales -- one genome
    returns log-dollars, another a z-score -- so averaging them directly lets
    whichever has the largest numbers dominate the blend. Ranks put every
    component on the same footing, which is what makes the average a vote
    rather than a sum.
    """
    return pd.DataFrame(np.where(panel.tradable, score, np.nan)) \
        .rank(axis=1, pct=True).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--split", type=float, default=0.75,
                    help="fraction used to SELECT; the rest tests")
    ap.add_argument("--sizes", default="3,5,10,20")
    ap.add_argument("--json", default="state/ensemble.json")
    a = ap.parse_args()

    load_env()
    cfg = load()
    panel = seal.training_panel(cfg, by="scripts/ensemble.py")
    feats = F.build(panel)

    T = panel.shape[0]
    fit_end = int(T * a.split)
    emb = int(cfg.cv.embargo_days)
    val_lo = min(T, fit_end + emb)
    if T - val_lo < 120:
        print("test window too short (%d sessions)" % (T - val_lo))
        return 1

    fit = cvmod._slice_panel(panel, 0, fit_end)
    val = cvmod._slice_panel(panel, val_lo, T)
    ff = {k: v[:fit_end] for k, v in feats.items()}
    vf = {k: v[val_lo:T] for k, v in feats.items()}
    print("[ens] select %s..%s (%d)   test %s..%s (%d)"
          % (fit.dates[0].date(), fit.dates[-1].date(), fit_end,
             val.dates[0].date(), val.dates[-1].date(), T - val_lo))

    h = a.horizon
    f_fwd = icmod.forward_returns(fit.close, h)
    v_fwd = icmod.forward_returns(val.close, h)

    state = json.loads(resolve("state/evolve_state.json").read_text("utf-8"))
    pool = [e for e in (state.get("hall_of_fame") or [])
            + (state.get("archive") or []) if e.get("genome")]

    rows = []
    for e in pool:
        g = Genome.from_dict(e["genome"])
        try:
            sf = g.score(ff, fit.tradable)
            sv = g.score(vf, val.tradable)
        except Exception:
            continue
        r = icmod.summarize(sf, f_fwd, fit.tradable, h)
        if np.isfinite(r["ic_t"]):
            rows.append({"t": r["ic_t"], "fp": e.get("fingerprint"),
                         "rf": ranks(sf, fit), "rv": ranks(sv, val)})
    if not rows:
        print("nothing scoreable")
        return 1

    # Ranked ONLY on the select window. Everything below is measured on data
    # that had no say in the ordering.
    rows.sort(key=lambda r: -r["t"])
    print("[ens] %d candidates ranked on the select window" % len(rows))

    out = []
    best_f = icmod.summarize(rows[0]["rf"], f_fwd, fit.tradable, h)
    best_v = icmod.summarize(rows[0]["rv"], v_fwd, val.tradable, h)
    out.append({"name": "best single", "k": 1,
                "select_t": best_f["ic_t"], "test_t": best_v["ic_t"],
                "test_ic": best_v["ic"]})

    for k in [int(x) for x in a.sizes.split(",") if x.strip()]:
        k = min(k, len(rows))
        af = np.nanmean(np.stack([r["rf"] for r in rows[:k]]), axis=0)
        av = np.nanmean(np.stack([r["rv"] for r in rows[:k]]), axis=0)
        sf = icmod.summarize(af, f_fwd, fit.tradable, h)
        sv = icmod.summarize(av, v_fwd, val.tradable, h)
        out.append({"name": "ensemble top %d" % k, "k": k,
                    "select_t": sf["ic_t"], "test_t": sv["ic_t"],
                    "test_ic": sv["ic"]})

    print("\n%-20s %10s %10s %12s" % ("", "SELECT t", "TEST t", "TEST IC"))
    print("-" * 56)
    for r in out:
        print("%-20s %+10.2f %+10.2f %+12.4f"
              % (r["name"], r["select_t"], r["test_t"], r["test_ic"]))

    best = max(out, key=lambda r: r["test_t"])
    single = out[0]["test_t"]
    print("\nbest on the TEST window : %s at t %+.2f"
          % (best["name"], best["test_t"]))
    print("best single there       : %+.2f" % single)
    print("-> %s" % ("combining helped out of sample"
                     if best["test_t"] > max(single, 2.0) + 0.5 else
                     "combining did not help; the in-sample gain was selection"))

    p = resolve(a.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"horizon": h, "split": a.split, "rows": out},
                            indent=1), encoding="utf-8")
    print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
