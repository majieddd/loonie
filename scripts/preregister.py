"""Commit a fixed set of candidates to the forward seal, before the data exists.

    python scripts/preregister.py [--n 5] [--dry-run]

The multiple-testing penalty is the price of searching. The bar a candidate
must clear is sqrt(2 ln k) on the number of hypotheses tested, and this search
evaluated enough genomes to put k in the tens of thousands, which sets the bar
near 4.7. The best IC the archive has produced is 4.01. That gap is not
closable by searching harder -- the search has not improved in hundreds of
generations -- and it is not an artefact of the wrong independence estimate,
which was checked.

It is closable by not searching. Fix a handful of candidates now, write them
down, and test only those on data that does not exist yet: k = 5 puts the bar
near 2.0, and 4.01 clears it with room. Nothing about the strategies changes.
What changes is that the number of hypotheses being tested is five rather than
sixty thousand, because the commitment was made before the evidence existed.

That only works if the commitment is real. Two things enforce it here:

  * The seal records a timestamp, and refuses at evaluation time to score any
    session dated before it. Data that already existed cannot test a
    commitment it could have informed.
  * open_holdout() refuses any genome not on the registered list. Swapping in
    a better candidate after seeing the data is the search again, with a
    sample size of one, and it would not feel like cheating at the time.

DIVERSITY MATTERS MORE THAN RANK. Registering the top five by score would
likely register one idea five times -- this archive's leaders are frequently
near-identical expressions -- and five copies of one hypothesis is one
hypothesis with a bar priced for five. So candidates are deduplicated by the
correlation of their IC series, not by fingerprint, which is the only measure
that notices two different expressions computing the same thing.
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

from loonie import features as F  # noqa: E402
from loonie import ic as icmod  # noqa: E402
from loonie import seal  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402
from loonie.genome import Genome  # noqa: E402

MAX_IC_CORR = 0.90       # above this, two candidates are the same hypothesis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default="state/evolve_state.json")
    a = ap.parse_args()

    load_env()
    cfg = load()
    sl = seal.Seal.load(cfg)
    if sl is None:
        print("no seal; run the search once to create one")
        return 1
    if sl.kind != "forward":
        print("seal is %r, not forward. Pre-registration only means something "
              "against a window that does not exist yet." % sl.kind)
        return 1
    if sl.registered and not a.dry_run:
        print("already registered %d candidates on %s:"
              % (len(sl.registered), sl.registered[0]["registered_at"][:10]))
        for r in sl.registered:
            print("  %s  %s" % (r["fingerprint"][:8], r["canonical"][:90]))
        print("\nRe-registering would let a later, better-informed choice "
              "replace an earlier one, which is the whole thing this prevents.")
        return 0

    panel = seal.training_panel(cfg, by="scripts/preregister.py")
    feats = F.build(panel)
    print("[pre] training window %s..%s | %d names"
          % (panel.dates[0].date(), panel.dates[-1].date(), len(panel.tickers)))

    state = json.loads(resolve(a.state).read_text(encoding="utf-8"))
    pool = [e for e in (state.get("hall_of_fame") or [])
            + (state.get("archive") or []) if e.get("genome")]
    print("[pre] %d candidates carrying a genome" % len(pool))

    scored = []
    for e in pool:
        g = Genome.from_dict(e["genome"])
        h = max(1, int(g.rebalance_days))
        try:
            s = g.score(feats, panel.tradable)
            series = icmod.ic_series(s, icmod.forward_returns(panel.close, h),
                                     panel.tradable)
            r = icmod.summarize(s, icmod.forward_returns(panel.close, h),
                                panel.tradable, h)
        except Exception:
            continue
        if not np.isfinite(r["ic_t"]):
            continue
        scored.append((r["ic_t"], r, e, g, np.nan_to_num(series)))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        print("nothing scoreable")
        return 1

    # Greedy selection: best first, then only candidates that are neither the
    # same expression nor behaviourally the same as one already chosen.
    #
    # Both checks are needed. The IC-series check alone let
    # `rank(f_rnd_intensity)` through twice at k=25/rb=2 and k=10/rb=21: the
    # two series are computed against DIFFERENT forward horizons, so they
    # correlate weakly even though they are one idea sized two ways. The
    # expression check alone would miss two different expressions that compute
    # the same thing, which is what the correlation is for.
    def core(entry):
        """The hypothesis, stripped of how much of it you bought."""
        return str(entry.get("canonical", "")).split("|")[0]

    chosen: list = []
    for ic_t, r, e, g, series in scored:
        if any(core(e) == core(pe) for _, _, pe, _, _ in chosen):
            continue
        dup = False
        for _, _, _, _, prev in chosen:
            ok = np.isfinite(series) & np.isfinite(prev)
            if ok.sum() > 100:
                c = float(np.corrcoef(series[ok], prev[ok])[0, 1])
                if abs(c) > MAX_IC_CORR:
                    dup = True
                    break
        if not dup:
            chosen.append((ic_t, r, e, g, series))
        if len(chosen) >= a.n:
            break

    bar_search = icmod.null_bar(
        (state.get("history") or [{}])[-1].get("trials_effective")
        or state.get("trials", 1))
    bar_pre = icmod.null_bar(len(chosen))

    print("\n%-10s %8s %8s %7s  %s" % ("genome", "IC", "IC t", "hit", "expression"))
    print("-" * 96)
    for ic_t, r, e, g, _ in chosen:
        print("%-10s %+8.4f %+8.2f %6.1f%%  %s"
              % (e["fingerprint"][:8], r["ic"], ic_t, 100 * r["ic_hit"],
                 e.get("canonical", "")[:52]))

    print("\nbar if chosen by search  (k=%s effective) : %.2f"
          % (int((state.get("history") or [{}])[-1].get("trials_effective") or 0),
             bar_search))
    print("bar for these %d pre-registered            : %.2f"
          % (len(chosen), bar_pre))
    print("best IC t among them                      : %.2f" % chosen[0][0])
    print("-> %s" % ("clears the pre-registered bar"
                     if chosen[0][0] > bar_pre else "does not clear even at k=%d"
                     % len(chosen)))

    if a.dry_run:
        print("\n(dry run; nothing registered)")
        return 0

    sl.register([g for _, _, _, g, _ in chosen],
                note=("top %d by cross-sectional IC t on %s..%s, deduplicated "
                      "at IC-series correlation %.2f; search stalled %s "
                      "generations before this commitment"
                      % (len(chosen), panel.dates[0].date(),
                         panel.dates[-1].date(), MAX_IC_CORR,
                         state.get("generation"))))
    print("\nregistered %d candidates against the forward seal committed %s"
          % (len(chosen), sl.committed_at))
    print("the window will not be evaluable until %d sessions accrue"
          % sl.min_sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
