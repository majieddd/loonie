"""Re-derive what the system believes about its own methods.

    python scripts/meta_review.py

Reads the whole experience corpus and rebuilds the method posteriors from it,
then prints what has actually been learned about how to search. Runs on a
four-hour schedule under the supervisor; the next search restart picks up the
result.

Posteriors are recomputed from scratch rather than incremented, so the belief
always reflects the entire corpus and a restart cannot double-count. Everything
here is credited on FORWARD alpha -- the window fitness never saw -- because a
method judged on its own fitness would win by inflating it.

Read the `labelled` column before believing any of it. With single-digit
observations a posterior is mostly prior, and the table says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loonie import config, experience, methods, registry  # noqa: E402


def main() -> int:
    config.load_env()
    hb = registry.Worker("meta", "meta", "method review")
    hb.beat(status="running", detail="reading experience corpus")

    exp = experience.summary()
    print("=" * 76)
    print("  META REVIEW — what has been learned about how to search")
    print("=" * 76)
    if not exp.get("rows"):
        print("  experience store is empty; nothing to learn from yet.")
        print("  It fills as the search gates candidates (run scripts/run_system.py).")
        hb.done("no experience yet")
        return 0

    print("  rows                %s" % f"{exp['rows']:,}")
    print("  with forward label  %s" % f"{exp.get('labelled', 0):,}")
    print("  distinct strategies %s" % f"{exp.get('distinct_strategies', 0):,}")
    print("  runs / methods      %s / %s" % (exp.get("runs"), exp.get("methods")))
    print("  promoted            %s" % exp.get("promoted"))
    print("  span                %s  ->  %s" % (exp.get("first"), exp.get("last")))
    print("  on disk             %.1f KB" % (exp.get("bytes", 0) / 1024))
    if exp.get("labelled"):
        print("  forward-positive    %.1f%% of labelled candidates"
              % (100 * exp.get("fwd_positive_rate", 0)))

    bandit = methods.MethodBandit()
    n = bandit.ingest_experience()
    print("-" * 76)
    print("  METHODS  (credited on forward alpha, never on fitness)")
    print("  %-16s %10s %10s %9s %9s  %s"
          % ("method", "posterior", "fwd rate", "labelled", "cycles", "established"))
    for r in bandit.table():
        print("  %-16s %10.3f %10s %9d %9d  %s"
              % (r["id"], r["posterior"],
                 "—" if r["forward_rate"] is None else "%.3f" % r["forward_rate"],
                 r["labelled"], r["cycles"],
                 "yes" if r["established"] else "not yet"))
    print("  (rebuilt from %d labelled rows)" % n)

    fams = experience.family_forward_rates()
    if fams:
        print("-" * 76)
        print("  FEATURE FAMILIES  (share of labelled candidates with positive")
        print("                     forward alpha — the long memory the in-run")
        print("                     bandit deliberately forgets)")
        for f, t in list(fams.items())[:12]:
            print("  %-18s %6.1f%%   (%d of %d)"
                  % (f, 100 * t["rate"], t["survived"], t["seen"]))

    established = [r for r in bandit.table() if r["established"]]
    print("=" * 76)
    if not established:
        print("  READ: no method has enough labelled outcomes to distinguish it")
        print("  from the prior. The bandit is still exploring, and any apparent")
        print("  ranking above is noise. Do not act on it.")
    else:
        best = established[0]
        print("  READ: %s leads on forward generalisation (%.1f%% of %d labelled)."
              % (best["id"], 100 * (best["forward_rate"] or 0), best["labelled"]))
        print("  The search will draw it more often; the floor keeps every other")
        print("  method reachable in case the regime changes.")
    print("=" * 76)

    hb.done("%d methods, %d labelled rows" % (len(bandit.table()), n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
