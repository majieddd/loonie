"""The always-on RSI loop: a local model proposes, the arithmetic referees.

    python scripts/run_researcher.py --daemon
    python scripts/run_researcher.py --cycles 3
    python scripts/run_researcher.py --dream-only

Two papers shape this. From "The Last AI Built by Humans" (Duan et al., 2026)
comes the autonomy ladder -- L1 improvement execution, L2 improvement
strategy, L3 experience acquisition, L4 environment adaptation, L5 recursive
meta-improvement -- and the insight that the improvement LOOP, not the
algorithm, is the unit of analysis: what generates experience, what changes,
what persists, and whether one round's result shapes the next.

From Dream-RSI (Zheng et al., 2026) comes the mechanism that makes the loop
affordable: a completed discovery history is a replay simulator, so
exploration policies can be evaluated by reading recorded outcomes instead of
re-running the search. Meta-level feedback that was delayed and expensive
becomes immediate.

The cycle here:

  1. DREAM      replay recorded search history against alternative policies,
                and report which allocation would have done better. Costs
                seconds, not the hours a live A/B would need.
  2. PROPOSE    the local model reads the warehouse and proposes ONE testable
                hypothesis, having first been told the current
                multiple-testing bar and that its proposal raises it.
  3. LEDGER     the proposal is recorded BEFORE any test, with a fingerprint
                over its substance rather than its title.
  4. REPORT     everything lands in state/ for the dashboard.

WHAT THIS IS HONESTLY NOT. It is L2 with a piece of L3. The model chooses
what to investigate and the bandits choose how to search, so improvement
STRATEGY is autonomous. But the acceptance rules, the gates, the schema of the
experience corpus and the code itself are human-defined and the model cannot
change them. That is the L5 boundary and nothing here crosses it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from loonie import dream as D  # noqa: E402
from loonie import registry as reg  # noqa: E402
from loonie import researcher as R  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402

REPORT = "state/researcher.json"


def dream_cycle(budget: int = 8, seeds: int = 60) -> dict:
    """Stage 1: evaluate search policies against recorded history."""
    worlds = D.load_worlds(min_nodes=15, n_worlds=3)
    if not worlds:
        return {"ok": False, "reason": "no replayable history yet"}
    s = D.summary(worlds)
    table = D.evaluate_all(worlds, budget=budget, seeds=seeds)
    best = table[0] if table else None
    incumbent = next((r for r in table if r["policy"] == "thompson"), None)
    return {
        "ok": True, "worlds": s["worlds"], "nodes": s["nodes"],
        "methods": s["methods"], "budget": budget, "table": table,
        "best_policy": best["policy"] if best else None,
        "incumbent_policy": "thompson",
        "gain_over_incumbent": (best["mean_value"] - incumbent["mean_value"])
        if best and incumbent else None,
        # Three worlds cannot support a t-statistic. Direction across every
        # world is the claim; significance is not.
        "caveat": ("replay can only visit candidates the history actually "
                   "contains, so it understates a genuinely better policy; "
                   "and %d worlds is too few for a significance test"
                   % s["worlds"]),
    }


def propose_cycle(model: str) -> dict:
    """Stages 2 and 3: one hypothesis, recorded before anything tests it."""
    before = R.bar()
    h = R.propose(model=model)
    if not h:
        return {"ok": False, "reason": "model returned no usable proposal",
                "bar": before}
    entry = R.record(h)
    after = R.bar()
    return {
        "ok": True, "hypothesis": h, "fingerprint": entry["fingerprint"],
        "duplicate": entry["duplicate"],
        "bar_before": before["bar"], "bar_after": after["bar"],
        "distinct_hypotheses": after["proposals_distinct"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--interval", type=float, default=900.0)
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--dream-only", action="store_true")
    a = ap.parse_args()

    load_env()
    load()

    models = R.available()
    if not models and not a.dream_only:
        print("no local model server on %s -- run `ollama serve`" % R.OLLAMA)
        return 1
    if models and a.model not in models:
        print("[rsi] %s not present; using %s" % (a.model, models[0]))
        a.model = models[0]

    hb = None
    try:
        hb = reg.Worker("researcher", "researcher", "local RSI loop")
    except Exception:
        pass

    n = 0
    try:
        while True:
            n += 1
            t0 = time.time()
            print("\n=== cycle %d ===" % n)

            dr = dream_cycle(budget=a.budget)
            if dr.get("ok"):
                print("[dream] %d worlds, %d nodes, %d methods, budget %d"
                      % (dr["worlds"], dr["nodes"], len(dr["methods"]),
                         dr["budget"]))
                for r in dr["table"]:
                    mark = "  <- incumbent" if r["policy"] == "thompson" else ""
                    print("  %-16s value %+.5f  regret %.5f%s"
                          % (r["policy"], r["mean_value"], r["mean_regret"], mark))
                if dr.get("gain_over_incumbent", 0):
                    print("[dream] best is %s, %+.5f over the live policy"
                          % (dr["best_policy"], dr["gain_over_incumbent"]))
            else:
                print("[dream] %s" % dr.get("reason"))

            pr = {"ok": False, "reason": "skipped"}
            if not a.dream_only:
                pr = propose_cycle(a.model)
                if pr.get("ok"):
                    h = pr["hypothesis"]
                    print("[propose] %s  (%s, %dd)"
                          % (h.get("title"), h.get("market"),
                             h.get("holding_days")))
                    print("          signal : %s" % str(h.get("signal"))[:100])
                    print("          reason : %s" % str(h.get("reason"))[:100])
                    print("[ledger]  %s%s | %d distinct | bar %.2f -> %.2f"
                          % (pr["fingerprint"],
                             "  DUPLICATE" if pr["duplicate"] else "",
                             pr["distinct_hypotheses"], pr["bar_before"],
                             pr["bar_after"]))
                else:
                    print("[propose] %s" % pr.get("reason"))

            doc = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "cycle": n, "seconds": round(time.time() - t0, 1),
                   "model": a.model, "dream": dr, "propose": pr}
            resolve(REPORT).write_text(json.dumps(doc, indent=1, default=str),
                                       encoding="utf-8")
            if hb:
                hb.beat(status="running",
                        detail="cycle %d | %s | bar %.2f"
                               % (n, dr.get("best_policy") or "no replay",
                                  pr.get("bar_after") or 0.0),
                        cycle=n)

            if not a.daemon and n >= a.cycles:
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n[rsi] interrupted")
    finally:
        if hb:
            hb.done("stopped after %d cycles" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
