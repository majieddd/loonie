"""Run every registered strategy and write the comparison table.

    python scripts/run_strategies.py [--json docs/data/strategies_table.json]

One table, four markets, the same measurements applied to each. The columns
are chosen so that no single one can carry a false impression on its own:

  total P/L      what it made over its own span, which differs by market
  consistency    share of ACTIVE periods that were positive
  max drawdown   printed beside consistency, deliberately

That last pairing is the point. A 95% win rate with a 60% drawdown is the
short-volatility signature -- many small wins and rare enormous losses -- and
the win rate alone is the most misleading number in trading. The two are shown
together so the shape of the return cannot hide behind either one.

Spans differ between markets and are NOT comparable as totals. Forex reaches
back to 1999, crypto to 2014, the wide equity panel to 2016, and the options
models to whenever VIX and SPY overlap. Annualised figures and t-statistics
are the comparable columns; total P/L is not.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from loonie import markets as M  # noqa: E402
from loonie import strategy_lib  # noqa: E402  (registers everything)
from loonie.config import load, load_env, resolve  # noqa: E402
from loonie.strategies import REGISTRY  # noqa: E402

assert strategy_lib  # imported for its registration side effect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/data/strategies_table.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    load_env()
    load()

    rows = []
    for sid, spec in REGISTRY.items():
        if a.only and a.only not in sid:
            continue
        t0 = time.time()
        try:
            res = spec["fn"]()
        except Exception as e:
            print("  ! %-15s %s: %s" % (sid, type(e).__name__, e))
            rows.append({"id": sid, "title": spec["title"],
                         "market": spec["market"], "ok": False,
                         "reason": "%s: %s" % (type(e).__name__, e)})
            continue
        st = res.stats()
        st["curve"] = res.curve()
        st["seconds"] = round(time.time() - t0, 1)
        rows.append(st)
        flag = "" if st.get("ok") else "  (%s)" % st.get("reason")
        print("  %-15s %5.1fs%s" % (sid, st["seconds"], flag))

    live = [r for r in rows if r.get("ok")]
    live.sort(key=lambda r: -r.get("cagr_pct", -1e9))

    print("\n%-34s %-8s %9s %9s %7s %8s %8s" %
          ("strategy", "market", "total P/L", "CAGR", "win%", "maxDD", "t"))
    print("-" * 92)
    for r in live:
        print("%-34s %-8s %8.1f%% %8.1f%% %6.0f%% %7.0f%% %8.2f"
              % (r["title"][:34], r["market"], r["total_pl_pct"],
                 r["cagr_pct"], r["win_rate_pct"], r["max_drawdown_pct"],
                 r["t_stat"]))

    dead = [r for r in rows if not r.get("ok")]
    for r in dead:
        print("  %-34s %-8s  unavailable: %s"
              % (r["title"][:34], r["market"], r.get("reason", "?")))

    # ---- the accounting that makes the table readable --------------------
    # Sorting N strategies by return and reporting the top one IS selection.
    # The largest |t| among N independent null draws concentrates near
    # sqrt(2 ln N), so that is the bar any winner has to clear before the
    # ranking means anything.
    import math
    n = len(live)
    bar = max(2.0, math.sqrt(2.0 * math.log(max(n, 2))))
    clears = [r for r in live if abs(r["t_stat"]) >= bar]
    best = max(live, key=lambda r: abs(r["t_stat"])) if live else None

    print()
    print("=" * 92)
    print("MULTIPLE TESTING")
    print("  strategies measured        : %d" % n)
    print("  bar sqrt(2 ln N)           : %.2f" % bar)
    print("  largest |t| observed       : %.2f  (%s)"
          % (abs(best["t_stat"]), best["title"]) if best else "")
    print("  clear the bar              : %d  %s"
          % (len(clears), ", ".join(r["title"] for r in clears) or "none"))
    measured = [r for r in clears if r["market"] != "options"]
    print("  ...of those, NOT modelled  : %d  %s"
          % (len(measured), ", ".join(r["title"] for r in measured) or "none"))
    print()
    print("  Hou, Xue & Zhang (2020) replicated 452 published anomalies and")
    print("  found 65% could not clear |t| >= 1.96, and 52% failed regardless")
    print("  after adjusting for multiple testing. Harvey, Liu & Zhu (2016)")
    print("  argue the honest hurdle for a new factor is nearer t = 3.0.")
    print("  The right prior for this table is that most of it does not work.")

    summary = {"n_strategies": n, "bar": bar,
               "clearing": [r["id"] for r in clears],
               "clearing_measured": [r["id"] for r in measured]}

    print("\nSpans differ by market and totals are NOT comparable across rows.")
    print("Win rate is shown beside max drawdown deliberately: a high rate with")
    print("a deep drawdown is the short-volatility shape, not a good strategy.")

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "markets": M.registry(), "strategies": rows,
           "multiple_testing": summary}
    p = resolve(a.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\nwrote %s (%.0f KB)" % (a.json, p.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
