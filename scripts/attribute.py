"""Regress the best strategies on the published factors and see what is left.

    python scripts/attribute.py [--top 12] [--json state/attribution.json]

Every gate in this system scores a strategy against an equal-weight universe
benchmark. That tells us whether selection beat holding everything. It cannot
tell us whether selection beat holding everything *for a reason nobody has
already packaged into an ETF*.

This asks the second question. For each candidate it runs the ordinary
backtest, then regresses the daily excess return on Mkt-RF, SMB, HML, RMW, CMA
and Mom. Two numbers come back: the t-stat the system already reports, and the
t-stat of the intercept once the six factors have had their turn.

When the second is much smaller than the first, the strategy is a factor
portfolio wearing a genetic program's clothes.
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
from loonie import factors  # noqa: E402
from loonie import features as F  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402
from loonie.data import load_panel  # noqa: E402
from loonie.genome import Genome  # noqa: E402


def candidates(state: dict, top: int) -> list:
    """Archive and hall-of-fame entries, best fitness first, deduped."""
    pool, seen = [], set()
    for src in ("hall_of_fame", "archive"):
        for e in state.get(src) or []:
            fp = e.get("fingerprint")
            if not e.get("genome") or fp in seen:
                continue
            seen.add(fp)
            pool.append((src, e))
    pool.sort(key=lambda se: -float(se[1].get("fitness") or -np.inf))
    return pool[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", default="state/attribution.json")
    ap.add_argument("--state", default="state/evolve_state.json")
    args = ap.parse_args()

    load_env()
    cfg = load()

    sp = resolve(args.state)
    if not sp.exists():
        print("no search state at %s" % args.state)
        return 1
    state = json.loads(sp.read_text(encoding="utf-8"))

    pool = candidates(state, args.top)
    if not pool:
        print("no candidates carrying a serialised genome")
        return 1

    print("[attr] generation %s, %d candidates"
          % (state.get("generation"), len(pool)))

    panel = load_panel(cfg, progress=False)
    feats = F.build(panel)
    print("[attr] panel %s..%s, %d names"
          % (panel.dates[0].date(), panel.dates[-1].date(), len(panel.tickers)))

    try:
        fac = factors.fetch()
        print("[attr] factors %s..%s" % (fac.index[0].date(), fac.index[-1].date()))
    except Exception as e:
        print("[attr] factor data unavailable: %s" % e)
        return 1

    rows = []
    for src, e in pool:
        g = Genome.from_dict(e["genome"])
        try:
            score = g.score(feats, panel.tradable)
            res = bt.run(panel, score, g, cfg)
        except Exception as ex:
            print("  ! %s: %s" % (e["fingerprint"][:8], ex))
            continue
        if not res.ok:
            print("  - %s: %s" % (e["fingerprint"][:8], res.reason))
            continue

        s = bt.summarize(res)
        a = factors.attribute(res.excess, res.dates, is_excess=True, factors=fac)
        row = {
            "fingerprint": e["fingerprint"],
            "source": src,
            "canonical": e.get("canonical", "")[:160],
            "fitness": e.get("fitness"),
            "bench_alpha_ann": s["alpha_ann"],
            "bench_alpha_t": s["alpha_tstat"],
            "ir": s["ir"],
            "factor": a,
            "verdict": factors.verdict(s["alpha_tstat"], a),
        }
        rows.append(row)

        print("\n  %s  %s" % (e["fingerprint"][:8], row["canonical"][:120]))
        print("    vs benchmark : alpha %+7.2f%%/yr   t %+6.2f   IR %+5.2f"
              % (100 * s["alpha_ann"], s["alpha_tstat"], s["ir"]))
        if a.get("ok"):
            print("    vs factors   : alpha %+7.2f%%/yr   t %+6.2f   R2 %5.2f  (n=%d)"
                  % (100 * a["alpha_ann"], a["alpha_tstat"], a["r2"], a["n"]))
            tilt = sorted(a["betas"].items(), key=lambda kv: -abs(kv[1]))
            print("    tilts        : " + "  ".join(
                "%s %+.2f(t%+.1f)" % (k, v, a["beta_tstats"][k])
                for k, v in tilt[:4]))
            print("    -> %s" % row["verdict"])
        else:
            print("    vs factors   : %s" % a.get("reason"))

    if not rows:
        print("\nnothing evaluable")
        return 1

    # ---------------------------------------------------------- the summary
    ok = [r for r in rows if r["factor"].get("ok")]
    print("\n" + "=" * 74)
    if ok:
        bt_t = np.array([r["bench_alpha_t"] for r in ok])
        ft_t = np.array([r["factor"]["alpha_tstat"] for r in ok])
        print("%d strategies attributed" % len(ok))
        print("  median t vs benchmark : %+.2f" % np.median(bt_t))
        print("  median t vs factors   : %+.2f" % np.median(ft_t))
        print("  median shrinkage      : %+.2f" % np.median(bt_t - ft_t))
        loss = sum(1 for r in ok if r["verdict"] == "explained by factors")
        keep = sum(1 for r in ok if r["verdict"] == "survives factors")
        print("  survive factors       : %d" % keep)
        print("  explained by factors  : %d" % loss)

        # Which factor is doing the work, averaged across the pool. A search
        # that keeps rediscovering one exposure is telling us what it found.
        names = ok[0]["factor"]["factors_used"]
        avg = {n: float(np.mean([r["factor"]["betas"][n] for r in ok]))
               for n in names}
        print("  mean tilt across pool : " + "  ".join(
            "%s %+.2f" % (k, v)
            for k, v in sorted(avg.items(), key=lambda kv: -abs(kv[1]))))

    out = resolve(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generation": state.get("generation"),
        "trials": state.get("trials"),
        "panel": {"start": str(panel.dates[0].date()),
                  "end": str(panel.dates[-1].date()),
                  "names": len(panel.tickers)},
        "rows": rows,
    }, indent=1, default=str), encoding="utf-8")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
