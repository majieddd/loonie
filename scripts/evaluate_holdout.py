"""Open the sealed holdout. Once.

    python scripts/evaluate_holdout.py

This is the end of the experiment, not a step in it. The window has never been
seen by the search; opening it consumes the single evaluation the seal allows
and writes the result into an append-only ledger before you have a chance to
react to it. If you do not like the number, the correct response is to collect
more market history -- not to adjust something and run this again, which the
seal will refuse anyway.

The verdict is computed, not eyeballed. "The strategy returned 30.4% and the
index returned 33.5%" is not a conclusion; it is two numbers. The conclusion
is whether the difference is distinguishable from zero, and that is what gets
printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from loonie import (backtest as bt, config, data, evolve, features,  # noqa: E402
                    metrics, notify, seal, universe)
from loonie.genome import Genome  # noqa: E402


def pick_strategy(cfg, which: str):
    p = config.resolve(evolve.STATE)
    if not p.exists():
        raise SystemExit("no evolution state; run scripts/run_evolve.py first")
    d = json.loads(p.read_text(encoding="utf-8"))
    hof, arch = d.get("hall_of_fame", []), d.get("archive", [])
    if which == "promoted":
        if not hof:
            raise SystemExit(
                "nothing has cleared the promotion gate.\n"
                "Evaluating an un-promoted candidate burns the one holdout\n"
                "evaluation you have on a strategy the search itself does not\n"
                "consider established. If you mean it, pass --which best."
            )
        entry = hof[-1]
    else:
        if not arch:
            raise SystemExit("archive is empty")
        entry = arch[0]
    return Genome.from_dict(entry["genome"] if "genome" in entry else entry), entry, d


def verdict(res, trials: int, n_years: float) -> dict:
    """Is the excess distinguishable from zero, after the search that found it?"""
    exc = res.ret - res.bench_ret
    t = bt._newey_west_tstat(exc)
    d1 = metrics.dsr_from_returns(exc, n_trials=1)
    dN = metrics.dsr_from_returns(exc, n_trials=max(trials, 1))
    haircut = metrics.haircut_sharpe(d1["sr_ann"], max(trials, 1), n_years)
    mtrl = metrics.min_track_record_length(
        d1["sr_daily"], len(exc), d1.get("skew", 0.0), d1.get("kurtosis", 3.0))
    beat = res.stats["cagr"] > res.stats["bench_cagr"]
    significant = (t >= 2.0) and (dN["dsr"] >= 0.95)
    return {
        "beat_benchmark": bool(beat),
        "excess_cagr": res.stats["excess_cagr"],
        "alpha_tstat": float(t),
        "dsr_naive": d1["dsr"],
        "dsr_corrected": dN["dsr"],
        "sr_excess_ann": d1["sr_ann"],
        "sr_threshold_ann": dN["sr_star_ann"],
        "haircut_sharpe_ann": haircut,
        "min_track_record_days": mtrl,
        "significant": bool(significant),
        "trials": int(trials),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["promoted", "best"], default="promoted")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()

    uni = universe.Universe.load(cfg)
    panel = data.load_panel(cfg, uni, progress=False)
    sl = seal.Seal.load(cfg)
    if sl is None:
        raise SystemExit("no seal exists. Run scripts/run_evolve.py first, "
                         "which creates one before the search starts.")

    print("=" * 72)
    print(sl.describe())
    print("=" * 72)
    if sl.remaining() <= 0:
        print("\nThis holdout is spent. Its ledger:\n")
        print(json.dumps(sl.ledger, indent=2))
        return 2

    g, entry, state = pick_strategy(cfg, a.which)
    trials = int(state.get("trials", 1))
    print("\nStrategy : %s" % g.canonical())
    print("Selected from %d evaluated candidates." % trials)
    print("CV alpha t-stat %.2f | IR %.2f"
          % (entry["cv"]["alpha_tstat"], entry["cv"]["mean_ir"]))

    if not a.yes:
        print("\nThis consumes the only holdout evaluation you have. Type "
              "'open' to proceed.")
        try:
            if input("> ").strip().lower() != "open":
                print("aborted; seal untouched")
                return 0
        except EOFError:
            print("non-interactive; pass --yes if you mean it")
            return 0

    sub = sl.open_holdout(panel, g, note="which=%s trials=%d" % (a.which, trials))
    print("\n[seal] opened. evaluation %d/%d consumed."
          % (sl.evaluations, sl.max_evaluations))

    feats = features.build(sub)
    bench = bt.equal_weight_benchmark(sub)
    score = g.score(feats, sub.tradable)
    res = bt.run(sub, score, g, cfg, bench_ret=bench)
    n_years = len(res.ret) / 252.0
    v = verdict(res, trials, n_years)
    sl.record_result({**res.stats, **v})

    print("\n" + "=" * 72)
    print("  SEALED HOLDOUT RESULT   %s -> %s   (%d sessions)"
          % (sub.dates[0].date(), sub.dates[-1].date(), len(res.ret)))
    print("=" * 72)
    s = res.stats
    print("  strategy CAGR        %8.2f%%" % (100 * s["cagr"]))
    print("  benchmark CAGR       %8.2f%%   (equal-weight eligible universe)"
          % (100 * s["bench_cagr"]))
    print("  excess               %8.2f%%" % (100 * s["excess_cagr"]))
    print("  strategy Sharpe      %8.2f" % s["sharpe"])
    print("  max drawdown         %8.2f%%   (benchmark %.2f%%)"
          % (100 * s["max_dd"], 100 * s["bench_max_dd"]))
    print("  annual turnover      %8.2f" % s["ann_turnover"])
    print("-" * 72)
    print("  alpha t-stat         %8.2f   (Newey-West; >= 2.0 to be meaningful)"
          % v["alpha_tstat"])
    print("  excess Sharpe (ann)  %8.2f" % v["sr_excess_ann"])
    print("  threshold Sharpe     %8.2f   <- what %d trials of noise produces"
          % (v["sr_threshold_ann"], v["trials"]))
    print("  Sharpe after haircut %8.2f" % v["haircut_sharpe_ann"])
    print("  DSR (uncorrected)    %8.2f" % v["dsr_naive"])
    print("  DSR (search-adjusted)%8.2f" % v["dsr_corrected"])
    mt = v["min_track_record_days"]
    print("  track record needed  %8s sessions for significance"
          % ("inf" if not np.isfinite(mt) else "%.0f" % mt))
    print("=" * 72)

    if v["significant"]:
        print("  VERDICT: the excess return survives correction for the %d\n"
              "  candidates evaluated. This is a real result. Paper trade it\n"
              "  for a meaningful period before risking anything." % v["trials"])
    elif v["beat_benchmark"]:
        print("  VERDICT: it beat the benchmark, but NOT significantly once the\n"
              "  search that produced it is priced in. This is the same outcome\n"
              "  the video reached -- a number that looks like a win and is\n"
              "  indistinguishable from luck. Do not trade it.")
    else:
        print("  VERDICT: it did not beat the benchmark. The honest reading is\n"
              "  that the stock picking added nothing over owning the universe.")
    print("=" * 72)

    f2 = lambda x: notify._fmt(x, dp=2)                      # noqa: E731
    pc = lambda x: notify._fmt(x, pct=True)                  # noqa: E731
    sections = [
        ("result", notify.table([s], [
            ("strategy CAGR", "cagr", True, pc),
            ("benchmark CAGR", "bench_cagr", True, pc),
            ("excess", "excess_cagr", True, pc),
            ("Sharpe", "sharpe", True, f2),
            ("max DD", "max_dd", True, pc),
            ("turnover", "ann_turnover", True, f2)])),
        ("significance", notify.table([v], [
            ("alpha t", "alpha_tstat", True, f2),
            ("excess SR", "sr_excess_ann", True, f2),
            ("noise threshold SR", "sr_threshold_ann", True, f2),
            ("DSR corrected", "dsr_corrected", True, f2),
            ("trials", "trials", True, str),
            ("significant", "significant", False, str)])),
        ("seal ledger", "<div class='card'><code>%s</code></div>"
         % json.dumps(sl.ledger[-3:], indent=2).replace("\n", "<br>")),
    ]
    p = notify.publish("holdout", "Sealed holdout result", sections,
                       "One evaluation, consumed. %s" % g.canonical()[:120])
    print("\nreport: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
