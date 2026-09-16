"""Walk-forward blind test of the SELECTION PROCEDURE, not of one strategy.

    python scripts/walkforward.py
    python scripts/walkforward.py --segments 10

The sealed holdout answers "is the strategy I picked any good?" exactly once.
This answers a different and more useful question as often as you like:

    If I had been running this system for the last eight years, picking the
    best strategy from what I knew at the time and trading it forward, would
    I have made money?

That is the question that matters, because you will never trade "the strategy
the search settled on in September 2026" — you will trade whatever it thinks
is best on each future day, and that changes. Backtesting the final champion
over history flatters you twice: once because it was chosen knowing that
history, and again because it is not the process you actually run.

Method. Chop the training window into N sequential segments. At each boundary,
rank every strategy in the archive using ONLY the segments before it, take the
winner, and record how it did on the *next* segment — which it has never been
scored on. Chain those forward and you get an equity curve of the procedure.

This touches only the training window. The holdout stays sealed; check with
`python scripts/status.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used below; without this every run dies on a UnicodeEncodeError in a
# print statement rather than in anything that matters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import numpy as np  # noqa: E402

from loonie import (backtest as bt, config, cv, data, evolve, features,  # noqa: E402
                    metrics, notify, registry, seal, universe)
from loonie.genome import Genome  # noqa: E402

TD = 252.0


def load_candidates(cfg, limit):
    p = config.resolve(evolve.STATE)
    if not p.exists():
        raise SystemExit("no evolution state; run scripts/run_evolve.py first")
    d = json.loads(p.read_text(encoding="utf-8"))
    pool = (d.get("archive") or [])[:limit]
    out = []
    for e in pool:
        if "genome" not in e:
            continue
        try:
            out.append((e.get("fingerprint"), e.get("canonical"),
                        Genome.from_dict(e["genome"])))
        except Exception:
            continue
    if not out:
        raise SystemExit("archive holds no reloadable genomes")
    return out, d


def ann(r):
    r = np.asarray(r, dtype=np.float64)
    if not len(r):
        return 0.0
    v = float(np.prod(1.0 + r))
    yrs = max(len(r) / TD, 1e-9)
    return v ** (1.0 / yrs) - 1.0 if v > 0 else -1.0


def honest_walkforward(cfg, train, feats, bench, sl, a):
    """The uncontaminated version: never let the searcher see the test segment.

    The fast path above reuses the archive, and the archive is the problem --
    every strategy in it earned its place by scoring well across the entire
    training window, the forward segments included. So "rank on the past, test
    on the future" still tests a pool that was picked knowing the future. On a
    first run that flattered the procedure to 45% annualised against a 12.9%
    universe, which is not a result, it is a leak.

    Here each segment gets its own search, run from scratch on data strictly
    before it, and only the winner of that search is carried forward. It is
    roughly N times slower and it is the only version whose number means
    anything.
    """
    T = train.shape[0]
    emb = a.embargo if a.embargo is not None else int(cfg.cv.embargo_days)
    edges = np.linspace(0, T, a.segments + 1).astype(int)
    rows, chained, chained_bench = [], [], []

    hb = registry.Worker("validate", "validate", "honest walk-forward")
    hb.beat(status="running", detail="fresh search per segment, past data only")
    print("  mode       HONEST — a fresh search per segment, past data only")
    print("  cost       %d searches x %d genomes; this takes a while"
          % (a.segments - 1, a.pop * a.gens))
    print("-" * 78)

    for s in range(1, a.segments):
        train_hi = edges[s]                      # searcher sees [0, train_hi)
        test_lo = edges[s] + emb                 # embargoed gap
        test_hi = edges[s + 1]
        if test_hi - test_lo < 40 or train_hi < 300:
            continue

        past = cv._slice_panel(train, 0, train_hi)
        past_feats = {k: v[:train_hi] for k, v in feats.items()}
        sub_cfg = config.Cfg(json.loads(json.dumps(dict(cfg), default=str)))
        sub_cfg["evolve"]["population"] = a.pop
        sub_cfg["evolve"]["null_samples_per_gen"] = 0
        sub_cfg["cv"]["n_splits"] = 4

        ev = evolve.Evolver(sub_cfg, past, past_feats,
                            seed=int(cfg.evolve.seed) + s, verbose=False)
        ev.seed_population(a.pop)
        for _ in range(a.gens):
            ev.step()
        if not ev.population:
            continue
        winner = ev.population[0].genome

        score = winner.score(feats, train.tradable)
        sub = cv._slice_panel(train, test_lo, test_hi)
        res = bt.run(sub, score[test_lo:test_hi], winner, cfg,
                     bench_ret=bench[test_lo:test_hi])
        if not res.ok:
            continue

        b = bench[test_lo:test_hi]
        n = min(len(res.ret), len(b))
        chained.append(res.ret[:n]); chained_bench.append(b[:n])
        rows.append({
            "seg": s + 1,
            "from": str(train.dates[test_lo].date()),
            "to": str(train.dates[test_hi - 1].date()),
            "pick": winner.fingerprint[:8],
            "expr": winner.canonical()[:54],
            "strat": ann(res.ret[:n]), "bench": ann(b[:n]),
            "excess": ann(res.ret[:n]) - ann(b[:n]),
        })
        hb.beat(status="running",
                detail="segment %d/%d — %s to %s" % (s + 1, a.segments,
                                                      rows[-1]["from"], rows[-1]["to"]),
                progress=s / max(1, a.segments - 1),
                segments_done=len(rows),
                last_excess=round(rows[-1]["excess"], 4))
        print("  seg %d  searched %d sessions, traded %s -> %s   %+7.2f%% vs %+7.2f%%"
              % (s + 1, train_hi, rows[-1]["from"], rows[-1]["to"],
                 100 * rows[-1]["strat"], 100 * rows[-1]["bench"]), flush=True)

    if not rows:
        print("  not enough history to chain an honest walk-forward")
        return 1
    rc = _report(rows, chained, chained_bench, sl, a.pop * a.gens, "honest")
    hb.done("%d segments chained" % len(rows))
    return rc


def _report(rows, chained, chained_bench, sl, n_trials, mode):
    R = np.concatenate(chained)
    B = np.concatenate(chained_bench)
    exc = R - B
    t = bt._newey_west_tstat(exc)
    won = sum(1 for r in rows if r["excess"] > 0)
    d = metrics.dsr_from_returns(exc, n_trials=max(1, n_trials))

    print("-" * 78)
    print("  %-10s %s -> %s   (%d sessions traded forward)"
          % (mode, rows[0]["from"], rows[-1]["to"], len(R)))
    print("  strategy   %7.2f%% annualised" % (100 * ann(R)))
    print("  benchmark  %7.2f%% annualised" % (100 * ann(B)))
    print("  excess     %7.2f%% annualised" % (100 * (ann(R) - ann(B))))
    print("  segments won %d of %d" % (won, len(rows)))
    print("  alpha t    %7.2f   (Newey-West; >= 2.0 to mean anything)" % t)
    print("  excess SR  %7.2f annualised" % d["sr_ann"])
    print("  DSR        %7.2f" % d["dsr"])
    print("=" * 78)
    if t >= 2.0 and won > len(rows) / 2:
        print("  READ: the procedure beat the universe out of sample with a")
        print("  t-stat that survives. The sealed holdout remains the final word.")
    elif ann(R) > ann(B):
        print("  READ: ahead, but not significantly. Indistinguishable from luck.")
    else:
        print("  READ: the procedure did NOT beat holding the universe.")
    print("=" * 78)

    f2 = lambda v: notify._fmt(v, dp=2)                       # noqa: E731
    pc = lambda v: notify._fmt(v, pct=True)                   # noqa: E731
    p = notify.write_report("walkforward", notify.render(
        "Walk-forward blind test (%s)" % mode, [
            ("segment by segment", notify.table(rows, [
                ("seg", "seg", True, str), ("from", "from", False, str),
                ("to", "to", False, str), ("picked", "pick", False, str),
                ("expression", "expr", False, str),
                ("strategy", "strat", True, pc), ("benchmark", "bench", True, pc),
                ("excess", "excess", True, pc)])),
            ("chained", notify.table([{
                "s": ann(R), "b": ann(B), "e": ann(R) - ann(B), "t": t,
                "sr": d["sr_ann"], "dsr": d["dsr"],
                "w": "%d/%d" % (won, len(rows))}], [
                ("strategy", "s", True, pc), ("benchmark", "b", True, pc),
                ("excess", "e", True, pc), ("alpha t", "t", True, f2),
                ("excess SR", "sr", True, f2), ("DSR", "dsr", True, f2),
                ("segments won", "w", False, str)])),
        ],
        "%s mode. Training window only; holdout untouched (%s of %s left)."
        % (mode, sl.remaining() if sl else "?",
           sl.max_evaluations if sl else "?")))
    print("  report: %s" % p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--embargo", type=int, default=None)
    ap.add_argument("--honest", action="store_true",
                    help="re-run a fresh search per segment using only prior "
                         "data (slow, and the only uncontaminated version)")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gens", type=int, default=15)
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()
    uni = universe.Universe.load(cfg)
    panel = data.load_panel(cfg, uni, progress=False)

    sl = seal.Seal.load(cfg)
    s0, s1 = seal.train_window(cfg)
    train = panel.slice_dates(s0, s1)
    if sl:
        sl.assert_train_clean(train)      # raises if the holdout leaked in

    feats = features.build(train)
    bench = bt.equal_weight_benchmark(train)

    if a.honest:
        return honest_walkforward(cfg, train, feats, bench, sl, a)

    cands, state = load_candidates(cfg, a.candidates)
    print("")
    print("  *** CONTAMINATED POOL ***")
    print("  These candidates were selected by a search that saw the WHOLE")
    print("  training window, including every segment tested below. Ranking")
    print("  them on earlier segments does not undo that. Treat the result as")
    print("  an upper bound, not a forecast. Use --honest for the real thing.")
    print("")

    T = train.shape[0]
    emb = a.embargo if a.embargo is not None else int(cfg.cv.embargo_days)
    edges = np.linspace(0, T, a.segments + 1).astype(int)

    print("=" * 78)
    print("  WALK-FORWARD BLIND TEST — the selection procedure, not one strategy")
    print("=" * 78)
    print("  window     %s → %s  (%d sessions, holdout untouched)"
          % (train.dates[0].date(), train.dates[-1].date(), T))
    print("  candidates %d archive strategies, re-ranked at every boundary"
          % len(cands))
    print("  segments   %d, %d-day embargo at each edge" % (a.segments, emb))
    print("-" * 78)

    # Score every candidate on every segment once; the walk-forward then only
    # needs to slice that matrix rather than re-run thousands of backtests.
    seg_ret = np.zeros((len(cands), a.segments), dtype=object)
    for ci, (_fp, _canon, g) in enumerate(cands):
        try:
            score = g.score(feats, train.tradable)
        except Exception:
            continue
        for s in range(a.segments):
            lo = edges[s] + (emb if s > 0 else 0)
            hi = edges[s + 1]
            if hi - lo < 40:
                seg_ret[ci, s] = None
                continue
            sub = cv._slice_panel(train, lo, hi)
            res = bt.run(sub, score[lo:hi], g, cfg, bench_ret=bench[lo:hi])
            seg_ret[ci, s] = res.ret if res.ok else None
        print("\r  scoring %d/%d" % (ci + 1, len(cands)), end="", flush=True)
    print()

    chained, chained_bench, rows = [], [], []
    for s in range(1, a.segments):
        # Rank using ONLY segments strictly before s.
        best, best_score = None, -np.inf
        for ci in range(len(cands)):
            past = [seg_ret[ci, k] for k in range(s) if seg_ret[ci, k] is not None]
            if len(past) < max(1, s // 2):
                continue
            r = np.concatenate(past)
            b = np.concatenate([bench[edges[k] + (emb if k > 0 else 0):edges[k + 1]]
                                for k in range(s)
                                if seg_ret[ci, k] is not None])
            n = min(len(r), len(b))
            exc = r[:n] - b[:n]
            sd = float(np.std(exc, ddof=1)) if n > 2 else 0.0
            sc = (float(np.mean(exc)) / sd * np.sqrt(TD)) if sd > 1e-12 else -np.inf
            if sc > best_score:
                best_score, best = sc, ci

        fwd = seg_ret[best, s] if best is not None else None
        if fwd is None:
            continue
        lo = edges[s] + emb
        hi = edges[s + 1]
        b = bench[lo:hi]
        n = min(len(fwd), len(b))
        chained.append(fwd[:n])
        chained_bench.append(b[:n])
        rows.append({
            "seg": s + 1,
            "from": str(train.dates[lo].date()),
            "to": str(train.dates[hi - 1].date()),
            "pick": (cands[best][0] or "")[:8],
            "expr": (cands[best][1] or "")[:54],
            "strat": ann(fwd[:n]),
            "bench": ann(b[:n]),
            "excess": ann(fwd[:n]) - ann(b[:n]),
        })

    if not rows:
        print("  not enough history to chain a walk-forward")
        return 1

    print()
    print("  %-4s %-11s %-11s %-9s %9s %9s %9s" %
          ("SEG", "FROM", "TO", "PICKED", "STRAT", "BENCH", "EXCESS"))
    for r in rows:
        print("  %-4d %-11s %-11s %-9s %8.2f%% %8.2f%% %8.2f%%" %
              (r["seg"], r["from"], r["to"], r["pick"],
               100 * r["strat"], 100 * r["bench"], 100 * r["excess"]))

    R = np.concatenate(chained)
    B = np.concatenate(chained_bench)
    exc = R - B
    t = bt._newey_west_tstat(exc)
    won = sum(1 for r in rows if r["excess"] > 0)

    d = metrics.dsr_from_returns(exc, n_trials=max(1, len(cands)))

    print("-" * 78)
    print("  chained    %s → %s   (%d sessions traded forward)"
          % (rows[0]["from"], rows[-1]["to"], len(R)))
    print("  strategy   %7.2f%% annualised" % (100 * ann(R)))
    print("  benchmark  %7.2f%% annualised   (equal-weight eligible universe)"
          % (100 * ann(B)))
    print("  excess     %7.2f%% annualised" % (100 * (ann(R) - ann(B))))
    print("  segments won %d of %d" % (won, len(rows)))
    print("  alpha t    %7.2f   (Newey-West; >= 2.0 to mean anything)" % t)
    print("  excess SR  %7.2f annualised" % d["sr_ann"])
    print("  DSR        %7.2f   (deflated for %d candidates re-ranked each step)"
          % (d["dsr"], len(cands)))
    print("=" * 78)

    if t >= 2.0 and won > len(rows) / 2:
        print("  READ: picking the best-so-far and trading it forward beat the")
        print("  universe with a t-stat that survives. That is the procedure")
        print("  working, not one lucky strategy. The sealed holdout is still the")
        print("  final word — it has never been scored and stays that way.")
    elif ann(R) > ann(B):
        print("  READ: ahead of the universe, but not significantly. This is the")
        print("  outcome the video reached — a number that looks like a win and")
        print("  cannot be distinguished from luck. Do not size up on it.")
    else:
        print("  READ: the procedure did NOT beat holding the universe. The")
        print("  strategy that looks best in-sample is not the one that wins")
        print("  next, which is exactly what a high PBO predicts.")
    print("=" * 78)

    f2 = lambda v: notify._fmt(v, dp=2)                       # noqa: E731
    pc = lambda v: notify._fmt(v, pct=True)                   # noqa: E731
    sections = [
        ("segment by segment", notify.table(rows, [
            ("seg", "seg", True, str), ("from", "from", False, str),
            ("to", "to", False, str), ("picked", "pick", False, str),
            ("expression", "expr", False, str),
            ("strategy", "strat", True, pc), ("benchmark", "bench", True, pc),
            ("excess", "excess", True, pc)])),
        ("chained result", notify.table([{
            "s": ann(R), "b": ann(B), "e": ann(R) - ann(B), "t": t,
            "sr": d["sr_ann"], "dsr": d["dsr"], "w": "%d/%d" % (won, len(rows)),
        }], [
            ("strategy", "s", True, pc), ("benchmark", "b", True, pc),
            ("excess", "e", True, pc), ("alpha t", "t", True, f2),
            ("excess SR", "sr", True, f2), ("DSR", "dsr", True, f2),
            ("segments won", "w", False, str)])),
    ]
    p = notify.write_report("walkforward", notify.render(
        "Walk-forward blind test", sections,
        "Re-ranking %d candidates at every boundary and trading the winner "
        "forward. Training window only — the holdout is untouched, %d of %d "
        "evaluations remain." % (len(cands),
                                 sl.remaining() if sl else 0,
                                 sl.max_evaluations if sl else 0)))
    print("  report: %s" % p)
    print("  holdout: %s" % ("%d of %d evaluations remaining"
                             % (sl.remaining(), sl.max_evaluations) if sl else "no seal"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
