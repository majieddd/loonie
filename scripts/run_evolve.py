"""The self-improvement daemon. Runs the search, indefinitely if you let it.

    python scripts/run_evolve.py --generations 40
    python scripts/run_evolve.py --daemon            # 24/7, resumes on restart

State is checkpointed every generation to state/evolve_state.json, so killing
this process and restarting it loses at most one generation. The holdout is
sliced off before the search ever sees the panel, and seal.assert_train_clean
raises if that slicing is wrong.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used below; without this every run dies on a UnicodeEncodeError in a
# print statement rather than in anything that matters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import numpy as np  # noqa: E402

from loonie import (config, data, evolve, experience, features,  # noqa: E402
                    methods, notify, publish, registry, seal, universe)


def build_report(ev: evolve.Evolver, panel) -> str:
    hist = ev.history[-1] if ev.history else {}
    ops = ev.bandit.table()
    op_rows = [{"op": k, **v} for k, v in sorted(
        ops.items(), key=lambda kv: -kv[1]["posterior_mean"])]

    elites = ev.leaderboard(10)
    elite_rows = [{
        "expr": e["canonical"][:90],
        "fit": e["fitness"],
        "ir": e["cv"]["mean_ir"],
        "t": e["cv"]["alpha_tstat"],
        "pos": e["cv"]["frac_positive"],
        "corr": e["cv"]["corr_bench"],
        "to": e["cv"]["ann_turnover"],
    } for e in elites]

    f2 = lambda v: notify._fmt(v, dp=2)                      # noqa: E731
    f3 = lambda v: notify._fmt(v, dp=3)                      # noqa: E731
    txt = lambda v: "<code>%s</code>" % v if v else "-"      # noqa: E731

    sections = [
        ("search state", notify.table([{
            "gen": hist.get("generation"),
            "trials": hist.get("trials"),
            "cells": hist.get("archive_cells"),
            "explore": hist.get("explore"),
            "promoted": hist.get("promoted_total"),
            "best_fit": hist.get("best_fitness"),
        }], [
            ("generation", "gen", True, str), ("trials", "trials", True, str),
            ("archive cells", "cells", True, str),
            ("explore rate", "explore", True, f2),
            ("promoted", "promoted", True, str),
            ("best fitness", "best_fit", True, f3),
        ])),
        ("elite strategies (quality-diversity archive)", notify.table(elite_rows, [
            ("expression", "expr", False, txt), ("fitness", "fit", True, f3),
            ("info ratio", "ir", True, f2), ("alpha t", "t", True, f2),
            ("folds +", "pos", True, f2), ("corr bench", "corr", True, f2),
            ("turnover", "to", True, f2),
        ])),
        ("mutation operators (Thompson posteriors)", notify.table(op_rows, [
            ("operator", "op", False, str), ("posterior", "posterior_mean", True, f3),
            ("win rate", "rate", True, f3), ("tried", "tried", True, str),
            ("won", "won", True, str),
        ])),
        ("data provenance", notify.table([panel.coverage], [
            ("provider", "provider", False, str),
            ("universe", "universe_size", True, str),
            ("fetched", "fetched", True, str),
            ("coverage", "coverage", True, lambda v: notify._fmt(v, pct=True)),
            ("departed & missing", "missing_departed", True, str),
        ])),
    ]
    return notify.render(
        "Evolution report - generation %s" % hist.get("generation"),
        sections,
        "%d trials evaluated. Fitness is information ratio vs an equal-weight "
        "portfolio of the same eligible universe, not total return."
        % ev.trials)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--daemon", action="store_true",
                    help="run forever, checkpointing every generation")
    ap.add_argument("--fresh", action="store_true", help="ignore saved state")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--method", default=None,
                    help="force a search method; default draws from the "
                         "method bandit (see loonie/methods.py)")
    ap.add_argument("--no-record", action="store_true",
                    help="skip writing to the experience store")
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--no-publish", action="store_true",
                    help="skip writing docs/data (the dashboard feed)")
    ap.add_argument("--push", action="store_true",
                    help="also git-push the dashboard data (throttled)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()

    # ---- pick how to search -------------------------------------------
    # A method is a behaviour-changing configuration of the search itself.
    # The bandit draws one by Thompson sampling over forward-validated
    # outcomes, so which way of searching gets used is itself learned.
    bandit = methods.MethodBandit()
    bandit.ingest_experience()
    method = (methods.BY_ID.get(a.method) if a.method
              else bandit.choose(np.random.default_rng()))
    if method is None:
        raise SystemExit("unknown method %r; have %s"
                         % (a.method, list(methods.BY_ID)))
    cfg = config.Cfg(method.apply(cfg))
    if a.population:
        cfg["evolve"]["population"] = int(a.population)

    print("=" * 72)
    print("  SELF-IMPROVING STRATEGY SEARCH")
    print("=" * 72)
    print("  method   %s -- %s" % (method.id, method.summary))
    print("           %s" % " ".join(method.rationale.split())[:150])

    uni = universe.Universe.load(cfg)
    panel = data.load_panel(cfg, uni, progress=not a.quiet)
    print(panel.describe())

    cov = panel.coverage.get("coverage", 0.0)
    floor = float(cfg.data.min_survivorship_coverage)
    if cov < floor:
        print("\n*** ABORT: universe coverage %.1f%% is below the configured "
              "floor of %.1f%%.\n    A backtest on this cache is optimistic by "
              "construction. Either widen\n    the window, lower the floor "
              "deliberately, or connect a provider that\n    carries delisted "
              "securities." % (100 * cov, 100 * floor))
        return 2

    # ---- seal the holdout BEFORE the search can see anything --------------
    sl = seal.Seal.create(cfg, panel)
    print("\n" + sl.describe())
    s0, s1 = seal.train_window(cfg)
    train = panel.slice_dates(s0, s1)
    sl.assert_train_clean(train)
    print("[seal] training window %s -> %s (%d sessions); holdout is not visible"
          % (train.dates[0].date(), train.dates[-1].date(), train.shape[0]))

    print("\n[features] building...")
    feats = features.build(train)
    print("[features] %d terminals over %d sessions x %d tickers"
          % (len(feats), train.shape[0], train.shape[1]))

    store = None if a.no_record else experience.ExperienceStore(
        method_id=method.id,
        context={"panel_start": str(train.dates[0].date()),
                 "panel_stop": str(train.dates[-1].date()),
                 "panel_sessions": int(train.shape[0]),
                 "panel_tickers": int(train.shape[1]),
                 "coverage": float(panel.coverage.get("coverage", 0))})
    ev = evolve.Evolver(cfg, train, feats, seed=a.seed, verbose=not a.quiet,
                        store=store)
    if not a.fresh:
        ev.load()

    bench = ev.bench
    ann = (np.prod(1 + bench)) ** (252 / len(bench)) - 1
    print("[benchmark] equal-weight universe: %.2f%% CAGR over the train window"
          % (100 * ann))
    print("[gate] %s\n" % dict(cfg.evolve.gate))

    gens = a.generations or int(cfg.evolve.generations)
    n_done = 0
    hb = registry.Worker("search", "search", "genetic program")
    hb.beat(status="running", detail="seeding population")
    try:
        while True:
            rec = ev.step()
            ev.save()
            n_done += 1

            hb.beat(
                status="running",
                detail="gen %d | fit %+.3f | IR %+.2f | alpha t %+.2f"
                       % (ev.generation, rec["best_fitness"], rec["best_ir"],
                          rec["best_alpha_t"]),
                progress=(n_done / gens) if not a.daemon and gens else None,
                generation=ev.generation, trials=ev.trials,
                trials_effective=rec.get("trials_effective"),
                best_fitness=round(rec["best_fitness"], 4),
                best_alpha_t=round(rec["best_alpha_t"], 3),
                best_dsr=round(rec["best_dsr"], 3),
                archive_cells=rec["archive_cells"],
                promoted=len(ev.hall_of_fame),
                demoted=len(getattr(ev, "demoted", [])),
                seconds_per_gen=rec["seconds"])

            # Feed the dashboard. Cheap (a few hundred KB of JSON); the git
            # push, if enabled, is throttled inside publish_dashboard.py so a
            # 20-second generation does not become 4,000 commits a day.
            if not a.no_publish:
                try:
                    publish.publish(cfg)
                    if a.push:
                        import subprocess
                        subprocess.run(
                            [sys.executable, "scripts/publish_dashboard.py",
                             "--push", "--quiet"],
                            cwd=str(config.ROOT), capture_output=True, timeout=180)
                except Exception as e:
                    print("[publish] skipped: %s: %s" % (type(e).__name__, e))

            if n_done % max(1, a.report_every) == 0:
                p = notify.write_report("evolve", build_report(ev, panel))
                print("[report] %s" % p)

            if not a.daemon and n_done >= gens:
                break
    except KeyboardInterrupt:
        print("\n[evolve] interrupted; state checkpointed")
        hb.done("interrupted")
    except Exception as exc:
        traceback.print_exc()
        hb.error(exc)
        return 1
    finally:
        # Flush unconditionally. A daemon is normally ended by Ctrl-C or a
        # supervisor terminate, and without this every row buffered since the
        # last 200-row flush is lost — which for a short run is all of them.
        if store is not None:
            store.flush()
            if store.written:
                print("[experience] recorded %d rows (method %s)"
                      % (store.written, method.id))

    p = notify.write_report("evolve", build_report(ev, panel))
    print("\n" + "=" * 72)
    print("  %d generations | %d trials | %d elites | %d promoted"
          % (ev.generation, ev.trials, ev.archive.coverage, len(ev.hall_of_fame)))
    print("  report: %s" % p)
    print("  state : %s" % config.resolve(evolve.STATE))
    if ev.hall_of_fame:
        print("\n  Promoted strategies (cleared every gate):")
        for h in ev.hall_of_fame[-5:]:
            print("   - %s" % h["canonical"][:100])
            print("     IR %.2f  alpha_t %.2f  DSR %.2f  corr %.2f"
                  % (h["cv"]["mean_ir"], h["cv"]["alpha_tstat"],
                     h["cv"].get("dsr", float("nan")), h["cv"]["corr_bench"]))
    else:
        print("\n  Nothing cleared the promotion gate. That is a real result,")
        print("  not a bug -- see docs/WHY.md. Most searches end here.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
