"""One-screen view of the whole system.

    python scripts/status.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loonie import allocator as alloc_mod, config, evolve, risk, seal  # noqa: E402


def box(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


def main() -> int:
    config.load_env()
    cfg = config.load()

    box("SEARCH")
    p = config.resolve(evolve.STATE)
    if not p.exists():
        print("  no evolution state -- run scripts/run_evolve.py")
    else:
        d = json.loads(p.read_text(encoding="utf-8"))
        h = (d.get("history") or [{}])[-1]
        print("  generation      %s" % d.get("generation"))
        print("  trials          %s  (feeds the multiple-testing correction)"
              % d.get("trials"))
        print("  archive cells   %s" % h.get("archive_cells"))
        print("  promoted        %s" % len(d.get("hall_of_fame", [])))
        print("  best fitness    %s" % _r(h.get("best_fitness")))
        print("  best alpha t    %s" % _r(h.get("best_alpha_t")))
        print("  best DSR        %s" % _r(h.get("best_dsr")))
        print("  best vs random  %s" % _r(h.get("best_vs_null_pct")))
        ns = d.get("null_summary", {})
        if ns.get("n"):
            print("  null pool       %d random genomes | p99 fitness %s"
                  % (ns["n"], _r(ns.get("p99_fitness"))))
        cov = (d.get("panel") or {}).get("coverage", {})
        if cov:
            print("  data            %s, %.1f%% universe coverage, %s losers missing"
                  % (cov.get("provider"), 100 * cov.get("coverage", 0),
                     cov.get("missing_departed")))
        if d.get("hall_of_fame"):
            print("\n  promoted strategies:")
            for e in d["hall_of_fame"]:
                print("   - %s" % e["canonical"][:96])
        else:
            print("\n  nothing has cleared the promotion gate.")

    box("HOLDOUT SEAL")
    sl = seal.Seal.load(cfg)
    print("  " + (sl.describe().replace("\n", "\n  ") if sl else "not created yet"))
    if sl and sl.ledger:
        print("\n  ledger:")
        for e in sl.ledger[-5:]:
            print("   - %s %s %s" % (e.get("at"), e.get("event"),
                                     (e.get("genome") or "")[:16]))

    box("ALLOCATOR (live learning)")
    al = alloc_mod.Allocator(cfg)
    rows = al.report()
    if not rows:
        print("  no live track records yet -- run scripts/run_trade.py")
    else:
        print("  %-20s %6s %12s %12s %10s" % ("strategy", "days", "mean daily",
                                              "cumulative", "SR ann"))
        for r in rows:
            print("  %-20s %6d %12.6f %12.4f %10.2f"
                  % (r["key"][:20], r["n"], r["mean_daily"], r["cumulative"],
                     r["sharpe_ann"]))
        w = al.weights()
        if w:
            print("\n  current Thompson weights:")
            for k, v in sorted(w.items(), key=lambda kv: -kv[1]):
                print("   %-20s %6.1f%%" % (k[:20], 100 * v))

    box("RISK")
    rm = risk.RiskManager(cfg)
    s = rm.state
    print("  halted          %s" % ("YES -- %s" % s.halt_reason if s.halted else "no"))
    print("  equity HWM      %.2f" % s.equity_high_water)
    print("  day             %s (start equity %.2f, %d orders)"
          % (s.day or "-", s.day_start_equity, s.orders_today))
    if s.breaches:
        print("  recent breaches:")
        for b in s.breaches[-3:]:
            print("   - %s %s" % (b["at"], b["reason"][:70]))

    box("BROKER")
    mode = str(cfg.trade.mode)
    print("  configured      %s (%s)" % (cfg.trade.broker, mode))
    print("  live locks      config=%s  env=%s  cli=(per-invocation)"
          % (cfg.trade.get("allow_live"), __import__("os").getenv("ALPACA_MODE", "paper")))
    try:
        from loonie.broker import get_broker
        b = get_broker(cfg)
        acct = b.account()
        print("  broker          %s | equity $%s | cash $%s | %d positions"
              % (b.name, format(acct.equity, ",.2f"), format(acct.cash, ",.2f"),
                 len(acct.positions)))
    except Exception as e:
        print("  broker          unavailable: %s" % e)
    print()
    return 0


def _r(v):
    try:
        return "%.3f" % float(v)
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    raise SystemExit(main())
