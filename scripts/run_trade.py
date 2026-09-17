"""Execution loop: turn promoted strategies into orders.

    python scripts/run_trade.py --dry-run          # print the orders, send none
    python scripts/run_trade.py                    # paper broker
    python scripts/run_trade.py --daemon           # paper, every session

Live trading needs all three locks open (see loonie/broker/base.py). Nothing
in this repository opens them for you, and --dry-run is the default posture
for a reason: read the order list before you let anything send it.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used below; without this every run dies on a UnicodeEncodeError in a
# print statement rather than in anything that matters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import numpy as np  # noqa: E402

from loonie import (allocator as alloc_mod, config, data, evolve,  # noqa: E402
                    features, notify, portfolio, publish, registry,
                    risk, universe)
from loonie.broker import get_broker  # noqa: E402
from loonie.genome import Genome  # noqa: E402


def load_strategies(cfg, limit: int = 5) -> dict:
    """Promoted genomes if any exist; otherwise the archive's best elites.

    Running un-promoted elites is legitimate in PAPER -- that is how a
    candidate builds the live track record the allocator learns from. It is
    not legitimate with real money, and the gate is what separates the two.
    """
    import json

    p = config.resolve(evolve.STATE)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        src = d.get("hall_of_fame") or d.get("archive", [])
        promoted = bool(d.get("hall_of_fame"))
    else:
        # CI runners have the committed strategies file but not the 1 MB
        # evolution state, which is deliberately gitignored to keep the repo
        # from gaining a megabyte of churn every twenty seconds.
        q = config.resolve("docs/data/strategies.json")
        if not q.exists():
            return {}
        d = json.loads(q.read_text(encoding="utf-8"))
        src = d.get("strategies", [])
        promoted = bool(d.get("promoted"))
        print("[trade] using committed strategies.json (gen %s, %s trials)"
              % (d.get("generation"), d.get("trials")))
    out = {}
    for e in src[:limit]:
        try:
            g = Genome.from_dict(e["genome"] if "genome" in e else e)
        except Exception as exc:
            print("[trade] skipping unreadable entry %s: %s"
                  % (e.get("fingerprint", "?"), exc))
            continue
        key = ("P:" if promoted else "C:") + e.get("fingerprint", g.fingerprint)
        out[key] = g
    return out


def build_report(target, orders, account, decision, strategies, alloc, live):
    f2 = lambda v: notify._fmt(v, dp=2)                       # noqa: E731
    f4 = lambda v: notify._fmt(v, dp=4)                       # noqa: E731
    money = lambda v: "$%s" % format(float(v or 0), ",.2f")   # noqa: E731

    pos_rows = [{
        "sym": s, "qty": p.qty, "avg": p.avg_price, "px": p.market_price,
        "mv": p.market_value, "pl": p.unrealized_pl,
    } for s, p in sorted(account.positions.items())]

    ord_rows = [{
        "sym": o.symbol, "side": o.side.upper(),
        "size": (o.notional if o.notional else o.qty),
        "kind": "notional" if o.notional else "shares",
        "status": o.status, "note": o.note,
    } for o in orders]

    tgt_rows = [{"sym": s, "w": w} for s, w in
                sorted(target.weights.items(), key=lambda kv: -kv[1])]

    alloc_rows = [{"key": k, "w": v,
                   "n": alloc.arms[k].n if k in alloc.arms else 0,
                   "mean": alloc.arms[k].mu if k in alloc.arms else 0.0}
                  for k, v in sorted(target.alloc.items(), key=lambda kv: -kv[1])]

    chk_rows = [{"check": c[0], "value": c[1], "limit": c[2],
                 "ok": "PASS" if c[3] else "BREACH"} for c in (decision.checks or [])]

    sections = [
        ("account", notify.table([{
            "mode": "LIVE" if live else "PAPER",
            "equity": account.equity, "cash": account.cash,
            "gross": account.gross_exposure, "n": len(account.positions),
        }], [("mode", "mode", False, str), ("equity", "equity", True, money),
             ("cash", "cash", True, money),
             ("gross exposure", "gross", True, lambda v: "%.1f%%" % (100 * v)),
             ("positions", "n", True, str)])),
        ("risk checks", notify.table(chk_rows, [
            ("check", "check", False, str), ("value", "value", True, f4),
            ("limit", "limit", True, f4), ("status", "ok", False, str)])),
        ("capital allocation (Thompson)", notify.table(alloc_rows, [
            ("strategy", "key", False, str),
            ("weight", "w", True, lambda v: "%.1f%%" % (100 * v)),
            ("live days", "n", True, str),
            ("mean daily excess", "mean", True, f4)])),
        ("target book", notify.table(tgt_rows, [
            ("symbol", "sym", False, str),
            ("target weight", "w", True, lambda v: "%.2f%%" % (100 * v))])),
        ("orders", notify.table(ord_rows, [
            ("symbol", "sym", False, str), ("side", "side", False, str),
            ("size", "size", True, f2), ("unit", "kind", False, str),
            ("status", "status", False, str), ("note", "note", False, str)])),
        ("current positions", notify.table(pos_rows, [
            ("symbol", "sym", False, str), ("qty", "qty", True, f2),
            ("avg", "avg", True, money), ("last", "px", True, money),
            ("value", "mv", True, money), ("unrealised", "pl", True, money)])),
    ]
    return notify.render(
        "%s session - %s" % ("LIVE" if live else "PAPER",
                             datetime.now().strftime("%Y-%m-%d %H:%M")),
        sections,
        "%d strategies live. %s"
        % (len(strategies),
           "Orders were NOT submitted (dry run)." if not orders
           else "Allocator weights reflect realised paper P&L, not backtest."))


def session(cfg, args) -> int:
    hb = registry.Worker("trade", "trade", "paper execution")
    hb.beat(status="running", detail="loading market data")
    uni = universe.Universe.load(cfg)
    panel = data.load_panel(cfg, uni, progress=False)
    feats = features.build(panel)

    strategies = load_strategies(cfg, args.max_strategies)
    if not strategies:
        print("[trade] no strategies in state/evolve_state.json -- run "
              "scripts/run_evolve.py first")
        return 3
    promoted = [k for k in strategies if k.startswith("P:")]
    print("[trade] %d strategies (%d promoted, %d unpromoted candidates)"
          % (len(strategies), len(promoted), len(strategies) - len(promoted)))
    if not promoted:
        print("[trade] NOTE: nothing has cleared the promotion gate. These are "
              "candidates\n        building a paper track record, not "
              "validated strategies.")
        if args.live:
            print("[trade] refusing to trade un-promoted candidates with real "
                  "money.")
            return 4

    broker = get_broker(cfg, cli_live_flag=args.live)
    is_live = getattr(broker, "is_live", False)

    marks = {}
    last_row = -1
    for j, sym in enumerate(panel.tickers):
        px = panel.close[last_row, j]
        if np.isfinite(px) and px > 0:
            marks[sym] = float(px)
    if hasattr(broker, "set_marks"):
        broker.set_marks(marks)

    account = broker.account()
    rm = risk.RiskManager(cfg)
    decision = rm.check(account.equity, account=account,
                        last_bar_date=panel.dates[-1])

    alloc = alloc_mod.Allocator(cfg)
    target = portfolio.target_portfolio(
        strategies, feats, panel, allocator=alloc if cfg.allocator.enabled else None,
        capital_pct=float(cfg.trade.capital_pct),
        max_position_pct=float(cfg.trade.max_position_pct))

    orders = portfolio.diff_to_orders(
        target.weights, account.positions, account.equity, marks,
        min_notional=float(cfg.trade.min_order_notional))

    print("[trade] last bar %s | equity $%s | %d positions | %d orders proposed"
          % (panel.dates[-1].date(), format(account.equity, ",.2f"),
             len(account.positions), len(orders)))
    for n in (target.notes or []):
        print("[trade] note: %s" % n)

    if not decision.allow:
        print("[trade] RISK HALT: %s" % decision.reason)
        if decision.halt and args.flatten_on_halt:
            n = broker.close_all()
            print("[trade] flattened %d positions" % n)
        orders = []
    elif args.dry_run:
        print("[trade] DRY RUN -- no orders submitted:")
        for o in orders:
            size = ("$%.2f" % o.notional) if o.notional else ("%.4f sh" % o.qty)
            print("    %-5s %-6s %-12s  %s" % (o.side.upper(), o.symbol, size, o.note))
    else:
        sent = 0
        # Sells first: they free the exposure the buys need. Submitting in the
        # diff's arbitrary order meant the buys were measured against a book
        # that had not been reduced yet, so a full book rejected every buy no
        # matter how many sells came later in the same batch.
        orders = sorted(orders, key=lambda o: 0 if str(o.side).lower() == "sell"
                        else 1)
        pending = 0.0
        for o in orders:
            notional = o.notional or (o.qty * marks.get(o.symbol, 0))
            ok = rm.check_order(o.symbol, notional, account.equity, account,
                                side=o.side, pending=pending)
            if not ok:
                print("[trade] skip %s: %s" % (o.symbol, ok.reason))
                continue
            broker.submit(o)
            rm.note_order()
            pending += (-notional if str(o.side).lower() == "sell" else notional)
            sent += 1
            print("    %-5s %-6s %-10s -> %s %s"
                  % (o.side.upper(), o.symbol,
                     ("$%.2f" % o.notional) if o.notional else ("%.4f" % o.qty),
                     o.status, o.note))
        print("[trade] submitted %d orders" % sent)
        account = broker.account()

    try:
        publish.publish(cfg)
    except Exception as e:
        print("[publish] skipped: %s: %s" % (type(e).__name__, e))

    hb.done(detail="%d orders, %d positions, equity $%s"
                   % (len(orders), len(account.positions),
                      format(account.equity, ",.2f")))
    hb.beat(equity=round(account.equity, 2),
            positions=len(account.positions),
            orders=len(orders),
            halted=bool(decision.halt))

    if cfg.report.enabled:
        p = notify.write_report("trade", build_report(
            target, orders, account, decision, strategies, alloc, is_live))
        print("[trade] report: %s" % p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the order list and submit nothing")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--interval", type=int, default=3600)
    ap.add_argument("--max-strategies", type=int, default=5)
    ap.add_argument("--flatten-on-halt", action="store_true")
    ap.add_argument("--i-understand-this-is-real-money", dest="live",
                    action="store_true",
                    help="third live-trading lock; the other two are in "
                         "config.yaml and the environment")
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()

    if a.live:
        print("!" * 72)
        print("  LIVE TRADING FLAG SET. Real money. Verify the other two locks.")
        print("!" * 72)

    if not a.daemon:
        return session(cfg, a)

    while True:
        try:
            session(cfg, a)
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print("[trade] session failed: %s: %s" % (type(e).__name__, e))
        time.sleep(max(60, a.interval))


if __name__ == "__main__":
    raise SystemExit(main())
