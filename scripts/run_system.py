"""Start everything. One command, runs forever.

    python scripts/run_system.py
    python scripts/run_system.py --serve --tunnel     # + dashboard + public URL

Supervises five jobs and restarts anything that dies:

    search      continuous   genetic program
    data        every 6h     price and macro tails, incremental
    validate    every 8h     honest walk-forward of the current champion
    trade       weekdays     one PAPER rebalance
    publish     every 60s    dashboard snapshot

Every job heartbeats into state/workers/, so the dashboard shows what is
running right now rather than only what has been computed.

PAPER ONLY. Nothing this launches can arm live trading.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loonie import config, orchestrator, registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=250)
    ap.add_argument("--serve", action="store_true",
                    help="also run the dashboard web server")
    ap.add_argument("--tunnel", action="store_true",
                    help="with --serve, expose a public https URL")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()

    pruned = registry.prune()
    if pruned:
        print("[system] pruned %d stale worker heartbeats" % pruned)

    web = None
    if a.serve:
        argv = [sys.executable, "-u", "scripts/serve.py", "--no-open"]
        if a.port:
            argv += ["--port", str(a.port)]
        if a.tunnel:
            argv.append("--tunnel")
        log = config.resolve("state/serve.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        web = subprocess.Popen(argv, cwd=str(config.ROOT),
                               stdout=open(log, "ab", buffering=0),
                               stderr=subprocess.STDOUT)
        print("[system] dashboard server started (pid %d); URL in %s"
              % (web.pid, log))

    print("=" * 70)
    print("  loonie — full system")
    print("=" * 70)
    orc = orchestrator.Orchestrator(cfg, population=a.population)
    try:
        orc.run(interval=a.interval)
    finally:
        if web is not None:
            try:
                web.terminate()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
