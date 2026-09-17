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


class _Tee:
    """Write to the console AND to state/system.log.

    The orchestrator logs with print(), so where its history ends up depends
    entirely on how someone launched it. Start it with stdout pointed anywhere
    else and state/system.log simply stops -- still present, still readable,
    quietly five hours out of date. That is worse than having no log, because
    the stale lines still look like the current state: it is how a run of
    failing publish jobs stayed invisible here for an entire evening.

    So the file is written from inside the process, whatever the caller does
    with stdout.
    """

    def __init__(self, stream, path):
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", errors="replace")

    def write(self, s):
        self.stream.write(s)
        try:
            self.fh.write(s)
            self.fh.flush()          # a log you have to wait for is not a log
        except Exception:
            pass                     # never let logging kill the supervisor
        return len(s)

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


sys.stdout = _Tee(sys.stdout, config.resolve("state/system.log"))
sys.stderr = _Tee(sys.stderr, config.resolve("state/system.log"))


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
