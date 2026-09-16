"""Download SEC company facts for the universe.

    python scripts/fetch_fundamentals.py [--limit N] [--refresh]

Free, permanent, and point-in-time correct: every XBRL fact carries the date it
was filed, which is the only date a backtest is allowed to use. Runs at roughly
six requests a second, inside the SEC's published cap of ten.

The raw documents are several megabytes each and are never stored. Only the
dozen concepts the features actually read survive to disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loonie import fundamentals as FU  # noqa: E402
from loonie.config import load, load_env  # noqa: E402
from loonie.universe import Universe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load()
    u = Universe.load(cfg)
    tickers = u.tickers_active_between(cfg.universe.start, cfg.universe.end)
    if args.limit:
        tickers = tickers[:args.limit]

    m = FU.cik_map(refresh=args.refresh)
    mapped = [t for t in tickers if FU._norm(t) in m]
    print("[sec] %d universe tickers, %d map to a CIK (%.1f%%)"
          % (len(tickers), len(mapped), 100 * len(mapped) / max(1, len(tickers))))

    t0 = time.time()

    def progress(i, n, ok, miss):
        if i % 20 == 0 or i == n:
            el = time.time() - t0
            print("\r[sec] %d/%d  ok=%d  missing=%d  %.0fs"
                  % (i, n, ok, miss, el), end="", flush=True)

    res = FU.fetch_universe(tickers, refresh=args.refresh, progress=progress)
    print()

    facts = res["facts"]
    print("[sec] %d tickers with facts, %d without"
          % (len(facts), len(res["missing"])))
    if res["missing"]:
        print("[sec] missing sample: %s" % ", ".join(sorted(res["missing"])[:20]))

    rows = sum(len(d) for d in facts.values())
    size = sum(p.stat().st_size for p in
               FU.resolve(FU.CACHE).glob("*.parquet")) / 1e6
    print("[sec] %d facts cached across %d files (%.1f MB)"
          % (rows, len(facts), size))

    # A quick look at what the oldest usable history is: features that need a
    # year of trailing data cannot start until a year after the first filing.
    if facts:
        first = min(d["filed"].min() for d in facts.values() if len(d))
        print("[sec] earliest filing in cache: %s" % first.date())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
