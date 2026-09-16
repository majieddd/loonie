"""Populate the local price cache over the point-in-time universe.

    python scripts/fetch_data.py [--provider yfinance|alpaca] [--refresh]

Prints a survivorship-coverage audit at the end. Read it. If coverage is low,
every backtest you run on this cache is optimistic by construction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loonie import config, data, universe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()
    uni = universe.Universe.load(cfg)

    rep = uni.survivorship_report(a.start or cfg.universe.start,
                                  a.end or cfg.universe.end)
    print("[universe] tickers ever in window : %d" % rep["tickers_ever_in_window"])
    print("[universe] departed since         : %d (%.1f%%)"
          % (rep["departed"], 100 * rep["departed_frac"]))

    panel = data.load_panel(cfg, uni, start=a.start, end=a.end,
                            provider=a.provider, refresh=a.refresh)
    print()
    print(panel.describe())
    print()
    print("[cache] %s" % config.resolve("data/cache/%s" % panel.coverage["provider"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
