"""Rebuild the analytical database from the files that remain the source of truth.

    python scripts/build_warehouse.py
    python scripts/build_warehouse.py --query "SELECT ..."

The warehouse is derived and disposable. Deleting data/loonie.duckdb loses
nothing, because every table here is rebuilt from the npz panels, the parquet
caches and the JSON results that produced it. That property is deliberate: an
analytical store that becomes the only copy of something has stopped being an
analytical store.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loonie import markets as M  # noqa: E402
from loonie import warehouse as W  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402


def stocks_panel():
    """The wide equity panel, wrapped so the loader sees one shape."""
    p = resolve("data/universe/wide_panel.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    close = z["close"]
    return M.MarketPanel(
        market="stocks", dates=pd.DatetimeIndex(z["dates"]),
        symbols=list(z["tickers"]), close=close, open=z["open"],
        high=z["high"], low=z["low"], volume=z["volume"],
        tradable=z["member"] & np.isfinite(close) & (close > 0),
        meta={"source": "yfinance (wide, rule-based membership)",
              "caveat": "listed survivors only; delisted names absent, and "
                        "worse here than in the S&P panel because the rule "
                        "reaches further down the liquidity curve"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="")
    a = ap.parse_args()

    load_env()
    load()
    con = W.connect()

    if a.query:
        print(con.execute(a.query).df().to_string(index=False))
        return 0

    t0 = time.time()
    print("[db] building %s" % W.DB)

    panels = [("stocks", stocks_panel())]
    for m in ("crypto", "forex", "volatility"):
        panels.append((m, M.load(m)))

    for name, panel in panels:
        if panel is None:
            print("  %-11s skipped (not built)" % name)
            continue
        n = W.load_market(con, name, panel)
        print("  %-11s %9s price rows" % (name, format(n, ",")))

    print("  %-11s %9s rows" % ("strategies", W.load_strategies(con)))
    print("  %-11s %9s rows" % ("congress", W.load_congress(con)))
    print("  %-11s %9s rows" % ("factors", W.load_factors(con)))

    s = W.summary(con)
    print("\nTABLES")
    for k, v in s.items():
        print("  %-16s %12s" % (k, format(v, ",")))

    size = resolve(W.DB).stat().st_size / 1e6
    print("\n%s  %.1f MB  built in %.0fs" % (W.DB, size, time.time() - t0))

    # A first question the warehouse makes askable in one statement, and which
    # previously needed a bespoke script per market.
    print("\nCOVERAGE BY MARKET")
    print(con.execute("""
        SELECT m.market, m.symbols, m.sessions,
               m.start_date, m.end_date,
               ROUND(AVG(s.sessions), 0) AS avg_sessions_per_symbol
        FROM markets m JOIN symbols s USING (market)
        GROUP BY ALL ORDER BY m.market
    """).df().to_string(index=False))

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
