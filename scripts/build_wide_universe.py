"""Build the wide universe and measure what breadth actually buys.

    python scripts/build_wide_universe.py [--names 3000] [--limit 0]

The S&P 500 panel gives ~500 names a day. Cross-sectional statistics scale
with that count, so the same underlying signal measured across 3,000 names
carries a t-statistic roughly sqrt(6) times larger. After 300,000 trials
failing to clear a bar on 500 names, breadth is the cheapest remaining lever.

Membership is a rule -- the N most liquid names by trailing median dollar
volume, recomputed monthly -- because point-in-time membership for a wide
index is not freely published, and applying today's constituent list to
history is precisely the bias this project exists to avoid.

The script prints a coverage audit next to the breadth number, and the two
should be read together. Reaching further down the liquidity curve means
reaching into names that fail more often, and the price provider drops
securities when they delist. Breadth buys statistical power. It buys nothing
in honesty, and this is the report that says so.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loonie import wide  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402

CACHE = "data/universe/wide_panel.npz"
BATCH = 300


def fetch_prices(symbols, start, end, batch=BATCH):
    """Batch-download OHLCV. Returns dict of field -> DataFrame."""
    import yfinance as yf

    frames = {f: [] for f in ("Open", "High", "Low", "Close", "Volume")}
    t0 = time.time()
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            d = yf.download(chunk, start=start, end=end, auto_adjust=True,
                            progress=False, threads=True, group_by="column")
        except Exception as e:
            print("\n[wide] batch %d failed (%s); continuing" % (i, e))
            continue
        if d is None or not len(d):
            continue
        for f in frames:
            if isinstance(d.columns, pd.MultiIndex):
                if f in d.columns.get_level_values(0):
                    frames[f].append(d[f])
            elif f.lower() in [str(c).lower() for c in d.columns]:
                frames[f].append(d[[f]].rename(columns={f: chunk[0]}))
        done = min(i + batch, len(symbols))
        print("\r[wide] %d/%d symbols  %.0fs"
              % (done, len(symbols), time.time() - t0), end="", flush=True)
    print()
    return {f: (pd.concat(v, axis=1) if v else pd.DataFrame())
            for f, v in frames.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=3000,
                    help="target universe size per date")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap symbols fetched (0 = all); for a quick trial")
    ap.add_argument("--min-dollar-vol", type=float, default=1e6)
    a = ap.parse_args()

    load_env()
    cfg = load()
    start, end = str(cfg.universe.start), "2026-12-31"

    syms = list(wide.fetch_symbols().symbol)
    if a.limit:
        syms = syms[:a.limit]
    print("[wide] %d candidate common stocks from the exchange directory"
          % len(syms))

    px = fetch_prices(syms, start, end)
    close = px["Close"]
    if close.empty:
        print("[wide] no price data")
        return 1

    # Drop names with almost no history: they contribute nothing to a
    # cross-section and inflate the "names ever" count misleadingly.
    usable = close.columns[close.notna().sum() >= 250]
    close = close[usable].sort_index()
    print("[wide] %d symbols with >= 250 sessions (%.0f%% of directory)"
          % (len(usable), 100 * len(usable) / max(len(syms), 1)))

    dates = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    fields = {}
    for f, key in (("close", "Close"), ("open", "Open"), ("high", "High"),
                   ("low", "Low"), ("volume", "Volume")):
        d = px[key]
        d = d.reindex(columns=usable).reindex(index=close.index)
        fields[f] = d.to_numpy(dtype="float32")

    member = wide.membership(
        fields["close"], fields["volume"], n_names=a.names,
        min_dollar_vol=a.min_dollar_vol,
        min_price=float(cfg.universe.min_price))
    rep = wide.summarise(member, dates)

    print()
    print("UNIVERSE BUILT BY RULE (top %d by trailing median dollar volume)" % a.names)
    print("  names ever in the universe : %d" % rep["names_ever"])
    print("  median names per day       : %.0f" % rep["median_per_day"])
    print("  range per day              : %d .. %d"
          % (rep["min_per_day"], rep["max_per_day"]))
    print("  churn per monthly refresh  : %.1f%%"
          % (100 * rep["median_churn_per_refresh"]))
    print("  first populated date       : %s" % rep["first_populated"])

    sp = len(__import__("loonie.universe", fromlist=["Universe"])
             .Universe.load(cfg).tickers_active_between(start, end))
    gain = (rep["median_per_day"] / 500.0) ** 0.5
    print()
    print("  S&P 500 panel, names ever  : %d" % sp)
    print("  breadth gain vs ~500/day   : %.1fx names, ~%.1fx on a t-statistic"
          % (rep["median_per_day"] / 500.0, gain))

    print()
    print("HONESTY NOTE -- read with the number above, not after it:")
    print("  This reaches further down the liquidity curve, where companies")
    print("  fail more often, and the price provider drops securities when")
    print("  they delist. Every name here is one that still exists today.")
    print("  Breadth buys statistical power and buys nothing in honesty.")

    out = resolve(CACHE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, dates=dates.values.astype("datetime64[D]"),
        tickers=np.array(list(usable), dtype=object), member=member,
        **{k: v for k, v in fields.items()})
    print("\nwrote %s (%.1f MB)" % (CACHE, out.stat().st_size / 1e6))

    resolve("state/wide_universe.json").write_text(
        json.dumps({"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "target_names": a.names, **rep,
                    "symbols_fetched": len(syms),
                    "symbols_usable": len(usable)}, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
