"""Turn the disclosure event study into a portfolio, and see what survives.

    python scripts/congress_backtest.py [--horizon 5] [--mode long_short]

An event study says the average disclosed buy beats the average disclosed sell
by 0.75% over the week after filing. That is not the same as a strategy, and
the gap between them is where most published edges die:

  * Events arrive on ~97 days a year carrying about two names each. A book
    built from them is concentrated, lumpy, and idle much of the time.
  * Five-day holds overlap. Position sizing has to handle a name appearing
    twice before the first hold expires.
  * The cash not deployed does nothing. An event study implicitly assumes
    every dollar is working; a portfolio has to say where it is.
  * Costs are charged per rebalance, not per event.

So this builds the actual book, day by day, and reports the equity curve with
costs taken out. The event study's number is an upper bound on what this can
produce, never a forecast of it.

IDLE CASH IS REPORTED, NOT HIDDEN. A strategy that is 8% invested and earns
0.7% on the invested part has made 0.06%, and quoting the first number is the
most common way an event study is oversold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loonie import congress as C  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402

WIDE = "data/universe/wide_panel.npz"
TD = 252.0


def build_weights(df, dates, tickers, member, horizon, mode, max_pos):
    """(T, N) target weights from disclosures, applied from the filing date.

    A name disclosed twice inside one holding window gets one position, not
    two: doubling up on a repeated disclosure is a sizing decision nobody
    made, and it concentrates exactly where the data is noisiest.
    """
    idx = {t: j for j, t in enumerate(tickers)}
    T, N = len(dates), len(tickers)
    raw = np.zeros((T, N))

    for r in df.itertuples():
        j = idx.get(str(r.ticker).upper())
        if j is None or r.side not in ("buy", "sell"):
            continue
        if r.side == "sell" and mode == "long_only":
            continue
        i = int(dates.searchsorted(pd.Timestamp(r.filing_date)))
        # The signal is actionable the session AFTER it is published: the
        # filing appears during a day whose close has already happened.
        i += 1
        if i >= T:
            continue
        stop = min(T, i + horizon)
        side = 1.0 if r.side == "buy" else -1.0
        raw[i:stop, j] = side          # set, not accumulate

    raw = np.where(member, raw, 0.0)
    gross = np.abs(raw).sum(axis=1, keepdims=True)
    w = np.divide(raw, np.maximum(gross, 1.0), where=gross > 0)
    return np.clip(w, -max_pos, max_pos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--mode", default="long_short",
                    choices=["long_short", "long_only", "tilt"])
    ap.add_argument("--tilt", type=float, default=0.30,
                    help="tilt mode: fraction of capital moved by the signal")
    ap.add_argument("--max-position", type=float, default=0.25)
    ap.add_argument("--cost-bps", type=float, default=2.6)
    ap.add_argument("--json", default="state/congress_backtest.json")
    a = ap.parse_args()

    load_env()
    load()
    p = resolve(WIDE)
    if not p.exists():
        print("no wide panel; run scripts/build_wide_universe.py first")
        return 1
    z = np.load(p, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    tickers = list(z["tickers"])
    close = z["close"].astype(np.float64)
    member = z["member"] & np.isfinite(close) & (close > 0)

    df = C.load()
    if df is None or not len(df):
        print("no congressional transactions cached")
        return 1

    # Trade only the span the disclosures cover, or years of flat zero before
    # the first filing will dilute every ratio below.
    lo = int(dates.searchsorted(pd.Timestamp(df["filing_date"].min())))
    hi = len(dates)
    dates_s = dates[lo:hi]
    close_s, member_s = close[lo:hi], member[lo:hi]

    if a.mode == "tilt":
        # The standard construction for a SPARSE signal, and the one the two
        # concentrated versions above argue for: hold the diversified
        # universe and move a slice of capital toward disclosed buys. Five
        # names carrying the whole book means idiosyncratic variance swamps a
        # 0.75% edge, which is exactly what long_short and long_only showed.
        base = member_s.astype(float)
        base /= np.maximum(base.sum(axis=1, keepdims=True), 1.0)
        sig = build_weights(df, dates_s, tickers, member_s, a.horizon,
                            "long_short", 1.0)
        w = (1.0 - a.tilt) * base + a.tilt * sig
        w = np.where(member_s | (sig != 0), w, 0.0)
        w = np.clip(w, -a.max_position, a.max_position)
    else:
        w = build_weights(df, dates_s, tickers, member_s, a.horizon, a.mode,
                          a.max_position)

    r1 = np.vstack([np.full((1, close_s.shape[1]), np.nan),
                    np.diff(close_s, axis=0) / close_s[:-1]])
    r1 = np.nan_to_num(r1)
    # Weights set at the close of t earn the return of t+1.
    held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross_ret = (held * r1).sum(axis=1)

    turn = np.abs(np.diff(np.vstack([np.zeros((1, w.shape[1])), w]),
                          axis=0)).sum(axis=1)
    cost = turn * a.cost_bps / 1e4
    net = gross_ret - cost

    bench = np.nanmean(np.where(member_s, r1, np.nan), axis=1)
    bench = np.nan_to_num(bench)

    invested = np.abs(held).sum(axis=1)
    days_live = int((invested > 0).sum())
    yrs = max(len(net) / TD, 1e-9)

    def cagr(x):
        v = float(np.prod(1.0 + x))
        return v ** (1.0 / yrs) - 1.0 if v > 0 else -1.0

    def sharpe(x):
        s = float(np.std(x, ddof=1))
        return float(np.mean(x)) / max(s, 1e-12) * np.sqrt(TD) if s > 0 else 0.0

    eq = np.cumprod(1.0 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    print("CONGRESS DISCLOSURE STRATEGY  (%s, %dd holds)" % (a.mode, a.horizon))
    print("  span                : %s .. %s (%d sessions)"
          % (dates_s[0].date(), dates_s[-1].date(), len(dates_s)))
    print("  days with a position: %d of %d (%.0f%%)"
          % (days_live, len(net), 100 * days_live / max(len(net), 1)))
    print("  median gross when on: %.0f%% of capital"
          % (100 * np.median(invested[invested > 0]) if days_live else 0))
    print("  median names held   : %.0f"
          % np.median((np.abs(held) > 0).sum(axis=1)[invested > 0])
          if days_live else 0)
    print()
    print("  return (net)        : %+.2f%%/yr" % (100 * cagr(net)))
    print("  benchmark           : %+.2f%%/yr" % (100 * cagr(bench)))
    print("  Sharpe (net)        : %+.2f" % sharpe(net))
    print("  max drawdown        : %.2f%%" % (100 * dd))
    print("  annual turnover     : %.1fx" % (turn.sum() / yrs))
    print("  cost drag           : %.2f%%/yr" % (100 * cost.sum() / yrs))
    print()
    print("  t-stat of daily net : %+.2f"
          % (np.mean(net) / max(np.std(net, ddof=1), 1e-12) * np.sqrt(len(net))))

    # The number that decides it. Absolute return in a bull market says more
    # about the market than the strategy; excess over the same universe is
    # what the disclosures are being asked to add.
    exc = net - bench
    exc_t = (np.mean(exc) / max(np.std(exc, ddof=1), 1e-12)) * np.sqrt(len(exc))
    print()
    print("  EXCESS over benchmark : %+.2f%%/yr"
          % (100 * (cagr(net) - cagr(bench))))
    print("  excess Sharpe         : %+.2f" % sharpe(exc))
    print("  excess t-stat         : %+.2f" % exc_t)
    print("  -> %s" % ("beats the universe" if exc_t > 2.0 else
                       "does not beat holding the universe"))

    # The honest denominator. A book that is 8% invested and earns 0.7% on the
    # invested part has made 0.06%, and quoting the first is how event studies
    # get oversold.
    if days_live:
        on_capital = np.mean(net[invested > 0])
        on_invested = np.mean((net / np.maximum(invested, 1e-9))[invested > 0])
        print("\n  mean daily on TOTAL capital  : %+.4f%%" % (100 * on_capital))
        print("  mean daily on INVESTED only  : %+.4f%%" % (100 * on_invested))
        print("  -> the gap is idle cash, and it is the difference between")
        print("     an event study and a portfolio")

    out = {"mode": a.mode, "horizon": a.horizon,
           "span": [str(dates_s[0].date()), str(dates_s[-1].date())],
           "days_live": days_live, "sessions": len(net),
           "cagr_net": cagr(net), "cagr_bench": cagr(bench),
           "sharpe": sharpe(net), "max_drawdown": dd,
           "turnover": float(turn.sum() / yrs),
           "cost_drag": float(cost.sum() / yrs)}
    resolve(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
