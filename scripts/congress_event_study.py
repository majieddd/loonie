"""Do disclosed congressional trades predict anything, once you can only act
on the disclosure?

    python scripts/congress_event_study.py [--horizon 5]

The naive version of this study keys on the TRANSACTION date and reports a
large edge. It is meaningless: the STOCK Act allows up to 45 days before the
trade becomes public (median 29 here, p90 72, and 15% breach the statute), and
the whole reason the delay matters is that nobody outside could act during it.

This keys on the FILING date -- the day the document appeared on the House
Clerk's site -- which is the first moment the information was purchasable.

WHY THE EVENT STUDY AND NOT AN IC. The signal is sparse: fewer than 0.4% of
panel cells carry a disclosure. A cross-sectional rank correlation across
2,439 names, of which nine have any signal, is a correlation between mostly
zeros and noise, and reports approximately nothing regardless of what the
disclosures actually predict. Measured that way the IC comes back +0.0001 at
t +0.38, which says nothing about Congress and everything about the
instrument.

WHY CLUSTERING. 1,988 events are not 1,988 independent observations: members
file in batches, several disclosures share a filing day, and neighbouring
events share overlapping return windows. Treating them as independent is
pseudo-replication. Here the correction happens to RAISE the t-statistic --
averaging within a cluster removes noise faster than it removes sample -- but
that is a fact about this data, not a reason to skip the correction.
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


def build_events(df, dates, tickers, close, member, horizon):
    """Signed excess return over `horizon` days from each FILING date."""
    idx = {t: j for j, t in enumerate(tickers)}
    r1 = np.vstack([np.full((1, close.shape[1]), np.nan),
                    np.diff(close, axis=0) / close[:-1]])
    # Equal-weight universe over the same window is the benchmark: without it
    # a month when everything rose reads as disclosure alpha.
    cum = np.nancumsum(np.nanmean(np.where(member, r1, np.nan), axis=1))

    rows = []
    for r in df.itertuples():
        j = idx.get(str(r.ticker).upper())
        if j is None or r.side not in ("buy", "sell"):
            continue
        i = int(dates.searchsorted(pd.Timestamp(r.filing_date)))
        if i + horizon >= len(dates) or not member[i, j]:
            continue
        a, b = close[i, j], close[i + horizon, j]
        if not (np.isfinite(a) and np.isfinite(b) and a > 0):
            continue
        exc = (b / a - 1.0) - (cum[i + horizon] - cum[i])
        rows.append({"date": pd.Timestamp(r.filing_date), "member": r.member,
                     "ticker": r.ticker, "side": r.side,
                     "signed_excess": (1.0 if r.side == "buy" else -1.0) * exc})
    return pd.DataFrame(rows)


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=1)
    return float(x.mean() / (s / np.sqrt(len(x)))) if len(x) > 2 and s > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--json", default="state/congress_event_study.json")
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
    print("[cong] %d disclosures, %d members, %d tickers, %s..%s"
          % (len(df), df["member"].nunique(), df["ticker"].nunique(),
             df["filing_date"].min().date(), df["filing_date"].max().date()))

    e = build_events(df, dates, tickers, close, member, a.horizon)
    if len(e) < 100:
        print("only %d usable events" % len(e))
        return 1

    by_day = e.groupby("date")["signed_excess"].mean()
    by_md = e.groupby(["date", "member"])["signed_excess"].mean()

    print("\nEVENT STUDY -- %d days from the FILING date" % a.horizon)
    print("  events                  : %d" % len(e))
    print("  distinct filing days    : %d" % e["date"].nunique())
    print("  naive t (all events)    : %+.2f   <- pseudo-replication"
          % tstat(e["signed_excess"]))
    print("  clustered by filing day : %+.2f   (n=%d)" % (tstat(by_day), len(by_day)))
    print("  clustered by member-day : %+.2f   (n=%d)" % (tstat(by_md), len(by_md)))

    q = by_day.quantile([0.01, 0.99])
    wins = by_day.clip(q.iloc[0], q.iloc[1])
    print("\n  mean excess per cluster : %+.3f%%" % (100 * by_day.mean()))
    print("  median                  : %+.3f%%" % (100 * by_day.median()))
    print("  share positive          : %.0f%%" % (100 * (by_day > 0).mean()))
    print("  winsorised 1/99         : %+.3f%%  t %+.2f   <- outlier check"
          % (100 * wins.mean(), tstat(wins)))

    years = {}
    print("\n  %-6s %9s %10s %8s" % ("year", "clusters", "mean exc", "t"))
    for yr, g0 in e.groupby(e["date"].dt.year):
        g = g0.groupby("date")["signed_excess"].mean()
        if len(g) < 20:
            continue
        years[int(yr)] = {"clusters": len(g), "mean": float(g.mean()),
                          "t": tstat(g)}
        print("  %-6d %9d %9.3f%% %8.2f" % (yr, len(g), 100 * g.mean(), tstat(g)))

    # Costs, so the number is net of what it takes to capture it.
    rt_bps = 2.0 * 2.6
    print("\n  round-trip execution    : %.1f bps (measured at this book size)"
          % rt_bps)
    print("  net of costs            : %+.3f%%"
          % (100 * by_day.mean() - rt_bps / 100.0))

    out = {"horizon": a.horizon, "events": len(e),
           "clusters": int(e["date"].nunique()),
           "t_naive": tstat(e["signed_excess"]),
           "t_cluster_day": tstat(by_day), "t_cluster_member_day": tstat(by_md),
           "mean_excess": float(by_day.mean()),
           "median_excess": float(by_day.median()),
           "share_positive": float((by_day > 0).mean()),
           "winsorised_mean": float(wins.mean()), "winsorised_t": tstat(wins),
           "by_year": years}
    resolve(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
