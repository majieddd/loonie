"""Point-in-time fundamentals from SEC filings.

Every one of the 57 terminals this search can reach is derived from price and
volume. That is a hypothesis space with a hole in it, and the factor
attribution found the hole: across the top archive entries the strategies load
-0.19 on RMW, significant in five of eight. RMW is profitability. The search is
placing a persistent bet on weak-profitability companies -- not because it
decided to, but because it cannot see profitability at all, and something has
to fill the space where that information would be.

This module fills it. It reads company facts from SEC XBRL, which is free,
permanent, and published for exactly this use.

THE FILING DATE IS THE ONLY HONEST KEY. Each XBRL fact carries two dates: the
period it describes (`end`) and the day it was submitted (`filed`). Apple's
FY2008 balance sheet has end=2008-09-27 and filed=2009-07-22 -- ten months
apart. A feature keyed to `end` would put September's assets into September's
factor, which nobody could have known until the following July. That mistake
produces a backtest that looks extraordinary and cannot be traded, and it is
invisible in every test that does not specifically look for it.

So the rule here is absolute: as of date t, a fact exists only if filed <= t.
Restatements fall out of the same rule for free -- an amended figure for an old
period simply has a later filing date, so the panel shows the original value
until the amendment was actually published and the revised one after.

WHAT THIS DOES NOT FIX. Coverage still tracks the price panel: companies that
left the index are the ones we can neither price nor map to a CIK. Fundamentals
do not repair survivorship, and the test suite asserts that they do not make it
worse -- a feature that is NaN precisely for the names that later delisted
would be a survivorship leak wearing a balance sheet.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

from .config import resolve

# The SEC asks for a descriptive User-Agent and rate-limits to 10 req/s. Both
# are conditions of access, not obstacles to it.
UA = "loonie-research (individual; majied.lafleur@gmail.com)"
RATE = 0.16                       # ~6 requests/second, inside the published cap
TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK%010d.json"

CACHE = "data/cache/sec"
MAP_CACHE = "data/cache/sec_cik.json"

# Instantaneous quantities: a balance-sheet line has one date, not a period.
STOCKS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "shares": ["EntityCommonStockSharesOutstanding",
               "CommonStockSharesOutstanding", "CommonStockSharesIssued"],
}

# Flows span a period. The tag lists are ordered by preference: XBRL naming
# changed with ASC 606 in 2018, so a single tag covers only part of the window
# and the older names have to remain reachable.
FLOWS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

FUNDAMENTAL_NAMES = [
    "f_gp_assets", "f_roa", "f_roe", "f_gross_margin", "f_op_margin",
    "f_accruals", "f_leverage", "f_asset_growth", "f_sales_growth",
    "f_rnd_intensity", "f_book_to_market", "f_earnings_yield",
    "f_fcf_yield", "f_days_since_filing",
]

EPS = 1e-12
_QUARTER = (80, 100)              # day span that counts as one fiscal quarter
_ANNUAL = (350, 380)


# =============================================================================
#  Fetching
# =============================================================================
def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        raw = fh.read()
        if fh.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        return raw


def cik_map(refresh: bool = False) -> dict:
    """Ticker -> CIK. Cached; the file changes only when registrants do."""
    p = resolve(MAP_CACHE)
    if p.exists() and not refresh:
        age_d = (time.time() - p.stat().st_mtime) / 86400.0
        if age_d < 30.0:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    rows = json.loads(_get(TICKER_MAP).decode("utf-8"))
    m = {r["ticker"].upper(): int(r["cik_str"]) for r in rows.values()}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m), encoding="utf-8")
    return m


def _norm(ticker: str) -> str:
    """Our universe writes share classes as BRK.B; SEC writes BRK-B."""
    return ticker.replace(".", "-").upper()


def _facts_path(ticker: str):
    return resolve(CACHE) / ("%s.parquet" % _norm(ticker))


def _extract(doc: dict) -> pd.DataFrame:
    """Flatten company facts down to the concepts we actually use.

    The raw document is several megabytes of hundreds of tags. Keeping it
    would cost gigabytes across the universe to store data we never read.
    """
    rows = []
    for taxonomy, tags in (doc.get("facts") or {}).items():
        for concept, spec in list(STOCKS.items()) + list(FLOWS.items()):
            for tag in spec:
                node = tags.get(tag)
                if not node:
                    continue
                for unit, points in (node.get("units") or {}).items():
                    if unit not in ("USD", "shares", "USD/shares"):
                        continue
                    for pt in points:
                        if pt.get("val") is None or not pt.get("filed"):
                            continue
                        rows.append((concept, tag, pt.get("start"), pt["end"],
                                     pt["filed"], float(pt["val"]),
                                     pt.get("form", "")))
                break            # first tag that exists wins for this concept

    if not rows:
        return pd.DataFrame(columns=["concept", "tag", "start", "end", "filed",
                                     "val", "form"])
    df = pd.DataFrame(rows, columns=["concept", "tag", "start", "end", "filed",
                                     "val", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(subset=["end", "filed"]).sort_values("filed")


def fetch_company(ticker: str, cik: int, refresh: bool = False,
                  max_age_days: float = 14.0) -> pd.DataFrame | None:
    """Company facts for one ticker, cached in compact form."""
    p = _facts_path(ticker)
    if p.exists() and not refresh:
        age_d = (time.time() - p.stat().st_mtime) / 86400.0
        if age_d < max_age_days:
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
    try:
        doc = json.loads(_get(FACTS % int(cik)).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None           # registrant files nothing in XBRL
        raise
    df = _extract(doc)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p)
    except Exception:
        pass
    return df


def fetch_universe(tickers, refresh: bool = False, progress=None) -> dict:
    """Fetch every ticker we can map, respecting the published rate limit."""
    m = cik_map(refresh=refresh)
    out, missing = {}, []
    for i, t in enumerate(tickers, 1):
        cik = m.get(_norm(t))
        if cik is None:
            missing.append(t)
            continue
        cached = _facts_path(t).exists()
        try:
            df = fetch_company(t, cik, refresh=refresh)
        except Exception:
            missing.append(t)
            continue
        if df is not None and len(df):
            out[t] = df
        if not cached:
            time.sleep(RATE)
        if progress:
            progress(i, len(tickers), len(out), len(missing))
    return {"facts": out, "missing": missing}


# =============================================================================
#  Point-in-time assembly
# =============================================================================
def _known_series(df: pd.DataFrame, concept: str, dates: pd.DatetimeIndex,
                  flow: bool) -> np.ndarray:
    """The value a reader would have had on each date. Nothing else.

    Walks filings in submission order and writes each new value forward from
    the day it was filed. Because the cursor only ever moves with `filed`, a
    restatement of an old period lands on the day it was published rather than
    on the period it corrects.
    """
    sub = df[df["concept"] == concept]
    if not len(sub):
        return np.full(len(dates), np.nan)

    if flow:
        span = (sub["end"] - sub["start"]).dt.days
        quarterly = sub[span.between(*_QUARTER)]
        annual = sub[span.between(*_ANNUAL)]
    else:
        quarterly, annual = sub, sub.iloc[0:0]

    # The value can only change on a filing date, so it is computed once per
    # filing and broadcast forward. The obvious loop -- reduce once per
    # calendar date -- does the same work a hundred times over and made a
    # rebuild of the panel take four minutes, long enough that the search
    # aged into "stale" on the dashboard every time it reloaded.
    ev = quarterly.sort_values("filed")
    if not len(ev):
        return np.full(len(dates), np.nan)

    breaks = ev["filed"].to_numpy()
    ends = ev["end"].to_numpy()
    vals = ev["val"].to_numpy(dtype=np.float64)

    uniq, first = np.unique(breaks, return_index=True)
    step = np.full(len(uniq), np.nan)
    latest: dict = {}
    bounds = list(first) + [len(ev)]
    for k in range(len(uniq)):
        for i in range(bounds[k], bounds[k + 1]):
            # A later filing for a period already seen is a restatement; it
            # replaces the old figure from this date forward, never before.
            latest[ends[i]] = vals[i]
        step[k] = _reduce(latest, flow)

    idx = np.searchsorted(uniq, dates.to_numpy(), side="right") - 1
    out = np.where(idx >= 0, step[np.maximum(idx, 0)], np.nan)

    if flow and len(annual):
        # Fall back to the annual figure wherever four quarters were never
        # filed -- common for foreign issuers and for the early years of the
        # XBRL mandate, when quarterly tagging was patchy.
        ann = _known_series(annual.assign(start=annual["end"]), concept,
                            dates, flow=False)
        out = np.where(np.isfinite(out), out, ann)
    return out


def _reduce(latest: dict, flow: bool) -> float:
    if not latest:
        return np.nan
    if not flow:
        return latest[max(latest)]
    # Trailing twelve months: the four most recent quarters we have seen.
    # A single quarter is seasonal, and comparing one company's Q4 against
    # another's Q2 across a cross-section would be measuring the calendar.
    ends = sorted(latest)[-4:]
    if len(ends) < 4:
        return np.nan
    return float(sum(latest[e] for e in ends))


def _last_filed(df: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Days since the most recent filing of any kind. A staleness clock."""
    if not len(df):
        return np.full(len(dates), np.nan)
    filed = np.sort(df["filed"].to_numpy())
    idx = np.searchsorted(filed, dates.to_numpy(), side="right") - 1
    out = np.full(len(dates), np.nan)
    ok = idx >= 0
    out[ok] = (dates.to_numpy()[ok] - filed[idx[ok]]) / np.timedelta64(1, "D")
    return out


def build(panel, facts: dict | None = None, verbose: bool = True) -> dict:
    """Fundamental features over the panel. All keyed to the filing date."""
    if facts is None:
        facts = load_cached(panel.tickers)
        if verbose:
            # Coverage depends on what happens to be cached, so a half-finished
            # download would otherwise change the feature set between runs with
            # nothing in the log to say so.
            have = sum(1 for t in panel.tickers if len(facts.get(t, ())))
            print("[fundamentals] %d/%d names have filings (%.1f%%)"
                  % (have, len(panel.tickers),
                     100.0 * have / max(1, len(panel.tickers))))

    dates = pd.DatetimeIndex(panel.dates)
    T, N = len(dates), len(panel.tickers)
    out = {k: np.full((T, N), np.nan, np.float32) for k in FUNDAMENTAL_NAMES}
    close = panel.bars["close"].astype(np.float64)

    for j, tk in enumerate(panel.tickers):
        df = facts.get(tk)
        if df is None or not len(df):
            continue

        s = {k: _known_series(df, k, dates, flow=False) for k in STOCKS}
        f = {k: _known_series(df, k, dates, flow=True) for k in FLOWS}

        assets = s["assets"]
        equity = s["equity"]
        rev = f["revenue"]
        ni = f["net_income"]
        mcap = close[:, j] * s["shares"]

        def over(a, b):
            b = np.where(np.abs(b) > EPS, b, np.nan)
            return a / b

        out["f_gp_assets"][:, j] = over(f["gross_profit"], assets)
        out["f_roa"][:, j] = over(ni, assets)
        out["f_roe"][:, j] = over(ni, np.where(equity > 0, equity, np.nan))
        out["f_gross_margin"][:, j] = over(f["gross_profit"], rev)
        out["f_op_margin"][:, j] = over(f["operating_income"], rev)
        out["f_accruals"][:, j] = over(ni - f["op_cash_flow"], assets)
        out["f_leverage"][:, j] = over(s["liabilities"], assets)
        out["f_rnd_intensity"][:, j] = over(f["rnd"], rev)
        out["f_book_to_market"][:, j] = over(equity, mcap)
        out["f_earnings_yield"][:, j] = over(ni, mcap)
        out["f_fcf_yield"][:, j] = over(f["op_cash_flow"] - np.nan_to_num(
            f["capex"], nan=0.0), mcap)
        out["f_asset_growth"][:, j] = _yoy(assets)
        out["f_sales_growth"][:, j] = _yoy(rev)
        out["f_days_since_filing"][:, j] = _last_filed(df, dates)

    for k in out:
        a = out[k]
        # Ratios built on tiny denominators produce values no analyst would
        # report. Clipping keeps one micro-cap's rounding error from becoming
        # the cross-section's entire spread.
        a = np.where(np.isfinite(a), a, np.nan)
        out[k] = np.clip(a, -50.0, 50.0).astype(np.float32)
    return out


def _yoy(a: np.ndarray, lag: int = 252) -> np.ndarray:
    prev = np.concatenate([np.full(min(lag, len(a)), np.nan), a[:-lag]]) \
        if lag < len(a) else np.full(len(a), np.nan)
    denom = np.where(np.abs(prev) > EPS, np.abs(prev), np.nan)
    return (a - prev) / denom


def load_cached(tickers) -> dict:
    """Whatever is already on disk. Never touches the network."""
    out = {}
    for t in tickers:
        p = _facts_path(t)
        if not p.exists():
            continue
        try:
            out[t] = pd.read_parquet(p)
        except Exception:
            continue
    return out


def coverage(panel, facts: dict | None = None) -> dict:
    """How much of the panel has usable fundamentals, and where it is thin."""
    facts = load_cached(panel.tickers) if facts is None else facts
    have = [t for t in panel.tickers if t in facts and len(facts[t])]
    return {
        "tickers": len(panel.tickers),
        "with_facts": len(have),
        "coverage": len(have) / max(1, len(panel.tickers)),
        "missing": sorted(set(panel.tickers) - set(have))[:25],
        "cached_files": len(list(resolve(CACHE).glob("*.parquet")))
        if resolve(CACHE).exists() else 0,
    }
