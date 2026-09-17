"""Congressional stock disclosures, keyed to the date the public could see them.

Members of Congress must disclose individual trades under the STOCK Act, and
the filings are public. The obvious idea is to follow them, and there is real
academic history behind it -- Ziobrowski et al. (2004) reported abnormal
returns on Senate portfolios, though later replications found much weaker
effects and some found none.

Whether it works here is an empirical question this module exists to make
askable. What it does NOT do is assume the answer.

THE 45-DAY PROBLEM. The STOCK Act allows up to 45 days between a transaction
and its disclosure. A backtest keyed to the TRANSACTION date buys at a price
nobody following the filings could have paid, and it will show a handsome
edge, because the whole point of the delay is that the information was private
during it. So every record here carries three dates and only one of them is
tradable:

    transaction_date  when the member traded        -- NOT tradable
    notification_date when they told the clerk      -- NOT tradable
    filing_date       when the document went public -- the only honest key

That is the same discipline the SEC fundamentals module uses, for the same
reason, and it is the single thing most likely to be got wrong by anyone
building this.

WHAT THE DATA IS. The House Clerk publishes a yearly ZIP with a tab-separated
index of every filing -- name, type, state, filing date, document id -- and the
Periodic Transaction Reports themselves as PDFs. Type "P" is the one that
matters; the rest are annual disclosures with no transaction detail. The PDFs
are machine-generated and extract cleanly, which is why this is feasible at
all.

WHAT IT IS NOT. Amounts are disclosed as ranges ($1,001-$15,000 and so on),
not exact sizes, so position weighting is approximate at best. Filings are
frequently amended. Some members file on paper and their PDFs are scans this
cannot read. And the Senate uses an entirely different system, not covered
here.
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

from .config import resolve

INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/%dFD.ZIP"
PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/%d/%s.pdf"
UA = "loonie-research (individual; majied.lafleur@gmail.com)"
CACHE = "data/congress"
RATE = 0.4                    # polite; this is a small government file server

# Disclosed amounts are ranges. The midpoint is a convention, not a
# measurement, and anything built on it inherits that.
AMOUNT_BANDS = {
    "$1,001 - $15,000": 8000, "$15,001 - $50,000": 32500,
    "$50,001 - $100,000": 75000, "$100,001 - $250,000": 175000,
    "$250,001 - $500,000": 375000, "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000, "$5,000,001 - $25,000,000": 15000000,
    "$25,000,001 - $50,000,000": 37500000, "$50,000,000 +": 50000000,
    "$1,000 - $15,000": 8000,
}

# A PTR row spans TWO lines, which is the thing that makes naive parsing
# wrong rather than merely incomplete:
#
#   GSK plc American Depositary Shares S 07/28/2025 08/11/2025 $1,001 - $15,000
#   (GSK) [ST]
#
# Side, both dates and the amount live on the first; the ticker lives on the
# second. A parser requiring all of them on one line silently keeps only the
# rows that happened not to wrap -- and the first version of this one did
# exactly that, returning "buy" for every single transaction because its
# side detection scanned the whole line and matched letters inside company
# names.
_ROW = re.compile(
    r"(?:^|\s)(?P<side>[PSE])\s+(?P<tx>\d{2}/\d{2}/\d{4})"
    r"(?:\s+(?P<notif>\d{2}/\d{2}/\d{4}))?"
    r"(?:\s*(?P<amt>\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+\s*\+))?")
_TICKER_LINE = re.compile(r"^\((?P<t>[A-Z][A-Z0-9.\-]{0,5})\)")
_SIDE = {"P": "buy", "S": "sell", "E": "exchange"}


def _get(url: str, timeout: float = 90.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


def fetch_index(year: int, refresh: bool = False) -> pd.DataFrame:
    """The year's filing index: who filed what, when, and the document id."""
    p = resolve("%s/index_%d.csv" % (CACHE, year))
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not refresh:
        return pd.read_csv(p, dtype={"DocID": str})

    z = zipfile.ZipFile(io.BytesIO(_get(INDEX_URL % year)))
    name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
    text = z.read(name).decode("utf-8", errors="replace")
    rows = [ln.split("\t") for ln in text.splitlines() if ln.strip()]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.to_csv(p, index=False)
    return df


def ptr_filings(year: int, refresh: bool = False) -> pd.DataFrame:
    """Just the Periodic Transaction Reports -- the ones naming trades."""
    df = fetch_index(year, refresh)
    df = df[df["FilingType"].astype(str).str.strip() == "P"].copy()
    df["filing_date"] = pd.to_datetime(df["FilingDate"], errors="coerce")
    df["member"] = (df["Last"].astype(str).str.strip() + ", "
                    + df["First"].astype(str).str.strip())
    return df.dropna(subset=["filing_date"])


# =============================================================================
#  Parsing
# =============================================================================
def parse_ptr(pdf_bytes: bytes) -> list:
    """Pull transactions out of one PTR. Returns [] on a scan it cannot read.

    Deliberately conservative: a row is kept only when a ticker, a transaction
    type and a date can all be identified. Half-parsed rows are worse than
    missing ones, because a wrong ticker is a trade in a company nobody
    mentioned.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    except Exception:
        return []
    if not text.strip():
        return []                      # a scan; nothing to read

    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    for i, line in enumerate(lines):
        m = _ROW.search(line)
        if not m:
            continue
        # The ticker is on the following line. No ticker means an asset with
        # no exchange symbol -- municipal bonds, private holdings, funds --
        # which cannot be traded against and is dropped rather than guessed.
        tk = None
        for nxt in lines[i + 1:i + 3]:
            t = _TICKER_LINE.match(nxt)
            if t:
                tk = t.group("t")
                break
        if not tk:
            continue
        out.append({
            "ticker": tk.replace(".", "-").upper(),
            "side": _SIDE[m.group("side")],
            "transaction_date": m.group("tx"),
            "notification_date": m.group("notif"),
            "amount_range": (re.sub(r"\s+", " ", m.group("amt")).strip()
                             if m.group("amt") else None),
        })
    return out


def fetch_transactions(years, limit: int | None = None, progress=None) -> pd.DataFrame:
    """Download and parse PTRs for the given years, caching per document."""
    raw_dir = resolve("%s/ptr" % CACHE)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, seen, failed = [], 0, 0

    for year in years:
        idx = ptr_filings(year)
        docs = list(idx.itertuples())
        if limit:
            docs = docs[:limit]
        for r in docs:
            doc = str(getattr(r, "DocID", "")).strip()
            if not doc:
                continue
            cached = raw_dir / ("%s_%s.json" % (year, doc))
            if cached.exists():
                try:
                    tx = json.loads(cached.read_text(encoding="utf-8"))
                except Exception:
                    tx = []
            else:
                try:
                    tx = parse_ptr(_get(PTR_URL % (year, doc), timeout=45))
                except Exception:
                    tx = []
                cached.write_text(json.dumps(tx), encoding="utf-8")
                time.sleep(RATE)
            if not tx:
                failed += 1
            for t in tx:
                rows.append({**t, "member": r.member, "year": year,
                             "doc_id": doc,
                             "filing_date": r.filing_date})
            seen += 1
            if progress:
                progress(seen, len(docs), len(rows), failed)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("transaction_date", "notification_date", "filing_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["amount_mid"] = df["amount_range"].map(
        lambda s: AMOUNT_BANDS.get(str(s).strip()))
    # The lag is the number this whole module is about, so it is computed
    # once here rather than rediscovered by every caller.
    df["disclosure_lag_days"] = (
        df["filing_date"] - df["transaction_date"]).dt.days
    return df.dropna(subset=["ticker", "filing_date"])


def save(df: pd.DataFrame) -> None:
    p = resolve("%s/transactions.parquet" % CACHE)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def load() -> pd.DataFrame:
    p = resolve("%s/transactions.parquet" % CACHE)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# =============================================================================
#  Signal
# =============================================================================
def build_signal(panel, df: pd.DataFrame | None = None,
                 decay_days: int = 63) -> dict:
    """A (T, N) feature: recent net congressional buying, by FILING date.

    A disclosure is applied from the day it became public and decays over
    `decay_days`. Applying it on the transaction date instead would be a
    lookahead of up to 45 days, and would be the single most flattering error
    available here.
    """
    import numpy as np

    df = load() if df is None else df
    T, N = panel.shape
    out = {"cong_net_buy": np.zeros((T, N), np.float32),
           "cong_days_since": np.full((T, N), np.nan, np.float32)}
    if df is None or not len(df):
        return out

    dates = pd.DatetimeIndex(panel.dates)
    idx = {t: j for j, t in enumerate(panel.tickers)}
    sign = {"buy": 1.0, "sell": -1.0, "exchange": 0.0}

    for r in df.itertuples():
        j = idx.get(str(r.ticker).upper())
        if j is None:
            continue
        i = int(dates.searchsorted(pd.Timestamp(r.filing_date)))
        if i >= T:
            continue
        stop = min(T, i + decay_days)
        n = stop - i
        if n <= 0:
            continue
        w = np.linspace(1.0, 0.0, n, dtype=np.float32)
        size = float(getattr(r, "amount_mid", 0) or 8000.0)
        out["cong_net_buy"][i:stop, j] += (
            sign.get(str(r.side), 0.0) * w * np.log1p(size / 1e4))
        out["cong_days_since"][i:stop, j] = np.arange(n, dtype=np.float32)
    return out


def summary(df: pd.DataFrame | None = None) -> dict:
    df = load() if df is None else df
    if df is None or not len(df):
        return {"transactions": 0}
    lag = df["disclosure_lag_days"].dropna()
    return {
        "transactions": len(df),
        "members": int(df["member"].nunique()),
        "tickers": int(df["ticker"].nunique()),
        "first_filing": str(df["filing_date"].min().date()),
        "last_filing": str(df["filing_date"].max().date()),
        "median_lag_days": float(lag.median()) if len(lag) else None,
        "p90_lag_days": float(lag.quantile(0.9)) if len(lag) else None,
        "over_45_days": float((lag > 45).mean()) if len(lag) else None,
        "buys": int((df["side"] == "buy").sum()),
        "sells": int((df["side"] == "sell").sum()),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
