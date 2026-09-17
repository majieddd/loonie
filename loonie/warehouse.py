"""One analytical database over everything this system has measured.

The data had accreted into four shapes: numpy archives for price panels,
parquet for filings and fundamentals, JSON for strategy results and search
state, and a pile of per-ticker files for SEC facts. Every cross-cutting
question -- why did illiquidity invert, which strategies overlap, what does
this symbol look like across markets -- needed a bespoke script that loaded
three formats and joined them by hand. That is how the altcoin-rotation
duplicate survived: nothing could compare two strategies without being
written first.

DuckDB is the right shape for this. It is a single file, it reads parquet
natively, it does no server, and it handles the one genuinely large table
here (about fourteen million price rows) without ceremony.

DESIGN NOTES, since the schema encodes decisions:

PRICES ARE LONG, NOT WIDE. A (dates x symbols) matrix is the right layout for
computing and the wrong one for asking questions. Long format costs disk and
buys every question that starts "for symbols where...".

MARKET IS A COLUMN, NOT A DATABASE. Stocks, crypto, forex and volatility live
in one prices table keyed by market, so a query can compare them without a
union. What stops that becoming a lie is that `markets` carries each store's
caveat, and any query crossing markets is expected to carry it too.

THE WAREHOUSE IS DERIVED AND DISPOSABLE. Nothing originates here. Every table
is rebuilt from the files that remain the source of truth, so a corrupted or
stale database is fixed by deleting it, and no analysis can quietly become the
only copy of something.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import resolve

DB = "data/loonie.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market      VARCHAR PRIMARY KEY,
    source      VARCHAR,
    symbols     INTEGER,
    sessions    INTEGER,
    start_date  DATE,
    end_date    DATE,
    caveat      VARCHAR,       -- what is wrong with this store, always
    built_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS symbols (
    market      VARCHAR,
    symbol      VARCHAR,
    first_date  DATE,
    last_date   DATE,
    sessions    INTEGER,
    PRIMARY KEY (market, symbol)
);

CREATE TABLE IF NOT EXISTS prices (
    market      VARCHAR,
    symbol      VARCHAR,
    date        DATE,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    tradable    BOOLEAN
);

CREATE TABLE IF NOT EXISTS strategies (
    id              VARCHAR PRIMARY KEY,
    title           VARCHAR,
    market          VARCHAR,
    description     VARCHAR,
    caveat          VARCHAR,
    is_modelled     BOOLEAN,   -- options rows are priced, not observed
    total_pl_pct    DOUBLE,
    cagr_pct        DOUBLE,
    win_rate_pct    DOUBLE,
    max_drawdown_pct DOUBLE,
    sharpe          DOUBLE,
    t_stat          DOUBLE,
    years           DOUBLE,
    periods         INTEGER,
    measured_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_curve (
    id          VARCHAR,
    date        DATE,
    equity      DOUBLE
);

CREATE TABLE IF NOT EXISTS congress_trades (
    member          VARCHAR,
    ticker          VARCHAR,
    side            VARCHAR,
    transaction_date DATE,
    filing_date     DATE,      -- the only tradable one
    amount_range    VARCHAR,
    amount_mid      DOUBLE,
    disclosure_lag_days INTEGER,
    doc_id          VARCHAR
);

CREATE TABLE IF NOT EXISTS factors (
    date        DATE PRIMARY KEY,
    mkt_rf      DOUBLE, smb DOUBLE, hml DOUBLE,
    rmw         DOUBLE, cma DOUBLE, mom DOUBLE, rf DOUBLE
);

CREATE TABLE IF NOT EXISTS runs (
    built_at    TIMESTAMP,
    table_name  VARCHAR,
    rows        BIGINT,
    note        VARCHAR
);
"""


def connect(read_only: bool = False):
    import duckdb

    p = resolve(DB)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def _note(con, table, rows, note=""):
    con.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                [datetime.now(timezone.utc), table, int(rows), note])


def _panel_to_long(market, dates, symbols, fields, tradable) -> pd.DataFrame:
    """Wide (dates x symbols) arrays to one row per symbol-date.

    Rows with no close at all are dropped: a wide panel is dense by
    construction and mostly NaN for any symbol that was not listed yet, and
    keeping those would multiply the table by five for no information.
    """
    T, N = fields["close"].shape
    out = {
        "market": np.repeat(market, T * N),
        "symbol": np.tile(np.asarray(symbols, dtype=object), T),
        "date": np.repeat(np.asarray(dates, dtype="datetime64[D]"), N),
    }
    for f in ("open", "high", "low", "close", "volume"):
        v = fields.get(f)
        out[f] = (np.asarray(v, dtype=np.float64).ravel() if v is not None
                  else np.full(T * N, np.nan))
    out["tradable"] = (np.asarray(tradable).ravel() if tradable is not None
                       else np.ones(T * N, bool))
    df = pd.DataFrame(out)
    return df[np.isfinite(df["close"])]


def load_market(con, market: str, panel) -> int:
    """Insert one market's prices, symbols and metadata."""
    fields = {f: getattr(panel, f, None)
              for f in ("open", "high", "low", "close", "volume")}
    df = _panel_to_long(market, panel.dates, panel.symbols, fields,
                        getattr(panel, "tradable", None))

    con.execute("DELETE FROM prices WHERE market = ?", [market])
    con.execute("DELETE FROM symbols WHERE market = ?", [market])
    con.execute("DELETE FROM markets WHERE market = ?", [market])
    con.register("_px", df)
    con.execute("INSERT INTO prices SELECT * FROM _px")
    con.unregister("_px")

    con.execute("""
        INSERT INTO symbols
        SELECT market, symbol, MIN(date), MAX(date), COUNT(*)
        FROM prices WHERE market = ? GROUP BY market, symbol
    """, [market])

    meta = getattr(panel, "meta", {}) or {}
    con.execute("INSERT INTO markets VALUES (?,?,?,?,?,?,?,?)", [
        market, meta.get("source", ""), len(panel.symbols),
        int(panel.shape[0]), pd.Timestamp(panel.dates[0]).date(),
        pd.Timestamp(panel.dates[-1]).date(), meta.get("caveat", ""),
        datetime.now(timezone.utc)])
    _note(con, "prices", len(df), market)
    return len(df)


def load_strategies(con, table_json: str = "docs/data/strategies_table.json") -> int:
    p = resolve(table_json)
    if not p.exists():
        return 0
    doc = json.loads(p.read_text(encoding="utf-8"))
    rows, curves = [], []
    now = datetime.now(timezone.utc)
    for s in doc.get("strategies", []):
        if not s.get("ok"):
            continue
        rows.append((s["id"], s["title"], s["market"], s.get("description", ""),
                     s.get("caveat", ""), s["market"] == "options",
                     s.get("total_pl_pct"), s.get("cagr_pct"),
                     s.get("win_rate_pct"), s.get("max_drawdown_pct"),
                     s.get("sharpe"), s.get("t_stat"), s.get("years"),
                     s.get("periods"), now))
        for d, e in (s.get("curve") or []):
            curves.append((s["id"], d, e))

    con.execute("DELETE FROM strategies")
    con.execute("DELETE FROM strategy_curve")
    if rows:
        con.executemany(
            "INSERT INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    if curves:
        con.executemany("INSERT INTO strategy_curve VALUES (?,?,?)", curves)
    _note(con, "strategies", len(rows))
    return len(rows)


def load_congress(con) -> int:
    from . import congress as C

    df = C.load()
    if df is None or not len(df):
        return 0
    keep = ["member", "ticker", "side", "transaction_date", "filing_date",
            "amount_range", "amount_mid", "disclosure_lag_days", "doc_id"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ("transaction_date", "filing_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    con.execute("DELETE FROM congress_trades")
    con.register("_cg", df)
    con.execute("INSERT INTO congress_trades SELECT %s FROM _cg"
                % ", ".join(keep))
    con.unregister("_cg")
    _note(con, "congress_trades", len(df))
    return len(df)


def load_factors(con) -> int:
    from . import factors as F

    try:
        df = F.fetch()
    except Exception:
        return 0
    df = df.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
                            "RMW": "rmw", "CMA": "cma", "Mom": "mom",
                            "RF": "rf"}).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    cols = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    con.execute("DELETE FROM factors")
    con.register("_ff", df)
    con.execute("INSERT INTO factors SELECT * FROM _ff")
    con.unregister("_ff")
    _note(con, "factors", len(df))
    return len(df)


def summary(con) -> dict:
    out = {}
    for t in ("markets", "symbols", "prices", "strategies", "strategy_curve",
              "congress_trades", "factors"):
        try:
            out[t] = int(con.execute(
                "SELECT COUNT(*) FROM %s" % t).fetchone()[0])
        except Exception:
            out[t] = 0
    return out
