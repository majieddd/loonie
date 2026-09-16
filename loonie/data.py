"""Price data: providers, caching, panel assembly, and a coverage audit.

Design rule: the panel knows what it is missing, and says so loudly.

Free data providers only carry securities that still exist. Ask yfinance for
SIVB, FRC, LEH, WCOM, XLNX, ATVI or TWTR and you get an empty frame. So a
"free" backtest over a 10-year window silently drops roughly a third of the
names that were actually in the index -- and every one of those drops is a
loser you would have owned. `Panel.coverage` measures that hole so the
backtester can refuse to pretend it isn't there.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .universe import Universe

warnings.filterwarnings("ignore", category=FutureWarning)
OHLCV = ("open", "high", "low", "close", "volume")


# =============================================================================
#  Providers
# =============================================================================
class Provider:
    name = "base"
    has_delisted = False
    earliest = "1990-01-01"

    def symbol(self, ticker: str) -> str:
        return ticker

    def fetch(self, ticker: str, start: str, end: str):
        raise NotImplementedError


class YFinance(Provider):
    """Free. SURVIVORSHIP-BIASED -- delisted securities return empty."""

    name = "yfinance"
    has_delisted = False
    earliest = "1970-01-01"

    def symbol(self, t: str) -> str:
        return t.replace(".", "-")

    def fetch(self, ticker, start, end):
        import yfinance as yf

        df = None
        for attempt in range(3):
            try:
                df = yf.download(
                    self.symbol(ticker), start=start, end=end, progress=False,
                    auto_adjust=True, threads=False, actions=False,
                )
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(1.5 * (attempt + 1))

        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        if "close" not in df.columns:
            return None
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        return df[[c for c in OHLCV if c in df.columns]].astype("float64")


class Alpaca(Provider):
    """Broker-native data. Carries many inactive symbols; history from 2016."""

    name = "alpaca"
    has_delisted = True
    earliest = "2016-01-01"

    def __init__(self):
        import os

        from alpaca.data.historical import StockHistoricalDataClient

        key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
        if not (key and sec):
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_API_SECRET not set. Copy .env.example "
                "to .env and paste your PAPER keys."
            )
        self._c = StockHistoricalDataClient(key, sec)

    def fetch(self, ticker, start, end):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        try:
            bars = self._c.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=self.symbol(ticker),
                    timeframe=TimeFrame.Day,
                    start=pd.Timestamp(start).to_pydatetime(),
                    end=pd.Timestamp(end).to_pydatetime(),
                    adjustment="all",
                )
            ).df
        except Exception:
            return None
        if bars is None or len(bars) == 0:
            return None
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.droplevel(0)
        bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
        bars.columns = [str(c).lower() for c in bars.columns]
        return bars[[c for c in OHLCV if c in bars.columns]].astype("float64")


PROVIDERS = {"yfinance": YFinance, "alpaca": Alpaca}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError("unknown provider %r; have %s" % (name, list(PROVIDERS)))
    return PROVIDERS[name]()


# =============================================================================
#  Panel
# =============================================================================
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list
    bars: dict                 # field -> (T, N) float32
    member: np.ndarray         # (T, N) bool, point-in-time index membership
    tradable: np.ndarray       # (T, N) bool, member & priced & liquid
    coverage: dict = field(default_factory=dict)

    @property
    def close(self) -> np.ndarray:
        return self.bars["close"]

    @property
    def shape(self):
        return len(self.dates), len(self.tickers)

    def slice_dates(self, start, end) -> "Panel":
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        m = (self.dates >= lo) & (self.dates <= hi)
        return Panel(
            dates=self.dates[m],
            tickers=list(self.tickers),
            bars={k: v[m] for k, v in self.bars.items()},
            member=self.member[m],
            tradable=self.tradable[m],
            coverage=dict(self.coverage),
        )

    def describe(self) -> str:
        T, N = self.shape
        c = self.coverage
        lines = [
            "Panel  %d sessions x %d tickers  [%s -> %s]"
            % (T, N, self.dates[0].date(), self.dates[-1].date()),
            "  provider           : %s" % c.get("provider"),
            "  index members ever : %s" % c.get("universe_size"),
            "  with usable data   : %s" % c.get("fetched"),
            "  COVERAGE           : %.1f%%" % (100 * c.get("coverage", 0)),
            "  departed & missing : %s  (losers the backtest cannot see)"
            % c.get("missing_departed"),
        ]
        if c.get("coverage", 1) < 0.8:
            lines.append("  *** SURVIVORSHIP-BIASED: results are optimistic. ***")
        return "\n".join(lines)


# =============================================================================
#  Loading
# =============================================================================
def _cache_path(provider: str, ticker: str) -> Path:
    d = resolve("data/cache/%s" % provider)
    d.mkdir(parents=True, exist_ok=True)
    return d / ("%s.parquet" % ticker.replace("/", "_"))


def fetch_ticker(prov: Provider, ticker: str, start: str, end: str,
                 refresh: bool = False, max_stale_days: int = 3,
                 min_refetch_hours: float = 6.0):
    """Fetch one ticker, parquet-cached, extending the tail incrementally.

    A daily system re-reads this cache every session. Re-downloading ten years
    of history for 745 names to learn what happened yesterday takes ten minutes
    and hammers the provider, so a cache that ends before `end` gets only its
    missing tail fetched and appended.

    Empty results are cached too: a security that genuinely no longer exists
    should not be re-requested every morning forever. But an empty cache older
    than `max_stale_days` is retried, because "no data" is sometimes just a
    rate limit wearing a disguise.
    """
    p = _cache_path(prov.name, ticker)
    cached = None
    if p.exists() and not refresh:
        try:
            cached = pd.read_parquet(p)
        except Exception:
            cached = None

    want_end = pd.Timestamp(end)
    if cached is not None:
        if cached.empty:
            # Epoch-vs-epoch. Mixing pd.Timestamp.now() (naive local) with a
            # unit="s" Timestamp (interpreted UTC) yields a NEGATIVE age west
            # of Greenwich, which silently disables every freshness check.
            age_days = (time.time() - p.stat().st_mtime) / 86400.0
            if age_days <= max_stale_days:
                return None
            cached = None                      # stale "missing" -> retry
        else:
            # Trust a cache written recently. Without this, every invocation
            # re-requests the tail for all 745 names -- five minutes and 745
            # API calls to learn nothing, on a system designed to be run
            # repeatedly through the day.
            age_h = (time.time() - p.stat().st_mtime) / 3600.0
            if age_h < min_refetch_hours:
                return cached

            last = pd.DatetimeIndex(cached.index).max()
            # Only business days count; a Monday cache is not stale on Sunday.
            gap = len(pd.bdate_range(last + pd.Timedelta(days=1),
                                     min(want_end, pd.Timestamp.now())))
            if gap <= 0:
                p.touch()          # verified current; restart the freshness clock
                return cached
            tail = prov.fetch(ticker, str((last + pd.Timedelta(days=1)).date()),
                              str(want_end.date()))
            if tail is None or len(tail) == 0:
                return cached
            merged = pd.concat([cached, tail])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            try:
                merged.to_parquet(p)
            except Exception:
                pass
            return merged

    df = prov.fetch(ticker, start, end)
    try:
        out = df if df is not None else pd.DataFrame(columns=list(OHLCV))
        out.to_parquet(p)
    except Exception:
        pass
    return df


def load_panel(cfg, universe=None, start=None, end=None, provider=None,
               refresh: bool = False, progress: bool = True) -> Panel:
    """Assemble an aligned (T, N) panel over the point-in-time universe."""
    universe = universe or Universe.load(cfg)
    start = str(start or cfg.universe.start)
    end = str(end or cfg.universe.end)

    want = provider or cfg.data.provider
    try:
        prov = get_provider(want)
    except Exception as e:
        fb = cfg.data.get("fallback", "yfinance")
        print("[data] provider %r unavailable (%s); falling back to %r"
              % (want, e, fb))
        prov = get_provider(fb)

    if pd.Timestamp(start) < pd.Timestamp(prov.earliest):
        print("[data] %s history starts %s; clamping start %s -> %s"
              % (prov.name, prov.earliest, start, prov.earliest))
        start = prov.earliest

    tickers = universe.tickers_active_between(start, end)
    frames = {}
    for i, t in enumerate(tickers, 1):
        df = fetch_ticker(prov, t, start, end, refresh)
        if df is not None and len(df) >= 60:
            frames[t] = df
        if progress and (i % 25 == 0 or i == len(tickers)):
            print("\r[data] %s: %d/%d (%d usable)"
                  % (prov.name, i, len(tickers), len(frames)), end="", flush=True)
    if progress:
        print()
    if not frames:
        raise RuntimeError("no data fetched -- check network / API keys")

    keep = sorted(frames)
    all_dates = set()
    for t in keep:
        all_dates |= set(frames[t].index)
    dates = pd.DatetimeIndex(sorted(all_dates))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]

    bars = {f: np.full((len(dates), len(keep)), np.nan, np.float32) for f in OHLCV}
    for j, t in enumerate(keep):
        df = frames[t].reindex(dates)
        for f in OHLCV:
            if f in df.columns:
                bars[f][:, j] = df[f].to_numpy(dtype="float32")

    member = universe.mask_for(dates, keep)
    priced = np.isfinite(bars["close"]) & (bars["close"] > 0)

    dollar_vol = bars["close"] * np.nan_to_num(bars["volume"])
    liq = _rolling_median(dollar_vol, 20) >= float(cfg.universe.min_dollar_volume)
    px_ok = np.nan_to_num(bars["close"]) >= float(cfg.universe.min_price)
    tradable = member & priced & liq & px_ok

    ever = set(tickers)
    today = universe.members_on(universe.change_dates[-1])
    missing = ever - set(keep)
    coverage = {
        "provider": prov.name,
        "provider_has_delisted": prov.has_delisted,
        "universe_size": len(ever),
        "fetched": len(keep),
        "coverage": len(keep) / max(1, len(ever)),
        "missing": len(missing),
        "missing_departed": len(missing - today),
        "missing_sample": sorted(missing)[:25],
    }
    return Panel(dates, keep, bars, member, tradable, coverage)


def _rolling_median(a: np.ndarray, win: int) -> np.ndarray:
    """Trailing rolling median down axis 0, NaN-safe, no lookahead."""
    T = a.shape[0]
    out = np.full_like(a, np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(T):
            lo = max(0, i - win + 1)
            out[i] = np.nanmedian(a[lo:i + 1], axis=0)
    return np.nan_to_num(out, nan=0.0)
