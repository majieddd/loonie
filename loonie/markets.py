"""One store per market, so a strategy can be told where its data came from.

The system began as equities and grew a single panel. Adding crypto, foreign
exchange and options to that panel would be a mistake in three directions at
once: they trade on different calendars, their returns have different
distributions, and -- most importantly -- the honesty problems are not the
same. Equities have survivorship bias. Crypto has exchange failure and
listings that did not exist for most of the history. Forex has none of those
and a different one: the rates are official daily fixings, not tradable
quotes.

So each market gets its own store, its own symbol registry and its own
coverage note, and anything reading them has to say which it is using.

    stocks   yfinance, ~2,400 liquid US names, 2016-
    crypto   yfinance, daily, BTC from 2014-
    forex    ECB daily reference rates, 40+ currencies, 1999-
    options  CBOE index volatility history; NO chains (see the caveat below)

WHAT EACH ONE IS NOT.

Stocks: the provider drops delisted securities, so every name still exists.
Measured at 82.7% coverage on the S&P panel and worse on the wide one.

Crypto: yfinance lists what survived. Dead exchanges and dead tokens are
absent, which is a harsher survivorship problem than equities -- the base rate
of total loss is far higher and the failures are exactly the interesting
cases. It also trades 365 days a year, so "annualised" means something
different.

Forex: ECB reference rates are a 14:15 CET daily fixing published for
accounting, NOT a price anyone traded at. Spreads, carry and the fact that FX
is quoted in pairs all sit outside this data. It is clean, long and honest
about being a fixing.

Options: THERE IS NO FREE HISTORICAL CHAIN DATA. What CBOE publishes is index
volatility history -- the VIX and its cousins -- which is an implied
volatility LEVEL, not an option price. Anything built on it is a model of what
an option would have cost, not a record of what one did cost, and this module
labels it as such so nothing downstream can forget.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import resolve

STORE = "data/markets"

ECB_HIST = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/%s_History.csv"
UA = "loonie-research (individual)"


@dataclass
class MarketPanel:
    """Aligned price history for one market, plus what is wrong with it."""
    market: str
    dates: pd.DatetimeIndex
    symbols: list
    close: np.ndarray
    volume: np.ndarray = None
    open: np.ndarray = None
    high: np.ndarray = None
    low: np.ndarray = None
    tradable: np.ndarray = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.close.shape

    # Daily moves beyond this are dropped, not clipped. They are not returns.
    #
    # yfinance back-adjusts for splits, so a company that has done several
    # reverse splits has its history multiplied up astronomically: JAGX peaks
    # at an adjusted $714,285,696 and GNLN spans a 9,342,787x range. Reverse
    # splits happen BECAUSE a stock collapsed, so these cluster entirely in
    # the illiquid end of the universe -- which is how "long illiquid, short
    # liquid" came back at t -3.47 and "low turnover" at t -7.59. Both were
    # measuring a handful of reverse-split artifacts, not a market effect.
    #
    # The min_price floor cannot catch them: in adjusted terms they look
    # expensive rather than cheap.
    MAX_DAILY_MOVE = 0.60

    def returns(self, sanitise: bool = True) -> np.ndarray:
        c = np.asarray(self.close, dtype=np.float64)
        prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (c - prev) / np.where(np.abs(prev) > 1e-12, prev, np.nan)
        if sanitise and self.market != "crypto":
            # Crypto genuinely moves more than 60% in a day; equities and FX
            # do not, and a "return" that large in this data is an adjustment
            # artefact rather than something anyone could have traded.
            r = np.where(np.abs(r) > self.MAX_DAILY_MOVE, np.nan, r)
        return np.nan_to_num(r)

    def describe(self) -> str:
        live = self.tradable.sum(axis=1) if self.tradable is not None else None
        n = float(np.median(live[live > 0])) if live is not None and (live > 0).any() else 0
        return ("%-7s %d sessions x %d symbols | %s..%s | median %d tradable/day"
                % (self.market, self.shape[0], self.shape[1],
                   self.dates[0].date(), self.dates[-1].date(), n))


def _path(market: str) -> "object":
    p = resolve("%s/%s.npz" % (STORE, market))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(panel: MarketPanel) -> None:
    arrays = {"close": panel.close}
    for f in ("open", "high", "low", "volume", "tradable"):
        v = getattr(panel, f)
        if v is not None:
            arrays[f] = v
    np.savez_compressed(
        _path(panel.market),
        dates=panel.dates.values.astype("datetime64[D]"),
        symbols=np.array(panel.symbols, dtype=object),
        meta=np.array([repr(panel.meta)], dtype=object), **arrays)


def load(market: str) -> MarketPanel | None:
    p = _path(market)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    got = {k: z[k] for k in z.files}
    meta = {}
    try:
        meta = eval(str(got.get("meta", ["{}"])[0]))    # written by us only
    except Exception:
        pass
    return MarketPanel(
        market=market, dates=pd.DatetimeIndex(got["dates"]),
        symbols=list(got["symbols"]), close=got["close"],
        volume=got.get("volume"), open=got.get("open"),
        high=got.get("high"), low=got.get("low"),
        tradable=got.get("tradable"), meta=meta)


# =============================================================================
#  Crypto
# =============================================================================
CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
    "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD",
    "XLM-USD", "ATOM-USD", "ETC-USD", "XMR-USD", "ALGO-USD", "VET-USD",
    "FIL-USD", "ICP-USD", "AAVE-USD", "UNI-USD", "MKR-USD", "EOS-USD",
]


def fetch_crypto(symbols=None, start="2014-01-01", min_days: int = 250) -> MarketPanel:
    """Daily crypto bars. Trades every day, including weekends."""
    import yfinance as yf

    symbols = symbols or CRYPTO_SYMBOLS
    d = yf.download(symbols, start=start, auto_adjust=True, progress=False,
                    threads=True, group_by="column")
    close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d
    keep = [c for c in close.columns if close[c].notna().sum() >= min_days]
    close = close[keep].sort_index()

    def grab(field):
        if isinstance(d.columns, pd.MultiIndex) and field in d.columns.get_level_values(0):
            return d[field].reindex(columns=keep).to_numpy(dtype="float32")
        return None

    c = close.to_numpy(dtype="float32")
    tradable = np.isfinite(c) & (c > 0)
    return MarketPanel(
        market="crypto", dates=pd.DatetimeIndex(close.index).tz_localize(None),
        symbols=list(keep), close=c, open=grab("Open"), high=grab("High"),
        low=grab("Low"), volume=grab("Volume"), tradable=tradable,
        meta={"source": "yfinance", "calendar": "365d",
              "caveat": "listed survivors only; dead tokens and exchanges absent"})


# =============================================================================
#  Forex
# =============================================================================
def fetch_forex(min_days: int = 1000) -> MarketPanel:
    """ECB daily euro reference rates, 1999 onward.

    Quoted as units of currency per EUR. Inverted here to USD-base crosses
    where possible, because every other market in this system is priced in
    dollars and a strategy comparing them should not have to remember which
    way one of them points.
    """
    req = urllib.request.Request(ECB_HIST, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as fh:
        blob = fh.read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=[c for c in df.columns if df[c].notna().sum() < min_days],
                 errors="ignore")
    if "USD" not in df.columns:
        raise RuntimeError("ECB file has no USD column")

    # X per EUR -> X per USD. USD itself becomes EUR per USD.
    usd = df["USD"]
    out = {}
    for c in df.columns:
        if c == "USD":
            out["EURUSD"] = usd
        else:
            out["USD" + c] = df[c] / usd
    fx = pd.DataFrame(out).dropna(how="all")

    c = fx.to_numpy(dtype="float32")
    return MarketPanel(
        market="forex", dates=pd.DatetimeIndex(fx.index), symbols=list(fx.columns),
        close=c, tradable=np.isfinite(c) & (c > 0),
        meta={"source": "ECB daily reference rates",
              "caveat": "14:15 CET accounting fixing, NOT a tradable quote; "
                        "no spread, no carry, no intraday"})


# =============================================================================
#  Options -- volatility only. There are no free historical chains.
# =============================================================================
VOL_INDICES = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "RVX", "VXN"]


def fetch_volatility(indices=None) -> MarketPanel:
    """CBOE index volatility history.

    This is NOT options data. It is the implied volatility the market was
    charging on index options, which is one input to an option price and not
    the price itself. Every strategy built on it is a MODEL of what a trade
    would have cost, and the meta dict says so, so the label travels with the
    numbers.
    """
    indices = indices or VOL_INDICES
    frames = {}
    for name in indices:
        try:
            req = urllib.request.Request(CBOE_VIX % name,
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read().decode("utf-8", errors="replace")
            d = pd.read_csv(io.StringIO(raw))
            dcol = next((c for c in d.columns if "date" in c.lower()), d.columns[0])
            ccol = next((c for c in d.columns if "close" in c.lower()), None)
            if ccol is None:
                continue
            d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
            s = d.dropna(subset=[dcol]).set_index(dcol)[ccol]
            frames[name] = pd.to_numeric(s, errors="coerce")
        except Exception:
            continue
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("no CBOE volatility history retrieved")
    vol = pd.DataFrame(frames).sort_index().dropna(how="all")
    c = vol.to_numpy(dtype="float32")
    return MarketPanel(
        market="volatility", dates=pd.DatetimeIndex(vol.index),
        symbols=list(vol.columns), close=c,
        tradable=np.isfinite(c) & (c > 0),
        meta={"source": "CBOE index volatility history",
              "is_option_prices": False,
              "caveat": "implied volatility LEVELS, not option prices. Any P&L "
                        "derived from these is modelled, not observed."})


def registry() -> list:
    """What is stored, how much of it, and what is wrong with each."""
    out = []
    for m in ("stocks", "crypto", "forex", "volatility"):
        p = load(m)
        if p is None:
            out.append({"market": m, "present": False})
            continue
        out.append({
            "market": m, "present": True,
            "symbols": len(p.symbols), "sessions": int(p.shape[0]),
            "start": str(p.dates[0].date()), "end": str(p.dates[-1].date()),
            "source": p.meta.get("source"), "caveat": p.meta.get("caveat"),
        })
    return out
