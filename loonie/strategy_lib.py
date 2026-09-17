"""The strategies themselves. Each registers into loonie/strategies.REGISTRY.

Grouped by market. Every one is deliberately simple and pre-specified: these
are hypotheses drawn from documented effects, not the output of a search, so
they carry no multiple-testing debt beyond the handful written here.
"""
from __future__ import annotations

import numpy as np

from . import markets as M
from .strategies import (COST_BPS, Result, _ranks, _roll, long_short_book,
                         register)


def _stock_panel():
    """The wide equity panel, loaded through the market store."""
    import pandas as pd

    from .config import resolve

    p = resolve("data/universe/wide_panel.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    close = z["close"].astype(np.float32)
    member = z["member"] & np.isfinite(close) & (close > 0)
    return M.MarketPanel(
        market="stocks", dates=pd.DatetimeIndex(z["dates"]),
        symbols=list(z["tickers"]), close=close, volume=z["volume"],
        open=z["open"], high=z["high"], low=z["low"], tradable=member,
        meta={"source": "yfinance wide", "caveat": "listed survivors only"})


def _bench(panel):
    r = panel.returns()
    return np.nan_to_num(np.nanmean(np.where(panel.tradable, r, np.nan), axis=1))


# =============================================================================
#  Equities -- the effects that survived a split on the wide universe
# =============================================================================
@register("stk_lowvol", "Low-Volatility Tilt", "stocks",
          "Long the calmest quintile of the liquid US universe, short the "
          "wildest, rebalanced monthly. The low-volatility anomaly is among "
          "the most replicated effects in equities: high-beta, high-variance "
          "names persistently underperform on a risk-adjusted basis.",
          "Survivorship works AGAINST this one. Failed high-volatility names "
          "are missing from the data, which should make the short leg look "
          "better than reality, not worse.")
def stk_lowvol(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_lowvol", "Low-Volatility Tilt", "stocks", "",
                      ok=False, reason="wide panel not built")
    r = p.returns()
    vol = _roll(r, 63, np.nanstd)
    score = -_ranks(vol, p.tradable)          # calm = high score
    net, _ = long_short_book(score, p.tradable, r, n, n, hold,
                             COST_BPS["stocks"])
    return _wrap("stk_lowvol", p, net)


@register("stk_mom121", "12-1 Momentum", "stocks",
          "Long the strongest decile by return over the last twelve months "
          "excluding the most recent one, short the weakest, held a month. "
          "Skipping the recent month avoids short-term reversal, which is a "
          "different and opposing effect.",
          "Momentum crashes. The worst months for this strategy cluster in "
          "sharp market rebounds, and a three-year sample may not contain one.")
def stk_mom121(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_mom121", "12-1 Momentum", "stocks", "",
                      ok=False, reason="wide panel not built")
    c = np.asarray(p.close, dtype=np.float64)
    T = c.shape[0]
    sh = lambda k: np.vstack([np.full((k, c.shape[1]), np.nan), c[:-k]])  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = sh(21) / sh(252) - 1.0
    score = _ranks(mom, p.tradable)
    net, _ = long_short_book(score, p.tradable, p.returns(), n, n, hold,
                             COST_BPS["stocks"])
    return _wrap("stk_mom121", p, net)


@register("stk_lowvol_mom", "Low-Vol + Momentum Blend", "stocks",
          "The average of the two rank signals above. They are close to "
          "uncorrelated -- one prefers calm names, the other strong ones -- "
          "so blending should raise the ratio of signal to idiosyncratic "
          "noise without adding a new hypothesis.",
          "A blend of two things measured on the same data is not a third "
          "independent piece of evidence.")
def stk_lowvol_mom(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_lowvol_mom", "Low-Vol + Momentum Blend", "stocks",
                      "", ok=False, reason="wide panel not built")
    c = np.asarray(p.close, dtype=np.float64)
    r = p.returns()
    sh = lambda k: np.vstack([np.full((k, c.shape[1]), np.nan), c[:-k]])  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = sh(21) / sh(252) - 1.0
    score = 0.5 * (-_ranks(_roll(r, 63, np.nanstd), p.tradable)
                   + _ranks(mom, p.tradable))
    net, _ = long_short_book(score, p.tradable, r, n, n, hold,
                             COST_BPS["stocks"])
    return _wrap("stk_lowvol_mom", p, net)


# =============================================================================
#  Crypto
# =============================================================================
@register("cry_trend", "Crypto Trend Following", "crypto",
          "Hold each coin only while it trades above its own 100-day average, "
          "equal-weighted across whichever qualify, otherwise sit in cash. "
          "Time-series momentum is the single most documented effect in "
          "crypto and the one most consistent with how the asset class "
          "actually moves.",
          "Survivorship is severe here: dead tokens and failed exchanges are "
          "simply absent, and total loss is a normal outcome in this market "
          "rather than a rare one.")
def cry_trend(window=100):
    p = M.load("crypto")
    if p is None:
        return Result("cry_trend", "Crypto Trend Following", "crypto", "",
                      ok=False, reason="crypto panel not fetched")
    c = np.asarray(p.close, dtype=np.float64)
    ma = _roll(c, window, np.nanmean)
    sig = (c > ma) & p.tradable
    w = sig / np.maximum(sig.sum(axis=1, keepdims=True), 1.0)
    held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross = (held * np.nan_to_num(p.returns())).sum(axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, w.shape[1])), w]),
                          axis=0)).sum(axis=1)
    return _wrap("cry_trend", p, gross - turn * COST_BPS["crypto"] / 1e4)


@register("cry_xmom", "Crypto Cross-Sectional Momentum", "crypto",
          "Long the third of coins with the strongest 30-day return, short "
          "the weakest third, rebalanced weekly. Tests whether relative "
          "strength within crypto predicts, separately from the market's own "
          "direction.",
          "A dollar-neutral crypto book still carries large residual risk: "
          "these assets correlate near one in a sell-off, which is exactly "
          "when the hedge is needed.")
def cry_xmom(hold=7, window=30, n=6):
    p = M.load("crypto")
    if p is None:
        return Result("cry_xmom", "Crypto Cross-Sectional Momentum", "crypto",
                      "", ok=False, reason="crypto panel not fetched")
    c = np.asarray(p.close, dtype=np.float64)
    sh = np.vstack([np.full((window, c.shape[1]), np.nan), c[:-window]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = c / sh - 1.0
    net, _ = long_short_book(_ranks(mom, p.tradable), p.tradable, p.returns(),
                             n, n, hold, COST_BPS["crypto"])
    return _wrap("cry_xmom", p, net)


# =============================================================================
#  Forex
# =============================================================================
@register("fx_trend", "FX Trend Following", "forex",
          "Long the third of currencies with the strongest 60-day trend "
          "against the dollar, short the weakest third, rebalanced monthly. "
          "Trend following in currencies is the oldest systematic strategy "
          "there is and the basis of most managed-futures programmes.",
          "ECB reference rates are a daily accounting fixing, not a tradable "
          "quote. There is no spread here and no interest-rate carry, and "
          "carry is where much of the real return in FX actually lives.")
def fx_trend(hold=21, window=60, n=10):
    p = M.load("forex")
    if p is None:
        return Result("fx_trend", "FX Trend Following", "forex", "",
                      ok=False, reason="forex panel not fetched")
    c = np.asarray(p.close, dtype=np.float64)
    sh = np.vstack([np.full((window, c.shape[1]), np.nan), c[:-window]])
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = c / sh - 1.0
    net, _ = long_short_book(_ranks(trend, p.tradable), p.tradable, p.returns(),
                             n, n, hold, COST_BPS["forex"])
    return _wrap("fx_trend", p, net)


@register("fx_reversal", "FX Mean Reversion", "forex",
          "The opposite bet at a shorter horizon: short whichever currencies "
          "rose most over five days, long those that fell most, held a week. "
          "Currencies are widely held to overshoot at short horizons and "
          "trend at long ones, so this and the trend strategy above should "
          "not both work.",
          "Same fixing caveat as above. If both this and FX trend show a "
          "profit on the same data, suspect the data before believing both.")
def fx_reversal(hold=5, window=5, n=10):
    p = M.load("forex")
    if p is None:
        return Result("fx_reversal", "FX Mean Reversion", "forex", "",
                      ok=False, reason="forex panel not fetched")
    c = np.asarray(p.close, dtype=np.float64)
    sh = np.vstack([np.full((window, c.shape[1]), np.nan), c[:-window]])
    with np.errstate(invalid="ignore", divide="ignore"):
        move = c / sh - 1.0
    net, _ = long_short_book(-_ranks(move, p.tradable), p.tradable, p.returns(),
                             n, n, hold, COST_BPS["forex"])
    return _wrap("fx_reversal", p, net)


def _wrap(sid, panel, net):
    spec = {k: v for k, v in REGISTRY_SPEC(sid).items() if k != "fn"}
    eq = np.cumprod(1.0 + np.nan_to_num(net))
    return Result(dates=panel.dates, equity=eq, returns=net,
                  benchmark=_bench(panel), **spec)


def REGISTRY_SPEC(sid):
    from .strategies import REGISTRY
    return REGISTRY[sid]


# =============================================================================
#  Options -- MODELLED. There is no free historical chain data.
# =============================================================================
#
# Everything below prices a trade from two observable series: the implied
# volatility the market was charging (CBOE VIX) and what the index actually
# did (SPY). That is enough to model the dominant term in a short-premium
# P&L -- the gap between implied and realised volatility -- and it is not
# enough to be a backtest.
#
# What is assumed rather than observed: the bid-ask on every leg, the skew
# (VIX is one number; real condors are priced off a smile), early assignment,
# and the fact that a real book cannot always be opened at the mid. The cost
# charge is deliberately punitive at 50 bps a round trip for that reason, and
# it is still a guess.
#
# These rows exist so the strategy list can carry the SHAPE of a short
# volatility return -- many small wins, rare large losses -- next to the
# equity strategies, honestly labelled. They are not evidence that an options
# book would have made this money.

def _spy_and_vix():
    """Daily SPY closes aligned to VIX. The two inputs the models need."""
    import warnings

    import pandas as pd
    import yfinance as yf

    vol = M.load("volatility")
    if vol is None or "VIX" not in vol.symbols:
        return None, None
    vix = pd.Series(vol.close[:, vol.symbols.index("VIX")].astype(float),
                    index=vol.dates).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = yf.download("SPY", start="1993-01-01", auto_adjust=True,
                        progress=False)
    if d is None or not len(d):
        return None, None
    px = d["Close"]
    if hasattr(px, "columns"):
        px = px.iloc[:, 0]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    idx = px.index.intersection(vix.index)
    return px.reindex(idx), vix.reindex(idx)


def _opt_result(sid, dates, rets, hold=21):
    """One row per HOLDING PERIOD, not per day -- so say so.

    Without periods_per_year the 442 monthly periods spanning 36 years get
    annualised as 442 trading days, and a strategy that made 6% a year reports
    266%.
    """
    import pandas as pd
    r = np.asarray(rets, dtype=float)
    spec = {k: v for k, v in REGISTRY_SPEC(sid).items() if k != "fn"}
    return Result(dates=pd.DatetimeIndex(dates), equity=np.cumprod(1.0 + r),
                  returns=r, benchmark=None,
                  periods_per_year=252.0 / max(hold, 1), **spec)


@register("opt_theta", "Short Volatility / Theta Decay", "options",
          "Sell a one-month at-the-money straddle on the index every month "
          "and hold it to expiry. The position profits whenever the index "
          "moves less than the implied volatility it was sold at, which is "
          "usually: the variance risk premium is one of the most persistent "
          "effects in derivatives.",
          "MODELLED, not backtested. P&L is computed from VIX against "
          "realised SPY movement; no free chain data exists, so bid-ask, skew "
          "and assignment are assumed rather than observed.")
def opt_theta(hold=21):
    px, vix = _spy_and_vix()
    if px is None:
        return Result("opt_theta", "Short Volatility / Theta Decay", "options",
                      "", ok=False, reason="SPY or VIX unavailable")
    p = px.to_numpy(float)
    v = vix.to_numpy(float) / 100.0
    rets, dates = [], []
    for t in range(0, len(p) - hold, hold):
        tau = hold / 252.0
        # Standard approximation for an ATM straddle premium as a fraction of
        # notional: 0.8 * sigma * sqrt(tau).
        premium = 0.8 * v[t] * np.sqrt(tau)
        moved = abs(p[t + hold] / p[t] - 1.0)
        rets.append(premium - moved - COST_BPS["options"] / 1e4)
        dates.append(vix.index[t + hold])
    return _opt_result("opt_theta", dates, rets, hold)


@register("opt_condor", "Iron Condor (monthly, 1-sigma wings)", "options",
          "Sell a one-month iron condor with short strikes one standard "
          "deviation out and defined-risk wings beyond them. Keeps the whole "
          "credit whenever the index finishes inside the wings, which on a "
          "one-sigma band is most months, and loses a capped amount when it "
          "does not.",
          "MODELLED, not backtested. Worse than the straddle: a condor's P&L "
          "depends on the volatility SMILE, and VIX is a single number that "
          "contains no skew at all.")
def opt_condor(hold=21, sigma_mult=1.0, wing=0.5):
    px, vix = _spy_and_vix()
    if px is None:
        return Result("opt_condor", "Iron Condor (monthly, 1-sigma wings)",
                      "options", "", ok=False, reason="SPY or VIX unavailable")
    p = px.to_numpy(float)
    v = vix.to_numpy(float) / 100.0
    rets, dates = [], []
    for t in range(0, len(p) - hold, hold):
        tau = hold / 252.0
        sd = v[t] * np.sqrt(tau)
        short_k = sigma_mult * sd
        long_k = short_k + wing * sd
        # Credit for a one-sigma condor, roughly a third of the wing width.
        # This is the crudest assumption in the model and it is load-bearing.
        credit = 0.33 * (long_k - short_k)
        moved = abs(p[t + hold] / p[t] - 1.0)
        if moved <= short_k:
            pnl = credit
        else:
            pnl = credit - min(moved - short_k, long_k - short_k)
        rets.append(pnl - COST_BPS["options"] / 1e4)
        dates.append(vix.index[t + hold])
    return _opt_result("opt_condor", dates, rets, hold)


@register("opt_vrp_timed", "Variance Premium, Regime-Timed", "options",
          "The same short straddle, opened only when implied volatility sits "
          "above its own trailing one-year median. The gap between implied "
          "and realised is widest when volatility is already elevated, so "
          "standing aside in calm regimes avoids collecting almost nothing "
          "for the same tail risk.",
          "MODELLED, not backtested. It also adds a timing rule to a strategy "
          "already in this table, which is one more hypothesis tested on the "
          "same data.")
def opt_vrp_timed(hold=21):
    px, vix = _spy_and_vix()
    if px is None:
        return Result("opt_vrp_timed", "Variance Premium, Regime-Timed",
                      "options", "", ok=False, reason="SPY or VIX unavailable")
    p = px.to_numpy(float)
    v = vix.to_numpy(float) / 100.0
    rets, dates = [], []
    for t in range(252, len(p) - hold, hold):
        med = np.nanmedian(v[t - 252:t])        # trailing only
        if v[t] <= med:
            rets.append(0.0)                    # stand aside, and pay nothing
        else:
            tau = hold / 252.0
            premium = 0.8 * v[t] * np.sqrt(tau)
            moved = abs(p[t + hold] / p[t] - 1.0)
            rets.append(premium - moved - COST_BPS["options"] / 1e4)
        dates.append(vix.index[t + hold])
    return _opt_result("opt_vrp_timed", dates, rets, hold)
