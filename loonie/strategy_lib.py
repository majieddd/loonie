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


# =============================================================================
#  The documented anomaly library
# =============================================================================
#
# Each of these is a published, named effect with a citation, implemented in
# its simplest faithful form. They are hypotheses drawn from the literature,
# not search output -- but the literature is itself the problem.
#
# Hou, Xue and Zhang (2020) replicated 452 published anomalies and found 65%
# could not clear |t| >= 1.96 once microcaps were handled properly, and 52%
# failed regardless after adjusting for multiple testing. Harvey, Liu and Zhu
# (2016) argue the honest hurdle for a NEW factor is nearer t = 3.0 precisely
# because so many have been tried.
#
# So the right expectation for everything below is that most of it will not
# work, and the ones that look best will mostly be the ones that got lucky on
# this particular span. The table reports the multiple-testing bar next to the
# results for that reason.

def _px(p):
    return np.asarray(p.close, dtype=np.float64)


def _shift(c, k):
    return np.vstack([np.full((k, c.shape[1]), np.nan), c[:-k]])


def _mom(c, back, skip=0):
    """Return from `back` periods ago to `skip` periods ago."""
    a, b = _shift(c, skip) if skip else c, _shift(c, back)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b - 1.0


def _ls(sid, p, score, hold, n, cost=None, invert=False):
    s = -score if invert else score
    net, _ = long_short_book(s, p.tradable, p.returns(), n, n, hold,
                             cost if cost is not None else COST_BPS[p.market])
    return _wrap(sid, p, net)


# ---------------------------------------------------------------- equities
@register("stk_strev", "Short-Term Reversal", "stocks",
          "Long last month's worst performers, short its best, held a month. "
          "Jegadeesh (1990) and Lehmann (1990): one-month returns reverse, "
          "which is the opposite sign to momentum at twelve months and one of "
          "the oldest documented effects in equities.",
          "Reversal is concentrated in small, illiquid names and is largely a "
          "liquidity-provision premium. On a liquid universe with realistic "
          "costs it is much weaker than the published version.")
def stk_strev(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_strev", "Short-Term Reversal", "stocks", "",
                      ok=False, reason="wide panel not built")
    return _ls("stk_strev", p, _ranks(_mom(_px(p), 21), p.tradable), hold, n,
               invert=True)


@register("stk_ltrev", "Long-Term Reversal", "stocks",
          "Long the worst performers of the last three years, short the best, "
          "rebalanced quarterly. De Bondt and Thaler (1985): extreme "
          "multi-year winners and losers revert, which they read as the "
          "market overreacting to long runs of news.",
          "Needs a long sample to test at all. Three-year formation on a "
          "ten-year panel leaves very few independent observations, so the "
          "t-statistic here is weak evidence either way.")
def stk_ltrev(hold=63, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_ltrev", "Long-Term Reversal", "stocks", "",
                      ok=False, reason="wide panel not built")
    return _ls("stk_ltrev", p, _ranks(_mom(_px(p), 756, 21), p.tradable),
               hold, n, invert=True)


@register("stk_52whigh", "52-Week High Proximity", "stocks",
          "Long names trading closest to their own 52-week high, short those "
          "furthest below it, held a month. George and Hwang (2004): nearness "
          "to the high predicts better than raw momentum, because traders "
          "anchor on the high and under-react when it is approached.",
          "Mechanically correlated with momentum. If both appear in this "
          "table they are not two independent pieces of evidence.")
def stk_52whigh(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_52whigh", "52-Week High Proximity", "stocks", "",
                      ok=False, reason="wide panel not built")
    c = _px(p)
    hi = _roll(c, 252, np.nanmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        prox = c / hi
    return _ls("stk_52whigh", p, _ranks(prox, p.tradable), hold, n)


@register("stk_bab", "Betting Against Beta", "stocks",
          "Long low-beta names, short high-beta ones, held a month. Frazzini "
          "and Pedersen (2014): investors who cannot use leverage bid up "
          "high-beta stocks instead, so beta is overpriced and the "
          "risk-adjusted payoff runs the other way.",
          "The published version levers the long leg to match beta. This does "
          "not, so it is the raw spread rather than the tradable factor, and "
          "will understate the effect the paper reports.")
def stk_bab(hold=21, n=150, window=252):
    p = _stock_panel()
    if p is None:
        return Result("stk_bab", "Betting Against Beta", "stocks", "",
                      ok=False, reason="wide panel not built")
    r = p.returns()
    mkt = np.nan_to_num(np.nanmean(np.where(p.tradable, r, np.nan), axis=1))
    T, N = r.shape
    beta = np.full((T, N), np.nan)
    for t in range(window, T, 21):
        w_r, w_m = r[t - window:t], mkt[t - window:t]
        vm = np.var(w_m)
        if vm > 1e-12:
            b = ((w_r - w_r.mean(0)) * (w_m - w_m.mean())[:, None]).mean(0) / vm
            beta[t:min(t + 21, T)] = b
    return _ls("stk_bab", p, _ranks(beta, p.tradable), hold, n, invert=True)


@register("stk_ivol", "Idiosyncratic Volatility", "stocks",
          "Long names with the lowest residual volatility against the market, "
          "short the highest, held a month. Ang, Hodrick, Xing and Zhang "
          "(2006) found high idiosyncratic volatility predicts LOW returns, "
          "which standard theory says should not happen at all.",
          "Closely related to the low-volatility tilt already in this table. "
          "Treat the two as one hypothesis measured twice, not two.")
def stk_ivol(hold=21, n=150, window=126):
    p = _stock_panel()
    if p is None:
        return Result("stk_ivol", "Idiosyncratic Volatility", "stocks", "",
                      ok=False, reason="wide panel not built")
    r = p.returns()
    mkt = np.nan_to_num(np.nanmean(np.where(p.tradable, r, np.nan), axis=1))
    T, N = r.shape
    iv = np.full((T, N), np.nan)
    for t in range(window, T, 21):
        w_r, w_m = r[t - window:t], mkt[t - window:t]
        vm = np.var(w_m)
        if vm > 1e-12:
            b = ((w_r - w_r.mean(0)) * (w_m - w_m.mean())[:, None]).mean(0) / vm
            iv[t:min(t + 21, T)] = (w_r - np.outer(w_m, b)).std(0)
    return _ls("stk_ivol", p, _ranks(iv, p.tradable), hold, n, invert=True)


@register("stk_max", "Lottery / MAX Effect", "stocks",
          "Short the names with the biggest single-day gain in the past "
          "month, long those with the smallest, held a month. Bali, Cakici "
          "and Whitelaw (2011): investors overpay for lottery-like payoffs, "
          "so stocks with recent extreme upside subsequently underperform.",
          "Overlaps idiosyncratic volatility and the low-vol tilt: extreme "
          "single-day moves happen in volatile names. Three rows here are "
          "close to the same bet.")
def stk_max(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_max", "Lottery / MAX Effect", "stocks", "",
                      ok=False, reason="wide panel not built")
    mx = _roll(p.returns(), 21, np.nanmax)
    return _ls("stk_max", p, _ranks(mx, p.tradable), hold, n, invert=True)


@register("stk_illiq", "Amihud Illiquidity Premium", "stocks",
          "Long the least liquid names by Amihud's measure -- absolute return "
          "per dollar traded -- short the most liquid, held a month. Amihud "
          "(2002): illiquid assets must offer higher expected returns to "
          "compensate for the cost of getting out.",
          "This deliberately buys what is expensive to trade, so it is the "
          "strategy in this table most likely to be destroyed by real costs. "
          "The 2.6 bps charged here is measured on LIQUID names and is "
          "certainly too low for the long leg.")
def stk_illiq(hold=21, n=100):
    p = _stock_panel()
    if p is None:
        return Result("stk_illiq", "Amihud Illiquidity Premium", "stocks", "",
                      ok=False, reason="wide panel not built")
    r = np.abs(p.returns())
    dv = _px(p) * np.nan_to_num(np.asarray(p.volume, dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        amihud = r / np.where(dv > 0, dv, np.nan)
    return _ls("stk_illiq", p, _ranks(_roll(amihud, 21, np.nanmean),
                                      p.tradable), hold, n)


@register("stk_turnover", "Low Turnover", "stocks",
          "Long names with the lowest share turnover, short the highest, held "
          "a month. Datar, Naik and Radcliffe (1998) and a long line after "
          "them: high-turnover stocks underperform, variously read as a "
          "liquidity premium or as a proxy for speculative interest.",
          "Turnover correlates with volatility and with the lottery measure. "
          "Another row that is not independent of the ones above it.")
def stk_turnover(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_turnover", "Low Turnover", "stocks", "",
                      ok=False, reason="wide panel not built")
    vol = np.nan_to_num(np.asarray(p.volume, dtype=np.float64))
    return _ls("stk_turnover", p, _ranks(_roll(vol, 21, np.nanmean),
                                         p.tradable), hold, n, invert=True)


@register("stk_mom_intermediate", "Intermediate Momentum (12-7)", "stocks",
          "Long the strongest names by return from twelve months ago to seven "
          "months ago, ignoring everything since. Novy-Marx (2012) argued "
          "momentum profits come from the INTERMEDIATE past, not the recent "
          "past, which if true means standard 12-1 momentum is mis-specified.",
          "A direct competitor to the 12-1 row. If both work they are the "
          "same effect; if only one does, this span is too short to say which.")
def stk_mom_intermediate(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_mom_intermediate", "Intermediate Momentum (12-7)",
                      "stocks", "", ok=False, reason="wide panel not built")
    return _ls("stk_mom_intermediate", p,
               _ranks(_mom(_px(p), 252, 126), p.tradable), hold, n)


@register("stk_vol_scaled_mom", "Volatility-Scaled Momentum", "stocks",
          "12-1 momentum with each position sized inversely to its own recent "
          "volatility. Barroso and Santa-Clara (2015): momentum's rare "
          "catastrophic drawdowns are predictable from its own volatility, "
          "and scaling by it roughly doubles the risk-adjusted return.",
          "Fixes momentum's worst property rather than its average one, so "
          "the improvement should show in drawdown and Sharpe, not in raw "
          "return. Judge it on those columns.")
def stk_vol_scaled_mom(hold=21, n=150):
    p = _stock_panel()
    if p is None:
        return Result("stk_vol_scaled_mom", "Volatility-Scaled Momentum",
                      "stocks", "", ok=False, reason="wide panel not built")
    r = p.returns()
    vol = _roll(r, 63, np.nanstd)
    score = _ranks(_mom(_px(p), 252, 21), p.tradable)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (score - 0.5) / np.where(vol > 1e-6, vol, np.nan)
    return _ls("stk_vol_scaled_mom", p, _ranks(scaled, p.tradable), hold, n)


@register("stk_turn_of_month", "Turn-of-the-Month Effect", "stocks",
          "Hold the whole universe only across the last trading day of each "
          "month and the first three of the next, in cash otherwise. Ariel "
          "(1987) and Lakonishok and Smidt (1988): essentially all of the "
          "market's historical return has accrued in that window.",
          "A calendar rule with no mechanism beyond flows, and the most "
          "likely of anything here to be a data artefact. It is also barely a "
          "strategy: it is the market, held a fifth of the time.")
def stk_turn_of_month():
    p = _stock_panel()
    if p is None:
        return Result("stk_turn_of_month", "Turn-of-the-Month Effect",
                      "stocks", "", ok=False, reason="wide panel not built")
    import pandas as pd
    d = pd.DatetimeIndex(p.dates)
    dom = d.day.to_numpy()
    eom = (pd.Series(d).groupby([d.year, d.month]).transform("max")
           == pd.Series(d)).to_numpy()
    on = eom | (dom <= 3)
    bench = _bench(p)
    net = np.where(on, bench, 0.0) - np.abs(np.diff(
        np.concatenate([[0.0], on.astype(float)]))) * COST_BPS["stocks"] / 1e4
    return _wrap("stk_turn_of_month", p, net)


@register("stk_sell_in_may", "Halloween / Sell in May", "stocks",
          "Hold the universe from November through April and stand aside from "
          "May through October. Bouman and Jacobsen (2002) documented the "
          "seasonal in 36 of 37 countries, which is either a remarkable "
          "regularity or a remarkable amount of data mining.",
          "No mechanism has ever been established. Ten years of data contains "
          "ten independent observations of an annual cycle, which is not "
          "enough to distinguish this from chance at any useful confidence.")
def stk_sell_in_may():
    p = _stock_panel()
    if p is None:
        return Result("stk_sell_in_may", "Halloween / Sell in May", "stocks",
                      "", ok=False, reason="wide panel not built")
    import pandas as pd
    m = pd.DatetimeIndex(p.dates).month.to_numpy()
    on = (m >= 11) | (m <= 4)
    bench = _bench(p)
    net = np.where(on, bench, 0.0) - np.abs(np.diff(
        np.concatenate([[0.0], on.astype(float)]))) * COST_BPS["stocks"] / 1e4
    return _wrap("stk_sell_in_may", p, net)


# ------------------------------------------------------------------ crypto
@register("cry_strev", "Crypto Short-Term Reversal", "crypto",
          "Long the coins that fell most over the past week, short those that "
          "rose most, rebalanced weekly. Short-horizon reversal is documented "
          "in crypto much as in equities, and is usually attributed to the "
          "cost of providing liquidity into sharp moves.",
          "Directly contradicts the crypto momentum row at a different "
          "horizon. That is not necessarily inconsistent -- reversal at a "
          "week and trend at a quarter can coexist -- but both winning "
          "handsomely should raise suspicion of the data.")
def cry_strev(hold=7, window=7, n=6):
    p = M.load("crypto")
    if p is None:
        return Result("cry_strev", "Crypto Short-Term Reversal", "crypto", "",
                      ok=False, reason="crypto panel not fetched")
    return _ls("cry_strev", p, _ranks(_mom(_px(p), window), p.tradable),
               hold, n, invert=True)


@register("cry_vol_target", "Crypto Volatility-Targeted Trend", "crypto",
          "The same trend rule as the crypto trend row, but scaling total "
          "exposure so that forecast portfolio volatility stays near 40% "
          "annualised. Volatility targeting is standard in managed futures "
          "and is the usual answer to an asset class whose risk varies by an "
          "order of magnitude between regimes.",
          "Targeting stabilises risk; it does not add return. If this beats "
          "plain trend on Sharpe but not on CAGR, that is the mechanism "
          "working exactly as intended and not an edge.")
def cry_vol_target(window=100, target=0.40):
    p = M.load("crypto")
    if p is None:
        return Result("cry_vol_target", "Crypto Volatility-Targeted Trend",
                      "crypto", "", ok=False, reason="crypto panel not fetched")
    c = _px(p)
    ma = _roll(c, window, np.nanmean)
    sig = (c > ma) & p.tradable
    w = sig / np.maximum(sig.sum(axis=1, keepdims=True), 1.0)
    held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross = (held * np.nan_to_num(p.returns())).sum(axis=1)
    # Trailing realised vol of the strategy itself, never the current bar.
    rv = np.full(len(gross), np.nan)
    for t in range(63, len(gross)):
        rv[t] = np.std(gross[t - 63:t]) * np.sqrt(365.0)
    lev = np.clip(np.where(np.isfinite(rv) & (rv > 1e-6), target / rv, 0.0),
                  0.0, 3.0)
    turn = np.abs(np.diff(np.concatenate([[0.0], lev]))) * 0.5
    return _wrap("cry_vol_target", p,
                 lev * gross - turn * COST_BPS["crypto"] / 1e4)


@register("cry_btc_relative", "Altcoin Rotation vs Bitcoin", "crypto",
          "Long the alt-coins outperforming Bitcoin over the past month, "
          "short Bitcoin itself, rebalanced weekly. Practitioner folklore "
          "holds that capital rotates from Bitcoin into alts in risk-on "
          "phases, which if true should show as persistent relative strength.",
          "This is folklore, not literature: it has no peer-reviewed support "
          "that I am aware of, and it is included precisely because the "
          "difference between a documented effect and a widely repeated one "
          "is what this table exists to measure.")
def cry_btc_relative(hold=7, window=30, n=6):
    p = M.load("crypto")
    if p is None or "BTC-USD" not in p.symbols:
        return Result("cry_btc_relative", "Altcoin Rotation vs Bitcoin",
                      "crypto", "", ok=False, reason="crypto panel or BTC missing")
    # Dividing every coin in a row by the same BTC factor is a per-row
    # CONSTANT, so it leaves the cross-sectional ranks untouched: the first
    # version of this was arithmetically identical to cross-sectional
    # momentum, and reported the same numbers to four significant figures.
    # The bet only means something if BTC is actually the short leg.
    c = _px(p)
    j_btc = p.symbols.index("BTC-USD")
    btc = c[:, j_btc]
    btc_mom = btc / np.concatenate([np.full(window, np.nan), btc[:-window]]) - 1.0
    alt_mom = _mom(c, window)
    T, N = c.shape
    w = np.zeros((T, N))
    for t in range(0, T, hold):
        excess = alt_mom[t] - btc_mom[t]
        excess[j_btc] = np.nan                    # BTC is the benchmark, not a pick
        ok = np.isfinite(excess) & p.tradable[t]
        idx = np.where(ok)[0]
        if len(idx) < n:
            continue
        pick = idx[np.argsort(-excess[idx])[:n]]
        row = np.zeros(N)
        row[pick] = 1.0 / n
        row[j_btc] = -1.0                          # funded by shorting Bitcoin
        w[t:min(t + hold, T)] = row
    held = np.vstack([np.zeros((1, N)), w[:-1]])
    gross = (held * np.nan_to_num(p.returns())).sum(axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, N)), w]), axis=0)).sum(axis=1)
    return _wrap("cry_btc_relative", p,
                 gross - turn * COST_BPS["crypto"] / 1e4)


# ------------------------------------------------------------------- forex
@register("fx_ppp", "FX Long-Horizon Reversion", "forex",
          "Long the currencies that have fallen most against the dollar over "
          "three years, short those that have risen most, rebalanced "
          "quarterly. A crude purchasing-power-parity bet: real exchange "
          "rates revert over multi-year horizons, one of the better "
          "established regularities in international finance.",
          "PPP works over five to ten years. Even 27 years of data contains "
          "only a handful of independent three-year observations, so this is "
          "under-powered by construction however it comes out.")
def fx_ppp(hold=63, window=756, n=10):
    p = M.load("forex")
    if p is None:
        return Result("fx_ppp", "FX Long-Horizon Reversion", "forex", "",
                      ok=False, reason="forex panel not fetched")
    return _ls("fx_ppp", p, _ranks(_mom(_px(p), window), p.tradable), hold, n,
               invert=True)


@register("fx_dollar", "Dollar Trend", "forex",
          "A single position: long a basket of all currencies against the "
          "dollar when the basket has risen over sixty days, short when it "
          "has fallen. The dollar factor is the first principal component of "
          "currency returns and explains most of the common variation in the "
          "asset class.",
          "One position rather than a cross-section, so the effective sample "
          "is far smaller than the session count suggests and the t-statistic "
          "should be read with that in mind.")
def fx_dollar(hold=21, window=60):
    p = M.load("forex")
    if p is None:
        return Result("fx_dollar", "Dollar Trend", "forex", "",
                      ok=False, reason="forex panel not fetched")
    basket = _bench(p)
    cum = np.cumsum(np.nan_to_num(basket))
    trend = cum - np.concatenate([np.full(window, np.nan), cum[:-window]])
    pos = np.where(np.isfinite(trend), np.sign(trend), 0.0)
    held = np.concatenate([[0.0], pos[:-1]])
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return _wrap("fx_dollar", p,
                 held * basket - turn * COST_BPS["forex"] / 1e4)


# ----------------------------------------------------------------- options
@register("opt_vix_carry", "VIX Term-Structure Carry", "options",
          "Short volatility exposure when the VIX curve is in contango -- "
          "three-month implied above spot -- and stand aside when it inverts. "
          "Contango means volatility futures roll down toward spot, and the "
          "roll is the return; inversion is the market pricing stress, which "
          "is when short-volatility positions are destroyed.",
          "MODELLED, not backtested. Uses real CBOE VIX and VIX3M history for "
          "the SIGNAL, which is genuine, but the P&L is still the modelled "
          "straddle rather than a traded VIX future.")
def opt_vix_carry(hold=21):
    px, vix = _spy_and_vix()
    vol = M.load("volatility")
    if px is None or vol is None or "VIX3M" not in vol.symbols:
        return Result("opt_vix_carry", "VIX Term-Structure Carry", "options",
                      "", ok=False, reason="VIX3M history unavailable")
    import pandas as pd
    v3 = pd.Series(vol.close[:, vol.symbols.index("VIX3M")].astype(float),
                   index=vol.dates).reindex(vix.index)
    p, v, t3 = px.to_numpy(float), vix.to_numpy(float) / 100.0, v3.to_numpy(float) / 100.0
    rets, dates = [], []
    for t in range(0, len(p) - hold, hold):
        dates.append(vix.index[t + hold])
        if not np.isfinite(t3[t]) or t3[t] <= v[t]:
            rets.append(0.0)                 # backwardation: stand aside
            continue
        tau = hold / 252.0
        premium = 0.8 * v[t] * np.sqrt(tau)
        moved = abs(p[t + hold] / p[t] - 1.0)
        rets.append(premium - moved - COST_BPS["options"] / 1e4)
    return _opt_result("opt_vix_carry", dates, rets, hold)


@register("opt_put_write", "Cash-Secured Put Write", "options",
          "Sell a one-month at-the-money put on the index each month and hold "
          "it to expiry, fully collateralised. The CBOE PUT index has tracked "
          "this since 1986; it earns the equity risk premium plus the "
          "variance premium while capping the upside.",
          "MODELLED, not backtested. A put write is a long-equity position "
          "with the tail left on and the upside sold, so comparing its return "
          "to a market-neutral strategy in this table is comparing different "
          "kinds of risk.")
def opt_put_write(hold=21):
    px, vix = _spy_and_vix()
    if px is None:
        return Result("opt_put_write", "Cash-Secured Put Write", "options",
                      "", ok=False, reason="SPY or VIX unavailable")
    p, v = px.to_numpy(float), vix.to_numpy(float) / 100.0
    rets, dates = [], []
    for t in range(0, len(p) - hold, hold):
        tau = hold / 252.0
        premium = 0.4 * v[t] * np.sqrt(tau)      # ATM put ~= half a straddle
        move = p[t + hold] / p[t] - 1.0
        rets.append(premium + min(move, 0.0) - COST_BPS["options"] / 1e4)
        dates.append(vix.index[t + hold])
    return _opt_result("opt_put_write", dates, rets, hold)
