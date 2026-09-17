"""What an order actually costs, as opposed to what the backtest charges it.

The paper broker used to fill every order instantly at the last close plus a
flat five basis points. Three things are wrong with that, and each one flatters
the result in the same direction:

YOU CANNOT TRADE A PRICE YOU HAVE ALREADY SEEN. The signal is computed from
Tuesday's close. Tuesday's close is gone by the time you have computed it. In
reality the order rests until Wednesday's open, and the gap between those two
prices is risk you carry without having chosen it. Filling at the close that
generated the signal is a one-bar lookahead sitting in the execution layer
rather than the feature layer, where nobody thinks to look for it.

COST DOES NOT SCALE WITH SIZE IN REAL MARKETS. A flat 5 bps says a $500 order
and a $500,000 order in the same name cost the same fraction, which is exactly
backwards -- the whole difficulty of running more money is that it is not.
Impact here follows the square-root law, the standard empirical form: cost in
volatility units grows with the square root of the fraction of daily volume
you consume. Small orders are nearly free, large ones are not, and the
strategy has to earn the difference.

THE SPREAD IS NOT A CONSTANT. A $400 megacap trades tighter than a $9 name
with a tenth of the volume. Charging both 5 bps subsidises trading illiquid
things, which is precisely where a naive search likes to go hunting.

None of this is conservatism for its own sake. The entire purpose of paper
trading before real money is to learn whether a strategy survives contact with
execution; a simulator that cannot express the ways execution kills strategies
will report that every strategy survives.

WHAT IS STILL MISSING, stated plainly so nobody mistakes this for a market
simulator: no order book, no queue position, no adverse selection, no intraday
timing, no borrow constraints on shorts, and gaps are taken as given rather
than modelled. It is a cost model, and the costs it produces are estimates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12

# Square-root impact coefficient. Empirical estimates cluster around 0.3-1.0 in
# units of daily volatility; 0.5 is a common central choice and is used here
# because being wrong towards expensive is the safe direction for a system
# deciding whether to risk real money.
IMPACT_COEF = 0.5

# Refuse to consume more than this share of a day's dollar volume in one order.
# Above roughly 10% the square-root model stops being trustworthy anyway, so
# the cap is as much about not believing our own numbers as about liquidity.
MAX_PARTICIPATION = 0.10


@dataclass
class Fill:
    """One simulated execution, with every cost component kept separate.

    Broken out rather than netted because the point of paper trading is to
    learn WHICH cost kills a strategy. A single "slippage" number tells you
    the strategy lost money on execution; these tell you whether it was the
    spread (trade less often), impact (trade smaller), or the gap (the signal
    decays overnight and the edge was never really there).
    """
    symbol: str
    side: str
    qty: float = 0.0
    ref_price: float = 0.0        # the close the decision was made on
    open_price: float = 0.0       # where it actually filled, before costs
    fill_price: float = 0.0       # after spread and impact
    gap_bps: float = 0.0          # close -> open, signed against us
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    commission: float = 0.0
    filled: bool = True
    note: str = ""

    @property
    def notional(self) -> float:
        return abs(self.qty * self.fill_price)

    @property
    def total_cost_bps(self) -> float:
        return self.gap_bps + self.spread_bps + self.impact_bps

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side,
            "qty": round(self.qty, 6),
            "ref_price": round(self.ref_price, 4),
            "open_price": round(self.open_price, 4),
            "fill_price": round(self.fill_price, 4),
            "gap_bps": round(self.gap_bps, 2),
            "spread_bps": round(self.spread_bps, 2),
            "impact_bps": round(self.impact_bps, 2),
            "commission": round(self.commission, 4),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "notional": round(self.notional, 2),
            "filled": self.filled, "note": self.note,
        }


def spread_bps(price: float, dollar_vol: float) -> float:
    """Estimated half-spread, in basis points, from price and liquidity.

    A stand-in for quote data we do not have. It reproduces the two effects
    that matter: spreads widen as price falls (tick size is a larger fraction
    of a cheap stock) and as volume thins. Calibrated loosely to US large-cap
    equities, where a liquid megacap sits near 1 bp and a thin small-cap runs
    tens of bps.
    """
    p = max(float(price), 0.01)
    dv = max(float(dollar_vol), 1.0)
    tick = 0.01 / p * 1e4 / 2.0                 # half a penny, in bps
    liq = 60.0 / math.sqrt(dv / 1e6 + 1.0)      # thins out with volume
    return float(np.clip(tick + liq, 0.5, 250.0))


def impact_bps(notional: float, adv_dollars: float, daily_vol: float,
               coef: float = IMPACT_COEF) -> float:
    """Square-root market impact, in basis points.

        impact = coef * sigma * sqrt(Q / ADV)

    sigma is the name's daily return volatility, Q/ADV the fraction of a day's
    volume the order consumes. The functional form is the standard empirical
    one; the point of using it rather than a constant is that it makes size
    expensive, which is the constraint every real strategy eventually meets.
    """
    adv = max(float(adv_dollars), 1.0)
    part = max(float(notional), 0.0) / adv
    sigma = max(float(daily_vol), 1e-4)
    return float(coef * sigma * math.sqrt(part) * 1e4)


def simulate(side: str, qty: float, ref_price: float, open_price: float,
             adv_dollars: float, daily_vol: float, commission_bps: float = 0.0,
             max_participation: float = MAX_PARTICIPATION,
             symbol: str = "") -> Fill:
    """Fill one order at the next open, charged for gap, spread and impact."""
    f = Fill(symbol=symbol, side=str(side).lower(), ref_price=float(ref_price))
    buy = f.side != "sell"

    if not np.isfinite(open_price) or open_price <= 0:
        f.filled, f.note = False, "no opening price"
        return f
    f.open_price = float(open_price)

    # The overnight gap, signed so that a move against us is positive cost.
    if ref_price and np.isfinite(ref_price) and ref_price > 0:
        move = (f.open_price - ref_price) / ref_price
        f.gap_bps = float((move if buy else -move) * 1e4)

    # Liquidity cap. An order larger than the market can absorb is partially
    # filled rather than silently granted, because pretending otherwise is how
    # a backtest "scales" to money it could never deploy.
    want = abs(float(qty))
    cap_qty = (max_participation * max(float(adv_dollars), 0.0)) / f.open_price
    if cap_qty > 0 and want > cap_qty:
        f.note = ("participation capped at %.0f%% of ADV (wanted %.0f, got %.0f)"
                  % (100 * max_participation, want, cap_qty))
        want = cap_qty
    if want <= 0:
        f.filled, f.note = False, f.note or "no liquidity"
        return f

    f.qty = want if buy else -want
    notional = want * f.open_price
    f.spread_bps = spread_bps(f.open_price, adv_dollars)
    f.impact_bps = impact_bps(notional, adv_dollars, daily_vol)

    # Both costs always work against the side being traded.
    adverse = (f.spread_bps + f.impact_bps) / 1e4
    f.fill_price = f.open_price * (1.0 + adverse if buy else 1.0 - adverse)
    f.commission = abs(want * f.fill_price) * commission_bps / 1e4
    return f


# =============================================================================
#  Per-symbol liquidity inputs, measured from the panel
# =============================================================================
def liquidity(panel, lookback: int = 21) -> dict:
    """Trailing dollar volume and return volatility per symbol.

    Trailing only. Using the same day's volume to price the cost of trading
    that day would be a small lookahead in the cost model itself -- the sort
    that makes execution look cheapest exactly when it was busiest.
    """
    close = np.asarray(panel.bars["close"], dtype=np.float64)
    vol = np.nan_to_num(np.asarray(panel.bars["volume"], dtype=np.float64))
    T = close.shape[0]
    lo = max(0, T - 1 - lookback)

    dv = close[lo:T - 1] * vol[lo:T - 1]
    prev = close[lo:T - 2] if T - 2 > lo else close[lo:T - 1]
    cur = close[lo + 1:T - 1] if T - 2 > lo else close[lo:T - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = (cur - prev) / np.where(np.abs(prev) > EPS, prev, np.nan)

    out = {}
    for j, sym in enumerate(panel.tickers):
        col_dv = dv[:, j]
        col_dv = col_dv[np.isfinite(col_dv) & (col_dv > 0)]
        col_r = rets[:, j] if rets.size else np.array([])
        col_r = col_r[np.isfinite(col_r)]
        out[sym] = {
            "adv": float(np.median(col_dv)) if len(col_dv) else 0.0,
            "vol": float(np.std(col_r, ddof=1)) if len(col_r) > 2 else 0.02,
        }
    return out


def summarise(fills) -> dict:
    """Aggregate cost report over a batch of fills."""
    done = [f for f in fills if f.filled and f.qty]
    if not done:
        return {"fills": 0}
    notional = sum(f.notional for f in done)
    w = lambda key: (sum(getattr(f, key) * f.notional for f in done)   # noqa: E731
                     / max(notional, EPS))
    return {
        "fills": len(done),
        "rejected": sum(1 for f in fills if not f.filled),
        "capped": sum(1 for f in done if "participation capped" in f.note),
        "notional": notional,
        "gap_bps": w("gap_bps"),
        "spread_bps": w("spread_bps"),
        "impact_bps": w("impact_bps"),
        "total_bps": w("gap_bps") + w("spread_bps") + w("impact_bps"),
        "commission": sum(f.commission for f in done),
        "cost_dollars": notional * (w("gap_bps") + w("spread_bps")
                                    + w("impact_bps")) / 1e4
                        + sum(f.commission for f in done),
    }
