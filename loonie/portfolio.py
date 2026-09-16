"""Turning strategies into orders.

Two steps, kept separate because they fail differently:

  target_portfolio()  -- what we want to hold, from the allocator's weights
                         over strategies and each strategy's own picks.
  diff_to_orders()    -- the smallest set of trades that gets us there, after
                         a no-trade band that stops the book churning itself
                         to death over rounding noise.

The no-trade band matters more than it looks. Without it, a 25-name equal
weight book re-solves to slightly different weights every session and pays
spread on all of it. `min_trade_frac` is the difference between 3x and 30x
annual turnover, and turnover is the most reliable way to convert a real edge
into a real loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .broker.base import Order


@dataclass
class Target:
    weights: dict          # symbol -> target weight of equity
    per_strategy: dict     # strategy key -> {symbol: weight}
    alloc: dict            # strategy key -> capital weight
    cash_weight: float = 0.0
    notes: list = None


def strategy_picks(genome, feats, panel, row: int = -1) -> dict:
    """The names one genome wants to hold, as weights summing to 1."""
    score = genome.score(feats, panel.tradable)
    s = score[row]
    elig = panel.tradable[row] & np.isfinite(s)
    n = int(elig.sum())
    if n == 0:
        return {}
    k = min(int(genome.n_positions), n)
    masked = np.where(elig, s, -np.inf)
    top = np.argpartition(-masked, k - 1)[:k]
    top = top[np.isfinite(masked[top])]
    if len(top) == 0:
        return {}

    if genome.weighting == "score":
        sv = masked[top].astype(np.float64)
        sv = sv - sv.min() + 1e-12
        w = sv / sv.sum()
    else:
        w = np.full(len(top), 1.0 / len(top))
    return {panel.tickers[int(j)]: float(wi) for j, wi in zip(top, w)}


def target_portfolio(strategies: dict, feats, panel, allocator=None,
                     capital_pct: float = 0.95, row: int = -1,
                     max_position_pct: float | None = None) -> Target:
    """Blend every live strategy's picks by the allocator's capital weights."""
    per = {}
    for key, g in strategies.items():
        picks = strategy_picks(g, feats, panel, row)
        if picks:
            per[key] = picks
    if not per:
        return Target({}, {}, {}, cash_weight=1.0, notes=["no strategy produced picks"])

    if allocator is not None:
        for key in per:
            allocator.register(key)
        alloc = allocator.weights(list(per.keys()))
    else:
        alloc = {k: 1.0 / len(per) for k in per}

    invested = sum(alloc.values())
    combined: dict = {}
    for key, picks in per.items():
        a = alloc.get(key, 0.0)
        if a <= 0:
            continue
        for sym, w in picks.items():
            combined[sym] = combined.get(sym, 0.0) + a * w

    scale = float(capital_pct) * min(1.0, invested)
    combined = {s: w * scale for s, w in combined.items() if w > 0}

    notes = []
    if invested < 1e-9:
        notes.append("allocator holds 100% cash: no strategy's posterior is "
                     "positive right now")

    # Cap position size HERE, not at the order gate. Score-weighting can hand a
    # single name 27% of the book; if the cap is only enforced when the order is
    # submitted, those orders are rejected one at a time and the portfolio ends
    # up under-invested and skewed toward whichever names happened to be small.
    # Water-fill instead: clip the offenders, redistribute the excess across the
    # names with headroom, repeat until it settles. Risk stays as the backstop
    # it should be, rather than doubling as the position sizer.
    if max_position_pct:
        cap = float(max_position_pct)
        budget = sum(combined.values())
        for _ in range(24):
            over = {s: w for s, w in combined.items() if w > cap + 1e-12}
            if not over:
                break
            excess = sum(w - cap for w in over.values())
            room = {s: cap - w for s, w in combined.items()
                    if s not in over and cap - w > 1e-12}
            for s in over:
                combined[s] = cap
            total_room = sum(room.values())
            if total_room <= 1e-12:
                notes.append(
                    "position cap %.1f%% binds on every name; %.1f%% of capital "
                    "left in cash" % (100 * cap, 100 * excess))
                break
            for s, r in room.items():
                combined[s] += excess * (r / total_room)
        held = sum(combined.values())
        if budget - held > 1e-9:
            notes.append("%.2f%% held in cash after position caps"
                         % (100 * (budget - held)))

    return Target(weights=combined, per_strategy=per, alloc=alloc,
                  cash_weight=1.0 - sum(combined.values()), notes=notes)


def diff_to_orders(target: dict, positions: dict, equity: float, marks: dict,
                   min_trade_frac: float = 0.005,
                   min_notional: float = 25.0) -> list:
    """Smallest trade list from current book to target, with a no-trade band."""
    cur = {}
    for sym, p in positions.items():
        px = marks.get(sym, p.market_price) or p.market_price
        cur[sym] = (p.qty * px) / max(equity, 1e-9)

    orders = []
    for sym in sorted(set(cur) | set(target)):
        tw = target.get(sym, 0.0)
        cw = cur.get(sym, 0.0)
        d = tw - cw
        if abs(d) < min_trade_frac:
            continue
        px = marks.get(sym)
        if px is None or px <= 0:
            continue
        notional = abs(d) * equity
        if notional < min_notional:
            continue

        if d > 0:
            orders.append(Order(symbol=sym, side="buy", qty=0.0,
                                notional=round(notional, 2),
                                note="target %.3f from %.3f" % (tw, cw)))
        else:
            pos = positions.get(sym)
            qty = min(abs(d) * equity / px, pos.qty if pos else 0.0)
            if qty <= 0:
                continue
            # Exiting completely? Sell the whole line, don't leave a stub.
            if tw <= 1e-9 and pos is not None:
                qty = pos.qty
            orders.append(Order(symbol=sym, side="sell", qty=round(qty, 6),
                                note="target %.3f from %.3f" % (tw, cw)))
    # Sells first: they fund the buys.
    orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
    return orders
