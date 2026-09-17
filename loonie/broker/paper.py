"""Local paper broker: a persistent simulated account with realistic frictions.

Useful for two things the real paper API is bad at: running without network or
keys, and replaying a strategy over history deterministically.

ORDERS DO NOT FILL WHEN YOU SUBMIT THEM. They rest until the next session's
open, which is when a decision made after a close could first be acted on.
The previous version filled instantly at the very close that generated the
signal -- a one-bar lookahead living in the execution layer, where nobody
thinks to look for it, and worth roughly the overnight gap on every trade.

So `submit()` queues, and `settle()` fills against the next open through
loonie/execution.py, charging the gap, a liquidity-dependent spread and
square-root market impact separately. A strategy whose edge is smaller than
its execution cost now looks like one.
"""
from __future__ import annotations

import json

from .. import execution
from ..config import resolve
from .base import Account, Broker, Order, Position, _now

STATE = "state/paper_account.json"


class PaperBroker(Broker):
    name = "paper"
    is_live = False

    def __init__(self, cfg, starting_cash: float | None = None):
        self.cfg = cfg
        self.slippage_bps = float(cfg.backtest.slippage_bps)
        self.commission_bps = float(cfg.backtest.commission_bps)
        self.path = resolve(STATE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._marks: dict = {}
        self.blotter: list = []
        self.pending: list = []
        if self.path.exists():
            self._load()
        else:
            self.cash = float(starting_cash or cfg.backtest.initial_capital)
            self._pos: dict = {}
            self._save()

    # ---------------------------------------------------------------- state
    def _load(self):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.cash = float(d["cash"])
        self._pos = {
            s: Position(s, float(p["qty"]), float(p["avg_price"]),
                        float(p.get("market_price", p["avg_price"])))
            for s, p in d.get("positions", {}).items()
        }
        self.blotter = d.get("blotter", [])[-2000:]
        self.pending = d.get("pending", [])

    def _save(self):
        self.path.write_text(json.dumps({
            "cash": self.cash,
            "positions": {s: {"qty": p.qty, "avg_price": p.avg_price,
                              "market_price": p.market_price}
                          for s, p in self._pos.items()},
            "blotter": self.blotter[-2000:],
            # Queued orders persist: a decision made after Tuesday's close is
            # still owed a Wednesday fill even if the process restarts in
            # between, and dropping it would quietly turn a missed trade into
            # a trade that never existed.
            "pending": self.pending,
            "updated": _now(),
        }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- marks
    def set_marks(self, marks: dict, as_of=None):
        """Latest closes, and the bar date they belong to.

        `as_of` is what lets settlement know whether a resting order has
        actually had its chance. Without it, an order queued from Tuesday's
        close would settle against Tuesday's OPEN -- a fill several hours
        before the signal that produced it existed. The obvious
        implementation of "fill at the open" fills backwards in time.
        """
        self._as_of = str(as_of) if as_of is not None else None
        self._marks.update({k: float(v) for k, v in marks.items()
                            if v is not None and v == v})
        for s, p in self._pos.items():
            if s in self._marks:
                p.market_price = self._marks[s]

    def last_price(self, symbol):
        return self._marks.get(symbol)

    # ------------------------------------------------------------- account
    def account(self) -> Account:
        eq = self.cash + sum(p.market_value for p in self._pos.values())
        return Account(cash=self.cash, equity=eq, buying_power=max(self.cash, 0.0),
                       positions=dict(self._pos))

    def positions(self):
        return dict(self._pos)

    def is_market_open(self) -> bool:
        return True

    # -------------------------------------------------------------- orders
    def submit(self, order: Order) -> Order:
        """Queue an order for the next open. Nothing fills here."""
        px = self._marks.get(order.symbol)
        if px is None or px <= 0:
            order.status, order.note = "rejected", "no mark for %s" % order.symbol
            return order

        qty = order.qty
        if order.notional is not None and not qty:
            qty = order.notional / px
        if qty <= 0:
            order.status, order.note = "rejected", "zero quantity"
            return order

        if order.side == "sell":
            held = self._pos[order.symbol].qty if order.symbol in self._pos else 0.0
            # Account for sells already queued, or two rebalances in a row
            # would each be allowed to sell the whole position.
            queued = sum(o["qty"] for o in self.pending
                         if o["symbol"] == order.symbol and o["side"] == "sell")
            qty = min(qty, max(0.0, held - queued))
            if qty <= 1e-9:
                order.status, order.note = "rejected", "no position to sell"
                return order

        self.pending.append({
            "symbol": order.symbol, "side": order.side, "qty": float(qty),
            "ref_price": float(px), "note": order.note or "",
            "submitted_at": _now(),
            # The bar whose CLOSE produced this decision. Settlement refuses
            # any open dated on or before it.
            "ref_bar": getattr(self, "_as_of", None),
        })
        order.status, order.note = "pending", "queued for next open"
        self._save()
        return order

    def settle(self, opens: dict, liq: dict | None = None,
               bar_date=None) -> list:
        """Fill every queued order against the next session's open.

        Sells settle first. They release the cash the buys need, and settling
        in submission order left buys short of cash that the same batch was
        about to produce.
        """
        if not self.pending:
            return []

        liq = liq or {}
        bar = str(bar_date) if bar_date is not None else None
        fills, unfilled = [], []
        queue = sorted(self.pending,
                       key=lambda o: 0 if o["side"] == "sell" else 1)

        for o in queue:
            # An order decided on bar T fills at the open of T+1 or later. The
            # same bar's open happened BEFORE the close that produced the
            # decision, so filling there would be trading on information that
            # did not exist yet -- the exact error this queue exists to stop.
            ref_bar = o.get("ref_bar")
            if bar is not None and ref_bar is not None and bar <= ref_bar:
                unfilled.append(o)
                continue
            sym = o["symbol"]
            info = liq.get(sym) or {}
            f = execution.simulate(
                side=o["side"], qty=o["qty"], ref_price=o.get("ref_price", 0.0),
                open_price=opens.get(sym, float("nan")),
                adv_dollars=info.get("adv", 0.0) or 0.0,
                daily_vol=info.get("vol", 0.02) or 0.02,
                commission_bps=self.commission_bps, symbol=sym)

            if not f.filled:
                # An order with no opening price has not failed, it has not
                # had its chance yet. Carrying it is what a resting order does.
                unfilled.append(o)
                fills.append(f)
                continue

            self._apply(f)
            fills.append(f)
            self.blotter.append({**f.as_dict(), "at": _now(),
                                 "submitted_at": o.get("submitted_at")})

        self.pending = unfilled
        self._save()
        return fills

    def _apply(self, f) -> None:
        """Move cash and positions for one completed fill."""
        qty = abs(f.qty)
        pos = self._pos.get(f.symbol)
        if f.side != "sell":
            cost = qty * f.fill_price + f.commission
            if cost > self.cash + 1e-6:
                qty = max(0.0, (self.cash - f.commission) / max(f.fill_price, 1e-9))
                cost = qty * f.fill_price + f.commission
                f.qty = qty
                f.note = (f.note + "; " if f.note else "") + "downsized to cash"
            if qty <= 0:
                f.filled, f.note = False, "insufficient cash"
                return
            self.cash -= cost
            if pos:
                total = pos.qty + qty
                pos.avg_price = (pos.avg_price * pos.qty
                                 + f.fill_price * qty) / max(total, 1e-12)
                pos.qty = total
            else:
                self._pos[f.symbol] = Position(f.symbol, qty, f.fill_price,
                                               f.open_price)
        else:
            held = pos.qty if pos else 0.0
            qty = min(qty, held)
            if qty <= 0:
                f.filled, f.note = False, "no position to sell"
                return
            self.cash += qty * f.fill_price - f.commission
            f.qty = -qty
            pos.qty -= qty
            if pos.qty <= 1e-9:
                self._pos.pop(f.symbol, None)

    def cancel_all(self) -> int:
        n = len(self.pending)
        self.pending = []
        self._save()
        return n

    def close_all(self) -> int:
        n = 0
        for sym in list(self._pos):
            o = self.submit(Order(symbol=sym, side="sell", qty=self._pos[sym].qty,
                                  note="close_all"))
            n += int(o.status == "filled")
        return n
