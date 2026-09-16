"""Local paper broker: a persistent simulated account with realistic frictions.

Useful for two things the real paper API is bad at: running without network or
keys, and replaying a strategy over history deterministically. Fills are at the
supplied mark plus a slippage charge in the direction that hurts.
"""
from __future__ import annotations

import json

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

    def _save(self):
        self.path.write_text(json.dumps({
            "cash": self.cash,
            "positions": {s: {"qty": p.qty, "avg_price": p.avg_price,
                              "market_price": p.market_price}
                          for s, p in self._pos.items()},
            "blotter": self.blotter[-2000:],
            "updated": _now(),
        }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- marks
    def set_marks(self, marks: dict):
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

        slip = px * self.slippage_bps / 1e4
        fill = px + slip if order.side == "buy" else px - slip
        fee = abs(qty * fill) * self.commission_bps / 1e4

        pos = self._pos.get(order.symbol)
        if order.side == "buy":
            cost = qty * fill + fee
            if cost > self.cash + 1e-6:
                qty = max(0.0, (self.cash - fee) / fill)
                cost = qty * fill + fee
                order.note = "downsized to available cash"
            if qty <= 0:
                order.status, order.note = "rejected", "insufficient cash"
                return order
            self.cash -= cost
            if pos:
                total = pos.qty + qty
                pos.avg_price = (pos.avg_price * pos.qty + fill * qty) / total
                pos.qty = total
            else:
                self._pos[order.symbol] = Position(order.symbol, qty, fill, px)
        else:
            held = pos.qty if pos else 0.0
            qty = min(qty, held)
            if qty <= 0:
                order.status, order.note = "rejected", "no position to sell"
                return order
            self.cash += qty * fill - fee
            pos.qty -= qty
            if pos.qty <= 1e-9:
                self._pos.pop(order.symbol, None)

        order.status, order.filled_qty, order.filled_price = "filled", qty, fill
        self.blotter.append({**order.as_dict(), "fee": fee, "at": _now()})
        self._save()
        return order

    def cancel_all(self) -> int:
        return 0  # fills are immediate

    def close_all(self) -> int:
        n = 0
        for sym in list(self._pos):
            o = self.submit(Order(symbol=sym, side="sell", qty=self._pos[sym].qty,
                                  note="close_all"))
            n += int(o.status == "filled")
        return n
