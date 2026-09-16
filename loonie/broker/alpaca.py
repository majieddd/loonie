"""Alpaca adapter. Paper by default; live requires all three locks open.

Paper keys are free and give you a real order lifecycle -- routing, partial
fills, rejects, halts, corporate actions -- against a simulated balance. That
is the environment the allocator should learn in, and for as long as possible.
"""
from __future__ import annotations

import os

from .base import Account, Broker, Order, Position, assert_live_allowed

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, cfg, cli_live_flag: bool = False):
        assert_live_allowed(cfg, cli_live_flag)

        from alpaca.trading.client import TradingClient

        self.cfg = cfg
        mode = str(cfg.trade.mode).lower()
        env_mode = os.getenv("ALPACA_MODE", "paper").lower()
        self.is_live = (mode == "live" and env_mode == "live" and cli_live_flag)

        key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
        if not (key and sec):
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_API_SECRET not set. Create a free paper "
                "account at https://app.alpaca.markets/paper/dashboard/overview, "
                "then copy .env.example to .env and paste the keys."
            )
        self._t = TradingClient(key, sec, paper=not self.is_live)
        self._data = None
        try:
            from alpaca.data.historical import StockHistoricalDataClient

            self._data = StockHistoricalDataClient(key, sec)
        except Exception:
            pass

    # ------------------------------------------------------------- account
    def account(self) -> Account:
        a = self._t.get_account()
        return Account(
            cash=float(a.cash), equity=float(a.equity),
            buying_power=float(a.buying_power), positions=self.positions(),
            blocked=bool(getattr(a, "trading_blocked", False)
                         or getattr(a, "account_blocked", False)),
        )

    def positions(self) -> dict:
        out = {}
        for p in self._t.get_all_positions():
            out[p.symbol] = Position(
                symbol=p.symbol, qty=float(p.qty),
                avg_price=float(p.avg_entry_price),
                market_price=float(p.current_price or p.avg_entry_price),
            )
        return out

    def is_market_open(self) -> bool:
        try:
            return bool(self._t.get_clock().is_open)
        except Exception:
            return False

    def last_price(self, symbol: str):
        if self._data is None:
            return None
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            r = self._data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol))
            return float(r[symbol].price)
        except Exception:
            return None

    # -------------------------------------------------------------- orders
    def submit(self, order: Order) -> Order:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        tif = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC,
               "cls": TimeInForce.CLS}.get(order.tif, TimeInForce.DAY)
        kwargs = {"symbol": order.symbol, "side": side, "time_in_force": tif}
        if order.notional is not None and not order.qty:
            kwargs["notional"] = round(float(order.notional), 2)
        else:
            kwargs["qty"] = round(float(order.qty), 6)

        try:
            r = self._t.submit_order(MarketOrderRequest(**kwargs))
            order.status = str(r.status).split(".")[-1].lower()
            order.client_id = str(r.id)
            order.filled_qty = float(r.filled_qty or 0)
            order.filled_price = float(r.filled_avg_price or 0)
        except Exception as e:
            order.status, order.note = "rejected", "%s: %s" % (type(e).__name__, e)
        return order

    def cancel_all(self) -> int:
        try:
            return len(self._t.cancel_orders() or [])
        except Exception:
            return 0

    def close_all(self) -> int:
        try:
            return len(self._t.close_all_positions(cancel_orders=True) or [])
        except Exception:
            return 0
