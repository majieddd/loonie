"""Broker adapters. Paper is the default and the locks ship closed."""
from .base import (Account, Broker, LiveTradingLocked, Order, Position,
                   assert_live_allowed)
from .paper import PaperBroker

__all__ = ["Account", "Broker", "Order", "Position", "PaperBroker",
           "LiveTradingLocked", "assert_live_allowed", "get_broker"]


def get_broker(cfg, cli_live_flag: bool = False):
    """Build the configured broker. Falls back to local paper when Alpaca
    is unavailable, so the daemon keeps running instead of dying."""
    name = str(cfg.trade.broker).lower()
    if name == "paper":
        return PaperBroker(cfg)
    if name == "alpaca":
        from .alpaca import AlpacaBroker
        try:
            return AlpacaBroker(cfg, cli_live_flag)
        except LiveTradingLocked:
            raise
        except Exception as e:
            print("[broker] alpaca unavailable (%s); using local paper broker" % e)
            return PaperBroker(cfg)
    raise ValueError("unknown broker %r" % name)
