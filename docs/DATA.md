# Data: what you can get for free, and what you cannot

The short version: **free data cannot support an honest long-horizon equity
backtest, and no amount of code fixes that.** This page documents exactly how
bad it is, measured rather than asserted, so you can decide what to buy.

## The measurement

Run against `fja05680/sp500` point-in-time membership, 1996-01-02 → 2026-08-18:

```
distinct tickers EVER in the S&P 500 : 1,209
still in the index today             :   503
left and never returned              :   706   (58%)
```

Now ask a free provider for those 706:

```
SIVB   (Silicon Valley Bank, failed 2023)   -> EMPTY
FRC    (First Republic, failed 2023)        -> EMPTY
LEH    (Lehman Brothers, failed 2008)       -> EMPTY
WCOM   (WorldCom, fraud, 2002)              -> EMPTY
ENRNQ  (Enron, fraud, 2001)                 -> EMPTY
XLNX   (Xilinx, acquired 2022)              -> EMPTY
ATVI   (Activision, acquired 2023)          -> EMPTY
TWTR   (Twitter, taken private 2022)        -> EMPTY
CERN   (Cerner, acquired 2022)              -> EMPTY
```

Free providers carry securities that still trade. Every name above is a
position your strategy would have held, and several are positions that went to
zero while you held them.

**This is not a data-quality nuisance. It is a directional bias, and it points
the way that flatters you.** A model that would have bought Silicon Valley
Bank in February 2023 does not get charged for it — the ticker simply is not in
the universe, so the trade never happens. Delete every bankruptcy from history
and a mediocre model reports a 26% CAGR.

## What this repo does about it

It cannot manufacture the missing prices, so it does the next best things:

1. **Point-in-time membership** (`loonie/universe.py`). On 2007-06-01 the
   backtester may only buy what was in the index on 2007-06-01. Membership is
   resolved per-date from a 2,720-row change log, not from a current snapshot.

2. **Coverage is measured and printed on every run** (`loonie/data.py`):

   ```
   Panel  2680 sessions x 615 tickers  [2016-01-04 -> 2026-08-31]
     provider           : yfinance
     index members ever : 745
     with usable data   : 615
     COVERAGE           : 82.6%
     departed & missing : 129  (losers the backtest cannot see)
   ```

3. **A hard floor.** `data.min_survivorship_coverage` (default 0.80) aborts the
   run rather than quietly producing an optimistic number. Lowering it is a
   deliberate, visible act.

## Providers

| provider | cost | history | delisted names | verdict |
|---|---|---|---|---|
| `yfinance` | free | 1970+ | **no** | fine for plumbing; ~83% coverage on 2016+; biased before that |
| `alpaca` | free w/ account | 2016+ | partial | best free option, and it is also the broker |
| Massive / Polygon / Norgate / CRSP | paid | deep | yes | what you need for a 1996 start |

The window in `config.yaml` defaults to 2016+ for a reason: that is where the
free path is least dishonest. A 1996 start on free data is not a longer
backtest, it is a more thoroughly biased one.

Stooq would otherwise be a good free source with delisted coverage, but it now
gates downloads behind a browser proof-of-work challenge. This repo does not
work around bot detection.

## Adding a paid provider

Subclass `Provider` in `loonie/data.py` and register it:

```python
class Massive(Provider):
    name = "massive"
    has_delisted = True
    earliest = "1996-01-01"

    def fetch(self, ticker, start, end):
        # return a DataFrame indexed by date with
        # open/high/low/close/volume, split+dividend adjusted,
        # running to the security's LAST TRADING DAY for delisted names
        ...

PROVIDERS["massive"] = Massive
```

The only requirement that matters: **delisted securities must return their
real price history up to the final trading day**, not an empty frame. That
last print — often a few cents — is the loss your backtest needs to take.

Then set `data.provider: massive` and re-run `scripts/fetch_data.py`. Coverage
should jump toward 100% and every backtested number will get worse. That is
what correct looks like.

## Corporate actions

`auto_adjust` / `adjustment="all"` gives split- and dividend-adjusted closes,
which is what the momentum and reversal features assume. Un-adjusted prices
produce phantom -50% returns on every 2-for-1 split, and a reversal strategy
will happily learn to buy them.

## A note on the sealed window

`config.holdout` carves off the most recent two years and `loonie/seal.py`
hashes it. If you refresh the cache after sealing, the hash changes and the
seal breaks on purpose — a result computed against different bytes is not the
test you sealed. Re-seal deliberately (`Seal.create(..., force=True)`) and
understand that you are starting the clock again.
