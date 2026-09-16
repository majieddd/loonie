# Why this is built the way it is

The video this project reproduces — *"My AI Stock Predictor is Now Improving
Itself"*, LosingLoonies, Aug 2026 — ends with an honest negative result, and
that ending is the most valuable thing in it. Worth restating precisely:

> Sealed window, never seen during development. Strategy returned **30.4%**.
> The market returned **33.5%**. "Statistically, no. It's indistinguishable
> from random. The 30% came from riding the index, not the model."

Everything in this repository is organised around the question of why that
happened, because the loop he ran was reasonable and the failure was not
obvious.

---

## What his loop was

```
propose a change (LLM) → run backtest → keep if better → repeat
```

Fourteen proposals. One kept. That is *good* discipline by the standards of
retail algo trading, and it still produced a result indistinguishable from
noise. Three reasons, in descending order of importance.

### 1. The objective was total return

If you rank the S&P 500 and hold 25 names, you will approximately get the S&P
500. Optimising CAGR cannot distinguish "picked well" from "held equities
during a bull market", because both produce a large number.

The fix is not subtle: **optimise the residual.** Regress strategy returns on
a benchmark and maximise what is left. In this codebase, fitness is the
information ratio of returns in excess of an *equal-weight portfolio of the
same eligible universe* — the return you would have got by owning everything
you were allowed to buy. A genome that tracks the index scores approximately
zero, by construction, no matter how good its CAGR is.

His self-improvement loop spent fourteen iterations climbing a hill that did
not point at the thing he wanted. The proposals were fine. The compass was
wrong.

### 2. "Better" was never given a statistical test

"Keep it if the backtest improved" is a selection procedure, and selection
procedures have error rates. Under the null that every strategy is worthless,
the expected maximum Sharpe over *N* trials grows roughly like `sqrt(2 log N)`.
Concretely, measured on this project's own data:

| trials | excess Sharpe you must beat to match noise |
|-------:|------------------------------------------:|
| 1      | 0.00 |
| 100    | ~0.95 |
| 1,000  | ~1.24 |
| 60,000 | **~1.50** |

At 14 trials the correction is small — but it is not zero, and nothing in his
loop computed it. So "it improved" was never distinguished from "it got
luckier". `loonie/metrics.py` computes the Deflated Sharpe Ratio, the
probability of backtest overfitting via CSCV, and a Newey-West alpha t-stat,
and `loonie/evolve.py` will not promote a candidate that fails any of them.

### 3. One sealed evaluation gives an answer, not a gradient

Sealing a holdout and opening it once at the end is correct. But it means you
learn whether you succeeded *after* all the decisions are made, and you learn
it once. During the search itself there was no signal about generalisation at
all.

This project runs purged, embargoed, block cross-validation on every
candidate, so out-of-sample consistency is a first-class input to fitness
rather than a final exam. The sealed window still exists, still opens once,
and is now enforced by a SHA-256 manifest and an append-only ledger rather
than by remembering a promise (`loonie/seal.py`).

---

## Does the LLM matter?

This was the original question: can pure code, rules and self-organising
algorithms do recursive self-improvement without a language model?

**Yes — and the LLM was never the load-bearing part.**

Strip the loop down and it is: *sample a neighbour of the current solution,
evaluate it, accept or reject.* That is stochastic local search. The LLM is a
mutation operator. It is a good one — it can propose semantically meaningful
edits like "rank by annualised return instead of raw return" — but it costs
seconds and dollars per proposal, which caps you at ~14.

The typed genetic program in `loonie/genome.py` proposes in ~100µs. Same loop
shape, roughly four orders of magnitude more proposals:

| | LLM loop | this |
|---|---|---|
| proposal cost | seconds, metered | ~0.1 ms |
| proposals per run | ~14 | 10⁴–10⁶ |
| semantic edits | yes | only within the grammar |
| multiple-testing correction | none | mandatory |

And here is the part that inverts the intuition: **more proposals is not
straightforwardly better.** Every additional candidate raises the bar the
winner must clear. Watch it happen in a real run of this engine — the leading
strategy is *unchanged* from generation 3 onward:

```
gen 3 | IR +1.03 | t +2.04 | DSR 0.45 | trials  529
gen 5 | IR +1.03 | t +2.04 | DSR 0.40 | trials  780
gen 7 | IR +1.03 | t +2.04 | DSR 0.37 | trials 1017
gen 9 | IR +1.03 | t +2.04 | DSR 0.35 | trials 1250
```

The strategy did not get worse. The *evidence* for it got weaker, because the
search that found it got bigger. Searching harder makes your best find less
credible, automatically and unavoidably. No amount of self-improvement escapes
this; it is the price of looking.

So the honest framing of "recursive self-improvement" for trading is not *make
the strategy better forever*. It is **spend your finite budget of statistical
credibility well.** That reframing is what the rest of the design follows from.

### What actually recurses here

Three loops, on three timescales, none of them needing a model:

1. **Genetic program** (per generation) — mutation and crossover over the
   expression grammar.
2. **Operator bandit** (per generation) — every mutation operator carries a
   Beta posterior over "did my child beat its parent?", sampled by Thompson
   sampling. The engine reallocates its own effort toward operators that still
   work and abandons ones that have stopped. In practice `hoist` — the
   *simplification* operator — dominates, which is the search discovering
   Occam's razor from data rather than being told it.
3. **Thompson allocator** (per trading day) — capital flows between live
   strategies based on realised paper P&L, not backtest results
   (`loonie/allocator.py`).

Loop 2 is the genuinely self-referential one: the search modifies its own
search policy based on measured outcomes. That is recursive self-improvement
in the only sense that survives contact with a noisy objective.

---

## The search will trade a rounding error if you let it

The most instructive failure in building this was not statistical. The fittest
genome in a run, the one that cleared all nine promotion gates, was:

```
ite(mul(mom_252, demean(-0.3359)), ma_ratio_50, ite(dollar_vol_21, rev_5, rev_5))
```

`demean` subtracts the cross-sectional mean, so `demean` of a *constant* is
identically zero. The condition is always zero, the true branch is unreachable,
and both arms of the inner `ite` are the same node. Mathematically the whole
expression is `rev_5` wearing nine extra nodes.

Except it was not. In float32, summing 500 identical values and subtracting the
mean gives **-2.98e-08**, not zero — and crucially, with a *consistent sign*.
So `mul(mom_252, -2.98e-08)` is positive exactly where `mom_252` is negative,
and the "dead" branch was live, switching on 12-month momentum. The search had
found a rounding error and was using it as a regime signal.

Measured cost, once `cs_demean` was changed to accumulate in float64:

| | IR | alpha | alpha t-stat |
|---|---:|---:|---:|
| float32 (noise channel open) | 1.354 | 0.332 | **2.955** |
| float64 (noise channel closed) | 1.028 | 0.220 | **2.04** |

**24% of the information ratio and 34% of the alpha were the artifact.** The
t-stat fell from comfortably clearing the gate to barely scraping it.

Two things worth taking from this. First, the optimiser was not malfunctioning
— it was doing its job perfectly against an objective that quietly contained a
channel nobody meant to offer it. Every degenerate solution looks like this
from the inside. Second, it survived *nine* statistical gates, including the
shuffle test and cost stress, because it was a real and stable pattern in the
data as computed. Statistical rigour does not protect you from a numerically
leaky objective; only reading the winning expression does.

The general lesson for any self-improving system: **the more capable your
search, the more carefully you have to audit what it is allowed to see.** A
weaker optimiser would never have found -3e-08.

---

## The correlated-trials problem

A subtlety that took a revision to get right. Charging the deflation for every
genome evaluated is *too harsh*: a GP's children are near-copies of their
parents, so 1,200 genomes that are 95% the same expression are not 1,200
independent shots at the data. Price them as if they were and nothing can ever
clear the gate — dishonesty in the opposite direction.

`metrics.effective_trials` builds the correlation matrix of candidate return
streams and takes its participation ratio — the effective-number-of-tests
estimator from genome-wide association studies, where the same problem appears
as correlated SNPs. M perfectly correlated candidates count as 1; M
independent ones count as M. Both numbers are reported, because the gap
between them tells you whether your search is exploring or just milling.

---

## The data problem is not a side issue

His second self-improvement round found survivorship bias, and he was right
that it matters enormously. Measured directly in this repo:

- **1,209** tickers have been in the S&P 500 since 1996
- **706** of them left and never returned
- yfinance returns **empty** for essentially all of them — SIVB, FRC, LEH,
  WCOM, XLNX, ATVI, TWTR, CERN

A backtest on "the S&P 500" built from a current constituent list has silently
deleted every bankruptcy, takeunder and slow bleed — which is to say, exactly
the trades that would have lost money. That is how a mediocre model reports
26% CAGR.

`loonie/universe.py` resolves membership point-in-time: on 2007-06-01 you may
only buy what was in the index on 2007-06-01. `loonie/data.py` measures how
much of that universe the configured provider can actually deliver and refuses
to run below `min_survivorship_coverage`. The free path reaches ~83% over
2016–2026, which is usable-with-a-warning and printed on every run. It is not
good enough for a 1996 start, and the system says so rather than quietly
producing a beautiful number.

**This is the one problem you cannot code your way out of.** Everything else
here is a matter of getting the statistics right. Survivorship-free data has
to be bought — the video's creator partnered with a vendor for exactly this
reason, and that was the correct call.

---

## What "success" looks like

Most searches should end with nothing promoted. That is the system working.

A market with millions of participants and decades of quant effort should not
hand a persistent edge to a laptop running a genetic program over 41 technical
features. If this engine promoted a strategy every time you ran it, the gate
would be broken.

What it buys you is the ability to tell the difference — between a strategy
that made money and one that had money happen to it. The video got that answer
once, at the end, by accident of good instincts. This gets it continuously, by
construction, and writes down the trial count that makes the answer mean
something.

---

## Further reading

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2016), *The Probability of Backtest
  Overfitting*
- Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* —
  the multiple-testing argument applied to the published factor literature
- López de Prado (2018), *Advances in Financial Machine Learning* — purged CV
  and embargoing
