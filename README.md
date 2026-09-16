# loonie

A self-improving cross-sectional equity strategy search that trades through a
real broker — built as a reproduction of, and a correction to, the loop in
[*"My AI Stock Predictor is Now Improving Itself"*](https://www.youtube.com/watch?v=noK0IwZAnyE)
(LosingLoonies, Aug 2026).

The video's loop was: **propose a change with an LLM → backtest → keep if
better → repeat.** Fourteen proposals, one kept. Final sealed test: strategy
**30.4%**, market **33.5%**, verdict *"statistically, indistinguishable from
random."*

That negative result is the interesting part, and this repo is organised
around why it happened. Same loop shape, three things different:

| | the video | here |
|---|---|---|
| proposal operator | LLM (~14 proposals) | typed genetic program (10⁴–10⁶) |
| objective | total return | **alpha vs equal-weight universe** |
| "is it better?" | backtest went up | purged CV + deflated Sharpe + PBO + shuffle test |
| holdout | promise not to re-run | SHA-256 seal + append-only ledger |
| survivorship | fixed late, manually | point-in-time universe, coverage measured every run |
| live learning | none | Thompson allocator over realised P&L |

**No language model runs anywhere in this system.** That was the question
behind the build, and the answer is in [`docs/WHY.md`](docs/WHY.md): the LLM
was never the load-bearing part of that loop. It was a mutation operator that
cost seconds per proposal.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # ~10 min, caches to data/cache/
python -m pytest tests/ -q            # 33 tests; the causality ones matter most
python scripts/run_evolve.py --generations 40
python scripts/status.py
python scripts/run_trade.py --dry-run # prints orders, sends nothing
```

Trading through Alpaca paper (free, no money at risk):

```bash
cp .env.example .env                  # paste PAPER keys from alpaca.markets
python scripts/run_trade.py           # paper broker
python scripts/run_trade.py --daemon  # every session
```

Run the search continuously:

```bash
python scripts/run_evolve.py --daemon
```

It checkpoints every generation to `state/evolve_state.json`, so killing and
restarting loses at most one generation.

---

## What it does

**Search.** Strategies are typed expression trees over 41 causal features
(momentum, volatility, liquidity, oscillators, microstructure, market-relative
risk) that score every eligible stock each day. Genetic programming mutates and
recombines them. MAP-Elites keeps a grid of *behaviourally distinct* winners —
binned by turnover, index-correlation and complexity — so the population cannot
collapse into one lineage that has overfit together.

**Fitness is alpha, not return.** The information ratio of returns in excess of
an equal-weight portfolio of the same eligible universe. A strategy that picks
25 S&P names and tracks the index scores ~0 by construction, however good its
CAGR looks. This is the single most important design decision in the repo.

**Promotion is a statistical claim, so it gets a statistical test.** To be
promoted a candidate must clear *every* gate: consistency across purged
embargoed folds, a Newey-West alpha t-stat ≥ 2, a deflated Sharpe priced for
the number of *independent* hypotheses tested, CSCV probability-of-overfitting
below threshold, turnover and correlation bounds — and a shuffle test.

The shuffle test is the one that directly answers the video's closing question.
Take the candidate's own signal, circularly shift it in time past any holding
period, and re-run. Same expression, same cross-sectional structure, aligned
with the wrong future. **A strategy that cannot beat its own decoupled replicas
has no timing information** — it is a static tilt wearing a prediction's
clothes. That is what "indistinguishable from random" means, computed per
candidate at search time instead of discovered at the end.

**Three recursive loops, no model:**

1. Genetic program — mutation/crossover over the grammar.
2. **Operator bandit** — each mutation operator carries a Beta posterior over
   *"did my child beat its parent?"*, drawn by Thompson sampling. The search
   reallocates its own effort toward what still works. In practice `hoist` (the
   simplification operator) dominates — the engine deriving Occam's razor from
   data rather than being told.
3. **Thompson allocator** — capital moves between live strategies based on
   realised paper P&L, not backtests.

Loop 2 is the genuinely self-referential one: the search modifying its own
search policy from measured outcomes.

---

## Two things that will bite you

### Searching harder makes your best find *less* credible

From a real run, leader unchanged from generation 3:

```
gen 3 | IR +1.03 | t +2.04 | DSR 0.45 | trials  529
gen 9 | IR +1.03 | t +2.04 | DSR 0.35 | trials 1250
```

The strategy didn't get worse — the evidence for it did, because the search
that found it got bigger. Every extra candidate raises the bar the winner must
clear. No amount of self-improvement escapes this. "Recursive self-improvement"
for trading is not *make it better forever*; it is **spend a finite budget of
statistical credibility well.**

### The free data is survivorship-biased and there is no code fix

Measured directly by this repo:

- **1,209** tickers have been in the S&P 500 since 1996
- **706** left and never returned
- yfinance returns **empty** for essentially all of them — SIVB, FRC, LEH,
  WCOM, XLNX, ATVI, TWTR, CERN

Every one of those is a loser you would have owned. Delete them and a mediocre
model reports 26% CAGR. `loonie/universe.py` resolves membership point-in-time;
`loonie/data.py` measures provider coverage and refuses to run below
`min_survivorship_coverage`. The free path reaches ~83% over 2016–2026 and says
so on every run.

This is the one problem you cannot engineer around — survivorship-free history
has to be bought. Everything else here is getting the statistics right.

---

## Trading for real

Paper trading works out of the box. **Live trading is behind three independent
locks that must all be opened by hand:**

1. `config.yaml` → `trade.allow_live: true`
2. environment → `ALPACA_MODE=live`
3. command line → `--i-understand-this-is-real-money`

They live in three different places, owned by three different actions, so no
single edit — and no automated process — can arm real money on its own. They
ship closed and nothing in this repo opens them for you. `run_trade.py` also
refuses to trade un-promoted candidates with real money regardless of the
locks.

Independent of the locks, kill switches in `loonie/risk.py` latch on daily
loss, drawdown, runaway order count, stale data and broker blocks. A latched
halt does not auto-resume when the number recovers — clearing it requires
`scripts/clear_halt.py --note "..."`, and the note goes into a permanent log.
Evolution proposes; risk disposes, and risk is outside the learning loop where
the search cannot tune it.

---

## Layout

```
loonie/
  universe.py    point-in-time S&P 500 membership     <- the honesty foundation
  data.py        providers, caching, coverage audit
  features.py    41 causal features (no lookahead, tested)
  genome.py      typed expression trees, mutation operators, simplifier
  backtest.py    cross-sectional engine, costs, delisting
  metrics.py     DSR, PBO/CSCV, effective trials, Newey-West
  cv.py          purged + embargoed block CV
  evolve.py      GP + MAP-Elites + operator bandit + null models
  seal.py        cryptographically sealed holdout
  allocator.py   Thompson sampling over live strategies
  portfolio.py   target book -> orders, position caps, no-trade band
  risk.py        latching kill switches
  macro.py       16 causal regime series (VIX, curve, credit, breadth)
  experience.py  parquet corpus: every candidate + its forward outcome
  methods.py     bandit over six ways of searching
  registry.py    per-worker heartbeats; liveness from timestamp age
  orchestrator.py  supervisor: five jobs, staggered, restarted on death
  publish.py     JSON snapshot the dashboard reads
  broker/        paper (local) + Alpaca (paper/live)
scripts/
  fetch_data.py  run_evolve.py  run_trade.py
  evaluate_holdout.py  status.py  clear_halt.py
  walkforward.py serve.py  run_system.py  meta_review.py
  publish_dashboard.py  setup_pages.sh  install_scheduler.ps1
docs/
  index.html     the dashboard (static; GitHub Pages)
  data/*.json    snapshot + series + strategies, rewritten each publish
.github/workflows/
  dashboard.yml  weekday paper rebalance + Pages deploy
docs/WHY.md      the full argument, with references
docs/RESEARCH.md seven papers read against this system
docs/memory.html a synthesis: model-agnostic memory substrate
```

---

## Honest expectations

**Most runs should promote nothing.** That is the system working. A market with
millions of participants and decades of quant effort should not hand a
persistent edge to a laptop running a genetic program over 41 technical
features. If this promoted a strategy every time, the gate would be broken.

What it buys you is the ability to tell the difference between a strategy that
made money and one that had money happen to it — continuously, with the trial
count written down, instead of once at the end.

Not investment advice. Backtested results are hypothetical. Paper trade for a
long time.

---

## What the literature says

Seven works read against this system, in [`docs/RESEARCH.md`](docs/RESEARCH.md)
([rendered](https://majieddd.github.io/loonie/research.html)). The useful output
is not a reading list — it is two places the literature says this design is
**wrong**, and one place it says the closest published work is.

The governing constraint, from a 2026 survey of 1,250 self-improvement papers:

> *"Every self-improvement loop is a claim that some signal can substitute for
> human judgment, and the loop's ceiling is exactly the quality of that
> substitute."*

Three collapse modes are named there. Checked one at a time: **model collapse**
does not apply (we never train on generated data — the exogenous fraction is
1.0 and cannot decay); **self-confirming loops** are avoided by construction
(the genetic program and the backtest share no parameters, which is the real
argument for the no-LLM design); **diversity collapse** we observed directly and
fixed with MAP-Elites.

Where it says we are wrong, now testable as methods:

- **Objective deception** (Stanley & Lehman) — we optimise alpha directly, and
  novelty search beats objective-driven search on exactly the deceptive problems
  where a good objective seems most valuable. Added as `novelty_search`.
- **Parsimony by node count is a heuristic** — the GP literature prefers
  description length. Added as `mdl_parsimony`, a BIC-style penalty that scales
  with sample size instead of being a constant chosen by hand.

And against the nearest published relative, AlgoEvolve (2026), which reports a
Sharpe of **5.60** from LLM-evolved trading programs — the paper's own caveat is
that its population mean is **1.21**. That 4.6× gap is the selection effect,
reported honestly and then not corrected for: no deflated Sharpe, no PBO, and
no disclosure of how many programs were evaluated, which makes correction
impossible even for a careful reader.

## A synthesis: model-agnostic memory

[`docs/memory.html`](https://majieddd.github.io/loonie/memory.html) converges
the same literature onto a different problem — **a memory substrate any local or
cloud model can plug into.**

The thesis: a memory system *is* a recursive self-improvement loop, and it
inherits every failure mode above. Three of them are invisible under recall@k —
diversity collapse in particular *improves* your retrieval metrics as it gets
worse. Five invariants follow, including that utility must be deflated by the
number of memories tried (the same order statistic as the deflated Sharpe), and
that the substrate must be text-and-structure rather than one model's vector
space — embeddings are a disposable per-model index, never the source of truth.

Written with a falsification plan attached. It is a hypothesis, not a result.

---

## Meta-learning: which way of searching actually works

Three loops already learn *inside* one method — which mutation operators help,
which feature families survive forward, which live strategies deserve capital.
None of them could ask whether genetic-programming-with-IR-fitness is the right
method at all, and that was picked by a person on the first afternoon with no
evidence.

**`loonie/methods.py`** is a bandit over six real, behaviour-changing
configurations of the search:

| method | what it changes |
|---|---|
| `qd_ir` | the incumbent: MAP-Elites + information-ratio fitness |
| `qd_consistency` | squares fold-agreement — six steady folds beat two spectacular ones |
| `elitist_ir` | drops quality-diversity entirely, so QD is measured against its own absence |
| `parsimony_hard` | 10× complexity pressure, shallower trees |
| `low_turnover` | 8× turnover cap, 9× cost stress |
| `wide_explore` | bigger population, deeper grammar — tests under-exploring vs over-fitting |

Each search cycle draws one by Thompson sampling. **Credit is forward alpha,
never fitness** — a method judged on its own score would win by inflating it.
Untried methods get first refusal, because a posterior built from zero
observations is a prior and acting on it as evidence is how a bandit convinces
itself of something it never measured.

Evidence is counted **once per distinct strategy**, not once per generation. A
leader that holds position for fifty generations is one observation; counting
fifty would let a method manufacture confidence by simply not improving.

## Training data that survives

**`loonie/experience.py`** keeps every gated candidate and its forward outcome
in a daily-partitioned parquet corpus. Before it existed, 340,000 strategies
had been evaluated and not one of those evaluations survived anywhere a later
run could read.

Price history is free and finite. What costs CPU-days is the *labelled pairing
of a strategy with how it actually generalised forward* — and that is what
accumulates here, across restarts and across `--fresh`.

```bash
python scripts/meta_review.py     # what has been learned, and whether to believe it
```

The review prints its own sample size and refuses to over-claim:

> *no method has enough labelled outcomes to distinguish it from the prior. The
> bandit is still exploring, and any apparent ranking above is noise. Do not
> act on it.*

The corpus is gitignored — it is machine-local and grows without bound. Back it
up deliberately.

---

## Running the whole system

```bash
python scripts/run_system.py --serve --tunnel
```

One supervisor, five jobs, each with a heartbeat:

| job | cadence | what |
|---|---|---|
| `search` | continuous | genetic program; restarted with backoff if it dies |
| `data` | every 6h | price and macro tails, incremental |
| `validate` | every 8h | **honest walk-forward of the current champion** |
| `trade` | weekdays | one paper rebalance |
| `publish` | every 60s | dashboard snapshot |

`validate` is the one that matters. Until it was scheduled, the only test that
ever caught anything was something a person had to remember to type — the
wrong shape for a system whose entire claim is that it improves itself. Its
results accumulate in `state/validation_history.json`, which is what turns
*"the search got a better score"* into *"the procedure did or did not keep
working"*. Those are different claims, and this project has now confused them
twice.

First runs are staggered (data +60s, trade +5m, validate +15m) so the
twenty-minute validation is not competing with the search for cores at the one
moment the search can least spare them.

**Live activity on the dashboard.** Every worker writes a heartbeat to
`state/workers/<id>.json` — one file per worker, atomically replaced, no lock.
A shared registry written by five processes needs a lock, and a lock held by a
process that gets killed mid-write leaves the dashboard reading half a JSON
object forever. Liveness is inferred from heartbeat *age*, never from a
self-reported flag: a hung or SIGKILLed process leaves `running: true` behind
forever but cannot fake a fresh timestamp.

## The search learns which inputs generalise

Two bandits now run, at different levels:

- **operator bandit** — which *edits* produce children that beat their parent
- **feature bandit** — which *input families* produce strategies that survive
  the held-back forward tail

The second is credited on forward validation rather than fitness, deliberately.
Crediting on fitness would only re-learn what fitness already rewards, and the
failure that prompted this layer was a leader with alpha t **4.07** in-sample
and **0.10** forward. Families that look good in-sample and evaporate out of
sample have to be pushed *down*, which only works if the credit signal comes
from the window fitness cannot see.

Weights are Thompson draws with a floor, so a family that stops generalising is
proposed less often but never banned — it keeps a tail of draws and can return
if the regime changes.

---

## Macro regime features

41 cross-sectional features compare stocks to each other. None of them knows
whether the day is March 2020 or a quiet Tuesday in 2017, so a strategy can
only express one idea and apply it identically in every environment.

`loonie/macro.py` adds 16 regime series — VIX, the MOVE index, the 3m/5y/10y
curve, HYG-over-LQD credit, the dollar, copper/gold, oil, utilities-over-
discretionary rotation, small-over-large breadth, market drawdown and realised
vol. All are **market-traded proxies, deliberately**: a yield or the VIX is
priced continuously and revised never, whereas CPI and payrolls are published
with a lag and then *restated*. A backtest reading the current value of a
revised series is reading a number nobody had on the day.

Two properties make them safe to hand to the search:

- **Causal** — forward-filled from the last known close, then trailing
  z-scored. Verified by `test_macro_features_are_causal`.
- **Centred** — the grammar's branch test is `> 0`, and raw VIX is always
  positive, so an un-centred series would send every branch the same way
  forever. Z-scored, `ite(m_vix, A, B)` reads as *"if volatility is above its
  own recent normal, do A, else B"* — the regime switch the `ite` node existed
  for and never previously had anything to condition on.

Adding them took effective independent hypotheses from **15,018 to 54,127** at
a similar raw trial count, which says the new axis genuinely diversified the
search rather than padding it.

## Blind testing without spending the seal

The sealed holdout answers *"is the strategy I picked any good?"* exactly once.
`scripts/walkforward.py` answers a more useful question as often as you like:

> If I had been running this system for the last eight years — picking the best
> strategy from what I knew at the time and trading it forward — would I have
> made money?

That is the question that matters, because you will never trade "the strategy
the search settled on in September 2026". You will trade whatever it thinks is
best on each future day, and that changes.

```bash
python scripts/walkforward.py --honest      # the version whose number means something
python scripts/walkforward.py               # fast, contaminated, upper bound only
```

**Use `--honest`.** The fast path re-ranks the existing archive at each segment
boundary, and the archive is the problem: every strategy in it earned its place
by scoring well across the *entire* training window, forward segments included.
Ranking on the past does not undo a pool picked knowing the future. On its
first run that flattered the procedure to **45% annualised against a 12.9%
universe** — which is not a result, it is a leak, and the script now says so in
capitals before printing the number.

`--honest` runs a fresh search per segment on data strictly before it, so the
searcher never sees the segment it is tested on. Roughly N times slower, and
the only version worth quoting.

---

## Live dashboard

The system publishes itself to a static site. Every generation and every trade
rewrites `docs/data/*.json`; GitHub Pages serves it; the page polls.

Two ways to run it. **Local** is the default and needs no GitHub account:

```bash
python scripts/serve.py                # dashboard on localhost + your LAN
python scripts/serve.py --tunnel       # + a public https URL (cloudflared)
python scripts/serve.py --preview      # browse design-demos/ (LAN only, never tunnelled)
```

Or double-click **`start-loonie.bat`**, which starts the search daemon *and*
the dashboard together. For a one-click desktop button that survives reboots:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -AtLogon
```

**Hosted** on GitHub Pages, when you want a stable URL:

```bash
bash scripts/setup_pages.sh <repo-name> --public   # one-time
python scripts/run_evolve.py --daemon --push       # search, pushes every 15 min
python scripts/run_trade.py                        # paper rebalance + publish
```

`serve.py` serves `docs/` and nothing else, on purpose: the project root holds
`.env`, your broker keys and 75 MB of cache, and none of that should be one
path traversal away from a public tunnel. `--preview` widens the root for
design review and is refused outright if `--tunnel` is also passed.

**What "real time" honestly means here.** GitHub Pages is static hosting — it
cannot run Python. So nothing streams. The page is as fresh as the last
publish, and it displays its own data age rather than implying a liveness it
does not have. For a daily rebalance and a ~20-second generation that is the
right granularity anyway.

Freshness comes from two independent sources, so the page keeps updating even
when your machine is off:

| source | what it does | cadence |
|---|---|---|
| local daemon | search progress, promotions, gates | every generation; pushes ≤ every 15 min |
| GitHub Actions | one PAPER rebalance + republish | each weekday after the close |

The Actions runner never sees the 1 MB evolution state (gitignored — it would
add a megabyte of churn every twenty seconds). It reads
`docs/data/strategies.json`, a few KB holding just the promoted genomes, which
changes only when something is actually promoted. It restores the 75 MB price
cache from `actions/cache` and fetches only the missing tail.

**Before you choose `--public`:** GitHub Pages on a private repo needs a paid
plan. On the free tier Pages serves public repos only — which makes your paper
positions, equity curve and evolved strategies world-readable. No credentials
are ever published (`.env` is gitignored and no key is written into
`docs/data`), so the exposure is your research and your book, not your
account. For paper trading that is usually fine. For live trading, a public
timestamped record of your holdings is a real information leak. The setup
script refuses to run at all if `trade.allow_live` is true.

---

## One more failure mode worth knowing about

The fittest genome in an early run — the one that cleared all nine gates — was
branching on `demean(-0.3359)`, which is mathematically zero and therefore a
dead branch. In float32 it evaluates to **-2.98e-08** with a consistent sign,
so the branch was live and switching on 12-month momentum. The search had found
a rounding error and was trading it.

Fixing `cs_demean` to accumulate in float64 cost the strategy **24% of its
information ratio and 34% of its alpha** (t-stat 2.96 → 2.04).

It survived nine statistical gates because it was a real, stable pattern in the
data *as computed*. Statistical rigour does not protect you from a numerically
leaky objective — only reading the winning expression does. Full account in
[`docs/WHY.md`](docs/WHY.md).
