# What the literature says, and what it says about *this*

Seven works, read against the system in this repository. The useful output is
not a reading list — it is two places the literature says we are **wrong**, and
one place it says the closest published work is.

---

## The papers

| # | work | what it establishes |
|---|---|---|
| 1 | [Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops](https://arxiv.org/abs/2607.07663) (2026, survey of 1,250 papers) | the evaluator is the ceiling; three named collapse modes |
| 2 | [AlgoEvolve: LLM-driven Meta-evolution of Algorithmic Trading Programs](https://arxiv.org/abs/2606.26173) (2026) | closest prior work: LLM-evolved trading programs with a meta-loop |
| 3 | [AutoML-Zero: Evolving ML Algorithms From Scratch](https://arxiv.org/abs/2003.03384) (Real, Liang, So, Le — ICML 2020) | evolution over primitives rediscovers real algorithms |
| 4 | [MAP-Elites / quality-diversity](https://arxiv.org/abs/1504.04909) (Mouret & Clune) | illumination beats optimisation on rugged spaces |
| 5 | [Why Greatness Cannot Be Planned](https://link.springer.com/book/10.1007/978-3-319-15524-1) (Stanley & Lehman, 2015) | **objective deception**: optimising the objective can prevent reaching it |
| 6 | [Genetic Programming, Validation Sets, and Parsimony Pressure](https://arxiv.org/abs/cs/0601044) | validation sets and parsimony both reduce GP overfitting, and both cost data |
| 7 | [The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) (Bailey & López de Prado) | selection bias under multiple testing |

---

## 1. The finding that matters most

From the 2026 survey, stated as flatly as it can be:

> *"Every self-improvement loop is a claim that some signal can substitute for
> human judgment, and the loop's ceiling is exactly the quality of that
> substitute."*

And the empirical regularity across the whole corpus:

> *"self-training works where answers are checkable (code, math) and degrades
> where they are not."*

This is the most important sentence in the literature for this project, because
trading sits in an awkward middle. The evaluator — a backtest — is **perfectly
checkable**: deterministic code over fixed data, near the top of the survey's
verification hierarchy. But the *quantity it estimates* is future return, which
is noisy and non-stationary. The loop is therefore well-grounded in mechanism
and weakly grounded in signal.

That is the exact position our own measurements keep reporting from: gates
passing cleanly in-sample while forward alpha sits at t = 0.10.

## 2. Three collapse modes, checked one at a time

The survey names three failure modes for self-improving systems. Where each
leaves us:

**Model collapse** — *"if the fraction of exogenous, externally grounded signal
vanishes asymptotically, degenerative dynamics follow."*

→ **Not applicable, structurally.** We never train on generated data. Every
evaluation runs against real market history the system cannot author. The
exogenous fraction is 1.0 and cannot decay. This is the one collapse mode a
trading system gets for free, and it is worth noticing that LLM self-training
loops do not.

**Self-confirming loops** — *"when generator and evaluator share weights,
biases correlate"*, over-rewarding high-confidence mistakes.

→ **Avoided by construction.** The generator is a typed genetic program; the
evaluator is a backtest. They share no parameters and no representation. This
is a genuine architectural advantage over LLM-based RSI, where the proposer and
the judge are the same weights — and it is the strongest argument for the
no-language-model design that started this project.

**Diversity collapse** — *"proposers converge to the narrow band of problems
that satisfy the reward."*

→ **Observed here, then fixed.** Without MAP-Elites the population collapsed
onto a single expression in three generations. The archive exists because of
that measurement, not because a paper recommended it.

## 3. Against the closest prior work

AlgoEvolve (2026) is the nearest published relative: LLM-driven evolution of
trading programs with a meta-evolutionary outer loop that evolves the prompts
guiding synthesis. It reports an **annualised Sharpe of 5.60**.

Read the paper's own caveat:

> *"AlgoEvolve's Sharpe here corresponds to the elite Prompt Genome selected
> under evolutionary pressure, while the population-level mean Sharpe remains
> ≈1.21; our claims are therefore comparative and selection-based rather than
> reflective of average deployable performance."*

That 4.6× gap between the selected elite and the population mean **is the
selection effect**, reported honestly and then not corrected for. The paper
applies no deflated Sharpe, no PBO and no trial counting — and, the detail that
makes correction impossible even for a careful reader, **does not report how
many programs were evaluated in total**.

This is less a criticism of their result than a description of the shape of the
field. The difference here is narrow but load-bearing: the trial count is an
*input to the gate*, so a 5.60 selected from an unreported population could not
be promoted by this system at all.

They also evaluate over roughly 200 trading days. Our walk-forward runs 1,575
sessions and still returns t ≈ 0.

## 4. Where the literature says WE are wrong

Two concrete gaps. Both are now methods in the registry, so the bandit settles
them with evidence instead of argument.

### (a) Objective deception — we optimise the objective directly

Stanley & Lehman's thesis is that for genuinely hard problems *"most beacons are
deceptive and will lead us astray"*, and that progress comes from collecting
stepping stones rather than climbing toward the goal. Novelty search — rewarding
behavioural difference and **ignoring the objective entirely** — outperforms
objective-driven search on precisely the deceptive problems where a good
objective would be most valuable.

Our fitness is alpha. MAP-Elites hedges this, since an archive *is* a
stepping-stone collector, but selection pressure inside each cell remains
objective-driven.

**Added as `novelty_search`**: fitness is mean distance to the k nearest
neighbours in behaviour space, with local competition; alpha enters only as a
tiebreak. If objective deception is real on this problem, the bandit finds out —
because it is credited on forward alpha either way.

### (b) Parsimony by node count is a heuristic, not a principle

The GP generalisation literature is explicit that bloat *"can be interpreted as
expanding the effective hypothesis class considered by the algorithm, which may
increase the risk of overfitting"*, and that parsimony pressure is an
*"algorithmic proxy"* for complexity control. Recent work prefers description
length as a principled criterion.

Our penalty was `0.01 × node_count` — a constant chosen by hand on the first
afternoon.

**Added as `mdl_parsimony`**: a BIC-style penalty,
`complexity × ln(n_sessions) / (2 × n_sessions)`, which scales with sample size
the way a model-selection criterion should.

## 5. Where the literature says we are right

- **AutoML-Zero** validates evolution over a primitive grammar: from basic maths
  operations it rediscovered two-layer networks, backpropagation and normalised
  gradients, and adapted its discoveries to the task — *"dropout-like techniques
  appear when little data is available."* Our terminals-and-operators grammar is
  the same shape of search space, and the macro-conditioned `ite` is the same
  shape of adaptation.
- **The validation-set finding** — a held-out set is the standard GP remedy, and
  its known drawback is that *"it removes a significant amount of data from the
  training set."* We hit that precisely: the 20% forward tail made early
  walk-forward segments unviable until the nested holdout was removed.
- **The survey's structural prescription** — *"audit structural change, not
  model judgment"*, keep the self-modification surface *"verifiable and
  diffs-able"* — is a description of what a human-readable expression tree with
  logged, reasoned demotions already is.

## 6. The honest summary

The literature does not say this approach is wrong. It says the ceiling is the
evaluator, and that in a noisy, non-stationary domain the evaluator is weak no
matter how clean the mechanism is. Everything expensive in this repository — the
forward tail, the deflation, the shuffle test, the scheduled walk-forward — is
spent raising that ceiling rather than on searching harder.

Which is consistent with what the system keeps reporting: **no edge**, measured
three times, at t between −0.16 and +0.18.
