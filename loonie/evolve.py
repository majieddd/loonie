"""The self-improvement engine.

This is the part the video handed to an LLM. The loop shape is identical --
propose a change, test it, keep it if it is better, repeat -- but three things
are different, and all three matter more than what generates the proposals:

  1. THE OBJECTIVE IS ALPHA, NOT RETURN.
     Fitness is the information ratio of returns *in excess of an equal-weight
     portfolio of the same eligible universe*, scaled by how consistently that
     excess shows up across folds. A genome that picks 25 S&P names and tracks
     the index scores ~0 here no matter how good its CAGR looks. The video
     optimised total return for fourteen iterations and arrived at a 30.4%
     strategy against a 33.5% index -- that outcome is what optimising the
     wrong objective looks like, and no amount of proposal quality fixes it.

  2. "BETTER" IS A STATISTICAL CLAIM, SO IT GETS A STATISTICAL TEST.
     Promotion requires clearing every gate in config: consistency across
     purged folds, a deflated Sharpe that prices in `self.trials` (the actual
     number of distinct genomes this engine has ever evaluated), a CSCV
     probability-of-overfitting below threshold, a Newey-West alpha t-stat,
     and a benchmark-correlation ceiling. Hill-climbing on a single backtest
     path is how you get 14 improvements that are all noise.

  3. THE SEARCH TUNES ITSELF.
     Each mutation operator carries a Beta posterior over "did my child beat
     its parent out-of-sample". Operators are drawn by Thompson sampling, so
     the engine reallocates its own effort toward whatever is still producing
     survivors and abandons operators that have stopped working. When OOS
     fitness plateaus it widens exploration on its own. That feedback loop --
     the search modifying its own search policy from measured outcomes -- is
     the recursive part, and it needs no language model to run.

MAP-Elites keeps a grid of behaviourally distinct winners (by turnover,
index-correlation and complexity) rather than one champion, because a
population that collapses onto a single lineage is a population that has
overfit together.
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import backtest as bt
from . import cv as cvmod
from . import ic as icmod
from . import metrics
from .config import resolve
from .genome import Genome, Grammar

STATE = "state/evolve_state.json"
EPS = 1e-12


# =============================================================================
#  Evaluation record
# =============================================================================
@dataclass
class Evaluation:
    genome: Genome
    fitness: float
    cv: dict
    descriptor: tuple
    returns: np.ndarray = None
    gates: list = field(default_factory=list)
    promoted: bool = False

    def as_dict(self):
        return {
            "fingerprint": self.genome.fingerprint,
            "canonical": self.genome.canonical(),
            # The full structure, not just the printed form -- run_trade.py has
            # to rebuild an executable genome from this file.
            "genome": self.genome.to_dict(),
            "operator": self.genome.operator,
            "born": self.genome.born,
            "fitness": float(self.fitness),
            "cv": self.cv,
            "descriptor": list(self.descriptor),
            "gates": self.gates,
            "promoted": self.promoted,
        }


# =============================================================================
#  Quality-diversity archive
# =============================================================================
class MapElites:
    """Grid of behaviourally distinct elites. Keeps the search from collapsing."""

    TURNOVER_BINS = (0.5, 1.5, 3.0, 6.0, 12.0, 25.0)
    CORR_BINS = (0.70, 0.85, 0.92, 0.96, 0.99)
    COMPLEX_BINS = (5, 10, 20, 40)

    def __init__(self):
        self.cells: dict = {}

    @staticmethod
    def _bin(value, edges):
        return int(np.searchsorted(np.asarray(edges), value))

    def descriptor(self, cvres, genome) -> tuple:
        return (
            self._bin(cvres.ann_turnover, self.TURNOVER_BINS),
            self._bin(abs(cvres.corr_bench), self.CORR_BINS),
            self._bin(genome.complexity, self.COMPLEX_BINS),
        )

    def add(self, ev: Evaluation) -> bool:
        cur = self.cells.get(ev.descriptor)
        if cur is None or ev.fitness > cur.fitness:
            self.cells[ev.descriptor] = ev
            return True
        return False

    def elites(self):
        return sorted(self.cells.values(), key=lambda e: -e.fitness)

    def sample(self, rng, n):
        pool = list(self.cells.values())
        if not pool:
            return []
        w = np.array([max(e.fitness, 0.0) + 1e-3 for e in pool])
        w = w / w.sum()
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=True, p=w)
        return [pool[int(i)] for i in idx]

    @property
    def coverage(self):
        return len(self.cells)


# =============================================================================
#  Operator bandit  -- the engine's model of its own search
# =============================================================================
class NullModel:
    """Empirical null distributions -- the honest version of "beat the market".

    Two nulls, answering two different questions:

    RANDOM-GENOME NULL. Fresh random expressions are evaluated every generation
    and their fitness pooled. This is "how good does an arbitrary strategy look
    on this data?" On 2180 sessions of 615 names, the answer is usually: better
    than you would guess, which is the whole problem.

    TIME-SHIFT NULL (per candidate, and the stronger of the two). Take the
    candidate's own score matrix and circularly shift it in time by a random
    offset larger than any holding period. The cross-sectional structure is
    preserved exactly -- same expression, same distribution of scores, same
    names favoured relative to each other -- but the signal is now aligned with
    the wrong future. If the real strategy cannot beat its own decoupled
    replicas, then whatever it is doing is not timing; it is a static tilt
    dressed up as a prediction.

    This is precisely the test the video's final result failed. It returned
    30.4% against a 33.5% index and the honest reading was "indistinguishable
    from random". Running that test at search time, on every candidate, is the
    difference between finding that out now and finding it out after you have
    traded it.
    """

    def __init__(self, cap=3000):
        self.fitness: list = []
        self.ir: list = []
        self.alpha_t: list = []
        self.cap = cap

    def add(self, ev: "Evaluation"):
        self.fitness.append(float(ev.fitness))
        self.ir.append(float(ev.cv["mean_ir"]))
        self.alpha_t.append(float(ev.cv["alpha_tstat"]))
        if len(self.fitness) > self.cap:
            self.fitness = self.fitness[-self.cap:]
            self.ir = self.ir[-self.cap:]
            self.alpha_t = self.alpha_t[-self.cap:]

    def percentile(self, value: float, which: str = "fitness") -> float:
        pool = getattr(self, which)
        if len(pool) < 20:
            return float("nan")
        return float(np.mean(np.asarray(pool) < value))

    def summary(self):
        if len(self.fitness) < 20:
            return {"n": len(self.fitness)}
        f = np.asarray(self.fitness)
        return {
            "n": len(f),
            "mean_fitness": float(f.mean()),
            "p95_fitness": float(np.percentile(f, 95)),
            "p99_fitness": float(np.percentile(f, 99)),
            "max_fitness": float(f.max()),
            "p95_alpha_t": float(np.percentile(self.alpha_t, 95)),
        }


class FeatureBandit:
    """Which INPUTS generalise forward — one level above the operator bandit.

    The operator bandit learns which edits produce children that beat their
    parent. It is blind to what those children are made of. This learns the
    other half: for each feature family, a Beta posterior over "did strategies
    built from me survive the held-back forward tail?"

    The credit signal is deliberately forward validation and not fitness.
    Crediting on fitness would just re-learn whatever the fitness function
    already rewards, and the measured failure that prompted this whole layer
    was a leader with alpha t 4.07 on full-window CV and 0.10 forward. Families
    that look good in-sample and evaporate out of sample must be pushed down,
    which only works if the credit comes from the window fitness cannot see.

    Weights are sampled by Thompson draw and applied when the grammar picks a
    terminal, so a family that stops generalising gets proposed less often
    without ever being banned — it keeps a tail of draws and can come back if
    the regime changes.
    """

    def __init__(self, families, prior=(1.0, 1.0), floor=0.15):
        self.families = sorted(set(families))
        self.a = {f: prior[0] for f in self.families}
        self.b = {f: prior[1] for f in self.families}
        self.seen = {f: 0 for f in self.families}
        self.survived = {f: 0 for f in self.families}
        self.floor = floor        # never starve a family completely

    def update(self, fams, survived: bool, weight: float = 1.0):
        for f in set(fams):
            if f not in self.a:
                continue
            self.seen[f] += 1
            if survived:
                self.a[f] += weight
                self.survived[f] += 1
            else:
                self.b[f] += weight

    def decay(self, factor=0.997):
        for f in self.families:
            self.a[f] = 1.0 + (self.a[f] - 1.0) * factor
            self.b[f] = 1.0 + (self.b[f] - 1.0) * factor

    def weights(self, rng) -> dict:
        """Thompson draw per family, floored so nothing is ever unreachable."""
        return {f: self.floor + (1.0 - self.floor) * float(rng.beta(self.a[f], self.b[f]))
                for f in self.families}

    def table(self):
        return {f: {"posterior_mean": self.a[f] / (self.a[f] + self.b[f]),
                    "seen": self.seen[f], "survived": self.survived[f],
                    "rate": self.survived[f] / max(1, self.seen[f])}
                for f in self.families}

    def to_dict(self):
        return {"a": self.a, "b": self.b, "seen": self.seen,
                "survived": self.survived}

    def load(self, d):
        for k in ("a", "b", "seen", "survived"):
            getattr(self, k).update(d.get(k, {}))


class OperatorBandit:
    """Thompson sampling over mutation operators, credited by OOS survival."""

    def __init__(self, names, prior=(1.0, 1.0)):
        self.names = list(names)
        self.a = {n: prior[0] for n in self.names}
        self.b = {n: prior[1] for n in self.names}
        self.tried = {n: 0 for n in self.names}
        self.won = {n: 0 for n in self.names}

    def sample(self, rng) -> str:
        draws = {n: rng.beta(self.a[n], self.b[n]) for n in self.names}
        return max(draws, key=draws.get)

    def update(self, name, success: bool, weight: float = 1.0):
        if name not in self.a:
            return
        self.tried[name] += 1
        if success:
            self.a[name] += weight
            self.won[name] += 1
        else:
            self.b[name] += weight

    def decay(self, factor=0.995):
        """Forget slowly, so an operator that stops working is re-explored."""
        for n in self.names:
            self.a[n] = 1.0 + (self.a[n] - 1.0) * factor
            self.b[n] = 1.0 + (self.b[n] - 1.0) * factor

    def table(self):
        return {
            n: {"rate": self.won[n] / max(1, self.tried[n]),
                "tried": self.tried[n], "won": self.won[n],
                "posterior_mean": self.a[n] / (self.a[n] + self.b[n])}
            for n in self.names
        }

    def to_dict(self):
        return {"a": self.a, "b": self.b, "tried": self.tried, "won": self.won}

    def load(self, d):
        self.a.update(d.get("a", {}))
        self.b.update(d.get("b", {}))
        self.tried.update(d.get("tried", {}))
        self.won.update(d.get("won", {}))


# =============================================================================
#  Evolver
# =============================================================================
class Evolver:
    def __init__(self, cfg, panel, feats, seed=None, verbose=True,
                 store=None):
        self.cfg = cfg
        self.panel = panel
        self.feats = feats
        self.verbose = verbose
        self.rng = np.random.default_rng(
            int(seed if seed is not None else cfg.evolve.seed))
        self.grammar = Grammar(list(feats.keys()), self.rng,
                               int(cfg.evolve.max_tree_depth))
        self.bandit = OperatorBandit(Grammar.ALL_OPERATORS)
        from . import features as featmod
        self.family_of = featmod.family_map(list(feats.keys()))
        self.features = FeatureBandit(set(self.family_of.values()))
        self.archive = MapElites()
        self.null = NullModel()
        # Append-only record of every gated candidate and its forward outcome.
        # Survives restarts and --fresh; this is the corpus methods.py learns
        # from. None during unit tests and inner walk-forward searches, where
        # recording would be noise.
        self.store = store

        # ---- forward validation tail -----------------------------------
        # Everything the fitness function touches -- folds, shuffle test, cost
        # stress, PBO -- lives in `self.panel`, which is only the FIRST part of
        # the training window. The tail is held back entirely.
        #
        # This exists because of a measured failure. With the gate running on
        # the full window, the leader cleared every test (alpha t 4.07, PBO
        # 0.257) while an honest sequential walk-forward of the same procedure
        # returned alpha t 0.10 -- no forward edge at all. The gates were not
        # broken; they were all answering "does this work across this history",
        # and CSCV in particular builds its training half from a RANDOM
        # combination of time blocks, so half the time it fits on blocks that
        # come AFTER the block it scores. On non-stationary data that is a
        # strictly easier question than the one live trading asks.
        #
        # A held-back tail asks the real question: fit on the past, and does it
        # still work on a stretch of future the search never saw?
        frac = float(cfg.evolve.get("validation_tail_frac", 0.20))
        T_all = panel.shape[0]
        emb = int(cfg.cv.embargo_days)
        fit_end = int(T_all * (1.0 - frac))
        val_lo = min(T_all, fit_end + emb)

        self.full_panel = panel
        self.panel = cvmod._slice_panel(panel, 0, fit_end)
        self.feats = {k: v[:fit_end] for k, v in feats.items()}

        if T_all - val_lo >= 120:
            self.val_panel = cvmod._slice_panel(panel, val_lo, T_all)
            self.val_feats = {k: v[val_lo:T_all] for k, v in feats.items()}
            self.val_bench = bt.equal_weight_benchmark(self.val_panel)
        else:
            self.val_panel = self.val_feats = self.val_bench = None

        self.bench = bt.equal_weight_benchmark(self.panel)
        self.folds = cvmod.block_folds(self.panel.shape[0], int(cfg.cv.n_splits),
                                       int(cfg.cv.embargo_days))

        # Forward returns for the IC gate, memoised per holding period. Every
        # genome carries its own rebalance length and there are only a handful
        # of distinct ones, so this is a few arrays rather than one per
        # candidate.
        self._fwd: dict = {}
        self._val_fwd: dict = {}
        # gate() needs the same score matrix evaluate() just built. Rebuilding
        # it costs 0.55s against 0.23s for the IC itself -- 70% of the gate
        # spent recomputing something we had. One entry, tagged with the
        # fingerprint it belongs to, so a cache hit in evaluate() can never
        # hand gate() another genome's scores.
        self._last_score: tuple = (None, None)

        self.population: list = []
        self.hall_of_fame: list = []
        self.demoted: list = []
        self.generation = 0
        self.trials = 0                 # distinct genomes ever evaluated
        self.seen: set = set()
        self._cache: OrderedDict = OrderedDict()
        self.history: list = []
        self.stall = 0
        self.best_fitness = -np.inf
        self.explore = 0.25             # adapts

    # ------------------------------------------------------------- fitness
    def evaluate(self, g: Genome) -> Evaluation | None:
        fp = g.fingerprint
        if fp in self._cache:
            self._cache.move_to_end(fp)
            return self._cache[fp]

        try:
            score = g.score(self.feats, self.panel.tradable)
        except Exception:
            return None
        if not np.isfinite(score).any():
            return None
        self._last_score = (fp, score)

        res = cvmod.evaluate(self.panel, score, g, self.cfg,
                             bench_ret=self.bench, folds=self.folds)
        if not res.ok:
            return None

        if fp not in self.seen:
            self.seen.add(fp)
            self.trials += 1

        fit = self._fitness(res, g)
        ev = Evaluation(genome=g, fitness=fit, cv=res.as_dict(),
                        descriptor=self.archive.descriptor(res, g),
                        returns=res.returns - res.bench)
        self._cache[fp] = ev
        if len(self._cache) > 4000:
            self._cache.popitem(last=False)
        return ev

    def _fitness(self, res, g: Genome) -> float:
        """Information ratio, discounted by inconsistency, complexity and
        index-hugging. Deliberately NOT return-based."""
        # Fitness shape is a METHOD choice (see methods.py), not a constant.
        # "consistency" squares the fold-agreement term, so a strategy winning
        # six folds of eight beats one winning enormously in two -- the second
        # profile is the one that has repeatedly failed forward here.
        mode = str(self.cfg.evolve.get("fitness_mode", "ir"))
        if mode == "novelty":
            # Novelty search with local competition (Stanley & Lehman). Reward
            # behavioural DIFFERENCE, not alpha: on deceptive problems the
            # objective is itself the thing leading the search astray. Alpha
            # survives only as a tiebreak, and the method is still credited on
            # forward alpha, so the question gets settled by evidence.
            fit = self._novelty(res, g) + 0.05 * res.mean_ir
        elif mode == "consistency":
            fit = res.mean_ir * (res.frac_positive ** 2)
        elif mode == "median_alpha":
            fit = res.median_alpha * (0.5 + 0.5 * res.frac_positive)
        else:
            fit = res.mean_ir * (0.5 + 0.5 * res.frac_positive)

        # Index-hugging penalty, ramped smoothly from 0.90. A hard cliff at the
        # gate threshold teaches the search to park at 0.9499 and collect the
        # reward -- which is exactly what it did the first time this ran.
        corr = abs(res.corr_bench)
        if corr > 0.90:
            fit -= 2.0 * ((corr - 0.90) / 0.10) ** 2

        if str(self.cfg.evolve.get("parsimony_mode", "linear")) == "bic":
            # BIC-style: complexity * ln(n) / 2n. Scales with sample size the
            # way a model-selection criterion should, instead of being a
            # constant someone picked on the first afternoon.
            n_obs = max(self.panel.shape[0], 2)
            fit -= g.complexity * float(np.log(n_obs)) / (2.0 * n_obs) * 100.0
        else:
            fit -= float(self.cfg.evolve.parsimony_penalty) * g.complexity
        # Turnover you cannot pay for.
        cap = float(self.cfg.evolve.gate.max_annual_turnover)
        if res.ann_turnover > cap:
            fit -= 0.5 * (res.ann_turnover / cap - 1.0)
        if res.total_trades < int(self.cfg.evolve.gate.min_trades):
            fit -= 1.0
        return float(fit) if np.isfinite(fit) else -np.inf

    # --------------------------------------------------- forward validation
    def validate(self, g: Genome) -> dict:
        """Backtest on the held-back tail. Nothing in fitness has seen this."""
        if self.val_panel is None:
            return {"val_alpha": float("nan"), "val_ir": float("nan"),
                    "val_t": float("nan")}
        try:
            score = g.score(self.val_feats, self.val_panel.tradable)
        except Exception:
            return {"val_alpha": float("nan"), "val_ir": float("nan"),
                    "val_t": float("nan")}
        res = bt.run(self.val_panel, score, g, self.cfg,
                     bench_ret=self.val_bench)
        if not res.ok:
            return {"val_alpha": float("nan"), "val_ir": float("nan"),
                    "val_t": float("nan")}
        out = {
            "val_alpha": float(res.stats["alpha_ann"]),
            "val_ir": float(res.stats["ir"]),
            "val_t": float(res.stats["alpha_tstat"]),
            "val_sessions": int(len(res.ret)),
        }
        # The same ranking question, on the window fitness has never seen. A
        # portfolio alpha over a short tail is almost pure noise; the IC over
        # the same tail uses every name on every day of it.
        try:
            h = max(1, int(g.rebalance_days))
            if h not in self._val_fwd:
                self._val_fwd[h] = icmod.forward_returns(
                    self.val_panel.close, h)
            r = icmod.summarize(score, self._val_fwd[h],
                                self.val_panel.tradable, h)
            out["val_ic"] = r["ic"]
            out["val_ic_t"] = r["ic_t"]
        except Exception:
            out["val_ic"] = out["val_ic_t"] = float("nan")
        return out

    # --------------------------------------------------------- cost stress
    def cost_stress(self, g: Genome, multiplier: float = 3.0) -> dict:
        """Does the alpha survive transaction costs being much worse than assumed?

        A raw turnover cap is the wrong instrument. Costs are already charged
        inside the backtest, so 22x turnover is not per se disqualifying -- the
        real question is whether the edge depends on the slippage estimate
        being right. A strategy that still clears at 3x assumed cost is robust;
        one that dies was never trading an anomaly, it was trading your cost
        model. That distinction is invisible to a turnover threshold.
        """
        import copy

        try:
            score = g.score(self.feats, self.panel.tradable)
        except Exception:
            return {"stress_alpha": float("nan"), "stress_ir": float("nan")}

        stressed = copy.deepcopy(dict(self.cfg))
        stressed["backtest"]["slippage_bps"] = (
            float(self.cfg.backtest.slippage_bps) * multiplier)
        stressed["backtest"]["commission_bps"] = (
            float(self.cfg.backtest.commission_bps) * multiplier)
        from .config import Cfg

        res = cvmod.evaluate(self.panel, score, g, Cfg(stressed),
                             bench_ret=self.bench, folds=self.folds)
        if not res.ok:
            return {"stress_alpha": float("nan"), "stress_ir": float("nan")}
        return {
            "stress_alpha": float(res.mean_alpha),
            "stress_ir": float(res.mean_ir),
            "stress_multiplier": float(multiplier),
        }

    def _novelty(self, res, g, k: int = 8) -> float:
        """Mean distance to the k nearest behaviours already in the archive.

        Behaviour is the MAP-Elites descriptor (turnover, benchmark
        correlation, complexity) taken as a continuous vector rather than a
        bin, so novelty is measured on the same axes the archive already
        considers meaningful. An empty archive means everything is novel.
        """
        here = np.array([
            float(res.ann_turnover), float(abs(res.corr_bench)),
            float(g.complexity)], dtype=np.float64)
        pool = self.archive.elites()
        if len(pool) < 2:
            return 1.0
        # Normalised so turnover (0-30) does not swamp correlation (0-1).
        scale = np.array([10.0, 0.25, 15.0])
        d = []
        for e in pool[:200]:
            c = e.cv
            other = np.array([
                float(c.get("ann_turnover", 0.0)),
                float(abs(c.get("corr_bench", 0.0))),
                float(e.genome.complexity)], dtype=np.float64)
            d.append(float(np.linalg.norm((here - other) / scale)))
        d.sort()
        return float(np.mean(d[:min(k, len(d))]))

    # ------------------------------------------------------------ null test
    def shuffle_test(self, g: Genome, n: int | None = None) -> dict:
        """Score the genome against time-shifted replicas of its own signal.

        Returns the fraction of replicas the real strategy beats. The replicas
        share the candidate's exact cross-sectional structure and differ only
        in that their signal is aligned with the wrong future, so this isolates
        timing information from static tilt.
        """
        n = int(n if n is not None else self.cfg.evolve.get("null_shuffles", 24))
        try:
            score = g.score(self.feats, self.panel.tradable)
        except Exception:
            return {"null_percentile": float("nan"), "n": 0}

        real = cvmod.evaluate(self.panel, score, g, self.cfg,
                              bench_ret=self.bench, folds=self.folds)
        if not real.ok:
            return {"null_percentile": float("nan"), "n": 0}

        T = score.shape[0]
        lo = max(252, T // 8)
        if T - lo <= lo:
            return {"null_percentile": float("nan"), "n": 0}

        irs = []
        for _ in range(n):
            shift = int(self.rng.integers(lo, T - lo))
            r = cvmod.evaluate(self.panel, np.roll(score, shift, axis=0), g,
                               self.cfg, bench_ret=self.bench, folds=self.folds)
            if r.ok:
                irs.append(r.mean_ir)
        # Require most replicas to have produced a usable backtest, but never
        # require more than were attempted. A fixed floor of 5 here meant any
        # configured `null_shuffles` below 5 was unsatisfiable: the test always
        # returned NaN, NaN is treated as a gate failure, and nothing could
        # ever be promoted -- silently, with no error anywhere.
        need = min(n, max(3, (2 * n) // 3))
        if len(irs) < need:
            return {"null_percentile": float("nan"), "n": len(irs),
                    "needed": need, "attempted": n}

        irs = np.asarray(irs)
        return {
            "null_percentile": float(np.mean(irs < real.mean_ir)),
            "null_mean_ir": float(irs.mean()),
            "null_p95_ir": float(np.percentile(irs, 95)),
            "real_ir": float(real.mean_ir),
            "n": len(irs),
        }

    # ---------------------------------------------------------------- gate
    def gate(self, ev: Evaluation, population_returns=None) -> list:
        """The promotion test. Every gate must pass; no partial credit."""
        g = self.cfg.evolve.gate
        cvd = ev.cv
        checks = [
            metrics.summarize_gate("cv_folds_positive", cvd["frac_positive"],
                                   ">=", float(g.min_cv_folds_positive)),
            metrics.summarize_gate("alpha_tstat", cvd["alpha_tstat"],
                                   ">=", float(g.min_alpha_tstat)),
            metrics.summarize_gate("n_trades", cvd["total_trades"],
                                   ">=", float(g.min_trades)),
            metrics.summarize_gate("ann_turnover", cvd["ann_turnover"],
                                   "<=", float(g.max_annual_turnover)),
            metrics.summarize_gate("corr_bench", abs(cvd["corr_bench"]),
                                   "<=", float(g.max_benchmark_corr)),
        ]
        # Deflated Sharpe on the EXCESS series, priced for the number of
        # *independent* hypotheses this search has really tested.
        eff = metrics.effective_trials(population_returns, self.trials) \
            if population_returns is not None else {"n_effective": self.trials,
                                                    "independence": 1.0}
        n_eff = int(max(1, round(eff["n_effective"])))
        d = metrics.dsr_from_returns(ev.returns, n_eff)
        checks.append(metrics.summarize_gate(
            "deflated_sharpe", d["dsr"], ">=", float(g.min_deflated_sharpe)))
        ev.cv["dsr"] = d["dsr"]
        ev.cv["sr_ann_excess"] = d["sr_ann"]
        ev.cv["sr_star_ann"] = d["sr_star_ann"]
        ev.cv["trials_total"] = self.trials
        ev.cv["trials_effective"] = n_eff
        ev.cv["independence"] = eff.get("independence", 1.0)

        # ---- the gate with power ------------------------------------------
        # Placed after n_eff because the threshold is a function of it: the
        # largest |t| this search would have found in noise, having looked
        # exactly as hard as it has. It rises as the search continues, so a
        # candidate cannot clear it simply by being evaluated late.
        try:
            h = max(1, int(ev.genome.rebalance_days))
            if h not in self._fwd:
                self._fwd[h] = icmod.forward_returns(self.panel.close, h)
            fp_last, cached = self._last_score
            score = (cached if fp_last == ev.genome.fingerprint
                     else ev.genome.score(self.feats, self.panel.tradable))
            r = icmod.summarize(score, self._fwd[h], self.panel.tradable, h)
            ev.cv.update({k: r[k] for k in
                          ("ic", "ic_t", "ic_n", "ic_ir", "ic_hit")})
            want = g.get("min_ic_tstat", "auto")
            bar = (icmod.null_bar(n_eff, float(g.get("ic_tstat_floor", 2.0)))
                   if str(want).lower() == "auto" else float(want))
            ev.cv["ic_bar"] = bar
            checks.append(metrics.summarize_gate("ic_tstat", r["ic_t"],
                                                 ">=", bar))
        except Exception:
            ev.cv["ic_t"] = float("nan")
            checks.append({"gate": "ic_tstat", "value": float("nan"),
                           "op": ">=", "threshold": float("nan"),
                           "pass": False})

        # PBO over the BEHAVIOUR archive, not the fitness leaderboard. CSCV asks
        # "does my selection procedure pick winners that keep winning?", which
        # only means something across genuinely different configurations. Run it
        # over the top 60 by fitness -- near-identical survivors -- and it sits
        # at 0.5 by construction, because choosing among interchangeable
        # strategies IS a coin flip and the metric is right to say so. The
        # archive is the diverse alternative set the method assumes.
        if population_returns is not None and population_returns.shape[1] >= 4:
            p = metrics.pbo_cscv(population_returns,
                                 int(self.cfg.cv.cscv_partitions))
            if np.isfinite(p.get("pbo", np.nan)):
                checks.append(metrics.summarize_gate(
                    "pbo", p["pbo"], "<=", float(g.max_pbo)))
                ev.cv["pbo"] = p["pbo"]

        # Cost stress and the shuffle test cost ~25 backtests each, so they run
        # last and only for a candidate that cleared everything cheap.
        if all(c["pass"] for c in checks):
            st = self.cost_stress(
                ev.genome, float(self.cfg.evolve.get("cost_stress_multiplier", 3.0)))
            ev.cv.update(st)
            if np.isfinite(st.get("stress_alpha", np.nan)):
                checks.append(metrics.summarize_gate(
                    "cost_stress_alpha", st["stress_alpha"], ">=",
                    float(g.get("min_stress_alpha", 0.0))))

        if all(c["pass"] for c in checks) and self.val_panel is not None:
            v = self.validate(ev.genome)
            ev.cv.update(v)
            if np.isfinite(v.get("val_alpha", np.nan)):
                # Credit the families this genome is built from by whether it
                # actually held up on the window fitness never saw.
                self.features.update(
                    [self.family_of.get(n, "other")
                     for n in ev.genome.feature_names()],
                    survived=v["val_alpha"] > 0)
                checks.append(metrics.summarize_gate(
                    "forward_alpha", v["val_alpha"], ">=",
                    float(g.get("min_forward_alpha", 0.0))))
                if np.isfinite(v.get("val_ic", np.nan)):
                    checks.append(metrics.summarize_gate(
                        "forward_ic", v["val_ic"], ">=",
                        float(g.get("min_forward_ic", 0.0))))
            else:
                checks.append({"gate": "forward_alpha", "value": float("nan"),
                               "op": ">=", "threshold": 0.0, "pass": False})

        if all(c["pass"] for c in checks):
            sh = self.shuffle_test(ev.genome)
            ev.cv.update(sh)
            if np.isfinite(sh.get("null_percentile", np.nan)):
                checks.append(metrics.summarize_gate(
                    "null_percentile", sh["null_percentile"], ">=",
                    float(g.get("min_null_percentile", 0.95))))
            else:
                checks.append({"gate": "null_percentile", "value": float("nan"),
                               "op": ">=", "threshold": float(
                                   g.get("min_null_percentile", 0.95)),
                               "pass": False})
        ev.gates = checks
        if self.store is not None:
            try:
                self.store.record("gated", ev.genome, cv=ev.cv, gates=checks,
                                  generation=self.generation,
                                  fitness=float(ev.fitness))
            except Exception:
                pass
        return checks

    def revalidate_hall_of_fame(self, population_returns=None):
        """Re-test everything already promoted against TODAY'S bar.

        Promotion was written as a one-way door: clear the gate once and stay
        in the hall of fame forever. That is wrong here, and wrong in a way
        specific to this project's own thesis.

        The bar is not fixed. The deflated-Sharpe threshold rises with the
        number of hypotheses the search has tested, so a strategy promoted at
        40,000 trials is being judged by a weaker standard than one promoted at
        160,000. Left alone, the hall of fame silently accumulates strategies
        that only ever cleared an easier test -- and `run_trade.py` keeps
        trading them. Observed in a live run: a promoted strategy sitting at
        PBO 0.771 against a 0.40 limit and DSR 0.319 against 0.50, still
        holding capital, purely because it was promoted early.

        So the gate is re-run every generation against the current trial count
        and the current archive. Demotions are recorded with the reason rather
        than deleted, because "this used to look real and no longer does" is
        the single most useful thing this system can tell you.
        """
        if not self.hall_of_fame:
            return []
        demoted = []
        keep = []
        for entry in self.hall_of_fame:
            gd = entry.get("genome")
            if not gd:
                keep.append(entry)
                continue
            try:
                ev = self.evaluate(Genome.from_dict(gd))
            except Exception:
                keep.append(entry)
                continue
            if ev is None:
                keep.append(entry)
                continue
            checks = self.gate(ev, population_returns)
            failed = [c for c in checks if not c["pass"]]
            if failed:
                entry["demoted_at_generation"] = self.generation
                entry["demoted_at_trials"] = self.trials
                entry["demoted_because"] = [
                    "%s %.4f %s %.4f" % (c["gate"], c["value"], c["op"],
                                         c["threshold"]) for c in failed
                ]
                entry["gates"] = checks
                demoted.append(entry)
                if self.store is not None:
                    try:
                        self.store.record(
                            "demoted", Genome.from_dict(gd), cv=ev.cv,
                            gates=checks, promoted=False,
                            generation=self.generation,
                            demote_reason="; ".join(entry["demoted_because"]))
                    except Exception:
                        pass
                self._log(
                    "DEMOTED  %s  no longer clears: %s"
                    % (entry.get("fingerprint"), "; ".join(entry["demoted_because"])))
            else:
                entry["gates"] = checks
                entry["cv"] = ev.cv
                keep.append(entry)

        if demoted:
            self.hall_of_fame = keep
            self.demoted.extend(demoted)
        demoted += self._dedupe_by_behaviour()
        return demoted

    def _dedupe_by_behaviour(self, threshold: float = 0.999):
        """Collapse promotions that are the same strategy in different spellings.

        Fingerprints distinguish expressions, not behaviour. The simplifier is
        deliberately conservative -- `x * 0 -> 0` is unsound when x is NaN, so
        it is not applied -- which leaves semantically identical genomes with
        different fingerprints. Observed live, two separately promoted entries:

            zscore(ite(mul(mom_252, 0), ma_ratio_50, rev_5))|k=15|rb=21|w=score
            rev_5|k=15|rb=21|w=score

        Their information ratios agreed to nine significant figures and they
        placed exactly the same 2,550 trades, because `mul(mom_252, 0)` is
        never positive so the branch is dead. One idea, two slots in the hall
        of fame -- and the Thompson allocator, which weights per strategy key,
        would therefore have given it twice the capital of a genuinely distinct
        peer. That is a sizing error dressed as diversification.

        So: correlate the realised excess-return series and treat anything
        above `threshold` as one strategy, keeping the structurally simpler
        expression. Occam breaks the tie, which is also the version a human
        stands a chance of understanding.
        """
        if len(self.hall_of_fame) < 2:
            return []

        series, entries = [], []
        for e in self.hall_of_fame:
            gd = e.get("genome")
            if not gd:
                continue
            try:
                ev = self.evaluate(Genome.from_dict(gd))
            except Exception:
                continue
            if ev is None or ev.returns is None or not len(ev.returns):
                continue
            series.append(np.asarray(ev.returns, dtype=np.float64))
            entries.append((e, Genome.from_dict(gd).complexity))

        drop = set()
        for i in range(len(entries)):
            if i in drop:
                continue
            for j in range(i + 1, len(entries)):
                if j in drop:
                    continue
                n = min(len(series[i]), len(series[j]))
                if n < 50:
                    continue
                a, b = series[i][:n], series[j][:n]
                if a.std() < EPS or b.std() < EPS:
                    continue
                if abs(float(np.corrcoef(a, b)[0, 1])) < threshold:
                    continue
                # Same behaviour: keep the simpler expression.
                loser = j if entries[j][1] >= entries[i][1] else i
                winner = i if loser == j else j
                e = entries[loser][0]
                e["demoted_at_generation"] = self.generation
                e["demoted_at_trials"] = self.trials
                e["demoted_because"] = [
                    "duplicate of %s (excess-return corr >= %.3f); kept the "
                    "simpler expression" % (entries[winner][0].get("fingerprint"),
                                            threshold)]
                drop.add(loser)
                self._log("DEDUPED  %s is the same strategy as %s"
                          % (e.get("fingerprint"),
                             entries[winner][0].get("fingerprint")))

        if not drop:
            return []
        removed = [entries[k][0] for k in sorted(drop)]
        keep_fps = {id(entries[k][0]) for k in range(len(entries))
                    if k not in drop}
        self.hall_of_fame = [e for e in self.hall_of_fame
                             if id(e) in keep_fps or e.get("genome") is None]
        self.demoted.extend(removed)
        return removed

    def sample_null(self, n: int):
        """Evaluate fresh random genomes to keep the empirical null calibrated."""
        for _ in range(int(n)):
            ev = self.evaluate(self.grammar.random_genome(self.generation))
            if ev is not None:
                self.null.add(ev)

    # ----------------------------------------------------------- lifecycle
    def seed_population(self, n=None):
        n = n or int(self.cfg.evolve.population)
        pop = []
        guard = 0
        while len(pop) < n and guard < n * 20:
            guard += 1
            ev = self.evaluate(self.grammar.random_genome(self.generation))
            if ev is not None:
                pop.append(ev)
                self.archive.add(ev)
        self.population = pop
        return pop

    def breed(self, n_children: int):
        parents = sorted(self.population, key=lambda e: -e.fitness)
        n_elite = max(2, int(len(parents) * float(self.cfg.evolve.elite_frac)))
        elite = parents[:n_elite]
        # Draw as many parents from the behaviour archive as from the fitness
        # elite. Breeding only from the leaderboard is how a population becomes
        # one lineage that has overfit together.
        # Drawing half the parents from the behaviour archive is itself a
        # method choice; elitist_ir turns it off so QD can be measured against
        # its own absence rather than assumed to be earning its slots.
        if bool(self.cfg.evolve.get("use_map_elites", True)):
            pool = elite + self.archive.sample(self.rng, max(4, 2 * n_elite))
        else:
            pool = elite

        children = []
        guard = 0
        while len(children) < n_children and guard < n_children * 12:
            guard += 1
            if self.rng.random() < self.explore * 0.35 or not pool:
                cand = self.grammar.random_genome(self.generation)
                op = "init"
                parent_fit = -np.inf
            else:
                p = pool[int(self.rng.integers(len(pool)))]
                mate = pool[int(self.rng.integers(len(pool)))] if len(pool) > 1 else None
                op = self.bandit.sample(self.rng)
                cand = self.grammar.apply(op, p.genome,
                                          mate.genome if mate else None,
                                          self.generation)
                parent_fit = p.fitness
            if cand is None:
                continue
            ev = self.evaluate(cand)
            if ev is None:
                self.bandit.update(op, False, 0.5)
                continue
            # Credit assignment: did this operator beat the parent it modified?
            self.bandit.update(op, ev.fitness > parent_fit)
            children.append(ev)
        return children

    def step(self):
        t0 = time.time()
        self.generation += 1
        n = int(self.cfg.evolve.population)

        # Re-draw the terminal weights so the grammar proposes from families
        # that have been surviving forward validation.
        self.grammar.set_feature_weights(
            self.features.weights(self.rng), self.family_of)
        self.sample_null(int(self.cfg.evolve.get("null_samples_per_gen", 12)))
        children = self.breed(n)
        merged = self.population + children
        # dedupe by fingerprint, keep best
        best_by_fp = {}
        for e in merged:
            fp = e.genome.fingerprint
            if fp not in best_by_fp or e.fitness > best_by_fp[fp].fitness:
                best_by_fp[fp] = e
        merged = sorted(best_by_fp.values(), key=lambda e: -e.fitness)

        for e in merged:
            self.archive.add(e)

        # Population = fitness elite + behaviourally distinct elites. Filling
        # every slot by fitness alone converges in ~3 generations onto one
        # expression and then stops searching; this keeps the frontier wide.
        if not bool(self.cfg.evolve.get("use_map_elites", True)):
            # elitist_ir: pure top-N. Kept as a method so quality-diversity can
            # be measured against its own absence instead of assumed to earn
            # the population slots it costs.
            self.population = merged[:n]
        else:
            keep_fit = int(n * 0.6)
            chosen = merged[:keep_fit]
            have = {e.genome.fingerprint for e in chosen}
            for e in self.archive.elites():
                if len(chosen) >= n:
                    break
                if e.genome.fingerprint not in have:
                    chosen.append(e)
                    have.add(e.genome.fingerprint)
            for e in merged[keep_fit:]:
                if len(chosen) >= n:
                    break
                if e.genome.fingerprint not in have:
                    chosen.append(e)
                    have.add(e.genome.fingerprint)
            self.population = sorted(chosen, key=lambda e: -e.fitness)
        self.bandit.decay()
        self.features.decay()

        # Forward-validate the leader EVERY generation, not only when the
        # cheap gates happen to pass. It is one extra backtest on a window
        # nothing else touches, and it is the only labelled signal the method
        # bandit and the feature bandit can learn from. Gating on it alone
        # produced six recorded candidates and zero labels -- a corpus with no
        # outcomes attached teaches nothing.
        if self.population and self.val_panel is not None:
            leader = self.population[0]
            if "val_alpha" not in leader.cv:
                try:
                    v = self.validate(leader.genome)
                    leader.cv.update(v)
                    if np.isfinite(v.get("val_alpha", np.nan)):
                        self.features.update(
                            [self.family_of.get(n, "other")
                             for n in leader.genome.feature_names()],
                            survived=v["val_alpha"] > 0)
                except Exception:
                    pass

        # ---- promotion test on the current leader ------------------------
        if not self.population:
            # Every candidate failed to evaluate -- too little usable history,
            # or a panel so short block_folds returns nothing. Re-seed once and
            # give up on the generation rather than dying on an index error, so
            # a caller running many small searches (walkforward --honest) can
            # skip the segment instead of losing the whole run.
            self.seed_population()
            if not self.population:
                self._log("generation %d: no viable candidates; skipping"
                          % self.generation)
                return {"generation": self.generation, "best_fitness": float("nan"),
                        "best_ir": float("nan"), "best_alpha_t": float("nan"),
                        "best_dsr": float("nan"), "best_pbo": float("nan"),
                        "best_corr": float("nan"), "best_vs_null_pct": float("nan"),
                        "median_fitness": float("nan"), "archive_cells": 0,
                        "trials": self.trials, "trials_effective": self.trials,
                        "independence": 1.0, "explore": float(self.explore),
                        "promoted_total": len(self.hall_of_fame),
                        "null_p99_fitness": float("nan"), "seconds": 0.0,
                        "degenerate": True}
        top = self.population[0]
        pop_ret = self._population_return_matrix()
        self.gate(top, pop_ret)
        passed = all(c["pass"] for c in top.gates)
        if passed and not any(h["fingerprint"] == top.genome.fingerprint
                              for h in self.hall_of_fame):
            top.promoted = True
            self.hall_of_fame.append(top.as_dict())
            if self.store is not None:
                try:
                    self.store.record("promoted", top.genome, cv=top.cv,
                                      gates=top.gates, promoted=True,
                                      generation=self.generation,
                                      fitness=float(top.fitness))
                except Exception:
                    pass
            self._log("PROMOTED  %s  fit=%.3f  IR=%.2f  DSR=%.2f  t=%.2f"
                      % (top.genome.fingerprint, top.fitness,
                         top.cv["mean_ir"], top.cv.get("dsr", float("nan")),
                         top.cv["alpha_tstat"]))

        # Re-test prior promotions against today's (higher) bar.
        self.revalidate_hall_of_fame(pop_ret)

        # ---- meta-adaptation: widen the search when it stops paying ------
        if top.fitness > self.best_fitness + 1e-4:
            self.best_fitness = top.fitness
            self.stall = 0
            self.explore = max(0.10, self.explore * 0.93)
        else:
            self.stall += 1
            if self.stall % 2 == 0:
                self.explore = min(0.85, self.explore * 1.30)

        rec = {
            "generation": self.generation,
            "best_fitness": float(top.fitness),
            "best_ir": float(top.cv["mean_ir"]),
            "best_alpha_t": float(top.cv["alpha_tstat"]),
            "best_dsr": float(top.cv.get("dsr", float("nan"))),
            "best_pbo": float(top.cv.get("pbo", float("nan"))),
            "best_corr": float(top.cv["corr_bench"]),
            "best_null_pct": float(top.cv.get("null_percentile", float("nan"))),
            "median_fitness": float(np.median([e.fitness for e in self.population])),
            "null_p99_fitness": float(self.null.summary().get(
                "p99_fitness", float("nan"))),
            "best_vs_null_pct": float(self.null.percentile(top.fitness)),
            "archive_cells": self.archive.coverage,
            "trials": self.trials,
            "trials_effective": int(top.cv.get("trials_effective", self.trials)),
            "independence": float(top.cv.get("independence", 1.0)),
            # Recorded so the stopping rule can measure stall from persisted
            # history rather than from a counter living in one process. The
            # supervisor restarts this worker on every source change, and an
            # in-process counter resets each time -- so a rule needing 300
            # stalled generations would never once reach 300.
            #
            # Read from `top`, like every other best_* field here. The first
            # version took the max over archive elites, but ic_t is only
            # computed inside gate(), which runs on the generation leader --
            # so most archive entries carry no IC at all and the maximum sat
            # frozen on whichever few had one. It logged 0.80 for 97
            # consecutive generations, which would have read as a dead flat
            # plateau rather than as a broken measurement.
            "best_ic_t": float(top.cv.get("ic_t", float("nan"))),
            "explore": float(self.explore),
            "promoted_total": len(self.hall_of_fame),
            "seconds": round(time.time() - t0, 2),
        }
        self.history.append(rec)
        self._log(
            "gen %3d | fit %+.3f | IR %+.2f | t %+.2f | DSR %.2f | corr %.2f "
            "| vs-null %.2f | cells %3d | trials %5d (eff %4d) | %.1fs"
            % (self.generation, rec["best_fitness"], rec["best_ir"],
               rec["best_alpha_t"], rec["best_dsr"], rec["best_corr"],
               rec["best_vs_null_pct"], rec["archive_cells"], rec["trials"],
               rec["trials_effective"], rec["seconds"]))
        return rec

    def run(self, generations=None):
        generations = generations or int(self.cfg.evolve.generations)
        if not self.population:
            self._log("seeding population (%d)..." % self.cfg.evolve.population)
            self.seed_population()
        for _ in range(generations):
            self.step()
            self.save()
        return self.history

    # ---------------------------------------------------------------- misc
    def _population_return_matrix(self, cap=80):
        """Return streams of behaviourally DISTINCT candidates, for CSCV.

        Drawn from the MAP-Elites archive rather than the fitness leaderboard,
        so the columns are genuinely different strategies rather than variants
        of one expression.
        """
        pool = self.archive.elites()[:cap] or self.population[:cap]
        cols = [e.returns for e in pool if e.returns is not None and len(e.returns)]
        if len(cols) < 4:
            return None
        L = min(len(c) for c in cols)
        return np.column_stack([c[:L] for c in cols])

    def _log(self, msg):
        if self.verbose:
            print("[evolve] %s" % msg, flush=True)

    def leaderboard(self, n=10):
        return [e.as_dict() for e in self.archive.elites()[:n]]

    # --------------------------------------------------------------- state
    def save(self, path=None):
        p = Path(path) if path else resolve(STATE)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generation": self.generation,
            "trials": self.trials,
            "best_fitness": (float(self.best_fitness)
                             if np.isfinite(self.best_fitness) else None),
            "explore": self.explore,
            "stall": self.stall,
            "bandit": self.bandit.to_dict(),
            "operator_table": self.bandit.table(),
            "feature_bandit": self.features.to_dict(),
            "feature_table": self.features.table(),
            "null_summary": self.null.summary(),
            "null_pool": {"fitness": self.null.fitness[-1500:],
                          "ir": self.null.ir[-1500:],
                          "alpha_t": self.null.alpha_t[-1500:]},
            "history": self.history[-500:],
            "hall_of_fame": self.hall_of_fame,
            "demoted": self.demoted,
            "archive": [e.as_dict() for e in self.archive.elites()],
            "population": [e.genome.to_dict() for e in self.population[:100]],
            "panel": {
                "start": str(self.panel.dates[0].date()),
                "stop": str(self.panel.dates[-1].date()),
                "tickers": self.panel.shape[1],
                "coverage": self.panel.coverage,
            },
        }
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return p

    def load(self, path=None):
        p = Path(path) if path else resolve(STATE)
        if not p.exists():
            return False
        d = json.loads(p.read_text(encoding="utf-8"))
        self.generation = int(d.get("generation", 0))
        self.trials = int(d.get("trials", 0))
        self.explore = float(d.get("explore", 0.25))
        self.stall = int(d.get("stall", 0))
        bf = d.get("best_fitness")
        self.best_fitness = float(bf) if bf is not None else -np.inf
        self.bandit.load(d.get("bandit", {}))
        self.features.load(d.get("feature_bandit", {}))
        np_pool = d.get("null_pool", {})
        self.null.fitness = list(np_pool.get("fitness", []))
        self.null.ir = list(np_pool.get("ir", []))
        self.null.alpha_t = list(np_pool.get("alpha_t", []))
        self.history = d.get("history", [])
        self.hall_of_fame = d.get("hall_of_fame", [])
        self.demoted = d.get("demoted", [])
        pop = []
        for gd in d.get("population", []):
            try:
                ev = self.evaluate(Genome.from_dict(gd))
                if ev:
                    pop.append(ev)
                    self.archive.add(ev)
            except Exception:
                continue
        self.population = sorted(pop, key=lambda e: -e.fitness)
        self._log("resumed at generation %d (%d trials, %d elites)"
                  % (self.generation, self.trials, self.archive.coverage))
        return True
