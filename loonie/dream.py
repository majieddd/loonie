"""Evaluate search policies against recorded history instead of re-running them.

From Dream-RSI (Zheng et al., 2026, Google/DeepMind): a completed discovery
history is not merely a log, it is a REPLAY SIMULATOR. The history already
records which attempts were made, what produced them, and what they scored --
so an alternative exploration policy can be evaluated by navigating those
recorded outcomes rather than by generating new ones. Meta-level feedback that
was delayed and expensive becomes immediate and nearly free.

This system needs exactly that. The genetic search plateaued after 359
generations having spent 300,000 trials, and there was no way to ask whether a
different search policy would have done better without running it for hours --
so the question was never asked. The method bandit picks among eight
strategies by Thompson sampling on forward-validated outcomes, which is
honest, but it can only learn from methods it actually deploys, one slow
generation at a time.

WHAT THE TREE IS HERE. The experience corpus records every gated candidate
with the method that produced it, the operator that mutated it, the generation
it was born in, its cross-validated scores, and -- critically -- its FORWARD
outcome on the held-back tail. Nodes are candidates; the score is the forward
result, not the fitness, because fitness is what the search optimised and
forward alpha is what it was trying to predict. Replaying against fitness
would measure how well a policy games the objective.

WHAT REPLAY CANNOT DO, stated plainly. A replayed policy can only visit
candidates that were actually generated. It cannot discover what a different
search would have found and this history never contained, so replay
systematically UNDERSTATES a genuinely better policy -- it measures how well a
policy selects from the recorded past, not how well it explores an open
future. That makes it a screening device, not a verdict: cheap enough to try
hundreds of policies, and never a substitute for deploying the survivor.
"""
from __future__ import annotations

import glob
import math
from dataclasses import dataclass, field

import numpy as np

from .config import resolve

# Dream-RSI's replay objective: quality, minus what it cost to get there.
BETA_COST = 0.0008          # per generation-evaluation attempt
BETA_PARALLEL = 0.0         # this search is sequential; no bonus to give


@dataclass
class Node:
    """One recorded attempt. Scores are forward outcomes, not fitness."""
    fingerprint: str
    method: str
    operator: str
    generation: int
    fitness: float
    fwd_alpha: float
    fwd_t: float
    gates_passed: int
    gates_total: int
    families: tuple = ()

    @property
    def score(self) -> float:
        """What the search was trying to produce, on the window it never saw."""
        return self.fwd_alpha if np.isfinite(self.fwd_alpha) else 0.0


@dataclass
class World:
    """A replayable discovery history, indexed by the decisions a policy makes."""
    nodes: list
    by_method: dict = field(default_factory=dict)
    run_id: str = ""

    @classmethod
    def build(cls, nodes, run_id="") -> "World":
        by = {}
        for n in nodes:
            by.setdefault(n.method, []).append(n)
        for m in by:
            by[m].sort(key=lambda n: n.generation)
        return cls(nodes=list(nodes), by_method=by, run_id=run_id)

    @property
    def methods(self) -> list:
        return sorted(self.by_method)

    def live_methods(self, cursor: dict) -> list:
        """Methods that still have an unseen node left to give."""
        return [m for m in self.methods
                if cursor.get(m, 0) < len(self.by_method[m])]

    def best_possible(self) -> float:
        return max((n.score for n in self.nodes), default=0.0)


def load_worlds(min_nodes: int = 20, n_worlds: int = 6) -> list:
    """Replayable worlds pooled from every recorded run.

    A world must contain at least two methods or there is no allocation
    decision to evaluate and every policy trivially ties.
    """
    import pandas as pd

    files = sorted(glob.glob(str(resolve("data/experience") / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not len(df):
        return []

    # The same candidate is recorded once per generation it survived in. For a
    # discovery tree each attempt should appear once, or a policy is credited
    # repeatedly for one piece of luck.
    df = df.sort_values("generation").drop_duplicates(
        subset=["run_id", "fingerprint"], keep="first")

    # ONE RUN USES ONE METHOD. run_evolve draws a method at startup and keeps
    # it for the whole run, so a per-run world offers a policy no choice at
    # all -- every policy scores identically because there is nothing to
    # allocate. The world has to pool runs so the decision "which method
    # deserves the next evaluation" becomes answerable.
    #
    # Worlds are then time slices of the pooled history. Several are needed
    # because a single one would rank policies on one arrangement of the past.
    df = df.sort_values(["generation", "epoch"]).reset_index(drop=True)
    # np.array_split on a DataFrame returns ndarrays and loses the columns;
    # split the index instead so each slice is still a frame.
    cuts = np.array_split(np.arange(len(df)), max(1, n_worlds))
    slices = [df.iloc[c] for c in cuts if len(c)]

    worlds = []
    for wi, g in enumerate(slices):
        if len(g) < min_nodes or g["method_id"].nunique() < 2:
            continue
        nodes = [Node(
            fingerprint=str(r.fingerprint), method=str(r.method_id),
            operator=str(r.operator),
            generation=int(r.generation or 0),
            fitness=float(r.fitness or 0.0),
            fwd_alpha=float(r.fwd_alpha) if r.fwd_alpha is not None else float("nan"),
            fwd_t=float(r.fwd_t) if r.fwd_t is not None else float("nan"),
            gates_passed=int(r.gates_passed or 0),
            gates_total=int(r.gates_total or 0),
            families=tuple(str(r.families or "").split(",")),
        ) for r in g.itertuples()]
        worlds.append(World.build(nodes, run_id="slice-%d" % wi))
    return worlds


# =============================================================================
#  Policies
# =============================================================================
class Policy:
    """Chooses which method to draw the next attempt from.

    Deliberately narrow. Dream-RSI lets an LLM rewrite the whole exploration
    policy as code; here the decision is which of the recorded search methods
    to spend the next evaluation on, because that is the only choice the
    recorded history can actually answer counterfactually. A policy that
    wanted to try an unrecorded method would be asking the replay a question
    it has no data for, and it would get a confident wrong answer.
    """
    name = "base"

    def reset(self):
        pass

    def choose(self, world: World, seen: dict, rng) -> str | None:
        raise NotImplementedError


class RoundRobin(Policy):
    name = "round_robin"

    def __init__(self):
        self.i = 0

    def reset(self):
        self.i = 0

    def choose(self, world, seen, rng):
        ms = world.methods
        if not ms:
            return None
        m = ms[self.i % len(ms)]
        self.i += 1
        return m


class Greedy(Policy):
    """Always spend on whichever method has produced the best node so far."""
    name = "greedy"

    def choose(self, world, seen, rng):
        best, bm = -np.inf, None
        for m, taken in seen.items():
            if taken:
                s = max(n.score for n in taken)
                if s > best:
                    best, bm = s, m
        # Nothing tried yet, or a tie: fall back to something unexplored.
        untried = [m for m in world.methods if not seen.get(m)]
        return untried[0] if untried else bm


class EpsilonGreedy(Policy):
    name = "epsilon_greedy"

    def __init__(self, eps=0.2):
        self.eps = eps

    def choose(self, world, seen, rng):
        if rng.random() < self.eps or not any(seen.values()):
            ms = world.methods
            return ms[int(rng.integers(len(ms)))] if ms else None
        return Greedy().choose(world, seen, rng)


class ThompsonLike(Policy):
    """What the live system does: sample each method's posterior, take the max.

    Included as the incumbent. A replay in which nothing beats this is a
    replay saying the current search policy is already reasonable, which is a
    useful answer and not a failed experiment.
    """
    name = "thompson"

    def choose(self, world, seen, rng):
        best, bm = -np.inf, None
        for m in world.methods:
            taken = seen.get(m) or []
            wins = sum(1 for n in taken if n.score > 0)
            draw = rng.beta(1.0 + wins, 1.0 + max(len(taken) - wins, 0))
            if draw > best:
                best, bm = draw, m
        return bm


POLICIES = [RoundRobin, Greedy, EpsilonGreedy, ThompsonLike]


# =============================================================================
#  Replay
# =============================================================================
def replay(policy: Policy, world: World, budget: int = 200,
           seed: int = 0) -> dict:
    """Spend `budget` evaluations under `policy`, using only recorded outcomes.

    Each decision draws the next unseen node from the chosen method, in the
    order it was actually generated. Nothing new is produced and nothing is
    re-run: this is reading, which is the entire point.
    """
    rng = np.random.default_rng(seed)
    policy.reset()
    seen: dict = {m: [] for m in world.methods}
    cursor = {m: 0 for m in world.methods}
    used = 0

    while used < budget:
        # Exhausted methods are hidden from the policy rather than refused
        # afterwards. Letting a greedy policy keep choosing a method that has
        # nothing left spins forever: `continue` costs no budget, so the loop
        # never terminates and never errors -- it just stops.
        live = world.live_methods(cursor)
        if not live:
            break
        view = World(nodes=world.nodes,
                     by_method={m: world.by_method[m] for m in live},
                     run_id=world.run_id)
        m = policy.choose(view, {k: seen[k] for k in live}, rng)
        if m is None or m not in view.by_method:
            break
        seen[m].append(world.by_method[m][cursor[m]])
        cursor[m] += 1
        used += 1

    taken = [n for v in seen.values() for n in v]
    best = max((n.score for n in taken), default=0.0)
    return {
        "policy": policy.name, "run_id": world.run_id,
        "evaluations": used,
        "best_score": best,
        "value": best - BETA_COST * used,          # the Dream-RSI objective
        "best_possible": world.best_possible(),
        "regret": world.best_possible() - best,
        "promoted": sum(1 for n in taken if n.gates_passed == n.gates_total
                        and n.gates_total > 0),
        "by_method": {m: len(v) for m, v in seen.items()},
    }


def evaluate_all(worlds, budget: int = 200, seeds: int = 8) -> list:
    """Every policy against every world, averaged over seeds.

    Averaging matters: two of these policies are stochastic, and a single seed
    would rank them on one draw of the random number generator.
    """
    out = []
    for cls in POLICIES:
        rows = []
        for w in worlds:
            for s in range(seeds):
                rows.append(replay(cls(), w, budget=budget, seed=s))
        if not rows:
            continue
        out.append({
            "policy": cls.name,
            "worlds": len(worlds),
            "runs": len(rows),
            "mean_value": float(np.mean([r["value"] for r in rows])),
            "mean_best": float(np.mean([r["best_score"] for r in rows])),
            "mean_regret": float(np.mean([r["regret"] for r in rows])),
            "mean_evals": float(np.mean([r["evaluations"] for r in rows])),
            "mean_promoted": float(np.mean([r["promoted"] for r in rows])),
        })
    out.sort(key=lambda r: -r["mean_value"])
    return out


def summary(worlds) -> dict:
    n_nodes = sum(len(w.nodes) for w in worlds)
    methods = sorted({m for w in worlds for m in w.methods})
    scored = [n.score for w in worlds for n in w.nodes]
    return {
        "worlds": len(worlds), "nodes": n_nodes, "methods": methods,
        "best_recorded": max(scored, default=0.0),
        "positive_forward": sum(1 for s in scored if s > 0),
        "share_positive": (sum(1 for s in scored if s > 0) / len(scored)
                           if scored else 0.0),
    }
