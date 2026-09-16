"""A bandit over ways of searching, credited by what actually generalises.

Three learning loops already run here: which mutation operators help, which
feature families survive forward, which live strategies deserve capital. All
three optimise *within* one method -- typed genetic programming, MAP-Elites
archive, information-ratio fitness. None of them can ask whether that method
is the right one, and it was chosen by a person on the first afternoon with no
evidence at all.

This is that question. A method here is a real, behaviour-changing
configuration of the search: what fitness rewards, whether quality-diversity
is used, how hard complexity is punished, how wide the exploration runs. Each
search cycle draws one by Thompson sampling, runs under it, and is credited by
the *forward* outcomes it produced -- read from the experience store, which
survives restarts and accumulates across runs.

Credit is forward validation, never fitness, for the same reason the feature
bandit uses it: a method that scores its own candidates generously would
otherwise win by inflating the number it is judged on. Forward alpha is the one
quantity no method can tune, because none of them can see the window it is
measured on.

The honest caveat: with six methods and a handful of labelled outcomes each,
this is a weak signal for a long time. It is deliberately floored so nothing
starves, and it reports its own sample size so a reader can tell whether the
posterior means anything yet. Early on it will not. That is not a reason to
pick the method by hand and call it settled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import resolve

STATE = "state/methods.json"


@dataclass
class Method:
    """A named, behaviour-changing configuration of the search."""

    id: str
    summary: str
    rationale: str
    delta: dict = field(default_factory=dict)

    def apply(self, cfg) -> dict:
        """Return a deep-copied config with this method's changes applied."""
        out = json.loads(json.dumps(dict(cfg), default=str))
        for dotted, value in self.delta.items():
            node = out
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return out


# The registry. Each entry changes what the search actually does -- these are
# not relabelled copies of one setup.
REGISTRY = [
    Method(
        "qd_ir",
        "MAP-Elites + information-ratio fitness",
        "The incumbent. Quality-diversity over turnover/correlation/complexity, "
        "fitness = mean IR scaled by fold consistency. Everything else is "
        "measured against this.",
        {"evolve.fitness_mode": "ir", "evolve.use_map_elites": True,
         "evolve.parsimony_penalty": 0.01},
    ),
    Method(
        "qd_consistency",
        "MAP-Elites + consistency-dominant fitness",
        "Squares the fold-consistency term instead of scaling by it. A strategy "
        "that wins in six folds of eight beats one that wins enormously in two "
        "-- the profile that has repeatedly failed forward here is the second.",
        {"evolve.fitness_mode": "consistency", "evolve.use_map_elites": True,
         "evolve.parsimony_penalty": 0.01},
    ),
    Method(
        "elitist_ir",
        "plain elitism, no quality-diversity",
        "Drops MAP-Elites entirely. QD is assumed to prevent lineage collapse, "
        "but that assumption has never been tested against its own absence on "
        "this data, and it costs population slots to maintain.",
        {"evolve.fitness_mode": "ir", "evolve.use_map_elites": False,
         "evolve.parsimony_penalty": 0.01},
    ),
    Method(
        "parsimony_hard",
        "heavy complexity penalty",
        "Ten times the parsimony pressure and a shallower tree. Simpler "
        "expressions have fewer ways to fit noise; if generalisation is the "
        "binding constraint this should win outright.",
        {"evolve.fitness_mode": "ir", "evolve.use_map_elites": True,
         "evolve.parsimony_penalty": 0.10, "evolve.max_tree_depth": 4},
    ),
    Method(
        "low_turnover",
        "turnover-averse",
        "Caps turnover at 8x and triples the cost stress. Most of the alpha "
        "found so far sits in high-turnover reversal, which is exactly where a "
        "wrong cost assumption does the most damage.",
        {"evolve.fitness_mode": "ir", "evolve.use_map_elites": True,
         "evolve.gate.max_annual_turnover": 8.0,
         "evolve.cost_stress_multiplier": 9.0},
    ),
    Method(
        "novelty_search",
        "novelty over objective (Stanley & Lehman)",
        "Fitness is distance to the k nearest neighbours in behaviour space; "
        "alpha is only a tiebreak. 'Why Greatness Cannot Be Planned' argues "
        "that on deceptive problems most beacons lead you astray, and that "
        "rewarding difference beats rewarding the goal. Every other method here "
        "climbs toward alpha; this one refuses to, and is still judged on "
        "forward alpha -- which is the only fair way to settle the question.",
        {"evolve.fitness_mode": "novelty", "evolve.use_map_elites": True,
         "evolve.parsimony_penalty": 0.01},
    ),
    Method(
        "mdl_parsimony",
        "description-length complexity penalty",
        "Replaces the hand-picked 0.01-per-node penalty with a BIC-style term, "
        "complexity * ln(n) / 2n, which scales with sample size the way a "
        "model-selection criterion should. The GP generalisation literature "
        "treats node-count parsimony as an algorithmic proxy and prefers "
        "description length; this tests whether the principled version is "
        "actually better here or just tidier.",
        {"evolve.fitness_mode": "ir", "evolve.use_map_elites": True,
         "evolve.parsimony_mode": "bic"},
    ),
    Method(
        "wide_explore",
        "high exploration, deeper trees",
        "Larger population, deeper grammar, more random injection. Tests "
        "whether the search is under-exploring rather than over-fitting -- the "
        "opposite diagnosis to parsimony_hard, and worth running against it.",
        {"evolve.fitness_mode": "ir", "evolve.use_map_elites": True,
         "evolve.max_tree_depth": 7, "evolve.population": 320,
         "evolve.parsimony_penalty": 0.004},
    ),
]

BY_ID = {m.id: m for m in REGISTRY}


class MethodBandit:
    """Thompson sampling over methods, credited by forward generalisation."""

    def __init__(self, floor: float = 0.05):
        self.a = {m.id: 1.0 for m in REGISTRY}
        self.b = {m.id: 1.0 for m in REGISTRY}
        self.seen = {m.id: 0 for m in REGISTRY}
        self.survived = {m.id: 0 for m in REGISTRY}
        self.fwd_alpha = {m.id: 0.0 for m in REGISTRY}
        self.cycles = {m.id: 0 for m in REGISTRY}
        self.floor = floor
        self.updated = None
        self.load()

    # ------------------------------------------------------------- crediting
    def ingest_experience(self) -> int:
        """Rebuild posteriors from every labelled row ever recorded.

        Recomputed from scratch rather than incremented, so the posterior always
        reflects the whole corpus and a restart cannot double-count.
        """
        from . import experience

        try:
            df = experience.load()
        except Exception:
            return 0
        if df.empty or "fwd_alpha" not in df or "method_id" not in df:
            return 0
        df = df[df["fwd_alpha"].notna()]
        if df.empty:
            return 0

        # One vote per distinct strategy per method. A leader that holds its
        # position for fifty generations is logged fifty times; counting each
        # as independent evidence would let a method manufacture confidence by
        # simply not improving. The claim being measured is "N distinct
        # strategies produced under this method generalised forward", not "N
        # generations elapsed".
        if "fingerprint" in df:
            df = df.drop_duplicates(subset=["method_id", "fingerprint"],
                                    keep="last")

        for mid in self.a:
            self.a[mid], self.b[mid] = 1.0, 1.0
            self.seen[mid] = self.survived[mid] = 0
            self.fwd_alpha[mid] = 0.0

        n = 0
        for mid, alpha in zip(df["method_id"], df["fwd_alpha"]):
            if mid not in self.a:
                continue
            ok = bool(alpha > 0)
            self.seen[mid] += 1
            self.fwd_alpha[mid] += float(alpha)
            if ok:
                self.a[mid] += 1.0
                self.survived[mid] += 1
            else:
                self.b[mid] += 1.0
            n += 1
        self.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save()
        return n

    # ------------------------------------------------------------- selecting
    def choose(self, rng=None) -> Method:
        rng = rng or np.random.default_rng()
        draws = {}
        for mid in self.a:
            d = float(rng.beta(self.a[mid], self.b[mid]))
            draws[mid] = self.floor + (1.0 - self.floor) * d
        # Untried methods get first refusal: a posterior built from zero
        # observations is a prior, and acting on it as evidence is how a
        # bandit convinces itself of something it never measured.
        untried = [m for m in self.a if self.seen[m] == 0]
        chosen = (str(rng.choice(untried)) if untried
                  else max(draws, key=draws.get))
        self.cycles[chosen] = self.cycles.get(chosen, 0) + 1
        self.save()
        return BY_ID[chosen]

    def table(self) -> list:
        rows = []
        for m in REGISTRY:
            seen = self.seen[m.id]
            rows.append({
                "id": m.id,
                "summary": m.summary,
                "posterior": self.a[m.id] / (self.a[m.id] + self.b[m.id]),
                "forward_rate": self.survived[m.id] / seen if seen else None,
                "mean_fwd_alpha": self.fwd_alpha[m.id] / seen if seen else None,
                "labelled": seen,
                "cycles": self.cycles.get(m.id, 0),
                # With a handful of observations the posterior is mostly prior.
                "established": seen >= 20,
            })
        rows.sort(key=lambda r: (-(r["forward_rate"] or -1), -r["posterior"]))
        return rows

    # ----------------------------------------------------------------- state
    def save(self):
        p = resolve(STATE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "a": self.a, "b": self.b, "seen": self.seen,
            "survived": self.survived, "fwd_alpha": self.fwd_alpha,
            "cycles": self.cycles, "updated": self.updated,
            "table": self.table(),
        }, indent=1), encoding="utf-8")

    def load(self):
        p = resolve(STATE)
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for k in ("a", "b", "seen", "survived", "fwd_alpha", "cycles"):
            getattr(self, k).update(d.get(k, {}))
        self.updated = d.get("updated")


def current() -> dict:
    """What the meta-learner currently believes, for the dashboard."""
    p = resolve(STATE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
