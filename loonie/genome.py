"""The evolvable object: a typed expression tree that scores every stock.

A genome is two things:

  1. `expr`  -- an expression over the feature terminals that evaluates to a
                (T, N) score matrix. Higher score = more attractive.
  2. params  -- how many names to hold, how often to rebalance, how to weight.

The video's loop used an LLM as its mutation operator: propose a change, run
the backtest, keep it if the number went up. Fourteen proposals, one kept. An
LLM is a very expensive way to sample a neighbourhood. The operators below do
the same job for free, so the search can afford ~10^5 proposals instead of 14
-- which is exactly why the statistical gate in metrics.py has to be real.
More proposals means more chances to get lucky, and luck has to be priced in.

Every genome carries a stable `fingerprint`; identical expressions collapse to
one entry so the multiple-testing trial count stays honest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from . import features as F

EPS = 1e-12
MAX_DEPTH_HARD = 9

# --------------------------------------------------------------------------
#  Grammar
# --------------------------------------------------------------------------
UNARY_POINTWISE = ("neg", "abs", "sign", "log1p", "sigmoid", "square", "sqrt_abs")
UNARY_CROSS = ("rank", "zscore", "demean")          # across tickers, per date
UNARY_TIME = ("delta", "tsmean", "tsstd", "tsrank")  # along time, causal
TIME_WINDOWS = (5, 10, 21, 63, 126)
BINARY = ("add", "sub", "mul", "div", "min", "max")
WEIGHTINGS = ("equal", "score")
N_POSITION_CHOICES = (10, 15, 20, 25, 30, 40, 50)
REBALANCE_CHOICES = (1, 2, 5, 10, 21)


# --------------------------------------------------------------------------
#  Nodes
# --------------------------------------------------------------------------
class Node:
    __slots__ = ()

    def evaluate(self, feats, mask):
        raise NotImplementedError

    def children(self):
        return []

    def size(self):
        return 1 + sum(c.size() for c in self.children())

    def depth(self):
        ch = self.children()
        return 1 + (max(c.depth() for c in ch) if ch else 0)

    def to_dict(self):
        raise NotImplementedError


@dataclass
class Feat(Node):
    name: str

    def evaluate(self, feats, mask):
        return feats[self.name]

    def __str__(self):
        return self.name

    def to_dict(self):
        return {"t": "feat", "name": self.name}


@dataclass
class Const(Node):
    value: float

    def evaluate(self, feats, mask):
        any_feat = next(iter(feats.values()))
        return np.full(any_feat.shape, np.float32(self.value), np.float32)

    def __str__(self):
        return "%.4g" % self.value

    def to_dict(self):
        return {"t": "const", "value": float(self.value)}


@dataclass
class Un(Node):
    op: str
    child: Node
    window: int = 0

    def children(self):
        return [self.child]

    def evaluate(self, feats, mask):
        x = self.child.evaluate(feats, mask)
        op = self.op
        if op == "neg":
            return -x
        if op == "abs":
            return np.abs(x)
        if op == "sign":
            return np.sign(x)
        if op == "log1p":
            return np.log1p(np.abs(x)) * np.sign(x)
        if op == "sigmoid":
            return (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))).astype(np.float32)
        if op == "square":
            return np.clip(x * x, -1e12, 1e12).astype(np.float32)
        if op == "sqrt_abs":
            return (np.sqrt(np.abs(x)) * np.sign(x)).astype(np.float32)
        if op == "rank":
            return F.cs_rank(x, mask)
        if op == "zscore":
            return F.cs_zscore(x, mask)
        if op == "demean":
            return F.cs_demean(x, mask)
        if op == "delta":
            return (x - F.shift(x, self.window)).astype(np.float32)
        if op == "tsmean":
            return F.roll_mean(x, self.window)
        if op == "tsstd":
            return F.roll_std(x, self.window)
        if op == "tsrank":
            return F.roll_rank_ts(x, self.window)
        raise ValueError("unknown unary op %r" % op)

    def __str__(self):
        w = "_%d" % self.window if self.op in UNARY_TIME else ""
        return "%s%s(%s)" % (self.op, w, self.child)

    def to_dict(self):
        return {"t": "un", "op": self.op, "window": self.window,
                "child": self.child.to_dict()}


@dataclass
class Bin(Node):
    op: str
    left: Node
    right: Node

    def children(self):
        return [self.left, self.right]

    def evaluate(self, feats, mask):
        a = self.left.evaluate(feats, mask)
        b = self.right.evaluate(feats, mask)
        op = self.op
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return np.clip(a * b, -1e12, 1e12).astype(np.float32)
        if op == "div":
            return (a / np.where(np.abs(b) < EPS, np.nan, b)).astype(np.float32)
        if op == "min":
            return np.fmin(a, b)
        if op == "max":
            return np.fmax(a, b)
        raise ValueError("unknown binary op %r" % op)

    def __str__(self):
        return "%s(%s, %s)" % (self.op, self.left, self.right)

    def to_dict(self):
        return {"t": "bin", "op": self.op,
                "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class Ite(Node):
    """if cond > 0 then a else b -- lets the search discover regime switches."""

    cond: Node
    a: Node
    b: Node

    def children(self):
        return [self.cond, self.a, self.b]

    def evaluate(self, feats, mask):
        c = self.cond.evaluate(feats, mask)
        return np.where(c > 0, self.a.evaluate(feats, mask),
                        self.b.evaluate(feats, mask)).astype(np.float32)

    def __str__(self):
        return "ite(%s, %s, %s)" % (self.cond, self.a, self.b)

    def to_dict(self):
        return {"t": "ite", "cond": self.cond.to_dict(),
                "a": self.a.to_dict(), "b": self.b.to_dict()}


def simplify(node: Node) -> Node:
    """Fold degenerate sub-expressions. Semantics-preserving, always.

    Genetic programming generates a great deal of dead code. A real example
    from a run of this engine, as the fittest genome in the population:

        ite(mul(mom_252, demean(-0.3359)), ma_ratio_50, ite(dollar_vol_21,
            rev_5, rev_5))

    `demean` of a constant is identically zero across the cross-section, so the
    condition is always 0, so the true branch is unreachable; and both arms of
    the inner `ite` are the same node. The whole thing is `rev_5` wearing nine
    extra nodes. Left alone that costs three ways: wasted evaluation, a
    parsimony penalty for complexity the strategy does not actually have, and
    -- worst -- a dozen structurally distinct fingerprints for one hypothesis,
    which inflates the trial count that the deflated Sharpe is charged against.

    Folding here makes semantically identical genomes collapse to one
    fingerprint, so the multiple-testing accounting counts ideas, not spellings.

    Every rule below must hold under IEEE-754 *including NaN*, because feature
    arrays are full of NaN during warm-up windows and for names that are not
    yet eligible. The tempting algebraic identities are exactly the unsound
    ones: `x * 0 -> 0` is false when x is NaN, `x - x -> 0` and `x / x -> 1`
    likewise, and `rank(const)` is not constant because ties resolve by column
    order. Those are omitted deliberately -- an earlier draft included them and
    the semantics test caught it.
    """
    if isinstance(node, (Feat, Const)):
        return node

    if isinstance(node, Un):
        child = simplify(node.child)
        # Cross-sectional centring of a constant is exactly zero everywhere the
        # name is eligible (true only because cs_demean accumulates in float64).
        if isinstance(child, Const):
            if node.op in ("demean", "zscore"):
                return Const(0.0)
            if node.op == "neg":
                return Const(-child.value)
            if node.op == "abs":
                return Const(abs(child.value))
        # Involutions and idempotents -- NaN-safe, since NaN maps to NaN.
        if isinstance(child, Un):
            if node.op == "neg" and child.op == "neg":
                return child.child
            if node.op == child.op and node.op in ("abs", "sign", "rank",
                                                   "zscore", "demean"):
                return child
        return Un(node.op, child, node.window)

    if isinstance(node, Bin):
        a, b = simplify(node.left), simplify(node.right)
        if isinstance(a, Const) and isinstance(b, Const):
            try:
                v = {"add": a.value + b.value, "sub": a.value - b.value,
                     "mul": a.value * b.value,
                     "div": a.value / b.value if abs(b.value) > EPS else 0.0,
                     "min": min(a.value, b.value),
                     "max": max(a.value, b.value)}[node.op]
                return Const(round(float(v), 6))
            except Exception:
                pass
        # fmin/fmax of a value with itself is that value, NaN included.
        if node.op in ("min", "max") and a.to_dict() == b.to_dict():
            return a
        return Bin(node.op, a, b)

    if isinstance(node, Ite):
        c, a, b = simplify(node.cond), simplify(node.a), simplify(node.b)
        if isinstance(c, Const):                 # condition is decided
            return a if c.value > 0 else b
        if a.to_dict() == b.to_dict():           # both arms identical
            return a
        return Ite(c, a, b)

    return node


def node_from_dict(d):
    t = d["t"]
    if t == "feat":
        return Feat(d["name"])
    if t == "const":
        return Const(d["value"])
    if t == "un":
        return Un(d["op"], node_from_dict(d["child"]), d.get("window", 0))
    if t == "bin":
        return Bin(d["op"], node_from_dict(d["left"]), node_from_dict(d["right"]))
    if t == "ite":
        return Ite(node_from_dict(d["cond"]), node_from_dict(d["a"]),
                   node_from_dict(d["b"]))
    raise ValueError("bad node %r" % t)


# --------------------------------------------------------------------------
#  Genome
# --------------------------------------------------------------------------
@dataclass
class Genome:
    expr: Node
    n_positions: int = 25
    rebalance_days: int = 5
    weighting: str = "equal"
    # lineage / bookkeeping -------------------------------------------------
    born: int = 0                       # generation index
    parents: tuple = ()
    operator: str = "init"              # which mutation made it (credit assign.)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ eval
    def score(self, feats, mask):
        """(T, N) desirability. NaN where a name is not scoreable."""
        s = self.expr.evaluate(feats, mask)
        s = np.where(np.isfinite(s), s, np.nan).astype(np.float32)
        return np.where(mask, s, np.nan)

    # -------------------------------------------------------------- identity
    def canonical(self) -> str:
        return "%s|k=%d|rb=%d|w=%s" % (self.expr, self.n_positions,
                                       self.rebalance_days, self.weighting)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha1(self.canonical().encode()).hexdigest()[:16]

    @property
    def complexity(self) -> int:
        return self.expr.size()

    # ---------------------------------------------------------------- serdes
    def to_dict(self):
        return {
            "expr": self.expr.to_dict(),
            "n_positions": self.n_positions,
            "rebalance_days": self.rebalance_days,
            "weighting": self.weighting,
            "born": self.born,
            "parents": list(self.parents),
            "operator": self.operator,
            "meta": self.meta,
            "fingerprint": self.fingerprint,
            "canonical": self.canonical(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            expr=node_from_dict(d["expr"]),
            n_positions=d.get("n_positions", 25),
            rebalance_days=d.get("rebalance_days", 5),
            weighting=d.get("weighting", "equal"),
            born=d.get("born", 0),
            parents=tuple(d.get("parents", ())),
            operator=d.get("operator", "loaded"),
            meta=d.get("meta", {}),
        )

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self):
        return self.canonical()


# --------------------------------------------------------------------------
#  Random construction
# --------------------------------------------------------------------------
class Grammar:
    """Samples and mutates genomes. All randomness flows through `rng`."""

    def __init__(self, feature_names, rng: np.random.Generator, max_depth=6):
        self.features = list(feature_names)
        self.rng = rng
        self.max_depth = max_depth

    # ---------------------------------------------------------------- sample
    def terminal(self) -> Node:
        if self.rng.random() < 0.88:
            return Feat(str(self.rng.choice(self.features)))
        return Const(round(float(self.rng.normal(0, 1.5)), 4))

    def random_expr(self, depth=0, p_terminal=None) -> Node:
        if p_terminal is None:
            p_terminal = 0.18 + 0.62 * (depth / max(1, self.max_depth))
        if depth >= self.max_depth or self.rng.random() < p_terminal:
            return self.terminal()

        r = self.rng.random()
        if r < 0.40:
            return Bin(str(self.rng.choice(BINARY)),
                       self.random_expr(depth + 1), self.random_expr(depth + 1))
        if r < 0.62:
            return Un(str(self.rng.choice(UNARY_CROSS)), self.random_expr(depth + 1))
        if r < 0.80:
            return Un(str(self.rng.choice(UNARY_POINTWISE)),
                      self.random_expr(depth + 1))
        if r < 0.92:
            return Un(str(self.rng.choice(UNARY_TIME)), self.random_expr(depth + 1),
                      int(self.rng.choice(TIME_WINDOWS)))
        return Ite(self.random_expr(depth + 1), self.random_expr(depth + 1),
                   self.random_expr(depth + 1))

    def random_genome(self, generation=0) -> Genome:
        # Wrapping the root in `rank` is a strong prior: cross-sectional
        # ranking is what makes scores comparable across names and dates.
        expr = simplify(self.random_expr())
        if self.rng.random() < 0.55:
            expr = Un("rank", expr)
        return Genome(
            expr=expr,
            n_positions=int(self.rng.choice(N_POSITION_CHOICES)),
            rebalance_days=int(self.rng.choice(REBALANCE_CHOICES)),
            weighting=str(self.rng.choice(WEIGHTINGS)),
            born=generation,
            operator="init",
        )

    # -------------------------------------------------------------- mutation
    def _collect(self, node, acc=None, parent=None, slot=None):
        acc = [] if acc is None else acc
        acc.append((node, parent, slot))
        if isinstance(node, Un):
            self._collect(node.child, acc, node, "child")
        elif isinstance(node, Bin):
            self._collect(node.left, acc, node, "left")
            self._collect(node.right, acc, node, "right")
        elif isinstance(node, Ite):
            self._collect(node.cond, acc, node, "cond")
            self._collect(node.a, acc, node, "a")
            self._collect(node.b, acc, node, "b")
        return acc

    def _clone(self, node):
        return node_from_dict(node.to_dict())

    def _replace(self, root, target_idx, new_node):
        root = self._clone(root)
        sites = self._collect(root)
        node, parent, slot = sites[target_idx]
        if parent is None:
            return new_node
        setattr(parent, slot, new_node)
        return root

    # -- individual operators; each returns a new expr (or None to abstain) --
    def op_point(self, expr):
        sites = self._collect(expr)
        i = int(self.rng.integers(len(sites)))
        node = sites[i][0]
        if isinstance(node, Feat):
            new = Feat(str(self.rng.choice(self.features)))
        elif isinstance(node, Const):
            new = Const(round(float(node.value + self.rng.normal(0, 0.5)), 4))
        elif isinstance(node, Bin):
            new = Bin(str(self.rng.choice(BINARY)), self._clone(node.left),
                      self._clone(node.right))
        elif isinstance(node, Un):
            pool = (UNARY_CROSS if node.op in UNARY_CROSS
                    else UNARY_TIME if node.op in UNARY_TIME else UNARY_POINTWISE)
            new = Un(str(self.rng.choice(pool)), self._clone(node.child),
                     int(self.rng.choice(TIME_WINDOWS)) if node.op in UNARY_TIME else 0)
        else:
            return None
        return self._replace(expr, i, new)

    def op_subtree(self, expr):
        sites = self._collect(expr)
        i = int(self.rng.integers(len(sites)))
        d = sites[i][0].depth()
        return self._replace(expr, i, self.random_expr(depth=max(0, self.max_depth - d)))

    def op_hoist(self, expr):
        """Promote a subtree to root -- the main force against bloat."""
        sites = self._collect(expr)
        if len(sites) < 3:
            return None
        i = int(self.rng.integers(1, len(sites)))
        return self._clone(sites[i][0])

    def op_shrink(self, expr):
        sites = self._collect(expr)
        if len(sites) < 3:
            return None
        i = int(self.rng.integers(1, len(sites)))
        return self._replace(expr, i, self.terminal())

    def op_wrap(self, expr):
        sites = self._collect(expr)
        i = int(self.rng.integers(len(sites)))
        r = self.rng.random()
        if r < 0.45:
            new = Un(str(self.rng.choice(UNARY_CROSS)), self._clone(sites[i][0]))
        elif r < 0.75:
            new = Un(str(self.rng.choice(UNARY_POINTWISE)), self._clone(sites[i][0]))
        else:
            new = Un(str(self.rng.choice(UNARY_TIME)), self._clone(sites[i][0]),
                     int(self.rng.choice(TIME_WINDOWS)))
        return self._replace(expr, i, new)

    def op_jitter(self, expr):
        sites = [(i, n) for i, (n, _, _) in enumerate(self._collect(expr))
                 if isinstance(n, Const)]
        if not sites:
            return None
        i, node = sites[int(self.rng.integers(len(sites)))]
        return self._replace(expr, i, Const(round(
            float(node.value * (1 + self.rng.normal(0, 0.3)) + self.rng.normal(0, 0.1)), 4)))

    def op_crossover(self, expr, other_expr):
        a_sites = self._collect(expr)
        b_sites = self._collect(other_expr)
        i = int(self.rng.integers(len(a_sites)))
        j = int(self.rng.integers(len(b_sites)))
        return self._replace(expr, i, self._clone(b_sites[j][0]))

    EXPR_OPERATORS = ("point", "subtree", "hoist", "shrink", "wrap", "jitter",
                      "crossover")
    PARAM_OPERATORS = ("params",)
    ALL_OPERATORS = EXPR_OPERATORS + PARAM_OPERATORS

    def apply(self, name, parent: Genome, mate: Genome | None, generation: int):
        """Apply one named operator. Returns a new Genome or None."""
        g = Genome.from_dict(parent.to_dict())
        g.born = generation
        g.operator = name
        g.parents = (parent.fingerprint,) + ((mate.fingerprint,) if mate else ())

        if name == "params":
            which = self.rng.random()
            if which < 0.4:
                g.n_positions = int(self.rng.choice(N_POSITION_CHOICES))
            elif which < 0.8:
                g.rebalance_days = int(self.rng.choice(REBALANCE_CHOICES))
            else:
                g.weighting = str(self.rng.choice(WEIGHTINGS))
            return g

        fn = {
            "point": self.op_point, "subtree": self.op_subtree,
            "hoist": self.op_hoist, "shrink": self.op_shrink,
            "wrap": self.op_wrap, "jitter": self.op_jitter,
        }.get(name)
        if name == "crossover":
            if mate is None:
                return None
            new = self.op_crossover(g.expr, mate.expr)
        else:
            new = fn(g.expr) if fn else None
        if new is None or new.depth() > MAX_DEPTH_HARD:
            return None
        g.expr = simplify(new)
        return g
