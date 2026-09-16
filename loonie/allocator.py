"""Capital allocation across live strategies -- the outer learning loop.

Evolution learns from history. This learns from fills.

Every promoted genome runs in paper and accrues a live track record. A Normal-
inverse-gamma posterior over each strategy's daily *excess* return is updated
from those realised returns, and weights come from Thompson sampling: draw one
plausible mean per strategy, allocate to the draws. Strategies that keep
delivering get sampled high more often; strategies that decay get squeezed out
without anyone deciding to fire them.

Two deliberate properties:

  * Observations are exponentially discounted (`halflife_days`), so the
    allocator tracks a changing market instead of averaging over a regime that
    ended in 2019.
  * Uncertainty earns capital. A strategy with three days of history has a wide
    posterior and will occasionally draw high -- that is exploration, and it is
    the only way a new promotion ever gets a chance against an incumbent.

This is the loop that makes the system self-correcting in production rather
than only at search time. Backtest alpha is a hypothesis; this prices it
against what actually filled.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import resolve

STATE = "state/allocator.json"


@dataclass
class Arm:
    """Normal-inverse-gamma posterior over one strategy's daily excess return."""

    key: str
    mu: float = 0.0          # posterior mean
    kappa: float = 1.0       # pseudo-observations behind mu
    alpha: float = 2.0       # inverse-gamma shape
    beta: float = 1e-4       # inverse-gamma scale
    n: int = 0
    last_update: str = ""
    cumulative: float = 0.0
    history: list = field(default_factory=list)

    def update(self, x: float, decay: float = 1.0):
        """Bayesian update with exponentially discounted prior mass."""
        self.kappa *= decay
        self.alpha = 2.0 + (self.alpha - 2.0) * decay
        self.beta *= decay

        k0, mu0 = self.kappa, self.mu
        self.kappa = k0 + 1.0
        self.mu = (k0 * mu0 + x) / self.kappa
        self.alpha += 0.5
        self.beta += 0.5 * (k0 / self.kappa) * (x - mu0) ** 2

        self.n += 1
        self.cumulative += x
        self.last_update = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.history.append(round(float(x), 8))
        self.history = self.history[-500:]

    def sample(self, rng: np.random.Generator) -> float:
        """One draw of the plausible mean -- the Thompson step."""
        var = self.beta / max(self.alpha, 1e-6)
        scale = math.sqrt(max(var / max(self.kappa, 1e-6), 1e-18))
        df = max(2.0 * self.alpha, 2.1)
        return float(self.mu + scale * rng.standard_t(df))

    def stats(self):
        var = self.beta / max(self.alpha, 1e-6)
        sd = math.sqrt(max(var, 0.0))
        return {
            "key": self.key, "n": self.n, "mean_daily": self.mu,
            "sd_daily": sd, "cumulative": self.cumulative,
            "sharpe_ann": (self.mu / sd * math.sqrt(252.0)) if sd > 1e-12 else 0.0,
            "last_update": self.last_update,
        }

    def to_dict(self):
        return dict(self.__dict__)


class Allocator:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.a = cfg.allocator
        self.rng = np.random.default_rng(seed)
        self.arms: dict = {}
        self.load()

    # ------------------------------------------------------------------ arms
    def register(self, key: str):
        if key not in self.arms:
            self.arms[key] = Arm(
                key=key, mu=float(self.a.prior_mean),
                beta=float(self.a.prior_std) ** 2 * 2.0)
        return self.arms[key]

    def observe(self, key: str, daily_excess_return: float):
        """Record one realised day of excess return for a live strategy."""
        arm = self.register(key)
        hl = max(float(self.a.halflife_days), 1.0)
        arm.update(float(daily_excess_return), decay=0.5 ** (1.0 / hl))
        self.save()

    # --------------------------------------------------------------- weights
    def weights(self, keys=None, n_draws: int = 400) -> dict:
        """Thompson-sampled capital weights over the registered strategies.

        Weight = the fraction of draws in which that strategy looked best,
        clipped to [min_weight, max_weight] and renormalised.
        """
        keys = list(keys or self.arms.keys())
        keys = [k for k in keys if k in self.arms]
        if not keys:
            return {}
        if len(keys) == 1:
            return {keys[0]: 1.0}

        wins = {k: 0 for k in keys}
        for _ in range(n_draws):
            draws = {k: self.arms[k].sample(self.rng) for k in keys}
            best = max(draws, key=draws.get)
            # Only back a positive expectation; otherwise the draw sits out.
            if draws[best] > 0:
                wins[best] += 1

        total = sum(wins.values())
        if total == 0:
            return {k: 0.0 for k in keys}   # nothing looks live -> hold cash

        lo, hi = float(self.a.min_weight), float(self.a.max_weight)
        w = {k: wins[k] / total for k in keys}
        w = {k: min(max(v, lo), hi) for k, v in w.items()}
        s = sum(w.values())
        return {k: v / s for k, v in w.items()} if s > 0 else {k: 0.0 for k in keys}

    def report(self):
        rows = [self.arms[k].stats() for k in self.arms]
        rows.sort(key=lambda r: -r["cumulative"])
        return rows

    # ----------------------------------------------------------------- state
    def save(self):
        p = resolve(STATE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"arms": {k: a.to_dict() for k, a in self.arms.items()},
             "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            indent=2), encoding="utf-8")

    def load(self):
        p = resolve(STATE)
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for k, v in d.get("arms", {}).items():
            v.pop("key", None)
            self.arms[k] = Arm(key=k, **v)
