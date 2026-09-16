"""Everything the system has ever learned, kept.

Until now every run started blind. `state/evolve_state.json` holds the current
population and is overwritten continuously; `--fresh` wipes it. Three hundred
and forty thousand strategies had been evaluated by the time this was written
and not one of those evaluations survived anywhere a later run could read.

That is a waste of the only genuinely scarce thing here. Price history is free
and finite; what costs CPU-days to produce is the *labelled* pairing of a
strategy with how it actually generalised forward. That pairing is training
data, and this module is where it accumulates.

WHAT IS RECORDED, AND WHY NOT EVERYTHING. Writing all 340k evaluations would
be mostly noise: the great majority are random genomes that failed on the
first cheap gate and carry no forward label. What earns a row is a candidate
that reached the promotion gate -- because only those get a forward-validation
score, which is the label -- plus every archive elite at each checkpoint, and
every promotion and demotion with its reason. Small, dense, and every row has
an outcome attached.

WHY PARQUET. Columnar, compressed, and readable by anything. A later analysis
that wants "every strategy whose momentum family cleared the forward gate
after 2026-09" should be a filter, not a parse of a million-line JSON file.
Daily partitions so appends never rewrite history.

Nothing here feeds back into a live decision on its own. It is the substrate
methods.py uses to ask which way of searching has actually been working.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

from .config import resolve

DIR = "data/experience"
FLUSH_EVERY = 200          # rows buffered before a write
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class ExperienceStore:
    """Append-only, daily-partitioned record of evaluations and outcomes."""

    def __init__(self, run_id: str | None = None, method_id: str = "default",
                 context: dict | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.method_id = method_id
        self.context = context or {}
        self.buffer: list = []
        self.written = 0
        self.dir = resolve(DIR)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- recording
    def record(self, kind: str, genome, cv: dict | None = None,
               forward: dict | None = None, gates: list | None = None,
               **extra):
        """Buffer one row. `kind` is gated | elite | promoted | demoted."""
        cv = cv or {}
        forward = forward or {}
        gates = gates or []

        try:
            from . import features as featmod
            fams = sorted({featmod.family_of(n) for n in genome.feature_names()})
            nfeat = len(set(genome.feature_names()))
        except Exception:
            fams, nfeat = [], 0

        row = {
            "schema": SCHEMA_VERSION,
            "ts": _now(),
            "epoch": time.time(),
            "run_id": self.run_id,
            "method_id": self.method_id,
            "kind": kind,
            # ---- the strategy ------------------------------------------
            "fingerprint": getattr(genome, "fingerprint", None),
            "canonical": getattr(genome, "canonical", lambda: None)(),
            "expr_json": json.dumps(genome.to_dict().get("expr"), default=str),
            "complexity": getattr(genome, "complexity", None),
            "n_positions": getattr(genome, "n_positions", None),
            "rebalance_days": getattr(genome, "rebalance_days", None),
            "weighting": getattr(genome, "weighting", None),
            "operator": getattr(genome, "operator", None),
            "born_generation": getattr(genome, "born", None),
            "families": ",".join(fams),
            "n_features": nfeat,
            # ---- in-sample cross-validation ----------------------------
            "cv_mean_ir": cv.get("mean_ir"),
            "cv_mean_alpha": cv.get("mean_alpha"),
            "cv_alpha_t": cv.get("alpha_tstat"),
            "cv_frac_positive": cv.get("frac_positive"),
            "cv_worst_fold": cv.get("worst_fold_alpha"),
            "cv_dsr": cv.get("dsr"),
            "cv_pbo": cv.get("pbo"),
            "cv_corr_bench": cv.get("corr_bench"),
            "cv_turnover": cv.get("ann_turnover"),
            "cv_trades": cv.get("total_trades"),
            "cv_stress_alpha": cv.get("stress_alpha"),
            "cv_null_pct": cv.get("null_percentile"),
            "trials_total": cv.get("trials_total"),
            "trials_effective": cv.get("trials_effective"),
            # ---- THE LABEL: how it did on data fitness never saw --------
            "fwd_alpha": forward.get("val_alpha", cv.get("val_alpha")),
            "fwd_ir": forward.get("val_ir", cv.get("val_ir")),
            "fwd_t": forward.get("val_t", cv.get("val_t")),
            "fwd_sessions": forward.get("val_sessions", cv.get("val_sessions")),
            # ---- gate outcome ------------------------------------------
            "gates_total": len(gates),
            "gates_passed": sum(1 for g in gates if g.get("pass")),
            "gates_failed": ",".join(g.get("gate", "") for g in gates
                                     if not g.get("pass")),
            "promoted": bool(extra.pop("promoted", kind == "promoted")),
        }
        row.update(self.context)
        row.update({k: v for k, v in extra.items() if not isinstance(v, (dict, list))})
        self.buffer.append(row)
        if len(self.buffer) >= FLUSH_EVERY:
            self.flush()

    # -------------------------------------------------------------- writing
    def flush(self) -> int:
        """Write the buffer to today's partition. Returns rows written."""
        if not self.buffer:
            return 0
        try:
            import pandas as pd
        except Exception:
            self.buffer.clear()
            return 0

        df = pd.DataFrame(self.buffer)
        path = self.dir / ("%s.parquet" % _today())
        try:
            if path.exists():
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            # Atomic: a reader mid-analysis must never see a partial file.
            fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
            os.close(fd)
            df.to_parquet(tmp, index=False)
            for attempt in range(40):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == 39:
                        raise
                    time.sleep(0.01)
        except Exception:
            self.buffer.clear()
            return 0

        n = len(self.buffer)
        self.written += n
        self.buffer.clear()
        return n

    def close(self):
        self.flush()


# =============================================================================
#  Reading
# =============================================================================
def load(since: str | None = None, kinds=None):
    """Every recorded evaluation, optionally filtered. Returns a DataFrame."""
    import pandas as pd

    d = resolve(DIR)
    if not d.exists():
        return pd.DataFrame()
    parts = []
    for p in sorted(d.glob("*.parquet")):
        if since and p.stem < since:
            continue
        try:
            parts.append(pd.read_parquet(p))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if kinds:
        df = df[df["kind"].isin(kinds)]
    return df


def summary() -> dict:
    """Headline counts, for the dashboard and the meta-learner."""
    try:
        df = load()
    except Exception:
        return {"rows": 0}
    if df.empty:
        return {"rows": 0}

    labelled = df[df["fwd_alpha"].notna()] if "fwd_alpha" in df else df.iloc[:0]
    out = {
        "rows": int(len(df)),
        "labelled": int(len(labelled)),
        "distinct_strategies": int(df["fingerprint"].nunique())
        if "fingerprint" in df else 0,
        "runs": int(df["run_id"].nunique()) if "run_id" in df else 0,
        "methods": int(df["method_id"].nunique()) if "method_id" in df else 0,
        "promoted": int(df["promoted"].sum()) if "promoted" in df else 0,
        "first": str(df["ts"].min()) if "ts" in df else None,
        "last": str(df["ts"].max()) if "ts" in df else None,
        "bytes": sum(p.stat().st_size for p in resolve(DIR).glob("*.parquet")),
    }
    if len(labelled):
        out["fwd_alpha_mean"] = float(labelled["fwd_alpha"].mean())
        out["fwd_positive_rate"] = float((labelled["fwd_alpha"] > 0).mean())
    return out


def family_forward_rates() -> dict:
    """Which feature families actually generalise, over all history.

    The in-run FeatureBandit learns this too, but forgets on restart and decays
    deliberately. This is the long memory: every labelled row ever recorded,
    which is the only basis on which "momentum has stopped working" could be
    said with any confidence.
    """
    try:
        df = load()
    except Exception:
        return {}
    if df.empty or "fwd_alpha" not in df:
        return {}
    df = df[df["fwd_alpha"].notna() & df["families"].notna()]
    if df.empty:
        return {}

    tally: dict = {}
    for fams, ok in zip(df["families"], df["fwd_alpha"] > 0):
        for f in str(fams).split(","):
            if not f:
                continue
            t = tally.setdefault(f, {"seen": 0, "survived": 0})
            t["seen"] += 1
            t["survived"] += int(bool(ok))
    for f, t in tally.items():
        t["rate"] = t["survived"] / max(1, t["seen"])
    return dict(sorted(tally.items(), key=lambda kv: -kv[1]["rate"]))
