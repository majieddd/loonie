"""The sealed holdout, enforced by code rather than by good intentions.

Everyone agrees you should keep a final untouched window. Almost nobody
actually does, because the enforcement mechanism is usually a person
remembering a promise while staring at a disappointing number. The temptation
is not to cheat outright -- it is to run the holdout "just to check", see a
bad result, change one thing, and run it again. Two evaluations and the
holdout is training data.

So this module makes peeking cost something:

  * The training panel is asserted clean -- any row dated on/after the seal
    boundary raises, so the search physically cannot see the window.
  * The holdout is hashed at creation. If its bytes change, the seal breaks.
  * Every evaluation is appended to a ledger with a timestamp, the genome
    fingerprint, and the result. The counter is monotonic and persisted.
  * Past `max_evaluations`, `open_holdout()` refuses. There is no flag to
    override it; you have to delete the manifest, and that is recorded too.

The ledger is the deliverable. A strategy whose holdout was opened once, with
the result written down before anyone could react to it, means something. One
opened eleven times means nothing, and now you can tell them apart.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve

MANIFEST = "state/holdout_seal.json"


class SealBroken(RuntimeError):
    pass


class HoldoutExhausted(RuntimeError):
    pass


@dataclass
class Seal:
    start: pd.Timestamp
    stop: pd.Timestamp
    digest: str
    created: str
    evaluations: int
    max_evaluations: int
    ledger: list
    path: Path
    contaminated: bool = False

    # ------------------------------------------------------------------ io
    @classmethod
    def _path(cls) -> Path:
        p = resolve(MANIFEST)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def load(cls, cfg):
        p = cls._path()
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            start=pd.Timestamp(d["start"]), stop=pd.Timestamp(d["stop"]),
            digest=d["digest"], created=d["created"],
            evaluations=int(d["evaluations"]),
            max_evaluations=int(d["max_evaluations"]),
            ledger=d.get("ledger", []), path=p,
            contaminated=bool(d.get("contaminated", False)),
        )

    def save(self):
        self.path.write_text(json.dumps({
            "start": str(self.start.date()), "stop": str(self.stop.date()),
            "digest": self.digest, "created": self.created,
            "evaluations": self.evaluations,
            "max_evaluations": self.max_evaluations,
            "ledger": self.ledger,
            "contaminated": self.contaminated,
        }, indent=2), encoding="utf-8")

    # -------------------------------------------------------------- create
    @classmethod
    def create(cls, cfg, panel, force: bool = False) -> "Seal":
        existing = cls.load(cfg)
        if existing and not force:
            return existing
        start = pd.Timestamp(cfg.holdout.start)
        stop = pd.Timestamp(cfg.holdout.end)
        sub = panel.slice_dates(start, stop)
        if sub.shape[0] < 60:
            raise SealBroken(
                "holdout window has only %d sessions; widen holdout.start/end"
                % sub.shape[0]
            )
        seal = cls(
            start=start, stop=stop, digest=digest_panel(sub),
            created=_now(), evaluations=0,
            max_evaluations=int(cfg.holdout.max_evaluations),
            ledger=[{"event": "created", "at": _now(),
                     "sessions": sub.shape[0], "tickers": sub.shape[1]}],
            path=cls._path(),
        )
        seal.save()
        return seal

    # --------------------------------------------------------------- guard
    def assert_train_clean(self, panel) -> None:
        """Raise if a panel handed to the search contains sealed dates."""
        if len(panel.dates) and pd.DatetimeIndex(panel.dates).max() >= self.start:
            raise SealBroken(
                "training panel reaches %s but the seal starts %s -- the search "
                "would be able to see the holdout. Slice the panel first."
                % (pd.DatetimeIndex(panel.dates).max().date(), self.start.date())
            )

    def remaining(self) -> int:
        return max(0, self.max_evaluations - self.evaluations)

    def open_holdout(self, full_panel, genome, note: str = ""):
        """Consume one evaluation. Returns the sealed panel slice."""
        if self.evaluations >= self.max_evaluations:
            raise HoldoutExhausted(
                "the holdout has already been evaluated %d/%d times.\n"
                "Ledger:\n%s\n"
                "This window is now training data. Extend the window with new "
                "market history, or accept the result you already have."
                % (self.evaluations, self.max_evaluations,
                   json.dumps(self.ledger, indent=2))
            )
        sub = full_panel.slice_dates(self.start, self.stop)
        d = digest_panel(sub)
        if d != self.digest:
            raise SealBroken(
                "holdout data changed since sealing.\n  sealed: %s\n  now:    %s\n"
                "Either the cache was refreshed or the window moved. Any result "
                "computed now is not the test you sealed." % (self.digest, d)
            )
        self.evaluations += 1
        self.ledger.append({
            "event": "opened", "at": _now(),
            "genome": getattr(genome, "fingerprint", str(genome)),
            "canonical": getattr(genome, "canonical", lambda: "")(),
            "note": note,
            "evaluation": self.evaluations,
        })
        self.save()
        return sub

    def record_exposure(self, by: str, what: str, note: str = "") -> None:
        """Record that the sealed window was READ outside a formal evaluation.

        The evaluation counter guards the front door. It does nothing about an
        analyst loading the full panel in a one-off script and looking -- which
        is how this window was actually compromised: several diagnostics called
        load_panel() without the sealed training range and read straight
        through the holdout, while the counter went on reporting 0 of 1.

        An unrecorded look is strictly worse than a recorded one, because the
        ledger keeps claiming a cleanliness the data no longer has. So this
        appends the exposure and sets `contaminated`, and nothing here ever
        clears that flag -- the only honest way out is a new window over
        market history that did not exist when the leak happened.
        """
        self.contaminated = True
        self.ledger.append({
            "event": "exposed", "at": _now(), "by": by,
            "what": what, "note": note,
        })
        self.save()

    def record_result(self, stats: dict) -> None:
        if self.ledger and self.ledger[-1].get("event") == "opened":
            self.ledger[-1]["result"] = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in stats.items()
            }
            self.save()

    def describe(self) -> str:
        return (
            "Holdout seal  [%s -> %s]\n"
            "  digest      : %s\n"
            "  created     : %s\n"
            "%s"
            "  evaluations : %d / %d  (%d remaining)"
            % (self.start.date(), self.stop.date(), self.digest, self.created,
               ("  CONTAMINATED: this window was read outside the ledger; the "
                "evaluation\n                count below understates what has "
                "been seen\n" if self.contaminated else ""),
               self.evaluations, self.max_evaluations, self.remaining())
        )


def digest_panel(panel) -> str:
    """Content hash of a panel slice. Order-stable, NaN-stable."""
    h = hashlib.sha256()
    h.update(str(len(panel.dates)).encode())
    h.update(",".join(str(d.date()) for d in panel.dates).encode())
    h.update(",".join(panel.tickers).encode())
    for f in sorted(panel.bars):
        a = np.nan_to_num(panel.bars[f], nan=-9.99e9).astype(np.float32)
        h.update(f.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:32]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def train_window(cfg) -> tuple:
    """The dates the search is allowed to see: start -> seal boundary - 1."""
    start = pd.Timestamp(cfg.universe.start)
    stop = pd.Timestamp(cfg.holdout.start) - pd.Timedelta(days=1)
    return start, stop


def training_panel(cfg, universe=None, progress: bool = False,
                   include_holdout: bool = False, by: str = "unknown"):
    """Load the panel the way a diagnostic is allowed to see it.

    `load_panel(cfg)` returns everything on disk, holdout included. That is
    correct for trading and for the holdout evaluator, and wrong for every
    analysis script -- which is not a hypothetical: attribute.py, feature_ic.py
    and power.py all called it plainly and read straight through the sealed
    window while the evaluation counter went on reporting 0 of 1.

    The counter guards one door. This guards the other. Diagnostics get the
    training window unless they say otherwise, and saying otherwise is logged
    to the ledger rather than being a quiet keyword argument.
    """
    from .data import load_panel
    from .universe import Universe

    universe = universe or Universe.load(cfg)
    if include_holdout:
        s = Seal.load(cfg)
        if s is not None:
            s.record_exposure(by=by, what="full panel including holdout",
                              note="include_holdout=True")
        return load_panel(cfg, universe, progress=progress)

    s0, s1 = train_window(cfg)
    return load_panel(cfg, universe, start=s0, end=s1, progress=progress)
