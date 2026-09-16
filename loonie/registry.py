"""Who is running, and what are they doing right now.

Every long-running process in this system writes a heartbeat here and the
dashboard reads them. Without it the site can only show the *state* of the
system -- a generation count, a book, a gate board -- and never the *activity*,
which is the thing you actually want at 3am when you are asking whether it is
still working or quietly dead.

Design: one file per worker, `state/workers/<id>.json`, and a worker only ever
writes its own. That is deliberate. A single shared registry file written by
four processes needs a lock, and a lock held by a process that gets killed
mid-write leaves the dashboard reading half a JSON object forever. Per-worker
files make concurrent writes impossible by construction, and a reader that
catches one mid-rename just skips that worker for one poll.

Writes are atomic: temp file plus `os.replace`, which is atomic on both POSIX
and Windows. A worker that dies mid-write leaves the previous heartbeat intact
rather than a truncated one.

Liveness is inferred from heartbeat age, never from a self-reported "running"
flag -- a process that is hung or SIGKILLed will happily leave `running: true`
behind forever, whereas it cannot fake a fresh timestamp.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import resolve

DIR = "state/workers"
STALE_AFTER = 180.0      # seconds without a heartbeat before we call it stale
DEAD_AFTER = 900.0       # ...and before we call it dead


def _dir() -> Path:
    d = resolve(DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Worker:
    """A process's handle on its own heartbeat file."""

    def __init__(self, worker_id: str, kind: str, label: str = ""):
        self.id = worker_id
        self.kind = kind
        self.label = label or kind
        self.path = _dir() / ("%s.json" % worker_id)
        self.started = time.time()
        self._doc = {
            "id": worker_id,
            "kind": kind,
            "label": self.label,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _now(),
            "status": "starting",
            "detail": "",
            "progress": None,
            "metrics": {},
            "cycle": 0,
            "errors": 0,
            "last_error": "",
            "heartbeat": _now(),
            "heartbeat_epoch": time.time(),
        }
        self._write()

    # ---------------------------------------------------------------- write
    def _write(self):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._doc, fh, default=str)
            os.replace(tmp, self.path)          # atomic on POSIX and Windows
            tmp = None
        except Exception:
            pass
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def beat(self, status: str | None = None, detail: str | None = None,
             progress: float | None = None, **metrics):
        """Update and flush. Cheap enough to call every few seconds."""
        if status is not None:
            self._doc["status"] = status
        if detail is not None:
            self._doc["detail"] = str(detail)[:240]
        if progress is not None:
            self._doc["progress"] = (None if progress != progress
                                     else max(0.0, min(1.0, float(progress))))
        if metrics:
            self._doc["metrics"].update(
                {k: v for k, v in metrics.items() if v is not None})
        self._doc["uptime_s"] = round(time.time() - self.started, 1)
        self._doc["heartbeat"] = _now()
        self._doc["heartbeat_epoch"] = time.time()
        self._write()

    def cycle(self, n: int | None = None):
        self._doc["cycle"] = (self._doc.get("cycle", 0) + 1) if n is None else n
        self.beat()

    def error(self, exc: BaseException | str):
        self._doc["errors"] = self._doc.get("errors", 0) + 1
        self._doc["last_error"] = ("%s: %s" % (type(exc).__name__, exc)
                                   if isinstance(exc, BaseException)
                                   else str(exc))[:240]
        self.beat(status="error")

    def done(self, detail: str = "", **metrics):
        """Mark a clean completion.

        Scheduled jobs exit when their work is done; without this they simply
        stop heartbeating and age into "dead", which puts a red light and a
        "no heartbeat" warning on the dashboard for a job that succeeded.
        """
        self.beat(status="finished", detail=detail, progress=1.0, **metrics)

    def retire(self):
        """Remove this worker's file. Called on a clean shutdown."""
        try:
            self.path.unlink()
        except Exception:
            pass


# ------------------------------------------------------------------- reading
def read_all() -> list:
    """Every worker's latest heartbeat, liveness inferred from its age."""
    out = []
    d = _dir()
    for p in sorted(d.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue                      # caught mid-rename; skip this poll
        age = time.time() - float(doc.get("heartbeat_epoch") or 0)
        doc["heartbeat_age_s"] = round(age, 1)
        doc["liveness"] = ("live" if age < STALE_AFTER
                           else "stale" if age < DEAD_AFTER else "dead")
        if doc.get("status") == "finished":
            doc["liveness"] = "finished"
        out.append(doc)
    out.sort(key=lambda w: (w["liveness"] != "live", w.get("kind", ""),
                            w.get("id", "")))
    return out


def prune(max_age_s: float = 86400.0) -> int:
    """Delete heartbeat files nothing has touched in a day."""
    n = 0
    for p in _dir().glob("*.json"):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - float(doc.get("heartbeat_epoch") or 0) > max_age_s:
                p.unlink()
                n += 1
        except Exception:
            continue
    return n


def summary() -> dict:
    ws = read_all()
    return {
        "workers": ws,
        "live": sum(1 for w in ws if w["liveness"] == "live"),
        "stale": sum(1 for w in ws if w["liveness"] == "stale"),
        "dead": sum(1 for w in ws if w["liveness"] == "dead"),
        "updated": _now(),
    }
