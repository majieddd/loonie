"""The outer loop: keep everything running, on a schedule, forever.

Until now each piece ran by hand. The search was a daemon, the trade was a
command, and the walk-forward — the only test that ever caught anything — was
something a person remembered to type. That is the wrong shape for a system
whose entire claim is that it improves itself: the component that decides
whether the improvement is real was the one component not on a schedule.

So: one supervisor, four jobs, and a heartbeat on every one of them.

    search      continuous   the genetic program; restarted if it dies
    data        every 6h     price and macro tails, incremental
    validate    every 8h     honest walk-forward of the current champion
    trade       weekdays     one paper rebalance after the close
    publish     every 60s    snapshot for the dashboard

`validate` is the one that matters. It re-runs the sequential walk-forward
with a fresh search per segment, so the number it reports has never been
contaminated by a candidate pool that saw the future. Its history accumulates
in `state/validation_history.json`, which is what turns "the search got a
better score" into "the procedure did or did not keep working" — and those are
different claims, as this project has now demonstrated twice.

Nothing here can arm live trading. The trade job shells out to run_trade.py
without the live flag, and run_trade.py needs two more locks besides.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import registry
from .config import ROOT, resolve

HISTORY = "state/validation_history.json"


@dataclass
class Job:
    name: str
    every_s: float                 # 0 = continuous child process
    argv: list
    last_run: float = 0.0
    runs: int = 0
    failures: int = 0
    last_status: str = "pending"
    last_detail: str = ""
    weekday_only: bool = False
    proc: object = None
    enabled: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def continuous(self) -> bool:
        return self.every_s <= 0

    def due(self, now: float) -> bool:
        if not self.enabled:
            return False
        if self.weekday_only and datetime.now(timezone.utc).weekday() >= 5:
            return False
        return (now - self.last_run) >= self.every_s


def default_jobs(cfg, population: int) -> list:
    """Job table, with staggered first runs.

    Without the stagger every scheduled job fires at t=0 and the twenty-minute
    validation competes with the search, the data fetch and the rebalance for
    the same cores on startup -- the one moment the search is least able to
    spare them. `first_in` pushes each job's first run out by seeding its
    last_run in the past.
    """
    py = sys.executable
    now = time.time()

    def stagger(job: "Job", first_in: float) -> "Job":
        job.last_run = now - job.every_s + first_in
        return job

    return [
        Job("search", 0, [py, "-u", "scripts/run_evolve.py", "--daemon",
                          "--population", str(population), "--report-every", "50"]),
        stagger(Job("data", 6 * 3600, [py, "-u", "scripts/fetch_data.py"]), 60),
        stagger(Job("validate", 8 * 3600,
                    [py, "-u", "scripts/walkforward.py", "--honest",
                     "--segments", "6", "--pop", "60", "--gens", "12"]), 900),
        stagger(Job("trade", 24 * 3600, [py, "-u", "scripts/run_trade.py",
                                         "--flatten-on-halt"],
                    weekday_only=True), 300),
        # Writes docs/data every minute so the LOCAL dashboard is current, and
        # pushes to the git remote at most every 15 minutes so GitHub Pages
        # stays fed without turning a 20-second generation into 4,000 commits
        # a day. The throttle lives inside publish_dashboard.py.
        Job("publish", 60, [py, "-u", "scripts/publish_dashboard.py",
                            "--quiet", "--push", "--throttle", "900"]),
    ]


class Orchestrator:
    def __init__(self, cfg, population: int = 250, once: bool = False):
        self.cfg = cfg
        self.once = once
        self.jobs = {j.name: j for j in default_jobs(cfg, population)}
        self.me = registry.Worker("orchestrator", "supervisor",
                                  "system supervisor")
        self.children: dict = {}
        self.started = time.time()

    # ------------------------------------------------------------ children
    def _spawn(self, job: Job):
        log = resolve("state/%s.log" % job.name)
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "ab", buffering=0)
        job.proc = subprocess.Popen(job.argv, cwd=str(ROOT), stdout=fh,
                                    stderr=subprocess.STDOUT)
        job.runs += 1
        job.last_run = time.time()
        job.last_status = "running"
        self.children[job.name] = fh
        self._log("started %s (pid %d)" % (job.name, job.proc.pid))

    def _reap(self, job: Job):
        """Collect a finished child and record how it went."""
        if job.proc is None:
            return
        rc = job.proc.poll()
        if rc is None:
            return
        job.last_status = "ok" if rc == 0 else "failed"
        if rc != 0:
            job.failures += 1
            job.last_detail = "exit %d" % rc
        job.proc = None
        fh = self.children.pop(job.name, None)
        if fh:
            try:
                fh.close()
            except Exception:
                pass
        if job.name == "validate" and rc == 0:
            self._record_validation()
        self._log("%s finished rc=%d" % (job.name, rc))

    # ---------------------------------------------------------- validation
    def _record_validation(self):
        """Pull the newest walk-forward report into a running history.

        A single walk-forward number is a measurement. The sequence of them is
        the only thing that can tell you whether the procedure is decaying, and
        decay is the failure mode a self-improving system is most prone to and
        least able to notice from the inside.
        """
        p = resolve("state/walkforward_last.json")
        if not p.exists():
            self._log("validate finished but wrote no result file")
            return
        try:
            entry = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            self._log("validate result unreadable: %s" % e)
            return

        hp = resolve(HISTORY)
        hist = []
        if hp.exists():
            try:
                hist = json.loads(hp.read_text(encoding="utf-8"))
            except Exception:
                hist = []
        if hist and hist[-1].get("at") == entry.get("at"):
            return                       # same run; do not double-record
        hist.append(entry)
        hp.write_text(json.dumps(hist[-200:], indent=1), encoding="utf-8")
        self._log("recorded validation: excess %s, alpha t %s"
                  % (entry.get("excess"), entry.get("alpha_t")))

    # ---------------------------------------------------------------- loop
    def _log(self, msg):
        print("[orchestrator] %s" % msg, flush=True)

    def status(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "jobs": {n: {"runs": j.runs, "failures": j.failures,
                         "status": j.last_status, "detail": j.last_detail,
                         "every_s": j.every_s,
                         "next_in_s": (None if j.continuous else
                                       max(0, j.every_s - (time.time() - j.last_run))),
                         "running": j.proc is not None}
                     for n, j in self.jobs.items()},
        }

    def tick(self):
        now = time.time()
        for job in self.jobs.values():
            self._reap(job)

            if job.continuous:
                # Restart a dead continuous worker, but back off so a job that
                # crashes on startup does not spin the CPU respawning forever.
                if job.proc is None:
                    backoff = min(300.0, 10.0 * (1 + job.failures))
                    if now - job.last_run >= backoff:
                        if job.runs:
                            job.failures += 1
                            self._log("%s died; restarting after %.0fs backoff"
                                      % (job.name, backoff))
                        self._spawn(job)
                continue

            if job.proc is None and job.due(now):
                self._spawn(job)

        running = [n for n, j in self.jobs.items() if j.proc is not None]
        self.me.beat(
            status="supervising",
            detail="running: " + (", ".join(running) if running else "idle"),
            **{"jobs_running": len(running),
               "failures": sum(j.failures for j in self.jobs.values())})

    def run(self, interval: float = 5.0):
        self._log("supervising %d jobs" % len(self.jobs))
        for n, j in self.jobs.items():
            self._log("  %-10s %s" % (n, "continuous" if j.continuous
                                      else "every %.0fh" % (j.every_s / 3600)))
        try:
            while True:
                self.tick()
                if self.once and all(j.runs for j in self.jobs.values()):
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            self._log("stopping")
        finally:
            self.shutdown()

    def shutdown(self):
        for job in self.jobs.values():
            if job.proc is not None:
                try:
                    job.proc.terminate()
                except Exception:
                    pass
        for fh in self.children.values():
            try:
                fh.close()
            except Exception:
                pass
        self.me.retire()


def validation_history() -> list:
    p = resolve(HISTORY)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
