"""Build the JSON snapshot the dashboard reads.

GitHub Pages is static hosting -- it cannot run Python. So the live-ness of the
dashboard comes from this: every time the search finishes a generation or the
trader finishes a session, a snapshot of the whole system is written to
`docs/data/`, committed, and pushed. Pages serves it; the page polls it.

That makes "real time" mean "as fresh as the last publish", which for a daily
rebalance and a ~20s generation is the right granularity anyway. Nothing here
streams, and the dashboard says exactly how old its data is rather than
implying a liveness it does not have.

Two files, deliberately split:
  snapshot.json  -- current state, rewritten each publish (~100 KB)
  series.json    -- append-only history for the charts (capped, ~200 KB)

The split matters because the page fetches the snapshot every 30s and the
series only on load. Putting the history in the snapshot would mean
re-downloading a growing file forever.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone


from . import orchestrator, registry
from .config import ROOT, resolve

OUT = "docs/data"
MAX_SERIES = 3000


def _read(path: str):
    p = resolve(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_snapshot(cfg) -> dict:
    """Assemble everything the dashboard shows, from the state files on disk."""
    ev = _read("state/evolve_state.json") or {}
    risk = _read("state/risk_state.json") or {}
    seal = _read("state/holdout_seal.json") or {}
    paper = _read("state/paper_account.json") or {}
    alloc = _read("state/allocator.json") or {}

    hist = ev.get("history") or []
    last = hist[-1] if hist else {}

    # ---- search ----------------------------------------------------------
    archive = ev.get("archive") or []
    hof = ev.get("hall_of_fame") or []
    search = {
        "generation": ev.get("generation", 0),
        "trials": ev.get("trials", 0),
        "trials_effective": last.get("trials_effective"),
        "independence": last.get("independence"),
        "archive_cells": last.get("archive_cells"),
        "explore": last.get("explore"),
        "best_fitness": last.get("best_fitness"),
        "best_ir": last.get("best_ir"),
        "best_alpha_t": last.get("best_alpha_t"),
        "best_dsr": last.get("best_dsr"),
        "best_pbo": last.get("best_pbo"),
        "best_corr": last.get("best_corr"),
        "best_null_pct": last.get("best_vs_null_pct"),
        "seconds_per_gen": last.get("seconds"),
        "promoted_count": len(hof),
        "demoted_count": len(ev.get("demoted") or []),
        "null_summary": ev.get("null_summary") or {},
        "operators": ev.get("operator_table") or {},
        "features": ev.get("feature_table") or {},
    }

    # ---- the promotion gate, as a pass/fail board ------------------------
    leader = archive[0] if archive else None
    gates = []
    if leader:
        for g in leader.get("gates", []):
            gates.append({
                "name": g.get("gate"),
                "value": g.get("value"),
                "op": g.get("op"),
                "threshold": g.get("threshold"),
                "pass": bool(g.get("pass")),
            })

    def _strategy(entry):
        c = entry.get("cv", {})
        return {
            "fingerprint": entry.get("fingerprint"),
            "expression": entry.get("canonical"),
            "operator": entry.get("operator"),
            "fitness": entry.get("fitness"),
            "ir": c.get("mean_ir"),
            "alpha": c.get("mean_alpha"),
            "alpha_t": c.get("alpha_tstat"),
            "folds_positive": c.get("frac_positive"),
            "worst_fold": c.get("worst_fold_alpha"),
            "dsr": c.get("dsr"),
            "pbo": c.get("pbo"),
            "corr": c.get("corr_bench"),
            "turnover": c.get("ann_turnover"),
            "trades": c.get("total_trades"),
            "stress_alpha": c.get("stress_alpha"),
            "null_pct": c.get("null_percentile"),
            "null_mean_ir": c.get("null_mean_ir"),
            "promoted": bool(entry.get("promoted")),
        }

    # ---- portfolio -------------------------------------------------------
    positions = []
    cash = float(paper.get("cash", 0.0))
    for sym, p in (paper.get("positions") or {}).items():
        qty = float(p.get("qty", 0))
        avg = float(p.get("avg_price", 0))
        mkt = float(p.get("market_price", avg))
        positions.append({
            "symbol": sym, "qty": qty, "avg_price": avg, "market_price": mkt,
            "market_value": qty * mkt,
            "unrealized": qty * (mkt - avg),
            "unrealized_pct": (mkt / avg - 1.0) if avg > 0 else 0.0,
        })
    positions.sort(key=lambda r: -r["market_value"])
    mv = sum(r["market_value"] for r in positions)
    equity = cash + mv
    start = float(cfg.backtest.initial_capital)
    for r in positions:
        r["weight"] = r["market_value"] / equity if equity > 0 else 0.0

    blotter = (paper.get("blotter") or [])[-60:]
    fills = [{
        "at": b.get("at"), "symbol": b.get("symbol"), "side": b.get("side"),
        "qty": b.get("filled_qty"), "price": b.get("filled_price"),
        "notional": float(b.get("filled_qty") or 0) * float(b.get("filled_price") or 0),
        "note": b.get("note"),
    } for b in reversed(blotter)]

    portfolio = {
        "equity": equity,
        "cash": cash,
        "invested": mv,
        "gross_exposure": (mv / equity) if equity > 0 else 0.0,
        "n_positions": len(positions),
        "total_return": (equity / start - 1.0) if start > 0 else 0.0,
        "positions": positions,
        "fills": fills,
        "updated": paper.get("updated"),
        "broker": str(cfg.trade.broker),
        "mode": str(cfg.trade.mode),
        "is_live": False,      # this repo never publishes a live-armed snapshot
    }

    # ---- risk ------------------------------------------------------------
    risk_out = {
        "halted": bool(risk.get("halted", False)),
        "halt_reason": risk.get("halt_reason", ""),
        "halted_at": risk.get("halted_at", ""),
        "equity_high_water": risk.get("equity_high_water", 0.0),
        "day": risk.get("day", ""),
        "day_start_equity": risk.get("day_start_equity", 0.0),
        "orders_today": risk.get("orders_today", 0),
        "breaches": (risk.get("breaches") or [])[-8:],
        "limits": {
            "max_daily_loss_pct": cfg.trade.risk.max_daily_loss_pct,
            "max_drawdown_pct": cfg.trade.risk.max_drawdown_pct,
            "max_orders_per_day": cfg.trade.risk.max_orders_per_day,
            "max_position_pct": cfg.trade.max_position_pct,
        },
    }
    hw = float(risk.get("equity_high_water") or equity or 1.0)
    risk_out["drawdown"] = (equity - hw) / hw if hw > 0 else 0.0
    dstart = float(risk.get("day_start_equity") or equity or 1.0)
    risk_out["day_pl"] = (equity - dstart) / dstart if dstart > 0 else 0.0

    # ---- seal ------------------------------------------------------------
    seal_out = {
        "start": seal.get("start"), "stop": seal.get("stop"),
        "digest": seal.get("digest"), "created": seal.get("created"),
        "evaluations": seal.get("evaluations", 0),
        "max_evaluations": seal.get("max_evaluations", 1),
        "ledger": seal.get("ledger") or [],
    }
    seal_out["remaining"] = max(
        0, seal_out["max_evaluations"] - seal_out["evaluations"])

    # ---- allocator -------------------------------------------------------
    arms = []
    for k, a in (alloc.get("arms") or {}).items():
        n = int(a.get("n", 0))
        mu = float(a.get("mu", 0.0))
        beta, al = float(a.get("beta", 0.0)), float(a.get("alpha", 2.0))
        var = beta / max(al, 1e-6)
        sd = var ** 0.5
        arms.append({
            "key": k, "n": n, "mean_daily": mu,
            "cumulative": a.get("cumulative", 0.0),
            "sharpe_ann": (mu / sd * (252 ** 0.5)) if sd > 1e-12 else 0.0,
            "last_update": a.get("last_update"),
        })
    arms.sort(key=lambda r: -r["cumulative"])

    # ---- who is running right now ----------------------------------------
    # Liveness is inferred from heartbeat age, never from a self-reported flag:
    # a hung or SIGKILLed process leaves "running: true" behind forever but
    # cannot fake a fresh timestamp.
    workers = registry.summary()
    # Redact local machine details. The dashboard needs to know a worker is
    # alive and what it is doing; it has never needed the hostname or the PID,
    # and this file is served publicly.
    for w in workers.get("workers", []):
        w.pop("host", None)
        w.pop("pid", None)
    vhist = orchestrator.validation_history()

    return {
        "generated_at": _now(),
        "workers": workers,
        "validation_history": vhist[-40:],
        "search": search,
        "gates": gates,
        "leader": _strategy(leader) if leader else None,
        "promoted": [_strategy(h) for h in hof],
        "demoted": [{**_strategy(h),
                     "demoted_at_generation": h.get("demoted_at_generation"),
                     "demoted_at_trials": h.get("demoted_at_trials"),
                     "demoted_because": h.get("demoted_because") or []}
                    for h in (ev.get("demoted") or [])],
        "elites": [_strategy(e) for e in archive[:12]],
        "portfolio": portfolio,
        "risk": risk_out,
        "seal": seal_out,
        "allocator": arms,
        "data": (ev.get("panel") or {}).get("coverage") or {},
        "panel": {k: v for k, v in (ev.get("panel") or {}).items()
                  if k != "coverage"},
        "config": {
            "gate": dict(cfg.evolve.gate),
            "universe_start": str(cfg.universe.start),
            "holdout_start": str(cfg.holdout.start),
            "population": cfg.evolve.population,
            "slippage_bps": cfg.backtest.slippage_bps,
        },
    }


def build_series(cfg, snapshot: dict) -> dict:
    """Append-only time series for the charts. Capped so it cannot grow forever."""
    p = resolve("%s/series.json" % OUT)
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    equity = prev.get("equity") or []
    point = {
        "t": snapshot["generated_at"],
        "equity": snapshot["portfolio"]["equity"],
        "cash": snapshot["portfolio"]["cash"],
        "positions": snapshot["portfolio"]["n_positions"],
    }
    if not equity or equity[-1]["equity"] != point["equity"]:
        equity.append(point)

    # Generation history comes straight from the search; it is already a series.
    ev = _read("state/evolve_state.json") or {}
    gens = [{
        "g": h.get("generation"), "fitness": h.get("best_fitness"),
        "ir": h.get("best_ir"), "alpha_t": h.get("best_alpha_t"),
        "dsr": h.get("best_dsr"), "trials": h.get("trials"),
        "trials_eff": h.get("trials_effective"),
        "cells": h.get("archive_cells"), "explore": h.get("explore"),
    } for h in (ev.get("history") or [])]

    return {
        "updated": snapshot["generated_at"],
        "equity": equity[-MAX_SERIES:],
        "generations": gens[-MAX_SERIES:],
    }


def build_strategies(cfg) -> dict:
    """The executable genomes, small enough to commit.

    `state/evolve_state.json` is ~1 MB and rewritten every generation -- commit
    that and the repo gains a megabyte of churn every twenty seconds. But a CI
    runner still needs to know what to trade. So this extracts just the genome
    trees of the promoted strategies (a few KB) into a file that changes only
    when something is actually promoted, which is rare by design.
    """
    ev = _read("state/evolve_state.json") or {}
    hof = ev.get("hall_of_fame") or []
    arch = ev.get("archive") or []
    out = []
    for e in (hof or arch[:3]):
        if "genome" not in e:
            continue
        out.append({
            "fingerprint": e.get("fingerprint"),
            "canonical": e.get("canonical"),
            "promoted": bool(hof),
            "genome": e["genome"],
            "cv": {k: v for k, v in (e.get("cv") or {}).items()
                   if isinstance(v, (int, float, bool))},
        })
    return {
        "updated": _now(),
        "generation": ev.get("generation", 0),
        "trials": ev.get("trials", 0),
        "promoted": bool(hof),
        "strategies": out,
    }


def publish(cfg, out_dir: str = OUT) -> dict:
    """Write snapshot.json + series.json + strategies.json."""
    d = ROOT / out_dir
    d.mkdir(parents=True, exist_ok=True)

    snap = build_snapshot(cfg)
    series = build_series(cfg, snap)
    strat = build_strategies(cfg)

    sp = d / "snapshot.json"
    yp = d / "series.json"
    gp = d / "strategies.json"
    sp.write_text(json.dumps(snap, indent=1, default=str), encoding="utf-8")
    yp.write_text(json.dumps(series, default=str), encoding="utf-8")
    gp.write_text(json.dumps(strat, indent=1, default=str), encoding="utf-8")
    return {"snapshot": sp, "series": yp, "strategies": gp,
            "bytes": sum(f.stat().st_size for f in (sp, yp, gp))}
