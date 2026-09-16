"""External theses, recorded as dated falsifiable claims rather than beliefs.

Someone sends you a compelling argument about where the market is going. The
tempting thing is to absorb it — to let it quietly shift what the system looks
for. That is exactly the failure this project exists to avoid: a plausible
narrative attached to a number that already happened is not an edge, and it is
indistinguishable from one until you check.

So claims get recorded here instead, with four things attached that an opinion
does not have:

  a CAPTURE DATE   — so forward performance is measured from when the claim was
                     made, not from whenever the numbers flatter it
  a PROXY          — something measurable; a claim with no observable
                     consequence is not being recorded as a claim
  a HORIZON        — the date by which it should have shown up
  a TRAILING READ  — what the proxy already did BEFORE the claim was made

The last one carries most of the weight. A thesis whose proxy has already run
several hundred percent is describing a trade that happened, however correct
its reasoning. Correct analysis of a completed move is not a forecast, and the
distinction is invisible unless you deliberately measure both sides of the
capture date.

Nothing here feeds the strategy search. The search is not permitted to read
theses, because a narrative that steers the hypothesis space is a narrative
that has escaped its own test.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from .config import resolve

STORE = "knowledge/theses.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> list:
    p = resolve(STORE)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("theses", [])
    except Exception:
        return []


def save(theses: list) -> None:
    p = resolve(STORE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"updated": _now(), "theses": theses}, indent=1),
                 encoding="utf-8")


def add(thesis: dict) -> list:
    """Record a thesis. Replaces any existing entry with the same id."""
    theses = [t for t in load() if t.get("id") != thesis.get("id")]
    thesis.setdefault("recorded", _now())
    theses.append(thesis)
    save(theses)
    return theses


# =============================================================================
#  Scoring
# =============================================================================
def _basket_excess(panel, tickers, lo: int, hi: int) -> float | None:
    """Annualised excess return of an equal-weight basket over the universe."""
    from . import backtest as bt

    idx = {t: i for i, t in enumerate(panel.tickers)}
    cols = [idx[t] for t in tickers if t in idx]
    if not cols or hi - lo < 20:
        return None

    r = bt._to_returns(panel.close)[lo:hi][:, cols]
    m = panel.tradable[lo:hi][:, cols]
    bench = bt.equal_weight_benchmark(panel)[lo:hi]

    w = np.where(m, 1.0, 0.0)
    prev = np.vstack([np.zeros((1, len(cols))), w[:-1]])
    n = prev.sum(axis=1, keepdims=True)
    basket = ((prev / np.maximum(n, 1.0)) * r).sum(axis=1)

    def ann(x):
        v = float(np.prod(1.0 + x))
        years = max(len(x) / 252.0, 1e-9)
        return v ** (1.0 / years) - 1.0 if v > 0 else -1.0

    return float(ann(basket) - ann(bench))


def score(panel, theses: list | None = None) -> list:
    """Measure every claim's proxy on both sides of its capture date.

    `trailing` is what the proxy did in the year BEFORE the claim; `forward` is
    what it has done since. A forecast is only being tested by the second.
    """
    import pandas as pd

    theses = load() if theses is None else theses
    dates = pd.DatetimeIndex(panel.dates)
    out = []

    for t in theses:
        cap = pd.Timestamp(t.get("captured") or t.get("recorded", "")[:10])
        at = int(dates.searchsorted(cap))
        rows = []
        for c in t.get("claims", []):
            px = c.get("proxy") or {}
            tickers = px.get("tickers") or []
            trailing = _basket_excess(panel, tickers, max(0, at - 252), at)
            forward = _basket_excess(panel, tickers, at, len(dates))
            elapsed = max(0, len(dates) - at)

            if forward is None or elapsed < 20:
                verdict = "too early"
            elif forward > 0.02:
                verdict = "holding"
            elif forward < -0.02:
                verdict = "not holding"
            else:
                verdict = "flat"

            rows.append({
                "claim": c.get("id"),
                "statement": c.get("statement"),
                "proxy": ", ".join(tickers) or "none",
                "trailing_excess": trailing,
                "forward_excess": forward,
                "sessions_since": elapsed,
                "verdict": verdict,
                # A thesis whose proxy already ran hard is describing a
                # completed move, however sound the reasoning behind it.
                "already_moved": bool(trailing is not None and trailing > 0.25),
                "horizon": c.get("horizon"),
                "assessment": c.get("assessment"),
            })
        out.append({
            "id": t.get("id"), "source": t.get("source"),
            "summary": t.get("summary"), "captured": str(cap.date()),
            "claims": rows,
        })
    return out


def summary(panel=None) -> dict:
    theses = load()
    if not theses:
        return {"theses": 0, "claims": 0}
    n_claims = sum(len(t.get("claims", [])) for t in theses)
    out = {"theses": len(theses), "claims": n_claims}
    if panel is not None:
        try:
            scored = score(panel, theses)
            flat = [c for t in scored for c in t["claims"]]
            out["scored"] = scored
            out["already_moved"] = sum(1 for c in flat if c["already_moved"])
            out["holding"] = sum(1 for c in flat if c["verdict"] == "holding")
            out["too_early"] = sum(1 for c in flat if c["verdict"] == "too early")
        except Exception:
            pass
    return out
