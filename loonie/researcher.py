"""A local model that proposes hypotheses, and a ledger that makes it pay for them.

This project started by asking whether the search could work without a
language model. It can, up to a point: the genetic program found nothing that
cleared its own multiple-testing bar in 300,000 trials. What a model adds is
not better search -- it is the ability to read the warehouse, notice something
across markets, and say what would test it. That is the part the grammar
cannot do, because a mutation operator cannot form a reason.

THE DANGER IS THE OBVIOUS ONE AND IT IS SEVERE. A model that can generate a
thousand plausible hypotheses is a machine for manufacturing false positives.
The bar a winner must clear is sqrt(2 ln N) on the number of hypotheses
actually tried, so a tireless proposer does not improve the odds of finding an
edge -- it raises the threshold for everything, including the hypotheses that
were already there. At 29 strategies the bar is 2.60. At 1,000 it is 3.72.

So the ledger is the point of this module, not the model:

  * EVERY proposal is recorded before it is tested, with a fingerprint. A
    hypothesis that is generated and quietly discarded still counts, because
    the discarding was done by looking.
  * The bar is recomputed from the ledger on every evaluation and travels with
    the result.
  * The model NEVER sees the outcome of a test before its proposal is
    committed. It can read past results between cycles, which is how it
    learns; it cannot revise a hypothesis after seeing how it did, which is
    how the count gets laundered.

The model is a proposer. The arithmetic is the referee.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timezone

from .config import resolve

OLLAMA = "http://localhost:11434"
LEDGER = "state/hypotheses.jsonl"

# Big enough to reason across a table, small enough to leave the machine
# usable while the search and the trader are also running.
DEFAULT_MODEL = "qwen3:14b-q4_K_M"

SYSTEM = """You are a quantitative research assistant working on a trading \
research system. You propose TESTABLE hypotheses about financial markets.

You will be shown a summary of what has already been measured. Propose ONE new
hypothesis that is not already in the list.

Rules you must follow:
- It must be falsifiable and mechanically testable from daily OHLCV data.
- It must name the market: stocks, crypto, forex, or macro.
- Prefer hypotheses with a stated ECONOMIC REASON, not pattern-fitting.
- Do not propose something already in the tested list, in substance or in
  wording. Cross-market applications of an existing idea ARE allowed and
  should be labelled as such.
- Be specific about the signal, the holding period, and the direction.

Reply with ONLY a JSON object, no prose before or after:
{"title": "...", "market": "stocks|crypto|forex|macro",
 "signal": "precise description of what to compute from OHLCV",
 "direction": "long high | long low",
 "holding_days": 21,
 "reason": "the economic mechanism you expect to drive it",
 "cross_market_of": "id of an existing strategy, or null"}"""


def _post(path: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode())


def available(timeout: float = 5.0) -> list:
    """Models the local server will serve, or [] if nothing is listening."""
    try:
        req = urllib.request.Request(OLLAMA + "/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return [m["name"] for m in json.loads(fh.read().decode())
                    .get("models", [])]
    except Exception:
        return []


# =============================================================================
#  The ledger
# =============================================================================
def _fingerprint(h: dict) -> str:
    """Identity of a hypothesis, so the same idea twice is counted once.

    Deliberately over the SUBSTANCE -- market, signal, direction, holding
    period -- and not the title. A model asked repeatedly for something new
    will happily rename the same idea, and a ledger keyed on titles would let
    it do that indefinitely without the bar moving.
    """
    key = "|".join(str(h.get(k, "")).strip().lower()
                   for k in ("market", "signal", "direction", "holding_days"))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def record(h: dict, source: str = "llm") -> dict:
    """Append a proposal. Returns it with its fingerprint and ordinal."""
    p = resolve(LEDGER)
    p.parent.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(h)

    seen = {e["fingerprint"] for e in read_ledger()}
    entry = {
        **h, "fingerprint": fp, "source": source,
        "proposed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duplicate": fp in seen,
    }
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def read_ledger() -> list:
    p = resolve(LEDGER)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def bar(extra: int = 0) -> dict:
    """The multiple-testing threshold implied by everything proposed so far.

    Counts DISTINCT hypotheses: a duplicate proposal is the same test, and
    charging for it twice would be as dishonest in the strict direction as
    not charging at all.
    """
    led = read_ledger()
    distinct = len({e["fingerprint"] for e in led})
    n = max(distinct + extra, 2)
    return {
        "proposals_total": len(led),
        "proposals_distinct": distinct,
        "n_for_bar": n,
        "bar": max(2.0, math.sqrt(2.0 * math.log(n))),
    }


# =============================================================================
#  The proposer
# =============================================================================
def context_for_model(max_strategies: int = 40) -> str:
    """What the model is allowed to see: what exists, and how it did."""
    lines = []
    try:
        from . import warehouse as W
        con = W.connect(read_only=True)
        rows = con.execute("""
            SELECT title, market, ROUND(cagr_pct,1) AS cagr,
                   ROUND(t_stat,2) AS t, is_modelled
            FROM strategies ORDER BY ABS(t_stat) DESC LIMIT ?
        """, [max_strategies]).fetchall()
        con.close()
        lines.append("ALREADY TESTED (title | market | %/yr | t-stat):")
        for r in rows:
            lines.append("  %s | %s | %s | %s%s"
                         % (r[0], r[1], r[2], r[3],
                            "  [MODELLED, not real data]" if r[4] else ""))
    except Exception as e:
        lines.append("(warehouse unavailable: %s)" % e)

    b = bar()
    lines.append("")
    lines.append("MULTIPLE TESTING: %d distinct hypotheses proposed so far. "
                 "Any winner must clear |t| >= %.2f. Every new proposal raises "
                 "this bar for every hypothesis, including the ones already "
                 "tested. Propose something you think is genuinely likely to "
                 "work, not merely something new."
                 % (b["proposals_distinct"], b["bar"]))
    return "\n".join(lines)


def propose(model: str = DEFAULT_MODEL, temperature: float = 0.7,
            timeout: float = 300.0) -> dict | None:
    """Ask the local model for one hypothesis. Returns None on any failure."""
    try:
        r = _post("/api/chat", {
            "model": model, "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": context_for_model()},
            ]}, timeout=timeout)
        text = (r.get("message") or {}).get("content", "")
    except Exception:
        return None

    # Models wrap JSON in prose or fences however they like; take the outermost
    # object rather than trusting the format instruction.
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        h = json.loads(text[i:j + 1])
    except Exception:
        return None
    if not h.get("signal") or h.get("market") not in (
            "stocks", "crypto", "forex", "macro"):
        return None
    h["holding_days"] = int(h.get("holding_days") or 21)
    h["model"] = model
    return h
