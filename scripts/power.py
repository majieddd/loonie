"""How large an edge would this experiment have to contain before it could see one?

    python scripts/power.py [--top 6]

Four honest walk-forwards have now returned "no edge". Before running a fifth,
it is worth asking whether the instrument could report anything else.

A t-statistic on excess return is (excess / tracking error) * sqrt(years). The
walk-forward spans 5.6 years, so a strategy needs an excess Sharpe near 0.85
just to reach t = 2. With the tracking error that a 25-name book carries
against a 500-name benchmark, that is an excess return around 12% a year.

No realistic long-only equity strategy delivers 12% a year of alpha. So the
gate `min_alpha_tstat: 2.0` is, on this measurement, asking for something that
does not exist -- and every "no edge" result so far is consistent with both a
dead search AND a good one, which means it has not been distinguishing them.

Tracking error is a design choice, not a fact of nature. It falls when the book
holds more names, and it falls much further when the market leg is removed. The
underlying signal is unchanged; only the noise around it moves. This measures
what each construction actually costs in detectable-edge terms, on real scores
from the live archive, rather than assuming.

The output to read is MIN DETECTABLE ALPHA: the smallest annual excess that
would register as significant. Lower is a sharper instrument.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from loonie import backtest as bt  # noqa: E402
from loonie import features as F  # noqa: E402
from loonie.config import load, load_env, resolve  # noqa: E402
from loonie import seal  # noqa: E402
from loonie.genome import Genome  # noqa: E402

TD = 252.0


def _ann(x: np.ndarray) -> float:
    v = float(np.prod(1.0 + x))
    yrs = max(len(x) / TD, 1e-9)
    return v ** (1.0 / yrs) - 1.0 if v > 0 else -1.0


def book(score, tradable, rets, k, rb, bps, short=False):
    """Equal-weight top-k, optionally dollar-neutral against the bottom k.

    Deliberately simpler than backtest.run: no position caps, no risk overlay.
    The question here is what the CONSTRUCTION does to variance, and extra
    machinery would only blur the comparison between rows.
    """
    T, N = score.shape
    w = np.zeros(N)
    out = np.zeros(T)
    turn = 0.0

    for t0 in range(0, T - 1, rb):
        t1 = min(t0 + rb, T - 1)
        s, elig = score[t0], tradable[t0] & np.isfinite(score[t0])
        n = int(elig.sum())
        new = np.zeros(N)
        if n:
            idx = np.where(elig)[0]
            order = idx[np.argsort(-s[idx], kind="stable")]
            kk = min(k, n // 2 if short else n)
            if kk:
                new[order[:kk]] = 1.0 / kk
                if short:
                    new[order[-kk:]] = -1.0 / kk
        turn += float(np.abs(new - w).sum())
        # Costs are charged on the turnover of the rebalance, then the book is
        # held flat to the next one -- the same convention as backtest.run.
        out[t0] = -float(np.abs(new - w).sum()) * bps
        w = new
        out[t0 + 1:t1 + 1] += rets[t0 + 1:t1 + 1] @ w
    return out, turn / max(T / TD, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--json", default="state/power.json")
    ap.add_argument("--include-holdout", action="store_true",
                    help="read the sealed window too; logged to the ledger")
    a = ap.parse_args()

    load_env()
    cfg = load()
    panel = seal.training_panel(cfg, by="scripts/power.py",
                                include_holdout=a.include_holdout)
    feats = F.build(panel)
    rets = bt._to_returns(panel.close)
    bench = bt.equal_weight_benchmark(panel)
    bps = (float(cfg.backtest.commission_bps)
           + float(cfg.backtest.slippage_bps)) / 1e4
    yrs = len(panel.dates) / TD

    state = json.loads(resolve("state/evolve_state.json").read_text("utf-8"))
    pool = [e for e in (state.get("archive") or []) if e.get("genome")]
    pool.sort(key=lambda e: -float(e.get("fitness") or -1e9))
    pool = pool[:a.top]
    print("[power] %d candidates | %.2f years | %d names"
          % (len(pool), yrs, len(panel.tickers)))

    designs = [
        ("top 25 long-only (current)", 25, False),
        ("top 50 long-only", 50, False),
        ("top 100 long-only", 100, False),
        ("top 25 long / bottom 25 short", 25, True),
        ("top 100 long / bottom 100 short", 100, True),
    ]

    rows = []
    for label, k, short in designs:
        te, exc = [], []
        for e in pool:
            g = Genome.from_dict(e["genome"])
            try:
                s = g.score(feats, panel.tradable)
            except Exception:
                continue
            r, _ = book(s, panel.tradable, rets, k,
                        max(1, int(g.rebalance_days)), bps, short)
            # A market-neutral book is its own benchmark; a long-only one is
            # measured against holding everything.
            d = r if short else r - bench
            te.append(float(np.std(d, ddof=1) * math.sqrt(TD)))
            exc.append(_ann(r) - (0.0 if short else _ann(bench)))
        if not te:
            continue
        mte = float(np.median(te))
        mde = 2.0 * mte / math.sqrt(yrs)      # excess needed for t = 2
        rows.append({"design": label, "k": k, "short": short,
                     "tracking_error": mte, "min_detectable_alpha": mde,
                     "median_excess": float(np.median(exc))})

    print("\n%-34s %9s %11s %11s" %
          ("construction", "track err", "MIN DETECT", "median exc"))
    print("-" * 68)
    base = rows[0]["min_detectable_alpha"] if rows else None
    for r in rows:
        print("%-34s %8.2f%% %10.2f%% %10.2f%%"
              % (r["design"], 100 * r["tracking_error"],
                 100 * r["min_detectable_alpha"], 100 * r["median_excess"]))

    if base:
        print("\nsharpening vs the current design:")
        for r in rows[1:]:
            f = base / max(r["min_detectable_alpha"], 1e-9)
            print("  %-34s %.1fx  (detects %.2f%% instead of %.2f%%)"
                  % (r["design"], f, 100 * r["min_detectable_alpha"],
                     100 * base))

    out = resolve(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"years": yrs, "rows": rows}, indent=1),
                   encoding="utf-8")
    print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
