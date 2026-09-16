"""Score every recorded external thesis against what actually happened.

    python scripts/review_theses.py

Measures each claim's proxy on both sides of its capture date. `trailing` is
the year before the claim was made; `forward` is everything since. Only the
second tests a forecast.

A claim flagged ALREADY MOVED had a proxy that ran more than 25% excess before
it was recorded. That does not make the reasoning wrong -- it usually means the
reasoning is right and the market agreed some time ago.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loonie import config, data, knowledge, universe  # noqa: E402


def main() -> int:
    config.load_env()
    cfg = config.load()
    theses = knowledge.load()
    if not theses:
        print("no theses recorded; see loonie/knowledge.py")
        return 0

    panel = data.load_panel(cfg, universe.Universe.load(cfg), progress=False)
    print("=" * 78)
    print("  EXTERNAL THESES  --  measured, not believed")
    print("=" * 78)

    for t in knowledge.score(panel):
        src = t.get("source") or {}
        print("\n  %s" % t["id"])
        print("  source    %s %s" % (src.get("platform", "?"), src.get("account", "")))
        print("  captured  %s" % t["captured"])
        if t.get("summary"):
            print("  %s" % " ".join(t["summary"].split())[:200])
        print("  %-24s %12s %12s  %-11s" %
              ("claim", "trailing 1y", "since", "verdict"))
        for c in t["claims"]:
            tr = "n/a" if c["trailing_excess"] is None else "%+.1f%%" % (100 * c["trailing_excess"])
            fw = "n/a" if c["forward_excess"] is None else "%+.1f%%" % (100 * c["forward_excess"])
            print("  %-24s %12s %12s  %-11s %s"
                  % (c["claim"], tr, fw, c["verdict"],
                     "ALREADY MOVED" if c["already_moved"] else ""))

    print("\n" + "=" * 78)
    print("  Trailing excess is what the proxy did BEFORE the claim was made.")
    print("  A large trailing number means the thesis describes a move that has")
    print("  happened. Correct analysis of a completed move is not a forecast.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
