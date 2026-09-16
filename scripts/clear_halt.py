"""Clear a latched risk halt. Deliberately manual.

    python scripts/clear_halt.py --note "reviewed: data outage, not strategy"

A halt is latched precisely so that resuming requires a person who has looked
at why it fired. The note is mandatory and goes into the permanent breach log.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loonie import config, risk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="why it is safe to resume")
    a = ap.parse_args()

    config.load_env()
    rm = risk.RiskManager(config.load())
    if not rm.state.halted:
        print("not halted; nothing to clear")
        return 0
    print("halt reason : %s" % rm.state.halt_reason)
    print("halted at   : %s" % rm.state.halted_at)
    rm.clear(a.note)
    print("cleared. logged: %s" % a.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
