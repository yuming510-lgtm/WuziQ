#!/usr/bin/env python3
"""Measure G V4 runtime and checkpoint-continuation facts before registration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_config import _atomic_json, audit_v3_continuation
from src.research.alphazero_g.v4_preflight import run_v4_preflight


def main() -> None:
    preflight = run_v4_preflight(ROOT)
    audit = audit_v3_continuation(ROOT)
    _atomic_json(ROOT / "research/g_v4/results/preflight/preflight.json", preflight)
    _atomic_json(ROOT / "research/g_v4/results/preflight/v3_continuation_audit.json", audit)
    print(json.dumps({
        "preflight": preflight,
        "continuation_audit": audit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
