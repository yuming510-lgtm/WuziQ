#!/usr/bin/env python3
"""Measure G V4.1 compute throughput before formal training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_1_preflight import run_v4_1_preflight


def main() -> None:
    result = run_v4_1_preflight(ROOT)
    destination = ROOT / "research/g_v4_1/results/preflight/preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(destination)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
