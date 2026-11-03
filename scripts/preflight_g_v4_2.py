#!/usr/bin/env python3
"""Run and save V4.2 resource/throughput preflight, never formal training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_2_preflight import run_v4_2_preflight


def main() -> None:
    result = run_v4_2_preflight(ROOT)
    path = ROOT / "research/g_v4_2/results/preflight/preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path.relative_to(ROOT)),
                      "cpu_seconds": result["preflight_cpu_seconds"],
                      "wall_seconds": result["preflight_wall_seconds"],
                      "rates": result["measurements"],
                      "estimates": result["estimates"]["levels"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
