#!/usr/bin/env python3
"""Freeze the V4 training, development, and blind opening pools."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_config import _atomic_json, build_v4_opening_schedule


def main() -> None:
    path = ROOT / "research/g_v4/config/openings.json"
    schedule = build_v4_opening_schedule(ROOT)
    _atomic_json(path, schedule)
    print(json.dumps({
        "path": str(path), "sha256": schedule["schedule_sha256"],
        "training_groups_per_seed": schedule["train_groups_per_seed"],
        "development_groups": schedule["development_groups"],
        "blind_groups": schedule["blind_groups"],
        "excluded_v3_opening_count": schedule["excluded_v3_opening_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
