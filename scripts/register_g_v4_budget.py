#!/usr/bin/env python3
"""Register the measured G V4 budget before any formal training starts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_config import register_v4_config


def main() -> None:
    config = register_v4_config(ROOT)
    print(json.dumps({
        "registered": config["registered"], "config_sha256": config["config_sha256"],
        "estimate": config["runtime_estimate"], "hard_cpu_hour_cap": config["hard_cpu_hour_cap"],
        "max_states": config["stages"]["maximum_total_new_states"],
        "minimum_states": config["stages"]["minimum_planned_new_states"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
