#!/usr/bin/env python3
"""Create the V4.2 fixed config and held-out opening schedule."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_2_registration import create_v4_2_registration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-per-arm", type=int, default=10_000)
    args = parser.parse_args()
    config, schedule = create_v4_2_registration(ROOT, states_per_arm=args.states_per_arm)
    print(json.dumps({"config_sha256": config["config_sha256"],
                      "opening_schedule_sha256": schedule["schedule_sha256"],
                      "historical_candidates": len(config["historical_candidates"]),
                      "excluded_opening_digests": len(schedule["excluded_opening_digests"]),
                      "training_groups": len(schedule["train_openings"]),
                      "development_groups": len(schedule["development_openings"]),
                      "blind_groups": len(schedule["blind_openings"])}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
