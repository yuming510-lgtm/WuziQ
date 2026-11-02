#!/usr/bin/env python3
"""Advance the registered G V4 experiment by one durable stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_runner import advance_v4_experiment, initialize_v4_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize", action="store_true", help="prepare only initial A/B checkpoints and resume states")
    parser.add_argument("--advance", action="store_true", help="run the next idempotent experiment stage")
    parser.add_argument("--advances", type=int, default=1, help="number of stages to advance before returning")
    args = parser.parse_args()
    if args.initialize:
        print(json.dumps(initialize_v4_jobs(ROOT), sort_keys=True))
    elif args.advance:
        if args.advances < 1:
            raise SystemExit("--advances must be positive")
        for index in range(args.advances):
            result = advance_v4_experiment(ROOT)
            result["advance_index"] = index + 1
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if result.get("completed") or result.get("kind") == "training_incomplete":
                break
    else:
        parser.error("choose --initialize or --advance")


if __name__ == "__main__":
    main()
