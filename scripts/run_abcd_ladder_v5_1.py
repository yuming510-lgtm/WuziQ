#!/usr/bin/env python3
"""Resume the independent V5.1 ABCD budget-version ladder."""
from __future__ import annotations

import argparse
import json

from src.research.abcd_ladder_v5_1_runner import repository_commit, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/abcd_ladder_v5_1.json")
    parser.add_argument("--output-dir", default="research/v5_1/results")
    parser.add_argument("--reuse-v5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--implementation-commit")
    args = parser.parse_args()
    progress = run_experiment(
        args.config, args.output_dir, reuse_v5=args.reuse_v5,
        max_games=args.max_games, max_groups=args.max_groups,
        implementation_commit=args.implementation_commit or repository_commit(),
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
