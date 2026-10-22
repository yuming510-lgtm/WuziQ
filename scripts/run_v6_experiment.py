#!/usr/bin/env python3
"""Run/resume the frozen V6 budget-ladder experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/experiment.json")
    parser.add_argument("--output-dir", default="research/v6/results")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--no-v5-1-reuse", action="store_true")
    parser.add_argument("--implementation-commit", default=None)
    args = parser.parse_args()
    result = run_experiment(
        ROOT / args.config,
        ROOT / args.output_dir,
        reuse_v5_1=not args.no_v5_1_reuse,
        max_groups=args.max_groups,
        implementation_commit=args.implementation_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
