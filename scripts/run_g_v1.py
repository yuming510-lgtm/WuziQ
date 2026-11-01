#!/usr/bin/env python3
"""Execute the registered G0 and G-F2 self-play training budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.runner import run_registered_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("research/g_v1/config/experiment.json"))
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else args.root / args.config
    result = run_registered_training(args.root, config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["completed"] or not result["training_budget_equal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
