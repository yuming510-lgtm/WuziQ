#!/usr/bin/env python3
"""Freeze or resume the mixed-engine F V1 data generation schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.generation import build_generation_config, run_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=731401)
    parser.add_argument("--openings-per-pair", type=int, default=5)
    parser.add_argument("--budget-seconds", type=float, default=0.05)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "research/f_v1/config/generation.json"
    output_dir = args.output or root / "research/f_v1/results"
    split_path = root / "research/f_v1/data/historical_split_manifest.json"
    if not split_path.exists():
        raise FileNotFoundError("prepare the historical dataset before generating mixed games")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        config = build_generation_config(
            seed=args.seed,
            openings_per_pair=args.openings_per_pair,
            budget_seconds=args.budget_seconds,
            excluded_opening_group_ids=tuple(split.get("opening_group_assignment", {})),
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temp = config_path.with_suffix(config_path.suffix + ".tmp")
        temp.write_text(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temp.replace(config_path)
    if args.config_only:
        print(json.dumps({"config": str(config_path), "maximum_games": config["maximum_games"]}, indent=2))
        return
    result = run_generation(config, output_dir, split_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
