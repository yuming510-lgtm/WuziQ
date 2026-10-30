#!/usr/bin/env python3
"""Prepare the frozen historical and train-only F V1 dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=731301)
    parser.add_argument("--validation-min", type=int, default=1000)
    parser.add_argument("--test-min", type=int, default=1000)
    parser.add_argument("--annotations", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "research/f_v1/data"
    result = prepare_dataset(
        args.root, output, args.seed, args.validation_min, args.test_min, args.annotations
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
