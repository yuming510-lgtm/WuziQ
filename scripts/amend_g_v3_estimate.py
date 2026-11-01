#!/usr/bin/env python3
"""Correct the pre-training G V3 evaluation game-count estimate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_config import amend_v3_evaluation_estimate_before_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    config = amend_v3_evaluation_estimate_before_training(args.root.resolve())
    print(json.dumps({
        "config_sha256": config["config_sha256"],
        "pre_training_amendments": config.get("pre_training_amendments", []),
        "runtime_estimate": config["runtime_estimate"],
    }, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
