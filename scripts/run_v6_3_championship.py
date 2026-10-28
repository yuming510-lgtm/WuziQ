#!/usr/bin/env python3
"""Run or resume the V6.3 C versus E5 paired tiebreak."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6_3.runner import run_championship


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_3/championship.json")
    parser.add_argument("--output-dir", default="research/v6_3/results")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--max-games", type=int, help="stop after this many newly completed games (for verified checkpoints)")
    args = parser.parse_args()
    result = run_championship(
        ROOT / args.config, ROOT / args.output_dir,
        max_games_this_run=args.max_games, initialize_only=args.initialize_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
