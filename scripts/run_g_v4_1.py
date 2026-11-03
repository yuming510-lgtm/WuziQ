#!/usr/bin/env python3
"""Run one registered G0 self-play arm, or one resumable evaluation batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_1_config import ARM_IDS
from src.research.alphazero_g.v4_1_evaluation import (
    run_blind, run_development, select_development_finalist,
)
from src.research.alphazero_g.v4_1_runner import run_v4_1_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm", choices=ARM_IDS)
    group.add_argument("--development", action="store_true")
    group.add_argument("--select-finalist", action="store_true")
    group.add_argument("--blind", action="store_true")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-matches", type=int, default=None)
    args = parser.parse_args()
    if args.arm:
        result = run_v4_1_arm(ROOT, args.arm, max_games=args.max_games)
    elif args.development:
        result = run_development(ROOT, max_matches=args.max_matches)
    elif args.select_finalist:
        result = select_development_finalist(ROOT)
    else:
        result = run_blind(ROOT, max_matches=args.max_matches)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
