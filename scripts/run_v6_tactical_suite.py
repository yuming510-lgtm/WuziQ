#!/usr/bin/env python3
"""Run the frozen V6 tactical corpus and persist its auditable JSON result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6.puzzles import run_tactical_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/v6/validation/tactical_suite.json")
    args = parser.parse_args()
    result = run_tactical_suite(ROOT / args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False))
    if not result["all_proofs_valid"] or not result["all_moves_legal"] or not result["all_boards_restored"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
