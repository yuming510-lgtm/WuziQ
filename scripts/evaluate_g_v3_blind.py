#!/usr/bin/env python3
"""Run/resume the once-only frozen G V3 blind test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_evaluation import run_v3_blind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = run_v3_blind(args.root)
    print(json.dumps({"blind_games_new": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
