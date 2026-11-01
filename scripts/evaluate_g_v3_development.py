#!/usr/bin/env python3
"""Resume one registered G V3 development/promotion checkpoint evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_evaluation import run_v3_development, run_v3_tactical_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tactics", action="store_true")
    args = parser.parse_args()
    rows = run_v3_development(args.root, node=args.node, seed=args.seed)
    tactics = run_v3_tactical_stage(args.root, node=args.node, seed=args.seed) if args.tactics else []
    print(json.dumps({"node": args.node, "seed": args.seed, "new_development_records": len(rows),
                      "new_tactical_records": len(tactics)}, sort_keys=True))


if __name__ == "__main__":
    main()
