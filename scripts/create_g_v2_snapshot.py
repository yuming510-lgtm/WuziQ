#!/usr/bin/env python3
"""Create a local, allowlisted G V2 recovery archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v2_snapshot import create_v2_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(create_v2_snapshot(root, output), sort_keys=True))


if __name__ == "__main__":
    main()
