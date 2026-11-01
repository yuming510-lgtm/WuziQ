#!/usr/bin/env python3
"""Build one local G V3 recovery archive for verified Drive rotation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_snapshot import create_v3_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root.resolve() / args.output
    print(json.dumps(create_v3_snapshot(args.root.resolve(), output), sort_keys=True))


if __name__ == "__main__":
    main()
