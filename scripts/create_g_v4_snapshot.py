#!/usr/bin/env python3
"""Create one verified local G V4 recovery archive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_snapshot import create_v4_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_v4_snapshot(ROOT, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
