#!/usr/bin/env python3
"""Verify and restore a G V2 recovery archive into an empty directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v2_snapshot import restore_v2_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore_v2_snapshot(args.archive, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
