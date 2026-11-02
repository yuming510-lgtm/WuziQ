#!/usr/bin/env python3
"""Restore and validate a G V4 archive into an empty directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_snapshot import restore_v4_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore_v4_snapshot(args.archive, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
