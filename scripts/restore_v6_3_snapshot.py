#!/usr/bin/env python3
"""Restore a previously downloaded V6.3 archive into a new empty directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6_3.snapshot import restore_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="local path to a Drive-downloaded V6.3 recovery zip")
    parser.add_argument("destination", help="new or empty destination directory")
    args = parser.parse_args()
    result = restore_snapshot(Path(args.snapshot), Path(args.destination))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
