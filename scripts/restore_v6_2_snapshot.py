#!/usr/bin/env python3
"""Restore one verified V6.2 snapshot into an empty output directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6_2.snapshot import restore_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = restore_snapshot(ROOT / args.snapshot, ROOT / args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
