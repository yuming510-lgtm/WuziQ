#!/usr/bin/env python3
"""Create a checksummed minimal archive of the current V6.2 progress."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6_2.snapshot import create_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_2/results")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(
        create_snapshot(ROOT / args.output_dir, ROOT / args.destination),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
