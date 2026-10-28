#!/usr/bin/env python3
"""Create a validated minimal local V6.3 recovery archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6_3.snapshot import create_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_3/results")
    parser.add_argument("--snapshot", default="research/v6_3/recovery/v6_3_recovery_latest.zip")
    args = parser.parse_args()
    result = create_snapshot(ROOT / args.output_dir, ROOT / args.snapshot)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
