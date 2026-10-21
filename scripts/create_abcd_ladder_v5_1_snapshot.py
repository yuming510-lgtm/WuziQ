#!/usr/bin/env python3
"""Create a minimal V5.1 recovery snapshot."""
from __future__ import annotations

import argparse
import json

from src.research.abcd_ladder_v5_1_snapshot import create_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="research/v5_1/results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(create_snapshot(args.results_dir, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
