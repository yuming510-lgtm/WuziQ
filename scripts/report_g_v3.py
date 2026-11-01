#!/usr/bin/env python3
"""Validate G V3 data and generate its Chinese report and plots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_report import generate_v3_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(generate_v3_report(args.root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
