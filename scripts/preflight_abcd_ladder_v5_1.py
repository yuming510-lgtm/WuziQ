#!/usr/bin/env python3
"""Check V5.1 schedule/resources, benchmark every fixed budget and estimate runtime."""
from __future__ import annotations

import argparse
import json

from src.research.abcd_ladder_v5_1_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/abcd_ladder_v5_1.json")
    parser.add_argument("--output", default="research/v5_1/results/preflight.json")
    args = parser.parse_args()
    result = run_preflight(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
