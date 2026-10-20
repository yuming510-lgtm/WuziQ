from __future__ import annotations

import argparse
import json

from src.research.bcd_high_compute_v4c_analysis import write_analysis
from src.research.bcd_high_compute_v4c import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize V4C results")
    parser.add_argument("--config", default="configs/bcd_high_compute_v4c.json")
    parser.add_argument("--results", default="research/v4c/results")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = write_analysis(args.results, load_config(args.config), require_complete=not args.allow_partial)
    print(json.dumps(summary["validation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
