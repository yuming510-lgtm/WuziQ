from __future__ import annotations

import argparse
import json

from src.research.bcd_high_compute_v4c_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V4C pre-match performance and resource checks")
    parser.add_argument("--config", default="configs/bcd_high_compute_v4c.json")
    parser.add_argument("--output", default="research/v4c/results/preflight.json")
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
