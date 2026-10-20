from __future__ import annotations

import argparse
import json

from src.research.bcd_high_compute_v4c import repository_commit
from src.research.bcd_high_compute_v4c_snapshot import create_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the minimal V4C recovery archive")
    parser.add_argument("--results", default="research/v4c/results")
    parser.add_argument("--output", default="research/v4c/results/v4c_recovery.zip")
    parser.add_argument("--code-version", default=repository_commit())
    args = parser.parse_args()
    print(json.dumps(create_snapshot(args.results, args.output, code_version=args.code_version), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
