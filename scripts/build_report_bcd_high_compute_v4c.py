from __future__ import annotations

import argparse
import json

from src.research.bcd_high_compute_v4c_report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the final V4C Chinese report and figures")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/bcd_high_compute_v4c.json")
    parser.add_argument("--results", default="research/v4c/results")
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--tests-failed", type=int, default=0)
    args = parser.parse_args()
    test_result = None if args.tests_passed is None else {"passed": args.tests_passed, "failed": args.tests_failed}
    result = build_report(args.root, args.config, args.results, test_result=test_result)
    print(json.dumps({"report_path": result["report_path"], "validation": result["validation"], "figures": result["figures"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
