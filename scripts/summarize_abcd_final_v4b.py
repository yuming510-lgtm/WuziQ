from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.abcd_final_v4b_report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate V4B games and generate its Chinese report")
    parser.add_argument("--results", default="research/v4b/results")
    args = parser.parse_args()
    result = build_report(args.results)
    print(json.dumps({"validation": result["validation"], "report_path": result["report_path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
