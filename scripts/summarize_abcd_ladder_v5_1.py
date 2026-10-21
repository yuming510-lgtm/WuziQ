#!/usr/bin/env python3
"""Replay, validate, summarize and render the V5.1 Chinese report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.abcd_ladder_v5_1_analysis import write_analysis
from src.research.abcd_ladder_v5_1_report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="research/v5_1/results")
    parser.add_argument("--analysis-dir", default="research/v5_1/analysis")
    parser.add_argument("--report-dir", default="research/v5_1")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summary = write_analysis(args.results_dir, args.analysis_dir,
                             require_complete=not args.allow_incomplete)
    preflight_path = Path(args.results_dir) / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    path = build_report(summary, args.results_dir, args.report_dir, preflight=preflight)
    print(json.dumps({"report": str(path), "games": summary["actual_games"],
                      "groups": summary["actual_groups"], "complete": summary["complete"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
