#!/usr/bin/env python3
"""Run V4 preflight, resumable matrix, local recovery, or report generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.abc_search_v4 import load_config, load_run_records, run_matrix
from src.research.abc_search_v4_analysis import write_report
from src.research.abc_search_v4_preflight import run_preflight
from src.research.abc_search_v4_snapshot import create_snapshot


DEFAULT_CONFIG = ROOT / "configs" / "abc_search_compute_v4.json"
DEFAULT_RESULTS = ROOT / "research" / "v4" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="measure resources and estimate full runtime")
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "preflight.json")

    run = commands.add_parser("run", help="execute or resume the official 252-run matrix")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    run.add_argument("--max-new-runs", type=int)

    analyze = commands.add_parser("analyze", help="validate results and write plots/report")
    analyze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analyze.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    analyze.add_argument(
        "--figures", type=Path, default=ROOT / "research" / "v4" / "figures"
    )

    snapshot = commands.add_parser("snapshot", help="create a local recovery ZIP")
    snapshot.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    snapshot.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "latest_snapshot.zip")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args.config, args.output, resource_path=ROOT)
        print(json.dumps(result["estimate"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        result = run_matrix(args.config, args.results, max_new_runs=args.max_new_runs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze":
        config = load_config(args.config)
        records_path = args.results / "search_runs.jsonl"
        records = load_run_records(records_path)
        preflight_path = args.results / "preflight.json"
        preflight_data = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
        execution_path = args.results / "execution.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else None
        estimate = preflight_data.get("estimate", {})
        report_preflight = {
            "estimated_seconds": estimate.get("estimated_official_compute_seconds", 0),
            "estimated_minutes": estimate.get("estimated_official_compute_minutes", 0),
        }
        result = write_report(
            config,
            records,
            args.results / "report_zh.md",
            args.figures,
            preflight=report_preflight if preflight_data else None,
            resources=preflight_data.get("resources_after"),
            execution=execution,
        )
        print(json.dumps({"report": result["report"], "figures": result["figures"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "snapshot":
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        manifest = create_snapshot(
            args.results / "config.json",
            args.results / "progress.json",
            args.results / "search_runs.jsonl",
            code_commit,
            args.output,
        )
        print(json.dumps({"snapshot": str(args.output), "files": manifest["files"]}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
