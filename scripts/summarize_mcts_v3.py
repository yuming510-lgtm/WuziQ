"""Summarize completed V3 games and generate the D3 budget-trend SVG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.mcts_v3_analysis import (
    summarize_v3_records,
    validate_v3_records,
    write_d3_budget_trend_svg,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "research/v3/results")
    parser.add_argument("--config", type=Path, default=ROOT / "research/v3/results/config.json")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "research/v3/results/summary.json"
    )
    parser.add_argument(
        "--chart", type=Path, default=ROOT / "research/v3/figures/d3_budget_trend.svg"
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    records = [
        record
        for path in sorted(args.results_dir.glob("*_games.jsonl"))
        for record in load_game_records(path)
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation = validate_v3_records(
        records, config, require_complete=not args.allow_incomplete
    )
    summary = summarize_v3_records(records, bootstrap_samples=args.bootstrap_samples)
    summary["experiment_id"] = config["experiment_id"]
    summary["code_commit"] = config.get("code_commit")
    summary["validation"] = validation
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_d3_budget_trend_svg(summary, args.chart)
    print(
        json.dumps(
            {
                "games": summary["game_count"],
                "phases": summary["phase_count"],
                "complete": validation["complete"],
                "summary": str(args.summary),
                "chart": str(args.chart),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
