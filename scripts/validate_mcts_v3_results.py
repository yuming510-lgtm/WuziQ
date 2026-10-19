"""Validate V3 game IDs, legal replays, paired seats, quotas, and completeness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.mcts_v3_analysis import validate_v3_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "research/v3/results")
    parser.add_argument("--config", type=Path, default=ROOT / "research/v3/results/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/v3/results/validation.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    records = [
        record
        for path in sorted(args.results_dir.glob("*_games.jsonl"))
        for record in load_game_records(path)
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = validate_v3_records(
        records, config, require_complete=not args.allow_incomplete
    )
    progress_path = args.results_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress.get("games_completed", -1)) != result["game_count"]:
            raise ValueError("progress.json games_completed differs from unique raw records")
        result["progress_file"] = progress
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "games": result["game_count"],
                "scheduled": result["scheduled_game_count"],
                "complete": result["complete"],
                "illegal_moves": result["illegal_moves"],
                "paired_openings": result["paired_opening_groups"],
                "unpaired_openings": result["unpaired_opening_group_count"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
