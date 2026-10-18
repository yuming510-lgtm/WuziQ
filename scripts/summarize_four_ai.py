"""Recompute board-size tournament summaries from the saved JSONL records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.metrics import aggregate_results
from src.research.runner import load_all_games


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "research/results")
    parser.add_argument("--output", type=Path, default=ROOT / "research/results/summary.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=5_102_026)
    args = parser.parse_args()

    records = load_all_games(args.results)
    summary = {}
    for board_size in (9, 15):
        games = [record for record in records if int(record["board_size"]) == board_size]
        summary[str(board_size)] = {
            "scope": "main" if board_size == 9 else "supplement",
            "board_size": board_size,
            "results": aggregate_results(
                games,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "games": len(records),
                "by_board_size": {
                    size: entry["results"]["game_count"]
                    for size, entry in summary.items()
                },
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
