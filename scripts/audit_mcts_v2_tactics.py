"""Replay V2 games and audit exact immediate win/block opportunities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.mcts_v2_tactics import audit_tactics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "research/v2/results")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/v2/results/tactical_audit.json"
    )
    args = parser.parse_args()

    records = [
        record
        for path in sorted(args.results_dir.glob("*_games.jsonl"))
        for record in load_game_records(path)
    ]
    result = audit_tactics(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "games_audited": result["games_audited"],
                "miss_count": result["miss_count"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
