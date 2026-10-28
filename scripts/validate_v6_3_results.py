#!/usr/bin/env python3
"""Replay every V6.3 game and reconcile the durable artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.v6.config import canonical, sha256
from src.research.v6_3.config import load_v6_3_config
from src.research.v6_3.schedule import build_schedule
from src.research.v6_3.validation import validate_v6_3_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_3/results")
    args = parser.parse_args()
    root = ROOT / args.output_dir
    config = load_v6_3_config(root / "config.json")
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    if canonical(schedule) != canonical(build_schedule(config)):
        raise ValueError("V6.3 schedule differs from deterministic preregistration")
    records = load_game_records(root / "9x9_games.jsonl")
    checked = validate_v6_3_records(records, schedule, require_complete=True)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if not progress.get("complete") or progress.get("completed_games") != checked["games"]:
        raise ValueError("V6.3 progress does not match final replay")
    if (progress.get("config_sha256") != sha256(canonical(config))
            or progress.get("schedule_sha256") != sha256(canonical(schedule))):
        raise ValueError("V6.3 frozen config or schedule hash mismatch")
    if sorted(progress.get("completed_ids", [])) != sorted(row["game_id"] for row in records):
        raise ValueError("V6.3 progress IDs do not exactly match the durable game ledger")

    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("new_games") != checked["games"]
                or summary.get("tiebreak_status") != checked["tiebreak_status"]
                or summary.get("champions") != checked["leaders"]):
            raise ValueError("V6.3 summary disagrees with replayed game data")
        for model in ("C", "E5"):
            for field in ("wins", "draws", "losses", "points"):
                if summary["standings"][model][field] != checked["standings"][model][field]:
                    raise ValueError(f"V6.3 summary has incorrect {model} {field}")
    if (root / "games.csv").is_file():
        with (root / "games.csv").open(encoding="utf-8", newline="") as handle:
            if sum(1 for _ in csv.DictReader(handle)) != checked["games"]:
                raise ValueError("V6.3 games.csv row count differs from raw games")
    if (root / "moves.csv").is_file():
        with (root / "moves.csv").open(encoding="utf-8", newline="") as handle:
            exported_moves = sum(1 for _ in csv.DictReader(handle))
        actual_moves = sum(len(game["moves"]) - 4 for game in records)
        if exported_moves != actual_moves:
            raise ValueError("V6.3 moves.csv row count differs from replayed decisions")

    result = {
        "valid": True,
        "games": checked["games"],
        "unique_ids": checked["unique_ids"],
        "replayed_moves": checked["replayed_moves"],
        "tiebreak_status": checked["tiebreak_status"],
        "standings": checked["standings"],
        "config_sha256": sha256(canonical(config)),
        "schedule_sha256": sha256(canonical(schedule)),
        "raw_games_sha256": sha256((root / "9x9_games.jsonl").read_bytes()),
        "complete": checked["complete"],
    }
    (root / "integrity_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
