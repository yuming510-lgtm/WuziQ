#!/usr/bin/env python3
"""Replay-check all V6.2 game records and recompute championship points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records, write_json_atomic
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.schedule import build_schedule
from src.research.v6_2.validation import validate_v6_2_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_2/results")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    config = load_v6_2_config()
    records = load_game_records(output / "9x9_games.jsonl")
    schedule = build_schedule(config)
    result = validate_v6_2_records(
        records, schedule, require_complete=not args.allow_incomplete
    )
    result["validated_at_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    write_json_atomic(output / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
