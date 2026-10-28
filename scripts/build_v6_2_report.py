#!/usr/bin/env python3
"""Build final V6.2 score tables, compute chart, and Chinese report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.v6_2.analysis import build_report
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.schedule import build_schedule
from src.research.v6_2.validation import validate_v6_2_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_2")
    args = parser.parse_args()
    root = ROOT / args.output_dir
    result_dir = root / "results"
    records = load_game_records(result_dir / "9x9_games.jsonl")
    config = load_v6_2_config()
    schedule = build_schedule(config)
    validation = validate_v6_2_records(records, schedule, require_complete=True)
    progress = json.loads((result_dir / "progress.json").read_text(encoding="utf-8"))
    preflight = json.loads((root / "preflight/preflight.json").read_text(encoding="utf-8"))
    environment = json.loads((result_dir / "environment.json").read_text(encoding="utf-8"))
    result = build_report(records, validation, progress, preflight, environment, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
