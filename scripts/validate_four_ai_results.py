"""Replay every saved game and verify the tournament schedule and quotas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.runner import load_all_games
from src.research.validation import validate_tournament


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/four_ai_experiment.json")
    parser.add_argument("--results", type=Path, default=ROOT / "research/results")
    parser.add_argument("--output", type=Path, default=ROOT / "research/results/validation.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    saved_config = json.loads((args.results / "config.json").read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("Result directory config differs from the requested experiment config")
    report = validate_tournament(load_all_games(args.results), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
