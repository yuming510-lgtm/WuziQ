#!/usr/bin/env python3
"""Train/resume F V1 from the frozen three-way dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "research/f_v1/config/training.json"
    data_dir = args.data or root / "research/f_v1/data"
    checkpoint_dir = args.checkpoint_dir or root / "research/f_v1/checkpoints"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = train(
        config,
        {"train": data_dir / "train.jsonl", "validation": data_dir / "validation.jsonl"},
        checkpoint_dir,
        resume=not args.no_resume,
    )
    summary = {key: str(value) if isinstance(value, Path) else value for key, value in result.items() if key != "model"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
