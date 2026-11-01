#!/usr/bin/env python3
"""Run/resume one frozen F V2 continuation training configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.training import train


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    initial = Path(config["initial_checkpoint"])
    config["initial_checkpoint"] = str(initial if initial.is_absolute() else root / initial)
    share = int(round(float(config["tactical_share"]) * 100))
    dataset = {
        "train": root / f"research/f_v2/data/train_tactical_{share:02d}.jsonl",
        "validation": root / "research/f_v2/data/validation.jsonl",
    }
    output = root / "research/f_v2/checkpoints" / str(config["run_name"])
    result = train(config, dataset, output, resume=not args.no_resume)
    summary = {
        "run_name": config["run_name"],
        "config": config,
        "dataset_sha256": hashlib.sha256(Path(dataset["train"]).read_bytes()).hexdigest(),
        "initial_checkpoint_sha256": result["initial_checkpoint_sha256"],
        "updates": result["updates"], "best_epoch": result["best_epoch"],
        "elapsed_seconds": result["elapsed_seconds"], "device": result["device"],
        "best_checkpoint": str(result["best_checkpoint"]),
        "state_checkpoint": str(result["state_checkpoint"]),
        "candidate_checkpoints": sorted(path.name for path in output.glob("epoch-*.pt")),
        "history": result["history"],
    }
    summary["best_checkpoint_sha256"] = hashlib.sha256(Path(result["best_checkpoint"]).read_bytes()).hexdigest()
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"},
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
