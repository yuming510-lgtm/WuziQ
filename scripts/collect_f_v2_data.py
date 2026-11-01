#!/usr/bin/env python3
"""Collect idempotent, train-only F1-vs-A DAgger examples for F V2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.v2_data import collect_dagger_data


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    schedule_path = args.schedule or root / "research/f_v2/config/experiment.json"
    checkpoint = args.checkpoint or root / "research/f_v2/checkpoints/f1_reference.pt"
    output = args.output or root / "research/f_v2/results/dagger"
    v1_data = root / "research/f_v1/data"
    experiment = json.loads(schedule_path.read_text(encoding="utf-8"))
    expected_hash = experiment["f1_checkpoint_sha256"]
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("frozen F1 checkpoint SHA-256 differs from experiment schedule")
    heldout = {
        str(row["state_key"])
        for split in ("validation.jsonl", "blind_test.jsonl")
        for row in _jsonl(v1_data / split)
    }
    result = collect_dagger_data(
        checkpoint, experiment["training_openings"], output,
        float(experiment["teacher_budget_seconds"]), int(experiment["seed"]) + 401,
        heldout_state_keys=heldout,
    )
    result.update({
        "f1_checkpoint_sha256": actual_hash,
        "schedule_sha256": experiment["schedule_sha256"],
        "heldout_states": len(heldout),
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "dagger_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
