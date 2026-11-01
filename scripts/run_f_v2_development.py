#!/usr/bin/env python3
"""Run one F1/F2 candidate on the frozen paired development openings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.model import load_f_checkpoint
from src.research.imitation_f.v2_evaluation import paired_opening_statistics, run_opening_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    schedule = json.loads((root / "research/f_v2/config/experiment.json").read_text(encoding="utf-8"))
    model_name = "F1" if args.candidate == "F1" else "F2"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(3)
    model = load_f_checkpoint(checkpoint, device="cpu")
    output_dir = root / "research/f_v2/results/development" / args.candidate
    records_path = output_dir / "matches.jsonl"
    summary = run_opening_matches(
        model, model_name, args.candidate, "A", schedule["development_openings"], records_path,
        "development", digest, int(schedule["seed"]) + 501, 0.05,
    )
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary["paired_statistics"] = paired_opening_statistics(
        rows, model_name, int(schedule["bootstrap_seed"]) + 1000,
        bootstrap_samples=2000,
    )
    summary["validation_policy_loss"] = model.checkpoint_metadata.get("validation", {}).get("policy_loss")
    summary["model_name"] = model_name
    summary["checkpoint"] = str(checkpoint.relative_to(root))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
