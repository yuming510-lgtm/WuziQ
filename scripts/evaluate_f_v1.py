#!/usr/bin/env python3
"""Compare fixed-seed untrained F with the selected checkpoint on blind data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.research.imitation_f.evaluation import evaluate_policy, evaluate_tactics, extract_direct_tactics, predict_rows
from src.research.imitation_f.model import PolicyValueNet, load_f_checkpoint


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "research/f_v1/data"
    results_dir = root / "research/f_v1/results"
    checkpoint = args.checkpoint or root / "research/f_v1/checkpoints/best.pt"
    config = json.loads((root / "research/f_v1/config/training.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    blind_rows = _read_jsonl(data_dir / "blind_test.jsonl")
    validation_rows = _read_jsonl(data_dir / "validation.jsonl")
    untrained = PolicyValueNet(seed=int(config["seed"]))
    trained = load_f_checkpoint(checkpoint, device="auto")
    baseline_metrics = evaluate_policy(untrained, blind_rows)
    trained_metrics = evaluate_policy(trained, blind_rows)
    tactics = extract_direct_tactics(blind_rows)
    baseline_tactics = evaluate_tactics(untrained, tactics)
    trained_tactics = evaluate_tactics(trained, tactics)
    validation_tactics = extract_direct_tactics(validation_rows)
    validation_tactic_metrics = evaluate_tactics(trained, validation_tactics)
    baseline_predictions = predict_rows(untrained, blind_rows)
    trained_predictions = predict_rows(trained, blind_rows)
    prediction_rows = []
    for before, after in zip(baseline_predictions, trained_predictions):
        prediction_rows.append({
            "state_key": after["state_key"],
            "baseline_action": before["predicted_action"],
            "baseline_correct": before["top1_correct"],
            "trained_action": after["predicted_action"],
            "trained_correct": after["top1_correct"],
            "teacher_actions": after["correct_actions"],
            "trained_policy_probability": after["predicted_probability"],
            "trained_value": after["value_prediction"],
        })
    annotation_rows = _read_jsonl(results_dir / "generated_annotations.jsonl")
    teacher_counts = {}
    for row in annotation_rows:
        name = row["teacher"]
        teacher_counts[name] = teacher_counts.get(name, 0) + 1
    training_state = torch.load(root / "research/f_v1/checkpoints/training_state.pt", map_location="cpu", weights_only=False)
    preflight = json.loads((results_dir / "preflight.json").read_text(encoding="utf-8"))
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "dataset": {
            "counts": manifest["counts"],
            "historical_games": manifest["historical"]["unique_games"],
            "historical_deduplicated_positions": manifest["historical"]["positions"],
            "generated_games": 36,
            "generated_annotations": len(annotation_rows),
            "teacher_annotation_counts": teacher_counts,
            "sha256": {key: value["sha256"] for key, value in manifest["files"].items()},
            "source_manifest_sha256": manifest["split_hash"],
        },
        "training": {
            "device": preflight["resource_snapshot"]["device"],
            "torch_version": torch.__version__,
            "seed": config["seed"],
            "epochs_completed": len(training_state.get("history", [])),
            "updates": training_state.get("update_step"),
            "elapsed_training_seconds": training_state.get("elapsed_training_seconds"),
            "training_started_at_utc": training_state.get("training_started_at_utc"),
            "best_epoch": trained.checkpoint_metadata.get("best_epoch"),
            "best_validation": trained.checkpoint_metadata.get("validation"),
            "checkpoint": "checkpoints/best.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_version": manifest["split_hash"],
            "optimizer": "AdamW",
            "loss": "masked soft-target cross entropy + terminal-outcome value MSE",
            "train_loss_curve_note": "training curve records last batch loss per epoch; validation metrics cover the full validation partition",
        },
        "blind_test": {"baseline_untrained": baseline_metrics, "trained": trained_metrics},
        "direct_tactics": {
            "sample_definition": "exact immediate own win, else single legal move blocking opponent's immediate five",
            "positions": len(tactics),
            "baseline_untrained": baseline_tactics,
            "trained": trained_tactics,
        },
        "validation_direct_tactics": {
            "positions": len(validation_tactics),
            "trained": validation_tactic_metrics,
        },
    }
    _atomic_json(results_dir / "metrics.json", result)
    _atomic_jsonl(results_dir / "blind_predictions.jsonl", prediction_rows)
    _atomic_jsonl(results_dir / "blind_direct_tactics.jsonl", tactics)
    print(json.dumps({
        "dataset": result["dataset"],
        "training": result["training"],
        "blind_test": result["blind_test"],
        "direct_tactics": {
            "positions": len(tactics),
            "baseline": {key: value for key, value in baseline_tactics.items() if key != "details"},
            "trained": {key: value for key, value in trained_tactics.items() if key != "details"},
        },
        "validation_direct_tactics": {
            "positions": len(validation_tactics),
            "trained": {key: value for key, value in validation_tactic_metrics.items() if key != "details"},
        },
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
