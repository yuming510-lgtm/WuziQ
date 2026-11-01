#!/usr/bin/env python3
"""Run the frozen 20-opening F1/F2 blind challenge and capability evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.evaluation import evaluate_policy, evaluate_tactics, extract_direct_tactics
from src.research.imitation_f.model import load_f_checkpoint
from src.gomoku import GomokuBoard
from src.research.imitation_f.v2_evaluation import paired_difference_ci, paired_opening_statistics, run_opening_matches


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    experiment = json.loads((root / "research/f_v2/config/experiment.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "research/f_v2/results/model_selection.json").read_text(encoding="utf-8"))
    f1_path = root / "research/f_v2/checkpoints/f1_reference.pt"
    f2_path = root / selection["selected_checkpoint"]
    f1_sha = hashlib.sha256(f1_path.read_bytes()).hexdigest()
    f2_sha = hashlib.sha256(f2_path.read_bytes()).hexdigest()
    if f1_sha != experiment["f1_checkpoint_sha256"] or f2_sha != selection["selected_checkpoint_sha256"]:
        raise ValueError("final blind checkpoint hashes do not match frozen selection metadata")
    train_groups = {str(row["opening_group_id"]) for row in experiment["training_openings"]}
    dev_groups = {str(row["opening_group_id"]) for row in experiment["development_openings"]}
    blind_groups = {str(row["opening_group_id"]) for row in experiment["blind_openings"]}
    excluded_groups = set(experiment["excluded_group_ids"])
    if len(blind_groups) != 20 or train_groups & blind_groups or dev_groups & blind_groups:
        raise ValueError("blind opening schedule overlaps training/development or is incomplete")
    if blind_groups & excluded_groups:
        raise ValueError("blind opening schedule reuses an excluded historical or preflight opening")
    for spec in experiment["blind_openings"]:
        if len(spec["opening"]) != 4:
            raise ValueError("blind openings must have four pre-registered plies")
        board = GomokuBoard(9)
        next_player = "B"
        for move in spec["opening"]:
            if next_player != move["player"]:
                raise ValueError("blind opening has an out-of-turn move")
            coord = int(move["row"]), int(move["col"])
            board.place_stone(str(move["player"]), coord)
            if board.check_victory(coord, str(move["player"])):
                raise ValueError("blind opening is terminal")
            next_player = "W" if next_player == "B" else "B"
    torch.set_num_threads(3)
    blind_dir = root / "research/f_v2/results/blind"
    models = {
        "F1": (load_f_checkpoint(f1_path, device="cpu"), f1_sha),
        "F2": (load_f_checkpoint(f2_path, device="cpu"), f2_sha),
    }
    summaries = {}
    record_sets = {}
    for name, (model, digest) in models.items():
        path = blind_dir / f"matches_{name.lower()}.jsonl"
        run_summary = run_opening_matches(
            model, name, name, "A", experiment["blind_openings"], path,
            "blind", digest, int(experiment["seed"]) + 601, 0.05,
        )
        records = _jsonl(path)
        record_sets[name] = records
        run_summary["paired_statistics"] = paired_opening_statistics(
            records, name, int(experiment["bootstrap_seed"]), bootstrap_samples=10_000,
        )
        summaries[name] = run_summary

    paired = paired_difference_ci(
        record_sets["F1"], record_sets["F2"], "F1", "F2",
        int(experiment["bootstrap_seed"]) + 17, bootstrap_samples=10_000,
    )
    data_dir = root / "research/f_v1/data"
    validation_rows = _jsonl(data_dir / "validation.jsonl")
    legacy_blind_rows = _jsonl(data_dir / "blind_test.jsonl")
    tactics = extract_direct_tactics(legacy_blind_rows)
    quality = {}
    for name, (model, digest) in models.items():
        quality[name] = {
            "checkpoint_sha256": digest,
            "validation_policy_value": evaluate_policy(model, validation_rows, device="cpu", latency_sample_size=100),
            "legacy_blind_policy_value": evaluate_policy(model, legacy_blind_rows, device="cpu", latency_sample_size=100),
            "legacy_blind_direct_tactics": evaluate_tactics(model, tactics, device="cpu"),
        }
    summary = {
        "schema_version": 1, "experiment_id": "wuziq-imitation-learning-f-v2",
        "schedule_sha256": experiment["schedule_sha256"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "selection_rule": selection["selection_rule"],
        "blind_opening_groups": len(experiment["blind_openings"]),
        "blind_opening_group_ids": [row["opening_group_id"] for row in experiment["blind_openings"]],
        "blind_seed": int(experiment["seed"]) + 601,
        "paired_f2_minus_f1": paired,
        "vs_a": summaries,
        "policy_value_and_tactics": quality,
        "results_paths": {name: str((blind_dir / f"matches_{name.lower()}.jsonl").relative_to(root))
                          for name in models},
    }
    blind_dir.mkdir(parents=True, exist_ok=True)
    (blind_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "vs_a": {name: row["paired_statistics"] for name, row in summaries.items()},
        "paired_f2_minus_f1": paired,
        "tactical_positions": len(tactics),
        "f2_metrics": quality["F2"]["legacy_blind_policy_value"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
