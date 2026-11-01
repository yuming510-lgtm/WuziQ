#!/usr/bin/env python3
"""Measure F V2 data, training, and paired-match throughput and freeze schedules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.dataset import canonicalize_position
from src.research.imitation_f.evaluation import _board_from_canonical, evaluate_policy, extract_direct_tactics, play_match
from src.research.imitation_f.generation import annotate_position, build_evaluation_openings
from src.research.imitation_f.model import PolicyValueNet, load_f_checkpoint, select_device
from src.research.imitation_f.training import _augment, _encode_rows
from src.research.imitation_f.v2_experiment import analyze_f1_a_losses, build_experiment_plan, estimate_run_seconds


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def _resource_snapshot(root: Path) -> dict:
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, raw = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable"):
                memory[name] = int(raw.strip().split()[0]) * 1024
    except OSError:
        pass
    return {
        "cpu_logical": os.cpu_count(),
        "memory_bytes": memory,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "cuda_available": torch.cuda.is_available(),
        "device": str(select_device("auto")),
        "torch_version": torch.__version__,
    }


def _benchmark_updates(rows: list[dict], device: torch.device, count: int = 100) -> float:
    x, policy, values = _encode_rows(rows[:64])
    x, policy, values = x.to(device), policy.to(device), values.to(device)
    model = PolicyValueNet(channels=32, blocks=2, seed=731501).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    started = time.perf_counter()
    model.train()
    for _ in range(count):
        xb, target = _augment(x, policy)
        logits, value = model(xb)
        legal = xb[:, 2].reshape(-1, 81).bool()
        logits = logits.masked_fill(~legal, -1e9)
        loss = -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean() + (value - values).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / count


def run_preflight(root: Path, teacher_budget_seconds: float = 0.2) -> dict:
    root = root.resolve()
    v1 = root / "research/f_v1"
    output = root / "research/f_v2"
    data_dir, result_dir, checkpoint_dir = output / "data", output / "results", output / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [json.loads(line) for line in (v1 / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    split_manifest = json.loads((v1 / "data/historical_split_manifest.json").read_text(encoding="utf-8"))
    old_eval = json.loads((v1 / "results/evaluation_openings.json").read_text(encoding="utf-8"))
    old_matches = [json.loads(line) for line in (v1 / "results/matches.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    loss_analysis = analyze_f1_a_losses(old_matches)
    _atomic_json(result_dir / "loss_analysis.json", loss_analysis)

    frozen_f1 = checkpoint_dir / "f1_reference.pt"
    if not frozen_f1.exists():
        shutil.copyfile(v1 / "checkpoints/best.pt", frozen_f1)
    f1_sha = hashlib.sha256(frozen_f1.read_bytes()).hexdigest()
    f1_model = load_f_checkpoint(frozen_f1, device="auto")
    device = select_device("auto")
    torch.set_num_threads(3)

    used_groups = set(split_manifest["opening_group_assignment"])
    used_groups.update(str(row["opening_group_id"]) for row in old_eval)
    pilot_openings = build_evaluation_openings(732001, 2, tuple(sorted(used_groups)))
    pilot_matches = []
    match_seconds = []
    f_turn_counts = []
    for index, opening in enumerate(pilot_openings):
        for color in ("B", "W"):
            started = time.perf_counter()
            record = play_match(
                f1_model, "A", opening["opening"], color,
                int(opening["opening_seed"]), opponent_budget_seconds=0.05,
            )
            elapsed = time.perf_counter() - started
            record["stage"] = "v2-preflight-only"
            record["pilot_opening_group_id"] = opening["opening_group_id"]
            pilot_matches.append(record)
            match_seconds.append(elapsed)
            f_turn_counts.append(sum(move.get("model") == "F" for move in record["moves"]))
    _atomic_jsonl(result_dir / "preflight_pilot_matches.jsonl", pilot_matches)

    annotation_rows = []
    annotation_seconds = []
    sample_indices = np.linspace(0, len(train_rows) - 1, num=8, dtype=int)
    for sample, index in enumerate(sample_indices):
        row = train_rows[int(index)]
        board, _ = _board_from_canonical(row["canonical_board"], row["to_move"])
        teacher = "C" if sample % 2 == 0 else "E5"
        started = time.perf_counter()
        result = annotate_position(board, row["to_move"], teacher, teacher_budget_seconds, 732051 + sample)
        annotation_seconds.append(time.perf_counter() - started)
        annotation_rows.append({"teacher": teacher, **result})

    step_seconds = _benchmark_updates(train_rows, device)
    tactics = extract_direct_tactics(train_rows)
    tactical_counts = {kind: sum(row["kind"] == kind for row in tactics) for kind in ("win", "block")}
    train_games = 32
    mean_game = float(np.mean(match_seconds))
    mean_f_turns = float(np.mean(f_turn_counts))
    mean_annotation = float(np.mean(annotation_seconds))
    added_rows = int(round(train_games * mean_f_turns))
    non_tactical = len(train_rows) - len(tactics) + int(round(added_rows * 0.9))
    tactical_total = len(tactics) + int(round(added_rows * 0.1))
    train_rows_30 = non_tactical + max(tactical_total, int(np.ceil(non_tactical * 0.30 / 0.70)))
    train_rows_40 = non_tactical + max(tactical_total, int(np.ceil(non_tactical * 0.40 / 0.60)))
    dev_versions_first = 7  # six epoch candidates plus the frozen F1 control
    dev_versions_max = 13  # two capped runs plus the frozen F1 control
    eval_games_first = dev_versions_first * 16 + 80
    eval_games_max = dev_versions_max * 16 + 80
    estimate_30 = estimate_run_seconds(
        train_games=train_games,
        mean_game_seconds=mean_game,
        mean_f_turns_per_game=mean_f_turns,
        mean_annotation_seconds=mean_annotation,
        train_rows=train_rows_30,
        seconds_per_update=step_seconds,
        expected_epochs=45,
        maximum_epochs=60,
        maximum_training_runs=2,
        evaluation_games=eval_games_first,
        mean_evaluation_game_seconds=mean_game,
        overhead_seconds=90,
    )
    estimate_40 = estimate_run_seconds(
        train_games=train_games,
        mean_game_seconds=mean_game,
        mean_f_turns_per_game=mean_f_turns,
        mean_annotation_seconds=mean_annotation,
        train_rows=max(train_rows_30, train_rows_40),
        seconds_per_update=step_seconds,
        expected_epochs=45,
        maximum_epochs=60,
        maximum_training_runs=2,
        evaluation_games=eval_games_max,
        mean_evaluation_game_seconds=mean_game,
        overhead_seconds=90,
    )
    estimate = dict(estimate_30)
    estimate["conservative_seconds"] = estimate_40["conservative_seconds"]
    estimate["calculation"] = {
        "data": "32 F1-vs-A paired-color games plus C/E5 labels at measured 200ms budget",
        "training": "45 expected or 60 maximum epochs per run; 30% and 40% tactical mixes; max two runs",
        "evaluation": f"8 opening groups per candidate, 6 candidate epochs/run; up to {eval_games_max} dev/final games",
        "seconds_per_update": step_seconds,
        "annotation_seconds_per_label": mean_annotation,
        "pilot_game_seconds": mean_game,
    }
    base_train_sha = hashlib.sha256((v1 / "data/train.jsonl").read_bytes()).hexdigest()
    excluded = used_groups | {str(row["opening_group_id"]) for row in pilot_openings}
    plan = build_experiment_plan(732101, 16, 8, 20, tuple(sorted(excluded)))
    plan.update({
        "teacher_budget_seconds": teacher_budget_seconds,
        "training_game_color_pairs": 16,
        "development_pairs": 8,
        "blind_pairs": 20,
        "bootstrap_seed": 732301,
        "expected_epochs": 45,
        "minimum_epochs": 30,
        "maximum_epochs": 60,
        "patience": 12,
        "tactical_shares": [0.30, 0.40],
        "maximum_training_runs": 2,
        "f1_checkpoint_sha256": f1_sha,
        "v1_train_sha256": base_train_sha,
        "preflight_pilot_group_ids": [row["opening_group_id"] for row in pilot_openings],
        "preflight_sha256": hashlib.sha256(json.dumps(estimate, sort_keys=True).encode()).hexdigest(),
    })
    _atomic_json(output / "config/experiment.json", plan)
    preflight = {
        "resources": _resource_snapshot(root),
        "pilot": {
            "evaluation_games": len(pilot_matches),
            "mean_game_seconds": mean_game,
            "f1_turns_per_game": mean_f_turns,
            "teacher_annotations": len(annotation_rows),
            "mean_teacher_annotation_seconds": mean_annotation,
            "max_teacher_annotation_seconds": max(annotation_seconds),
            "teacher_budget_seconds": teacher_budget_seconds,
            "training_updates": 100,
            "seconds_per_update": step_seconds,
        },
        "historical_train": {"positions": len(train_rows), "tactical_positions": len(tactics), "tactical_by_kind": tactical_counts},
        "estimate": estimate,
        "f1_a_loss_analysis": loss_analysis,
        "f1_checkpoint_sha256": f1_sha,
        "schedule_sha256": plan["schedule_sha256"],
    }
    _atomic_json(result_dir / "preflight.json", preflight)
    return preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--teacher-budget-seconds", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.root, args.teacher_budget_seconds), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
