#!/usr/bin/env python3
"""Freeze G V1 training/evaluation quotas after measured preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from src.research.alphazero_g.experiment import register_g_budget, validate_g_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", type=Path, default=Path("research/g_v1/results/preflight.json"))
    parser.add_argument("--schedule", type=Path, default=Path("research/g_v1/config/openings.json"))
    parser.add_argument("--output", type=Path, default=Path("research/g_v1/config/experiment.json"))
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--environment-steps", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--cpu-hour-cap", type=float, default=2.0)
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    preflight_path, schedule_path, output_path = map(resolve, (args.preflight, args.schedule, args.output))
    if output_path.exists():
        raise FileExistsError(f"registered experiment already exists: {output_path}")
    preflight = json.loads(preflight_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    validate_g_schedule(schedule)
    parameters = {
        "seed": schedule["seed"],
        "simulations_per_move": args.simulations,
        "environment_steps_per_arm": args.environment_steps,
        "updates_per_arm": args.updates,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "stage_environment_steps": [args.environment_steps // 2, args.environment_steps - args.environment_steps // 2],
        "stage_updates": [args.updates // 2, args.updates - args.updates // 2],
        "development_checkpoint_nodes": 3,
        "cpu_hour_cap": args.cpu_hour_cap,
    }
    registered = register_g_budget(preflight, schedule, parameters)
    output = {
        **registered,
        "experiment_id": "wuziq-alphazero-lite-g-v1",
        "base_commit": "89fecf2a12c58cc27c985f588d5023ad9d198c80",
        "preflight_commit": preflight["code_commit"],
        "training_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "opening_schedule_path": "research/g_v1/config/openings.json",
        "opening_schedule": schedule,
        "opening_schedule_sha256": schedule["schedule_sha256"],
        "network": {"board_size": 9, "channels": 32, "residual_blocks": 2, "input_planes": 3, "policy_actions": 81},
        "initialization": {
            "G0": {"type": "random", "seed": 510700},
            "G-F2": {"type": "frozen_f2", "sha256": registered["f2_sha256"]},
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "CosineAnnealingLR eta_min=0.2*initial_lr over total registered updates",
            "loss": "visit-policy soft-target cross entropy + terminal value MSE",
            "augmentation": "uniform D4 board symmetries applied synchronously to state and visit target",
            "root_dirichlet_alpha": 0.3,
            "root_noise_fraction": 0.25,
            "temperature_moves": 12,
            "late_temperature": 0.1,
            "replay_sampling": "uniform by game then position; arm-isolated",
            "teacher_rows": 0,
            "f_supervised_rows": 0,
            "artificial_tactical_rewards": 0,
        },
        "checkpoint_nodes": ["initial", "stage-1", "stage-2"],
        "development_selection_rule": "For each arm, select the checkpoint with the highest mean PUCT-mode score over A/B/F1/F2 on the fixed development opening pairs; ties select the earlier checkpoint. Network-mode results are diagnostic only.",
        "blind_rule": "The blind opening pool is evaluated once after checkpoint selection and is never used for training or selection.",
        "opponent_hashes": {
            "A_B_code_base": "89fecf2a12c58cc27c985f588d5023ad9d198c80",
            "B_budget_seconds": 0.05,
            "F1_sha256": "8d3cbeefb3b0ec75a4ec769b9852327f7411e8836f87a4add80d9cff2516e51f",
            "F2_sha256": registered["f2_sha256"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(json.dumps({
        "registered": True,
        "output": str(output_path),
        "simulations": args.simulations,
        "environment_steps_per_arm": args.environment_steps,
        "updates_per_arm": args.updates,
        "development_games_per_checkpoint": registered["evaluation"]["development_games_per_checkpoint"],
        "blind_games": registered["evaluation"]["blind_games_per_selected_checkpoint"],
        "estimate": registered["estimates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
