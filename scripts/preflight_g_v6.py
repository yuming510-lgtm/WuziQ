from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from src.research.alphazero_g.v5_training import _policy_update, _run_game
from src.research.alphazero_g.v5_reference import ReferencePolicyValueNet
from src.research.alphazero_g.v6_registration import (
    BLIND_PAIRS,
    DEVELOPMENT_PAIRS,
    MILESTONES,
    SIMULATIONS_PER_MOVE,
    TARGET_STATES,
    validate_v5_resume_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _cgroup_quota() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota == "max":
            return None
        return int(quota) / int(period)
    except (OSError, ValueError):
        return None


def estimate_workload(
    *,
    measured_game_seconds: float,
    measured_game_states: int,
    observed_mean_states_per_game: float,
    optimizer_update_seconds: float,
    evaluation_seconds_per_game: dict[int, float],
    tactical_seconds_per_position: dict[int, float],
    preflight_seconds: float,
) -> dict[str, Any]:
    remaining_states = TARGET_STATES - 1_514
    if remaining_states <= 0 or measured_game_states < 1 or observed_mean_states_per_game <= 0:
        raise ValueError("invalid G V6 remaining-state or benchmark measurements")
    games_needed = int(math.ceil(remaining_states / observed_mean_states_per_game))
    game_cost_per_state = measured_game_seconds / measured_game_states
    training_selfplay = remaining_states * game_cost_per_state
    training_updates = games_needed * max(0.0, optimizer_update_seconds)
    development_games_per_budget = len(MILESTONES) * 6 * DEVELOPMENT_PAIRS * 2
    blind_games_per_budget = 2 * BLIND_PAIRS * 2
    evaluation_games_by_budget = {
        "64": development_games_per_budget + blind_games_per_budget,
        "400": development_games_per_budget + blind_games_per_budget,
    }
    evaluation_cpu = sum(evaluation_games_by_budget[str(budget)] * evaluation_seconds_per_game[budget]
                         for budget in (64, 400))
    tactics_per_budget = len(MILESTONES) * 64
    tactic_cpu = sum(tactics_per_budget * tactical_seconds_per_position[budget]
                     for budget in (64, 400))
    audit_report_reserve = 0.25 * 3600
    base = preflight_seconds + training_selfplay + training_updates + evaluation_cpu + tactic_cpu + audit_report_reserve
    estimates = {
        "optimistic": base * 0.8,
        "normal": base,
        "conservative": base * 1.6,
    }
    return {
        "remaining_training_states": remaining_states,
        "estimated_remaining_games": games_needed,
        "estimated_remaining_optimizer_updates": games_needed,
        "measured_game_states": int(measured_game_states),
        "measured_game_cpu_seconds": float(measured_game_seconds),
        "measured_game_cpu_seconds_per_state": float(game_cost_per_state),
        "measured_optimizer_update_cpu_seconds": float(optimizer_update_seconds),
        "planned_development_games": development_games_per_budget * 2,
        "planned_development_games_per_budget": development_games_per_budget,
        "planned_blind_games": blind_games_per_budget * 2,
        "planned_blind_games_per_budget": blind_games_per_budget,
        "evaluation_games_by_budget": evaluation_games_by_budget,
        "tactical_positions_by_budget": {"64": tactics_per_budget, "400": tactics_per_budget},
        "cpu_seconds_by_component": {
            "preflight": float(preflight_seconds),
            "selfplay": float(training_selfplay),
            "optimizer_updates": float(training_updates),
            "paired_evaluations": float(evaluation_cpu),
            "tactics": float(tactic_cpu),
            "audit_report_reserve": float(audit_report_reserve),
        },
        "estimated_total_cpu_hours": {key: value / 3600 for key, value in estimates.items()},
    }


def run_preflight(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    registration_path = root / "research/g_v6/config/registration.json"
    if not registration_path.is_file():
        raise FileNotFoundError("run scripts/register_g_v6.py before V6 preflight")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    validation = validate_v5_resume_checkpoint(root)
    source_state = torch.load(
        root / "research/g_v5/checkpoints/reference400_empty/training_state.pt",
        map_location="cpu", weights_only=False,
    )
    model = ReferencePolicyValueNet(9, seed=int(source_state["base_seed"]))
    model.load_policy_param(source_state["model_state"])
    model.optimizer.load_state_dict(source_state["optimizer_state"])
    torch.set_num_threads(1)
    started = time.process_time()
    benchmark_game, states, policies, values = _run_game(
        size=9, n_in_row=5, simulations=SIMULATIONS_PER_MOVE,
        model=model, game_id="g-v6-preflight-discarded", seed=696105,
    )
    game_cpu = float(benchmark_game["runner_cpu_seconds"])

    # Exercise one real V5 objective update on a throwaway model/replay copy.
    update_model = ReferencePolicyValueNet(9, seed=int(source_state["base_seed"]))
    update_model.load_policy_param(source_state["model_state"])
    update_model.optimizer.load_state_dict(source_state["optimizer_state"])
    import random
    previous_python, previous_numpy, previous_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        random.setstate(source_state["python_rng_state"])
        np.random.set_state(source_state["numpy_rng_state"])
        torch.set_rng_state(source_state["torch_rng_state"])
        replay = __import__("collections").deque(source_state["replay_buffer"])
        update_started = time.process_time()
        _policy_update(
            update_model, replay,
            batch_size=int(registration["v5_training_parameters"]["batch_size"]),
            epochs=int(registration["v5_training_parameters"]["epochs"]),
            learning_rate=float(registration["v5_training_parameters"]["learning_rate"]),
            kl_target=float(root.joinpath("research/g_v5/config/experiment.json").is_file()
                            and json.loads(root.joinpath("research/g_v5/config/experiment.json").read_text())["training"]["kl_target"]),
            lr_multiplier=float(source_state["lr_multiplier"]),
        )
        update_cpu = time.process_time() - update_started
    finally:
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.set_rng_state(previous_torch)

    v5_games = []
    for path in (root / "research/g_v5/raw/selfplay/reference400_empty").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("source") == "self_play" and int(row.get("board_size", 0)) == 9:
            v5_games.append(row)
    if not v5_games:
        raise ValueError("V5 reference400 self-play game records are missing")
    mean_states = float(np.mean([row["training_state_count"] for row in v5_games]))
    game_cpu_rates = [float(row["runner_cpu_seconds"]) / int(row["training_state_count"]) for row in v5_games]

    evaluation_costs: dict[int, float] = {}
    match_paths = list((root / "research/g_v5/results/matches").rglob("v5m-*.json"))
    for budget in (64, 400):
        costs = []
        for path in match_paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            if int(row.get("simulations_per_reference_turn", -1)) == budget:
                costs.append(float(row.get("total_cpu_seconds", 0.0)))
        if not costs:
            raise ValueError(f"V5 has no measured evaluation games at PUCT-{budget}")
        evaluation_costs[budget] = float(np.mean(costs))

    tactics = json.loads((root / "research/g_v6/config/tactics.json").read_text(encoding="utf-8"))
    tactical_costs: dict[int, float] = {}
    for budget in (64, 400):
        tactical_started = time.process_time()
        from src.research.alphazero_g.v5_evaluation import evaluate_reference_tactics
        evaluate_reference_tactics(model, tactics[:2], simulations=budget, seed=696120 + budget)
        tactical_costs[budget] = (time.process_time() - tactical_started) / 2.0

    preflight_cpu = time.process_time() - started
    workload = estimate_workload(
        measured_game_seconds=game_cpu,
        measured_game_states=len(states),
        observed_mean_states_per_game=mean_states,
        optimizer_update_seconds=update_cpu,
        evaluation_seconds_per_game=evaluation_costs,
        tactical_seconds_per_position=tactical_costs,
        preflight_seconds=preflight_cpu,
    )
    logical = os.cpu_count() or 1
    quota = None
    try:
        value = Path("/sys/fs/cgroup/cpu.max").read_text().split()[0]
        if value != "max":
            quota = int(value) / int(Path("/sys/fs/cgroup/cpu.max").read_text().split()[1])
    except (OSError, ValueError):
        pass
    wall_capacity = quota or float(logical)
    result = {
        "format_version": 1,
        "experiment_id": registration["experiment_id"],
        "benchmark_is_training_data": False,
        "resource": {
            "cpu_model": platform.processor(),
            "logical_cpu_count": logical,
            "cpu_quota": quota,
            "memory_limit_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text())
                                   if Path("/sys/fs/cgroup/memory.max").read_text().strip().isdigit() else None,
            "disk_free_bytes": __import__("shutil").disk_usage(root).free,
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "resume_validation": validation,
        "benchmark": {
            "board_size": 9, "n_in_row": 5,
            "configured_playouts_per_move": SIMULATIONS_PER_MOVE,
            "actual_decision_states": len(states),
            "game_cpu_seconds": game_cpu,
            "game_wall_seconds": float(benchmark_game["runner_wall_seconds"]),
            "optimizer_update_cpu_seconds": float(update_cpu),
            "average_historical_states_per_game": mean_states,
            "average_historical_game_cpu_seconds": float(np.mean([row["runner_cpu_seconds"] for row in v5_games])),
            "historical_game_seconds_per_state_median": float(np.median(game_cpu_rates)),
            "evaluation_game_seconds_by_budget": {str(k): v for k, v in evaluation_costs.items()},
            "tactic_seconds_per_position_by_budget": {str(k): v for k, v in tactical_costs.items()},
        },
        "workload_estimate": workload,
        "estimated_wall_hours_at_cpu_quota": {
            key: value / wall_capacity * 1.15
            for key, value in workload["estimated_total_cpu_hours"].items()
        },
        "cpu_cap_seconds": 36_000,
    }
    path = root / "research/g_v6/results/preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temp, path)
    progress_path = root / "research/g_v6/results/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["v6_cumulative_cpu_seconds"] = preflight_cpu
    progress["phase"] = "preflight_complete"
    progress["preflight_cpu_seconds"] = preflight_cpu
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_preflight(), ensure_ascii=False, indent=2))
