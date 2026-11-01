"""Exact-quota, per-game persistent dual-arm self-play training runner."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .experiment import validate_g_schedule
from .replay import ReplayBuffer, append_complete_game, budgeted_prefix, load_game_log
from .selfplay import play_selfplay_game
from .training import (
    GTrainer,
    initialize_arm,
    save_g_checkpoint,
    save_training_state,
    load_training_state,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_config(config: dict[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(config, dict):
        return config, Path.cwd()
    path = Path(config).resolve()
    return json.loads(path.read_text()), path.parents[2]


def _budgeted_game(game: dict[str, Any]) -> dict[str, Any]:
    count = game.get("budgeted_environment_steps")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(game["experience_rows"]):
        raise ValueError("self-play log has invalid budgeted_environment_steps")
    replay_game = copy.deepcopy(game)
    replay_game["experience_rows"] = copy.deepcopy(game["experience_rows"][:count])
    return replay_game


def _compute_progress(games: list[dict[str, Any]], stages: list[int]) -> dict[str, Any]:
    steps_by_stage = [0 for _ in stages]
    for game in games:
        stage = int(game.get("stage_index", -1))
        if stage < 0 or stage >= len(stages):
            raise ValueError("self-play game has a stage index outside the registered plan")
        count = int(game.get("budgeted_environment_steps", -1))
        if count < 0 or count > len(game["experience_rows"]):
            raise ValueError("self-play game has invalid budgeted step count")
        steps_by_stage[stage] += count
    for index, (count, limit) in enumerate(zip(steps_by_stage, stages, strict=True)):
        if count > limit:
            raise ValueError(f"stage {index} exceeded its registered environment-step quota")
    return {
        "steps_by_stage": steps_by_stage,
        "budgeted_environment_steps": sum(steps_by_stage),
        "actual_environment_steps": sum(len(game["experience_rows"]) for game in games),
        "completion_overrun_steps": sum(int(game.get("completion_overrun_steps", 0)) for game in games),
        "game_count": len(games),
        "actual_simulations": sum(int(game["stats"]["simulations"]) for game in games),
        "completion_overrun_simulations": sum(
            sum(int(row.get("simulations", 0)) for row in game["experience_rows"][int(game["budgeted_environment_steps"]):])
            for game in games
        ),
        "training_history": [],
    }


def _rebuild_replay(games: list[dict[str, Any]], replay: ReplayBuffer) -> None:
    for game in games:
        if int(game.get("budgeted_environment_steps", 0)):
            replay.add_complete_game(_budgeted_game(game))


def run_training_arm(
    root: str | Path,
    config: dict[str, Any],
    arm_id: str,
    *,
    stop_after_games: int | None = None,
) -> dict[str, Any]:
    """Run/resume one training arm, persisting after every complete game/update stage."""
    if arm_id not in {"G0", "G-F2"}:
        raise ValueError("arm_id must be G0 or G-F2")
    root_path = Path(root).resolve()
    stages = list(map(int, config["stage_environment_steps"]))
    stage_updates = list(map(int, config["stage_updates"]))
    arm_config = config["arms"][arm_id]
    if arm_config["environment_steps"] != sum(stages) or arm_config["updates"] != sum(stage_updates):
        raise ValueError("arm quotas do not match registered stage sums")
    train_openings = config["opening_schedule"]["train_openings"]
    if not train_openings:
        raise ValueError("registered schedule has no self-play training openings")
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    device = config.get("device", "cpu")
    arm_dir = root_path / "research/g_v1"
    checkpoint_dir = arm_dir / "checkpoints" / arm_id
    result_dir = arm_dir / "results/selfplay"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    game_log = result_dir / f"{arm_id}.jsonl"
    state_file = checkpoint_dir / "training_state.pt"
    progress_file = checkpoint_dir / "progress.json"
    game_records, seen_ids = load_game_log(game_log)
    init_spec = config["initialization"][arm_id]
    model = initialize_arm(
        arm_id,
        seed=int(init_spec.get("seed", config["seed"])),
        f2_checkpoint=(root_path / "research/f_v2/checkpoints/best_f2.pt") if arm_id == "G-F2" else None,
        device=device,
    )
    total_updates = sum(stage_updates)
    trainer = GTrainer(
        model,
        seed=int(config["seed"]) + (0 if arm_id == "G0" else 1_000_000),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        expected_updates=total_updates,
        device=device,
    )
    replay = ReplayBuffer(int(config["replay_capacity"]), seed=int(config["seed"]) + (0 if arm_id == "G0" else 1_000_000))
    if not (checkpoint_dir / "initial.pt").exists():
        save_g_checkpoint(checkpoint_dir / "initial.pt", model, {
            "arm_id": arm_id,
            "stage": "initial",
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        })
    progress: dict[str, Any]
    if state_file.exists():
        checkpoint_replay = ReplayBuffer(int(config["replay_capacity"]), seed=0)
        saved_progress = load_training_state(state_file, trainer, checkpoint_replay)
        checkpoint_replay_state = checkpoint_replay.state_dict()
        _rebuild_replay(game_records, replay)
        replay_state = replay.state_dict()
        replay_state["rng_state"] = checkpoint_replay_state["rng_state"]
        replay.load_state_dict(replay_state)
        progress = _compute_progress(game_records, stages)
        saved_game_count = int(saved_progress.get("game_count", 0))
        extra_games = game_records[saved_game_count:]
        progress["training_history"] = list(saved_progress.get("training_history", []))
        progress["stage_update_counts"] = list(saved_progress.get("stage_update_counts", [0] * len(stages)))
        progress["active_wall_seconds"] = float(saved_progress.get("active_wall_seconds", 0.0)) + sum(
            float(game.get("runner_wall_seconds", 0.0)) for game in extra_games
        )
        progress["active_cpu_seconds"] = float(saved_progress.get("active_cpu_seconds", 0.0)) + sum(
            float(game.get("runner_cpu_seconds", 0.0)) for game in extra_games
        )
    else:
        if game_records:
            raise ValueError("self-play games exist without an exact-resume training state")
        progress = _compute_progress([], stages)
        progress["stage_update_counts"] = [0] * len(stages)
        progress["active_wall_seconds"] = 0.0
        progress["active_cpu_seconds"] = 0.0
    if trainer.update_count > total_updates:
        raise ValueError("resume state exceeds registered optimizer update budget")
    progress["optimizer_updates"] = trainer.update_count
    progress["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root_path, text=True).strip()
    if not state_file.exists():
        save_training_state(state_file, trainer, replay, progress=progress)
        _atomic_json(progress_file, progress)

    per_arm_cpu_cap = float(config["hard_cpu_hour_cap"]) * 3600 / 2 if "hard_cpu_hour_cap" in config else None
    if per_arm_cpu_cap is not None and progress["active_cpu_seconds"] > per_arm_cpu_cap:
        raise RuntimeError("resumed arm already exceeds its registered CPU budget share")

    games_run_this_call = 0
    for stage_index, (stage_limit, stage_update_limit) in enumerate(zip(stages, stage_updates, strict=True)):
        if trainer.update_count >= sum(stage_updates[:stage_index + 1]):
            continue
        if trainer.update_count != sum(stage_updates[:stage_index]):
            raise ValueError("optimizer resume index does not align with registered stage boundary")
        if progress["steps_by_stage"][stage_index] < stage_limit:
            while progress["steps_by_stage"][stage_index] < stage_limit:
                if stop_after_games is not None and games_run_this_call >= stop_after_games:
                    return {**progress, "completed": False, "arm_id": arm_id, "optimizer_updates": trainer.update_count}
                game_index = sum(int(game.get("stage_index") == stage_index) for game in game_records)
                opening_group = train_openings[game_index % len(train_openings)]
                remaining = stage_limit - progress["steps_by_stage"][stage_index]
                seed = int(config["seed"]) + stage_index * 1_000_003 + game_index * 97
                game_id = f"g-v1-{arm_id}-stage-{stage_index + 1}-game-{game_index:05d}"
                game_started = time.perf_counter()
                cpu_started = time.process_time()
                game = play_selfplay_game(
                    trainer.model,
                    opening_group["opening"],
                    arm_id=arm_id,
                    game_id=game_id,
                    opening_group_id=opening_group["opening_group_id"],
                    simulations=int(arm_config["simulations_per_move"]),
                    seed=seed,
                    temperature_moves=int(config["training"]["temperature_moves"]),
                    late_temperature=float(config["training"]["late_temperature"]),
                    device=device,
                )
                accepted, overrun = budgeted_prefix(game["experience_rows"], remaining)
                game["stage_index"] = stage_index
                game["stage_game_index"] = game_index
                game["budgeted_environment_steps"] = len(accepted)
                game["completion_overrun_steps"] = overrun
                game["runner_wall_seconds"] = time.perf_counter() - game_started
                game["runner_cpu_seconds"] = time.process_time() - cpu_started
                if not append_complete_game(game_log, game, seen_ids):
                    raise ValueError(f"runner tried to repeat game id {game_id}")
                game_records.append(game)
                if accepted:
                    replay.add_complete_game(_budgeted_game(game))
                previous_progress = progress
                progress = _compute_progress(game_records, stages)
                progress["optimizer_updates"] = trainer.update_count
                progress["stage_update_counts"] = list(previous_progress["stage_update_counts"])
                progress["training_history"] = list(previous_progress.get("training_history", []))
                progress["active_wall_seconds"] = float(previous_progress.get("active_wall_seconds", 0)) + game["runner_wall_seconds"]
                progress["active_cpu_seconds"] = float(previous_progress.get("active_cpu_seconds", 0)) + game["runner_cpu_seconds"]
                save_training_state(state_file, trainer, replay, progress=progress)
                _atomic_json(progress_file, progress)
                if per_arm_cpu_cap is not None and progress["active_cpu_seconds"] > per_arm_cpu_cap:
                    raise RuntimeError("arm exceeded its registered CPU budget share")
                games_run_this_call += 1
                if stop_after_games is not None and games_run_this_call >= stop_after_games:
                    return {**progress, "completed": False, "arm_id": arm_id, "optimizer_updates": trainer.update_count}

        if progress["steps_by_stage"][stage_index] != stage_limit:
            raise ValueError("stage did not reach its exact registered budgeted environment-step quota")
        metrics = trainer.train_updates(
            replay,
            updates=stage_update_limit,
            batch_size=int(config["batch_size"]),
        )
        progress["optimizer_updates"] = trainer.update_count
        progress["stage_update_counts"][stage_index] = stage_update_limit
        progress["training_history"] = list(progress.get("training_history", [])) + [{
            "stage_index": stage_index,
            "budgeted_environment_steps": progress["budgeted_environment_steps"],
            **metrics,
            "replay_positions": replay.position_count,
        }]
        progress["active_wall_seconds"] = float(progress.get("active_wall_seconds", 0)) + float(metrics["wall_seconds"])
        progress["active_cpu_seconds"] = float(progress.get("active_cpu_seconds", 0)) + float(metrics["cpu_seconds"])
        checkpoint_path = checkpoint_dir / f"stage-{stage_index + 1}.pt"
        save_g_checkpoint(checkpoint_path, trainer.model, {
            "arm_id": arm_id,
            "stage": stage_index + 1,
            "budgeted_environment_steps": progress["budgeted_environment_steps"],
            "optimizer_updates": trainer.update_count,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        })
        save_training_state(state_file, trainer, replay, progress=progress)
        _atomic_json(progress_file, progress)
        _atomic_json_lines(checkpoint_dir / "training_history.jsonl", progress["training_history"])
        if per_arm_cpu_cap is not None and progress["active_cpu_seconds"] > per_arm_cpu_cap:
            raise RuntimeError("arm exceeded its registered CPU budget share")
    complete = (
        progress["budgeted_environment_steps"] == sum(stages)
        and trainer.update_count == total_updates
    )
    if complete:
        save_g_checkpoint(checkpoint_dir / "final.pt", trainer.model, {
            "arm_id": arm_id,
            "stage": "final",
            "budgeted_environment_steps": progress["budgeted_environment_steps"],
            "optimizer_updates": trainer.update_count,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        })
    return {**progress, "completed": complete, "arm_id": arm_id, "optimizer_updates": trainer.update_count}


def run_registered_training(
    root: str | Path,
    config_path: str | Path,
    *,
    stop_after_games_per_arm: int | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = json.loads(Path(config_path).read_text())
    result = {}
    for arm in ("G0", "G-F2"):
        result[arm] = run_training_arm(
            root_path, config, arm, stop_after_games=stop_after_games_per_arm
        )
    result["completed"] = all(result[arm]["completed"] for arm in ("G0", "G-F2"))
    result["training_budget_equal"] = all(
        result["G0"][key] == result["G-F2"][key]
        for key in ("budgeted_environment_steps", "optimizer_updates")
    )
    total_cpu = sum(result[arm]["active_cpu_seconds"] for arm in ("G0", "G-F2"))
    result["total_active_cpu_seconds"] = total_cpu
    if total_cpu > float(config["hard_cpu_hour_cap"]) * 3600:
        raise RuntimeError("dual-arm run exceeded its registered total CPU budget")
    _atomic_json(root_path / "research/g_v1/results/training/summary.json", result)
    return result
