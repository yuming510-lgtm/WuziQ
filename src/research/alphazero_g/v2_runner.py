"""Exact-resume, equal-budget self-play training for the three G V2 candidates."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .replay import ReplayBuffer, append_complete_game, budgeted_prefix, load_game_log
from .selfplay import play_selfplay_game
from .training import GTrainer, initialize_arm, load_training_state, save_g_checkpoint, save_training_state
from .v2_schedule import validate_v2_schedule

LR_IDS = {"lr_1e4", "lr_3e5", "lr_1e3"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def budgeted_game_prefix(game: dict[str, Any], budgeted_steps: int) -> dict[str, Any]:
    """Return a validated replay view containing only the accepted state prefix."""
    rows = game.get("experience_rows")
    if not isinstance(rows, list) or isinstance(budgeted_steps, bool) or not isinstance(budgeted_steps, int):
        raise ValueError("budgeted prefix needs complete game rows and an integer quota")
    total_steps = len(rows)
    if not 0 <= budgeted_steps <= total_steps:
        raise ValueError("budgeted prefix exceeds the complete game's training rows")
    replay = copy.deepcopy(game)
    replay["experience_rows"] = copy.deepcopy(rows[:budgeted_steps])
    replay["budgeted_environment_steps"] = budgeted_steps
    replay["completion_overrun_steps"] = total_steps - budgeted_steps
    replay["replay_prefix"] = True
    return replay


def _config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _code_commit(root: Path) -> str:
    for repository in (root, Path(__file__).resolve().parents[3]):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return "unknown"


def _trainer_inputs(root: Path, config: dict[str, Any], lr_id: str) -> tuple[Any, GTrainer, ReplayBuffer]:
    if lr_id not in LR_IDS:
        raise ValueError("unknown G V2 candidate learning-rate id")
    rates = config.get("candidate_learning_rates", {})
    expected = {"lr_1e4": 1e-4, "lr_3e5": 3e-5, "lr_1e3": 1e-3}
    if rates != expected:
        raise ValueError("G V2 requires the three frozen learning-rate candidates")
    device = str(config.get("device", "cpu"))
    seed = int(config["seed"])
    model = initialize_arm("G-F2", seed=seed, device=device)
    model.checkpoint_metadata = {
        **getattr(model, "checkpoint_metadata", {}),
        "g_v2_candidate_id": lr_id,
        "g_v2_experiment_id": config["experiment_id"],
    }
    trainer = GTrainer(
        model,
        seed=seed + {"lr_1e4": 10_001, "lr_3e5": 20_003, "lr_1e3": 30_007}[lr_id],
        learning_rate=float(rates[lr_id]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        expected_updates=int(config["updates_per_candidate"]),
        device=device,
    )
    replay = ReplayBuffer(
        int(config["replay_capacity"]),
        seed=seed + {"lr_1e4": 40_009, "lr_3e5": 50_021, "lr_1e3": 60_023}[lr_id],
    )
    return model, trainer, replay


def initialize_v2_candidate(root: str | Path, config: dict[str, Any], lr_id: str) -> dict[str, Any]:
    """Create an isolated F2-derived model, optimizer and replay buffer."""
    model, trainer, replay = _trainer_inputs(Path(root).resolve(), config, lr_id)
    if getattr(model, "checkpoint_metadata", {}).get("checkpoint_sha256") != config.get("f2_sha256"):
        raise ValueError("candidate initialization did not load the registered frozen F2 checkpoint")
    return {"model": model, "trainer": trainer, "replay": replay}


def _load_schedule(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    schedule_path = Path(str(config["opening_schedule_path"]))
    if not schedule_path.is_absolute():
        schedule_path = root / schedule_path
    schedule = json.loads(schedule_path.read_text())
    validate_v2_schedule(schedule)
    if schedule.get("schedule_sha256") != config.get("opening_schedule_sha256"):
        raise ValueError("training opening schedule checksum differs from the registered config")
    return schedule


def _progress_from_games(games: list[dict[str, Any]], stages: list[int], history: list[dict[str, Any]]) -> dict[str, Any]:
    steps_by_stage = [0] * len(stages)
    for game in games:
        stage = int(game.get("stage_index", -1))
        count = int(game.get("budgeted_environment_steps", -1))
        if stage < 0 or stage >= len(stages) or count < 0:
            raise ValueError("candidate log has an invalid stage or budgeted state count")
        steps_by_stage[stage] += count
    if any(count > limit for count, limit in zip(steps_by_stage, stages, strict=True)):
        raise ValueError("candidate exceeded a registered stage state budget")
    budgeted = sum(steps_by_stage)
    return {
        "steps_by_stage": steps_by_stage,
        "budgeted_environment_steps": budgeted,
        "actual_environment_steps": sum(len(game["experience_rows"]) for game in games),
        "completion_overrun_steps": sum(int(game.get("completion_overrun_steps", 0)) for game in games),
        "game_count": len(games),
        "actual_simulations": sum(int(game["stats"]["simulations"]) for game in games),
        "optimizer_updates": 0,
        "training_history": history,
        "active_cpu_seconds": sum(
            float(game.get("runner_cpu_seconds", game["stats"].get("cpu_seconds", 0))) for game in games
        ) + sum(float(row.get("cpu_seconds", 0)) for row in history),
        "active_wall_seconds": sum(
            float(game.get("runner_wall_seconds", game["stats"].get("wall_seconds", 0))) for game in games
        ) + sum(float(row.get("wall_seconds", 0)) for row in history),
    }


def run_v2_candidate(
    root: str | Path,
    config: dict[str, Any],
    lr_id: str,
    *,
    until_node: int | None = None,
    stop_after_games: int | None = None,
) -> dict[str, Any]:
    """Run or resume one candidate to a registered cumulative state node."""
    if lr_id not in LR_IDS:
        raise ValueError("unknown G V2 candidate learning-rate id")
    root_path = Path(root).resolve()
    stages = list(map(int, config["stage_environment_steps"]))
    nodes = list(map(int, config["stage_nodes"]))
    if (len(stages) != len(nodes) or sum(stages) != int(config["states_per_candidate"])
            or [sum(stages[:index + 1]) for index in range(len(stages))] != nodes):
        raise ValueError("candidate state stages do not match the registered quota")
    if any(node <= 0 for node in nodes) or any(right <= left for left, right in zip(nodes, nodes[1:])):
        raise ValueError("registered G V2 milestone nodes must be strictly increasing")
    expected_stage_updates = [
        (node // 16) - (nodes[index - 1] // 16 if index else 0)
        for index, node in enumerate(nodes)
    ]
    if list(map(int, config["stage_updates"])) != expected_stage_updates:
        raise ValueError("candidate stage updates do not match the one-per-16-state cadence")
    target = int(until_node) if until_node is not None else nodes[-1]
    if target not in nodes:
        raise ValueError("until_node must be one of the frozen milestone nodes")
    if int(config["simulation_budget"]) != 64 or int(config["update_every_budgeted_states"]) != 16:
        raise ValueError("G V2 runner requires registered PUCT64 and one batch update per 16 states")
    schedule = _load_schedule(root_path, config)
    if not schedule["train_openings"]:
        raise ValueError("G V2 schedule has no training openings")
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    model, trainer, replay = _trainer_inputs(root_path, config, lr_id)
    out_dir = root_path / "research/g_v2"
    checkpoint_dir = out_dir / "checkpoints" / lr_id
    game_path = out_dir / "results/selfplay" / f"{lr_id}.jsonl"
    state_path = checkpoint_dir / "training_state.pt"
    progress_path = checkpoint_dir / "progress.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    games, seen_ids = load_game_log(game_path)
    saved_progress: dict[str, Any] = {}
    saved_replay_rng: dict[str, Any] | None = None
    if state_path.exists():
        saved_replay = ReplayBuffer(int(config["replay_capacity"]), seed=0)
        saved_progress = load_training_state(state_path, trainer, saved_replay)
        saved_replay_rng = saved_replay.state_dict()["rng_state"]
        rebuilt = ReplayBuffer(int(config["replay_capacity"]), seed=0)
        for game in games:
            count = int(game.get("budgeted_environment_steps", 0))
            if count:
                rebuilt.add_complete_game(budgeted_game_prefix(game, count))
        replay_state = rebuilt.state_dict()
        replay_state["rng_state"] = saved_replay_rng
        replay.load_state_dict(replay_state)
        saved_game_count = int(saved_progress.get("game_count", 0))
        if saved_game_count > len(games):
            raise ValueError("candidate state refers to more complete games than the durable log")
        history = list(saved_progress.get("training_history", []))
        progress = _progress_from_games(games, stages, history)
        progress["active_cpu_seconds"] = float(saved_progress.get("active_cpu_seconds", 0)) + sum(
            float(game.get("runner_cpu_seconds", game["stats"].get("cpu_seconds", 0)))
            for game in games[saved_game_count:]
        )
        progress["active_wall_seconds"] = float(saved_progress.get("active_wall_seconds", 0)) + sum(
            float(game.get("runner_wall_seconds", game["stats"].get("wall_seconds", 0)))
            for game in games[saved_game_count:]
        )
        progress["training_history"] = history
    else:
        if games:
            raise ValueError("complete candidate games exist without an exact-resume training state")
        initial_path = checkpoint_dir / "initial.pt"
        if not initial_path.exists():
            save_g_checkpoint(initial_path, model, {
                "candidate_id": lr_id, "stage": "initial", "node": 0,
                "f2_sha256": config["f2_sha256"], "config_sha256": _config_digest(config),
            })
        progress = _progress_from_games([], stages, [])
        progress["optimizer_updates"] = 0
        save_training_state(state_path, trainer, replay, progress=progress)
        _atomic_json(progress_path, progress)

    total_budgeted = int(progress["budgeted_environment_steps"])
    if total_budgeted > target:
        raise ValueError("candidate already passed the requested registered milestone")
    expected_updates = total_budgeted // int(config["update_every_budgeted_states"])
    if trainer.update_count > expected_updates:
        raise ValueError("saved candidate has more optimizer updates than its budgeted states allow")
    if expected_updates > trainer.update_count:
        metrics = trainer.train_updates(replay, updates=expected_updates - trainer.update_count,
                                        batch_size=int(config["batch_size"]))
        metric = {**metrics, "reason": "reconcile_durable_games_after_interruption"}
        progress["training_history"].append(metric)
        progress = _progress_from_games(games, stages, progress["training_history"])
        progress["optimizer_updates"] = trainer.update_count
        save_training_state(state_path, trainer, replay, progress=progress)
        _atomic_json(progress_path, progress)

    games_run_this_call = 0
    while int(progress["budgeted_environment_steps"]) < target:
        if stop_after_games is not None and games_run_this_call >= int(stop_after_games):
            return {**progress, "completed": False, "lr_id": lr_id, "target_node": target}
        current = int(progress["budgeted_environment_steps"])
        stage_index = next(index for index, node in enumerate(nodes) if current < node)
        stage_remaining = nodes[stage_index] - current
        game_index = len(games)
        opening_group = schedule["train_openings"][game_index % len(schedule["train_openings"])]
        game_id = f"g-v2-{_config_digest(config)[:12]}-{lr_id}-game-{game_index:06d}"
        game_seed = int(config["seed"]) + 700_001 + game_index * 97
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        game = play_selfplay_game(
            trainer.model, opening_group["opening"], arm_id=f"G-F2-{lr_id}", game_id=game_id,
            opening_group_id=opening_group["opening_group_id"], simulations=64, seed=game_seed,
            temperature_moves=int(config.get("temperature_moves", 12)),
            late_temperature=float(config.get("late_temperature", 0.1)), device=trainer.device.type,
        )
        accepted, overrun = budgeted_prefix(game["experience_rows"], stage_remaining)
        game["candidate_lr_id"] = lr_id
        game["stage_index"] = stage_index
        game["stage_game_index"] = sum(int(item.get("stage_index", -1)) == stage_index for item in games)
        game["budgeted_environment_steps"] = len(accepted)
        game["completion_overrun_steps"] = overrun
        game["runner_cpu_seconds"] = time.process_time() - cpu_started
        game["runner_wall_seconds"] = time.perf_counter() - wall_started
        game["code_commit"] = _code_commit(root_path)
        if not append_complete_game(game_path, game, seen_ids):
            raise ValueError(f"candidate runner attempted to repeat game id {game_id}")
        games.append(game)
        if accepted:
            replay.add_complete_game(budgeted_game_prefix(game, len(accepted)))
        progress = _progress_from_games(games, stages, list(progress.get("training_history", [])))
        target_updates = int(progress["budgeted_environment_steps"]) // int(config["update_every_budgeted_states"])
        if trainer.update_count > target_updates:
            raise ValueError("optimizer update counter exceeded the registered sample cadence")
        if target_updates > trainer.update_count:
            metrics = trainer.train_updates(
                replay, updates=target_updates - trainer.update_count, batch_size=int(config["batch_size"])
            )
            metrics = {**metrics, "states_seen": int(progress["budgeted_environment_steps"]),
                       "game_id": game_id, "lr_id": lr_id}
            progress["training_history"].append(metrics)
            progress = _progress_from_games(games, stages, progress["training_history"])
        progress["optimizer_updates"] = trainer.update_count
        if trainer.update_count != target_updates:
            raise ValueError("candidate optimizer schedule did not reach the required update index")
        progress["replay_positions"] = replay.position_count
        progress["replay_games"] = replay.game_count
        progress["code_commit"] = game["code_commit"]
        progress["candidate_id"] = lr_id
        save_training_state(state_path, trainer, replay, progress=progress)
        _atomic_json(progress_path, progress)
        games_run_this_call += 1

    node_path = checkpoint_dir / f"node_{target}.pt"
    save_g_checkpoint(node_path, trainer.model, {
        "candidate_id": lr_id, "stage": f"node_{target}", "node": target,
        "budgeted_environment_steps": int(progress["budgeted_environment_steps"]),
        "optimizer_updates": trainer.update_count,
        "learning_rate": float(config["candidate_learning_rates"][lr_id]),
        "f2_sha256": config["f2_sha256"],
        "config_sha256": _config_digest(config),
        "training_data_source": config.get("data_source", "own self-play only"),
    })
    final_node = nodes[-1]
    if target == final_node:
        final_path = checkpoint_dir / "final.pt"
        save_g_checkpoint(final_path, trainer.model, {
            "candidate_id": lr_id, "stage": "final", "node": target,
            "budgeted_environment_steps": int(progress["budgeted_environment_steps"]),
            "optimizer_updates": trainer.update_count,
            "learning_rate": float(config["candidate_learning_rates"][lr_id]),
            "f2_sha256": config["f2_sha256"], "config_sha256": _config_digest(config),
        })
    return {**progress, "completed": True, "lr_id": lr_id, "target_node": target,
            "checkpoint": str(node_path)}
