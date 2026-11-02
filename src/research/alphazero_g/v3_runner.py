"""Matched multi-seed self-play runner with exact checkpoint/log recovery."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .replay import ReplayBuffer, append_complete_game, budgeted_prefix, load_game_log
from .selfplay import play_selfplay_game
from .training import (
    F2_SHA256,
    GTrainer,
    initialize_arm,
    save_g_checkpoint,
)
from .v3_bn import capture_bn_buffers
from .v3_budget import (
    BudgetExhausted, ensure_v3_cpu_room, record_v3_cpu_event,
)
from .v3_schedule import validate_v3_schedule

ARMS: tuple[str, str] = ("G0", "G-F2")


def _config_digest(config: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _budgeted_game_prefix(game: dict[str, Any], accepted_rows: list[dict[str, Any]], overrun: int) -> dict[str, Any]:
    """Build the replay-only view while retaining the complete terminal game log."""
    result = copy.deepcopy(game)
    result["experience_rows"] = copy.deepcopy(accepted_rows)
    result["budgeted_environment_steps"] = len(accepted_rows)
    result["completion_overrun_steps"] = int(overrun)
    result["replay_prefix"] = True
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _identity(root: Path) -> tuple[str, str]:
    repository = _repo_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != "research/alphazero-g-v3":
        raise ValueError("G V3 runner requires research/alphazero-g-v3")
    return branch, commit


def load_v3_config(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = root_path / "research/g_v3/config/experiment.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("registered") is not True or config.get("branch") != "research/alphazero-g-v3":
        raise ValueError("G V3 experiment config is not registered for the V3 branch")
    if config.get("base_commit") != "c95eb5aa919c769af11ea6b7fc22df2594ae1c28":
        raise ValueError("G V3 config does not reference the frozen G V2 baseline")
    if config.get("config_sha256") != _config_digest(config):
        raise ValueError("G V3 experiment config hash does not match its payload")
    schedule_path = root_path / str(config["opening_schedule_path"])
    schedule_bytes = schedule_path.read_bytes()
    schedule = json.loads(schedule_bytes)
    validate_v3_schedule(schedule)
    if hashlib.sha256(schedule_bytes).hexdigest() != config.get("opening_schedule_sha256"):
        raise ValueError("G V3 opening schedule differs from the registered schedule hash")
    preflight_path = root_path / "research/g_v3/results/preflight/preflight.json"
    probe_path = root_path / "research/g_v3/results/bn_probe/probe.json"
    if (not preflight_path.is_file() or
            hashlib.sha256(preflight_path.read_bytes()).hexdigest() != config.get("preflight_sha256")):
        raise ValueError("G V3 measured preflight differs from the registered evidence hash")
    if (not probe_path.is_file() or
            hashlib.sha256(probe_path.read_bytes()).hexdigest() != config.get("bn_probe_sha256")):
        raise ValueError("G V3 BatchNorm probe differs from the registered evidence hash")
    if schedule.get("seeds") != config.get("seeds"):
        raise ValueError("G V3 schedule seed list differs from the registered config")
    f2_path = root_path / "research/f_v2/checkpoints/best_f2.pt"
    if not f2_path.is_file() or hashlib.sha256(f2_path.read_bytes()).hexdigest() != F2_SHA256:
        raise ValueError("frozen F2 checkpoint does not match the registered V3 baseline")
    return config


def initialize_v3_arm(
    root: str | Path,
    *,
    arm: Literal["G0", "G-F2"],
    seed: int,
    bn_policy: Literal["update", "freeze"] | None = None,
) -> tuple[Any, GTrainer, ReplayBuffer]:
    """Construct fully isolated model, optimizer and replay state for one route/seed."""
    root_path = Path(root).resolve()
    branch, _ = _identity(root_path)
    if arm not in ARMS:
        raise ValueError("arm must be G0 or G-F2")
    config = load_v3_config(root_path)
    if int(seed) not in list(map(int, config["seeds"])):
        raise ValueError("seed is not in the registered G V3 seed list")
    selected_bn = str(config["bn_policy"] if bn_policy is None else bn_policy)
    if selected_bn != config["bn_policy"]:
        raise ValueError("BN policy override differs from the registered common mode")
    model = initialize_arm(
        arm, seed=int(seed),
        f2_checkpoint=root_path / "research/f_v2/checkpoints/best_f2.pt" if arm == "G-F2" else None,
        device="cpu",
    )
    if arm == "G-F2" and model.checkpoint_metadata.get("checkpoint_sha256") != F2_SHA256:
        raise ValueError("G-F2 failed to load the hash-verified frozen F2 checkpoint")
    model.checkpoint_metadata = {
        **getattr(model, "checkpoint_metadata", {}),
        "g_v3_arm": arm,
        "g_v3_seed": int(seed),
        "g_v3_branch": branch,
        "g_v3_config_sha256": config["config_sha256"],
    }
    stages = list(map(int, config["stage_nodes"]))
    trainer = GTrainer(
        model,
        seed=int(seed) + (0 if arm == "G0" else 1_000_003) + 87_217,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        expected_updates=int(config["states_per_seed"]) // int(config["update_every_budgeted_states"]),
        device="cpu",
        bn_mode=selected_bn,
    )
    replay = ReplayBuffer(
        int(config["replay_capacity"]),
        seed=int(seed) + (0 if arm == "G0" else 1_000_033) + 91_037,
    )
    return model, trainer, replay


def _cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return copy.deepcopy(value)


def save_v3_state(
    path: str | Path,
    trainer: GTrainer,
    replay: ReplayBuffer,
    *,
    arm: str,
    seed: int,
    progress: dict[str, Any],
    initial_bn_buffers: dict[str, Any],
) -> str:
    """Atomically persist train state; replay rows are rebuilt from fsynced games."""
    if arm not in ARMS or int(seed) < 0:
        raise ValueError("invalid G V3 state identity")
    config_sha = str(progress.get("config_sha256", ""))
    if len(config_sha) != 64:
        raise ValueError("training state progress must carry the registered config hash")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    replay_state = replay.state_dict()
    payload = {
        "format_version": 1,
        "arm": arm,
        "seed": int(seed),
        "config_sha256": config_sha,
        "trainer": _cpu_state(trainer.state_dict()),
        "replay": {
            "format_version": replay_state["format_version"],
            "capacity": replay_state["capacity"],
            "rng_state": copy.deepcopy(replay_state["rng_state"]),
            "game_ids": [str(game["game_id"]) for game in replay_state.get("games", [])],
        },
        "progress": copy.deepcopy(progress),
        "initial_bn_buffers": _cpu_state(initial_bn_buffers),
        "global_rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_v3_state(
    path: str | Path,
    trainer: GTrainer,
    replay: ReplayBuffer,
    *,
    arm: str,
    seed: int,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Restore trainer and RNGs after the caller rebuilds replay from game logs."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported G V3 training state")
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("G V3 state identity does not match the requested arm/seed")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("G V3 training state config hash differs from the registered config")
    if int(payload["replay"].get("capacity", -1)) != replay.capacity:
        raise ValueError("G V3 replay capacity differs from the saved state")
    if payload.get("progress", {}).get("config_sha256") != expected_config_sha256:
        raise ValueError("G V3 progress payload config hash differs from state metadata")
    trainer.load_state_dict(payload["trainer"])
    replay._rng.bit_generator.state = copy.deepcopy(payload["replay"]["rng_state"])
    random.setstate(payload["global_rng"]["python"])
    np.random.set_state(payload["global_rng"]["numpy"])
    torch.set_rng_state(payload["global_rng"]["torch_cpu"])
    if torch.cuda.is_available() and payload["global_rng"]["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(payload["global_rng"]["torch_cuda"])
    return copy.deepcopy(payload["progress"])


def _progress_from_games(
    games: list[dict[str, Any]],
    nodes: list[int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    budgeted = sum(int(game.get("budgeted_environment_steps", -1)) for game in games)
    if budgeted < 0 or budgeted > nodes[-1]:
        raise ValueError("G V3 game log exceeds the registered state quota")
    stage_counts: list[int] = []
    previous = 0
    for node in nodes:
        stage_counts.append(max(0, min(node, budgeted) - previous))
        previous = node
    stage_updates = [
        max(0, min(nodes[index], budgeted) // 16 - (nodes[index - 1] // 16 if index else 0))
        for index in range(len(nodes))
    ]
    return {
        "stage_nodes": nodes,
        "budgeted_environment_steps": budgeted,
        "actual_experience_rows": sum(len(game["experience_rows"]) for game in games),
        "completion_overrun_steps": sum(int(game.get("completion_overrun_steps", 0)) for game in games),
        "game_count": len(games),
        "completed_game_ids": [str(game["game_id"]) for game in games],
        "actual_simulations": sum(int(game["stats"]["simulations"]) for game in games),
        "selfplay_cpu_seconds": sum(float(game.get("runner_cpu_seconds", game["stats"].get("cpu_seconds", 0))) for game in games),
        "selfplay_wall_seconds": sum(float(game.get("runner_wall_seconds", game["stats"].get("wall_seconds", 0))) for game in games),
        "stage_environment_steps": stage_counts,
        "stage_optimizer_updates": stage_updates,
        "training_history": copy.deepcopy(history),
        "optimizer_updates": sum(int(row.get("updates", 0)) for row in history),
        "optimizer_cpu_seconds": sum(float(row.get("cpu_seconds", 0.0)) for row in history),
        "optimizer_wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in history),
    }


def _append_stage_record(root: Path, row: dict[str, Any]) -> None:
    path = root / "research/g_v3/results/training/stages.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            records.add(str(old["checkpoint_id"]))
    if row["checkpoint_id"] in records:
        return
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("failed to append G V3 stage record")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def run_v3_arm(
    root: str | Path,
    *,
    arm: str,
    seed: int,
    until_node: int | None = None,
    stop_after_games: int | None = None,
) -> dict[str, Any]:
    """Run/resume one route and seed to a registered cumulative state node."""
    if arm not in ARMS:
        raise ValueError("arm must be G0 or G-F2")
    if stop_after_games is not None and (isinstance(stop_after_games, bool) or stop_after_games < 0):
        raise ValueError("stop_after_games must be a nonnegative integer")
    root_path = Path(root).resolve()
    _identity(root_path)
    config = load_v3_config(root_path)
    if int(seed) not in list(map(int, config["seeds"])):
        raise ValueError("seed is not registered")
    config_sha = str(config["config_sha256"])
    nodes = list(map(int, config["stage_nodes"]))
    if not nodes or nodes != sorted(set(nodes)) or nodes[-1] != int(config["states_per_seed"]):
        raise ValueError("registered V3 milestone nodes do not match the state quota")
    target = int(until_node) if until_node is not None else nodes[-1]
    if target not in nodes:
        raise ValueError("until_node must be a registered milestone node")
    if int(config["simulation_budget"]) != 64 or int(config["update_every_budgeted_states"]) != 16:
        raise ValueError("V3 runner requires the registered PUCT64 / one-update-per-16-state schedule")
    torch.set_num_threads(1)

    model, trainer, replay = initialize_v3_arm(root_path, arm=arm, seed=int(seed))
    checkpoint_dir = root_path / "research/g_v3/checkpoints" / arm / f"seed-{int(seed)}"
    game_path = root_path / "research/g_v3/results/selfplay" / f"{arm}-seed-{int(seed)}.jsonl"
    state_path = checkpoint_dir / "training_state.pt"
    progress_path = checkpoint_dir / "progress.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    games, seen_ids = load_game_log(game_path)

    if state_path.exists():
        state_payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if state_payload.get("arm") != arm or int(state_payload.get("seed", -1)) != int(seed):
            raise ValueError("persisted training state has another route/seed identity")
        saved_progress = state_payload.get("progress", {})
        saved_count = int(saved_progress.get("game_count", -1))
        saved_ids = list(saved_progress.get("completed_game_ids", []))
        if saved_count < 0 or saved_count > len(games) or saved_ids != [str(game["game_id"]) for game in games[:saved_count]]:
            raise ValueError("training state game cursor conflicts with the durable game log")
        if len(games) - saved_count > 1:
            raise ValueError("too many games are ahead of the last atomic optimizer checkpoint")
        for game in games:
            count = int(game.get("budgeted_environment_steps", -1))
            if count < 0:
                raise ValueError("durable V3 game is missing its budgeted state prefix")
            if count:
                accepted_rows = game["experience_rows"][:count]
                replay.add_complete_game(_budgeted_game_prefix(
                    game, accepted_rows, int(game.get("completion_overrun_steps", 0)),
                ))
        history = list(saved_progress.get("training_history", []))
        initial_bn_buffers = state_payload.get("initial_bn_buffers")
        if not isinstance(initial_bn_buffers, dict):
            raise ValueError("training state is missing initial BatchNorm buffers")
        loaded_progress = load_v3_state(
            state_path, trainer, replay, arm=arm, seed=int(seed),
            expected_config_sha256=config_sha,
        )
        if loaded_progress.get("completed_game_ids") != saved_ids:
            raise ValueError("restored training state cursor differs from its saved game IDs")
        if trainer.bn_mode != config["bn_policy"]:
            raise ValueError("restored checkpoint BatchNorm mode differs from the registered mode")
        progress = _progress_from_games(games, nodes, history)
        if progress["budgeted_environment_steps"] > target:
            raise ValueError("durable games already pass the requested registered milestone")
        target_updates = progress["budgeted_environment_steps"] // int(config["update_every_budgeted_states"])
        if trainer.update_count > target_updates:
            raise ValueError("saved optimizer is ahead of the durable accepted experience rows")
        reconciled_log_ahead = len(games) > saved_count
        if trainer.update_count < target_updates:
            ensure_v3_cpu_room(root_path, float(config["hard_cpu_hour_cap"]) * 3600)
            recovered = trainer.train_updates(
                replay, updates=target_updates - trainer.update_count,
                batch_size=int(config["batch_size"]),
            )
            history.append({
                **recovered,
                "states_seen": int(progress["budgeted_environment_steps"]),
                "reason": "reconcile_durable_game_after_interruption",
            })
            progress = _progress_from_games(games, nodes, history)
            progress["optimizer_updates"] = trainer.update_count
            progress["last_state_reconciled"] = True
            progress["config_sha256"] = config_sha
            progress["arm"] = arm
            progress["seed"] = int(seed)
            progress["replay_positions"] = replay.position_count
            progress["replay_games"] = replay.game_count
            state_hash = save_v3_state(
                state_path, trainer, replay, arm=arm, seed=int(seed),
                progress=progress, initial_bn_buffers=initial_bn_buffers,
            )
            progress["training_state_sha256"] = state_hash
            _atomic_json(progress_path, progress)
            record_v3_cpu_event(
                root_path,
                f"optimizer:{arm}:seed-{int(seed)}:states-{int(progress['budgeted_environment_steps'])}",
                float(recovered.get("cpu_seconds", 0.0)),
                category="optimizer",
            )
        elif reconciled_log_ahead:
            progress["optimizer_updates"] = trainer.update_count
            progress["last_state_reconciled"] = True
            progress["config_sha256"] = config_sha
            progress["arm"] = arm
            progress["seed"] = int(seed)
            progress["replay_positions"] = replay.position_count
            progress["replay_games"] = replay.game_count
            state_hash = save_v3_state(
                state_path, trainer, replay, arm=arm, seed=int(seed),
                progress=progress, initial_bn_buffers=initial_bn_buffers,
            )
            progress["training_state_sha256"] = state_hash
            _atomic_json(progress_path, progress)
    else:
        if games:
            raise ValueError("complete V3 games exist without a matching exact-resume training state")
        if arm == "G-F2" and model.checkpoint_metadata.get("checkpoint_sha256") != config["f2_sha256"]:
            raise ValueError("G-F2 initial checkpoint does not match frozen F2 metadata")
        initial_bn_buffers = capture_bn_buffers(model)
        initial_checkpoint = checkpoint_dir / "node_0.pt"
        initial_sha = save_g_checkpoint(initial_checkpoint, model, {
            "experiment": "g_v3", "arm": arm, "seed": int(seed), "node": 0,
            "initialization": "seeded_random" if arm == "G0" else "frozen_f2",
            "f2_sha256": F2_SHA256 if arm == "G-F2" else None,
            "config_sha256": config_sha,
        })
        progress = _progress_from_games([], nodes, [])
        progress.update({
            "arm": arm,
            "seed": int(seed),
            "config_sha256": config_sha,
            "initial_checkpoint": initial_checkpoint.relative_to(root_path).as_posix(),
            "initial_checkpoint_sha256": initial_sha,
            "replay_positions": 0,
            "replay_games": 0,
        })
        state_hash = save_v3_state(
            state_path, trainer, replay, arm=arm, seed=int(seed),
            progress=progress, initial_bn_buffers=initial_bn_buffers,
        )
        progress["training_state_sha256"] = state_hash
        _atomic_json(progress_path, progress)

    if int(progress["budgeted_environment_steps"]) > target:
        raise ValueError("training state is ahead of the requested milestone")
    opening_path = root_path / str(config["opening_schedule_path"])
    schedule = json.loads(opening_path.read_text(encoding="utf-8"))
    train_groups = schedule["train_openings_by_seed"][str(seed)]
    if not train_groups:
        raise ValueError("registered V3 schedule contains no training openings")
    games_run_this_call = 0
    hard_cap = float(config["hard_cpu_hour_cap"]) * 3600
    while int(progress["budgeted_environment_steps"]) < target:
        if stop_after_games is not None and games_run_this_call >= stop_after_games:
            return {**progress, "completed": False, "target_node": target}
        try:
            ensure_v3_cpu_room(root_path, hard_cap)
        except BudgetExhausted as exc:
            return {**progress, "completed": False, "target_node": target, "budget_exhausted": str(exc)}
        current = int(progress["budgeted_environment_steps"])
        stage_index = next(index for index, node in enumerate(nodes) if current < node)
        stage_remaining = nodes[stage_index] - current
        game_index = len(games)
        opening_group = train_groups[game_index % len(train_groups)]
        game_id = f"g-v3-{config_sha[:12]}-{arm}-s{int(seed)}-game-{game_index:06d}"
        game_seed = int(seed) + 700_001 + game_index * 97
        game_started = time.perf_counter()
        cpu_started = time.process_time()
        game = play_selfplay_game(
            trainer.model,
            opening_group["opening"],
            arm_id=f"{arm}-seed-{int(seed)}",
            game_id=game_id,
            opening_group_id=opening_group["opening_group_id"],
            simulations=64,
            seed=game_seed,
            temperature_moves=int(config.get("temperature_moves", 12)),
            late_temperature=float(config.get("late_temperature", 0.1)),
            device=trainer.device.type,
        )
        accepted, overrun = budgeted_prefix(game["experience_rows"], stage_remaining)
        game.update({
            "stage_index": stage_index,
            "stage_game_index": sum(int(item.get("stage_index", -1)) == stage_index for item in games),
            "training_opening_kind": opening_group["kind"],
            "budgeted_environment_steps": len(accepted),
            "completion_overrun_steps": overrun,
            "runner_cpu_seconds": time.process_time() - cpu_started,
            "runner_wall_seconds": time.perf_counter() - game_started,
            "code_commit": _identity(root_path)[1],
            "config_sha256": config_sha,
        })
        if not append_complete_game(game_path, game, seen_ids):
            raise ValueError(f"G V3 runner attempted to duplicate game ID {game_id}")
        record_v3_cpu_event(
            root_path, f"selfplay:{game_id}", float(game["runner_cpu_seconds"]),
            category="selfplay",
        )
        games.append(game)
        if accepted:
            replay.add_complete_game(_budgeted_game_prefix(game, accepted, overrun))
        progress = _progress_from_games(games, nodes, list(progress.get("training_history", [])))
        progress.update({"arm": arm, "seed": int(seed), "config_sha256": config_sha})
        target_updates = progress["budgeted_environment_steps"] // int(config["update_every_budgeted_states"])
        if trainer.update_count > target_updates:
            raise ValueError("G V3 optimizer updates exceed the registered state cadence")
        new_training_batch = False
        if target_updates > trainer.update_count:
            try:
                ensure_v3_cpu_room(root_path, hard_cap)
            except BudgetExhausted as exc:
                progress["budget_exhausted"] = str(exc)
                raise
            metrics = trainer.train_updates(
                replay,
                updates=target_updates - trainer.update_count,
                batch_size=int(config["batch_size"]),
            )
            progress["training_history"].append({
                **metrics,
                "states_seen": int(progress["budgeted_environment_steps"]),
                "game_id": game_id,
                "stage_index": stage_index,
            })
            new_training_batch = True
        progress["optimizer_updates"] = trainer.update_count
        if trainer.update_count != target_updates:
            raise ValueError("G V3 optimizer schedule did not match accepted environment states")
        progress["replay_positions"] = replay.position_count
        progress["replay_games"] = replay.game_count
        progress["simulations_per_move"] = 64
        progress["bn_policy"] = trainer.bn_mode
        state_hash = save_v3_state(
            state_path, trainer, replay, arm=arm, seed=int(seed),
            progress=progress, initial_bn_buffers=initial_bn_buffers,
        )
        progress["training_state_sha256"] = state_hash
        _atomic_json(progress_path, progress)
        if new_training_batch:
            latest_training = progress["training_history"][-1]
            record_v3_cpu_event(
                root_path,
                f"optimizer:{arm}:seed-{int(seed)}:states-{int(latest_training['states_seen'])}",
                float(latest_training.get("cpu_seconds", 0.0)),
                category="optimizer",
            )
        games_run_this_call += 1

        if int(progress["budgeted_environment_steps"]) in nodes:
            node = int(progress["budgeted_environment_steps"])
            node_path = checkpoint_dir / f"node_{node}.pt"
            checkpoint_sha = save_g_checkpoint(node_path, trainer.model, {
                "experiment": "g_v3", "arm": arm, "seed": int(seed), "node": node,
                "budgeted_environment_steps": node,
                "optimizer_updates": trainer.update_count,
                "f2_sha256": F2_SHA256 if arm == "G-F2" else None,
                "config_sha256": config_sha,
                "training_data_source": "own PUCT-64 self-play only",
            })
            stage_row = {
                "checkpoint_id": f"{arm}-seed-{int(seed)}@{node}",
                "arm": arm,
                "seed": int(seed),
                "node": node,
                "game_count": len(games),
                "optimizer_updates": trainer.update_count,
                "checkpoint": node_path.relative_to(root_path).as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "progress": {key: progress[key] for key in (
                    "budgeted_environment_steps", "actual_experience_rows", "completion_overrun_steps",
                    "actual_simulations", "selfplay_cpu_seconds", "selfplay_wall_seconds",
                    "optimizer_cpu_seconds", "optimizer_wall_seconds",
                    "replay_positions", "replay_games", "stage_optimizer_updates",
                )},
                "training_metrics": progress["training_history"][-1] if progress["training_history"] else None,
                "config_sha256": config_sha,
            }
            _append_stage_record(root_path, stage_row)
            progress["last_checkpoint"] = node_path.relative_to(root_path).as_posix()
            progress["last_checkpoint_sha256"] = checkpoint_sha
            if node == nodes[-1]:
                final_path = checkpoint_dir / "final.pt"
                progress["final_checkpoint"] = final_path.relative_to(root_path).as_posix()
                progress["final_checkpoint_sha256"] = save_g_checkpoint(final_path, trainer.model, {
                    "experiment": "g_v3", "arm": arm, "seed": int(seed), "node": node,
                    "optimizer_updates": trainer.update_count, "config_sha256": config_sha,
                    "f2_sha256": F2_SHA256 if arm == "G-F2" else None,
                    "training_data_source": "own PUCT-64 self-play only",
                })
            state_hash = save_v3_state(
                state_path, trainer, replay, arm=arm, seed=int(seed),
                progress=progress, initial_bn_buffers=initial_bn_buffers,
            )
            progress["training_state_sha256"] = state_hash
            _atomic_json(progress_path, progress)

    return {**progress, "completed": int(progress["budgeted_environment_steps"]) == target, "target_node": target,
            "checkpoint": progress.get("last_checkpoint")}


def initialize_v3_starts(root: str | Path) -> list[tuple[int, str]]:
    """Create/verify every route-seed node-0 checkpoint before promotion state exists."""
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    recovery_target = int(config["stage_nodes"][-1])
    initialized: list[tuple[int, str]] = []
    for seed in map(int, config["seeds"]):
        for arm in ARMS:
            result = run_v3_arm(
                root_path, arm=arm, seed=seed, until_node=recovery_target, stop_after_games=0,
            )
            checkpoint = root_path / f"research/g_v3/checkpoints/{arm}/seed-{seed}/node_0.pt"
            if not checkpoint.is_file() or int(result["budgeted_environment_steps"]) < 0:
                raise ValueError(f"failed to initialize {arm} seed {seed} node-0 state")
            initialized.append((seed, arm))
    return initialized
