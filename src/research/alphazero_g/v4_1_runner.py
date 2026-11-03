"""Crash-safe, one-factor G0 self-play training arms for G V4.1."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .replay import ReplayBuffer, append_complete_game, load_game_log
from .selfplay import play_selfplay_game
from .training import GTrainer, load_g_checkpoint, save_g_checkpoint
from .v3_runner import load_v3_state, save_v3_state
from .v4_1_config import ARM_IDS, BASE_COMMIT, load_v4_1_config


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


def _append_jsonl_unique(path: Path, payload: dict[str, Any], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = str(payload[key])
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if str(row.get(key)) == identity:
                if row != payload:
                    raise ValueError(f"conflicting duplicate {key}: {identity}")
                return
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("failed to append durable G V4.1 record")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def budgeted_game_prefix(game: dict[str, Any], *, remaining_states: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(remaining_states, bool) or int(remaining_states) != remaining_states or remaining_states < 0:
        raise ValueError("remaining_states must be a nonnegative integer")
    rows = game.get("experience_rows")
    if not isinstance(rows, list):
        raise ValueError("self-play game has no experience rows")
    accepted_count = min(int(remaining_states), len(rows))
    accepted = copy.deepcopy(rows[:accepted_count])
    total_steps = len(game.get("moves", [])) - len(game.get("opening", []))
    if total_steps != len(rows):
        raise ValueError("self-play experience count differs from completed game plies")
    replay_game = copy.deepcopy(game)
    replay_game.update({
        "experience_rows": accepted,
        "budgeted_environment_steps": accepted_count,
        "completion_overrun_steps": total_steps - accepted_count,
        "replay_prefix": True,
    })
    return accepted, replay_game


def deterministic_game_seed(game_index: int, opening_seed: int) -> int:
    if isinstance(game_index, bool) or int(game_index) != game_index or game_index < 0:
        raise ValueError("game_index must be a nonnegative integer")
    payload = f"wuziq-g-v4-1:{int(game_index)}:{int(opening_seed)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _paths(root: Path, arm_id: str) -> dict[str, Path]:
    checkpoint_dir = root / "research/g_v4_1/checkpoints" / arm_id
    return {
        "checkpoint_dir": checkpoint_dir,
        "node0": checkpoint_dir / "node_0.pt",
        "node10k": checkpoint_dir / "node_10000.pt",
        "state": checkpoint_dir / "training_state.pt",
        "progress": checkpoint_dir / "progress.json",
        "game_log": root / "research/g_v4_1/results/selfplay" / f"{arm_id}.jsonl",
        "stages": root / "research/g_v4_1/results/training/stages.jsonl",
        "ledger": root / "research/g_v4_1/results/cpu_ledger.json",
    }


def _record_cpu(root: Path, event_id: str, cpu_seconds: float, wall_seconds: float, category: str) -> None:
    path = root / "research/g_v4_1/results/cpu_ledger.json"
    state = json.loads(path.read_text()) if path.exists() else {"format_version": 1, "events": []}
    existing = {str(row["event_id"]): row for row in state["events"]}
    row = {"event_id": event_id, "cpu_seconds": float(cpu_seconds),
           "wall_seconds": float(wall_seconds), "category": category}
    if event_id in existing:
        if existing[event_id] != row:
            raise ValueError(f"conflicting CPU event {event_id}")
        return
    if sum(float(item["cpu_seconds"]) for item in state["events"]) + cpu_seconds > 6 * 3600:
        raise RuntimeError("G V4.1 six CPU-hour hard cap reached")
    state["events"].append(row)
    _atomic_json(path, state)


def _cpu_used(root: Path) -> float:
    path = root / "research/g_v4_1/results/cpu_ledger.json"
    if not path.exists():
        return 0.0
    state = json.loads(path.read_text())
    return sum(float(row["cpu_seconds"]) for row in state.get("events", []))


def _reconcile_training_cpu(root: Path, arm_id: str, games: list[dict[str, Any]],
                            progress: dict[str, Any]) -> None:
    """Repair a ledger append interrupted after the durable train-state save."""
    path = root / "research/g_v4_1/results/cpu_ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else {"events": []}
    recorded = {str(row["event_id"]) for row in ledger.get("events", [])}
    history = {str(row.get("game_id")): row for row in progress.get("training_history", [])
               if row.get("game_id") is not None}
    for game in games:
        game_id = str(game["game_id"])
        event_id = f"train:{arm_id}:{game_id}"
        if event_id in recorded:
            continue
        update = history.get(game_id, {})
        selfplay_cpu = float(game.get("runner_cpu_seconds", game.get("stats", {}).get("cpu_seconds", 0.0)))
        optimizer_cpu = float(update.get("cpu_seconds", 0.0))
        selfplay_wall = float(game.get("runner_wall_seconds", game.get("stats", {}).get("wall_seconds", 0.0)))
        optimizer_wall = float(update.get("wall_seconds", 0.0))
        _record_cpu(root, event_id, selfplay_cpu + optimizer_cpu,
                    selfplay_wall + optimizer_wall, "train")


def _budgeted_replay_copy(game: dict[str, Any], count: int) -> dict[str, Any]:
    _, replay_game = budgeted_game_prefix(game, remaining_states=count)
    return replay_game


def _rebuild_initial_replay(root: Path, config: dict[str, Any], replay: ReplayBuffer) -> int:
    source = root / config["initial_replay"]["source"]
    games, _ = load_game_log(source)
    remaining = int(config["initial_replay"]["budgeted_states"])
    used = 0
    for game in games:
        available = int(game.get("budgeted_environment_steps", len(game["experience_rows"])))
        take = min(available, remaining - used)
        if take <= 0:
            break
        replay.add_complete_game(_budgeted_replay_copy(game, take))
        used += take
    if used != remaining:
        raise ValueError(f"V3 G0 replay rebuild wanted {remaining} states but found {used}")
    return used


def _add_new_games(replay: ReplayBuffer, games: list[dict[str, Any]], count: int) -> int:
    used = 0
    for game in games[:count]:
        accepted = int(game.get("budgeted_environment_steps", -1))
        if accepted < 0:
            raise ValueError("G V4.1 self-play record lacks accepted-state count")
        if accepted:
            replay.add_complete_game(_budgeted_replay_copy(game, accepted))
            used += accepted
    return used


def _initialize_state(root: Path, config: dict[str, Any], arm_id: str, replay: ReplayBuffer,
                      trainer: GTrainer, paths: dict[str, Path]) -> dict[str, Any]:
    initial_states = _rebuild_initial_replay(root, config, replay)
    if paths["state"].exists():
        payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
        saved_progress = payload.get("progress", {})
        if saved_progress.get("v4_1_arm_id") != arm_id:
            raise ValueError("G V4.1 training state belongs to another single-factor arm")
        games, _ = load_game_log(paths["game_log"])
        saved_count = int(saved_progress.get("completed_games", -1))
        if saved_count < 0 or len(games) < saved_count or len(games) > saved_count + 1:
            raise ValueError("G V4.1 complete-game log and atomic optimizer state disagree")
        _add_new_games(replay, games, saved_count)
        saved_ids = list(payload.get("replay", {}).get("game_ids", []))
        actual_ids = [str(game["game_id"]) for game in replay.state_dict()["games"]]
        if saved_ids != actual_ids:
            raise ValueError("G V4.1 replay reconstructed from durable games differs from checkpoint")
        progress = load_v3_state(
            paths["state"], trainer, replay, arm="G0", seed=int(config["random_seed"]),
            expected_config_sha256=str(config["config_sha256"]),
        )
        if int(progress.get("initial_replay_states", -1)) != initial_states:
            raise ValueError("G V4.1 restored initial replay size differs from registration")
        if len(games) == saved_count + 1:
            ahead = games[-1]
            accepted = int(ahead.get("budgeted_environment_steps", -1))
            if accepted < 0:
                raise ValueError("ahead-of-checkpoint game has no accepted state count")
            if accepted:
                replay.add_complete_game(_budgeted_replay_copy(ahead, accepted))
            total_new = int(progress["budgeted_environment_steps"]) + accepted
            target_updates = total_new // int(config["training"]["update_every_budgeted_states"])
            if target_updates < trainer.update_count:
                raise ValueError("G V4.1 optimizer is ahead of durable game records")
            if target_updates > trainer.update_count:
                metrics = trainer.train_updates(
                    replay, updates=target_updates - trainer.update_count,
                    batch_size=int(config["training"]["batch_size"]),
                    sampling_mode=str(config["arms"][arm_id]["sampling_mode"]),
                )
                progress.setdefault("training_history", []).append({
                    **metrics, "states_seen": total_new, "game_id": ahead["game_id"],
                    "reason": "reconcile_fsynced_game_after_restart",
                })
            progress["budgeted_environment_steps"] = total_new
            progress["completed_games"] = len(games)
            progress["optimizer_updates"] = trainer.update_count
            progress["replay_positions"] = replay.position_count
            progress["replay_games"] = replay.game_count
            progress["resumed_reconciliation"] = True
            digest = save_v3_state(
                paths["state"], trainer, replay, arm="G0", seed=int(config["random_seed"]),
                progress={**progress, "config_sha256": config["config_sha256"]},
                initial_bn_buffers={},
            )
            progress["training_state_sha256"] = digest
            _atomic_json(paths["progress"], progress)
        return progress
    if paths["game_log"].exists() and paths["game_log"].stat().st_size:
        raise ValueError("G V4.1 self-play log exists without an atomic train-state checkpoint")
    model_sha = save_g_checkpoint(paths["node0"], trainer.model, {
        "experiment": "g_v4_1", "arm_id": arm_id, "node": 0,
        "initial_checkpoint_id": config["initial_checkpoint_id"],
        "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
        "optimizer_reset": True, "config_sha256": config["config_sha256"],
    })
    progress: dict[str, Any] = {
        "phase": "TRAINING", "v4_1_arm_id": arm_id, "config_sha256": config["config_sha256"],
        "initial_checkpoint_id": config["initial_checkpoint_id"],
        "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
        "initial_node0_sha256": model_sha, "initial_replay_states": initial_states,
        "budgeted_environment_steps": 0, "completed_games": 0,
        "optimizer_updates": 0, "replay_positions": replay.position_count,
        "replay_games": replay.game_count, "actual_simulations": 0,
        "selfplay_cpu_seconds": 0.0, "optimizer_cpu_seconds": 0.0,
        "training_history": [], "sampling_mode": config["arms"][arm_id]["sampling_mode"],
        "learning_rate": config["arms"][arm_id]["learning_rate"],
    }
    digest = save_v3_state(
        paths["state"], trainer, replay, arm="G0", seed=int(config["random_seed"]),
        progress=progress, initial_bn_buffers={},
    )
    progress["training_state_sha256"] = digest
    _atomic_json(paths["progress"], progress)
    return progress


def run_v4_1_arm(
    root: str | Path,
    arm_id: str,
    *,
    max_games: int | None = None,
) -> dict[str, Any]:
    """Run one registered 10K G0 arm; fsync each complete game and train state."""
    root_path = Path(root).resolve()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    config = load_v4_1_config(root_path)
    if arm_id not in ARM_IDS:
        raise ValueError("unknown G V4.1 arm")
    if max_games is not None and (isinstance(max_games, bool) or int(max_games) != max_games or max_games < 1):
        raise ValueError("max_games must be a positive integer or null")
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    arm = config["arms"][arm_id]
    total_states = int(config["training"]["new_states_per_arm"])
    paths = _paths(root_path, arm_id)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    seed = int(config["random_seed"])
    model = load_g_checkpoint(root_path / config["initial_checkpoint"], device="cpu")
    trainer = GTrainer(
        model, seed=seed, learning_rate=float(arm["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        expected_updates=total_states // int(config["training"]["update_every_budgeted_states"]),
        device="cpu", bn_mode=str(config["training"]["bn_policy"]),
    )
    replay = ReplayBuffer(int(config["initial_replay"]["capacity_positions"]), seed=seed + 17)
    if paths["state"].exists():
        payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
        saved_count = int(payload.get("progress", {}).get("completed_games", -1))
        logged, _ = load_game_log(paths["game_log"])
        if saved_count >= 0:
            _rebuild_initial_replay(root_path, config, replay)
            _add_new_games(replay, logged, saved_count)
            if list(payload.get("replay", {}).get("game_ids", [])) != [
                str(game["game_id"]) for game in replay.state_dict()["games"]
            ]:
                raise ValueError("G V4.1 replay game IDs do not match the atomic checkpoint")
            progress = load_v3_state(
                paths["state"], trainer, replay, arm="G0", seed=seed,
                expected_config_sha256=str(config["config_sha256"]),
            )
            # A single complete game may be durable while its optimizer state is not.
            if len(logged) == saved_count + 1:
                game = logged[-1]
                accepted = int(game.get("budgeted_environment_steps", -1))
                if accepted < 0:
                    raise ValueError("durable G V4.1 game has no accepted-state count")
                if accepted:
                    replay.add_complete_game(_budgeted_replay_copy(game, accepted))
                states = int(progress["budgeted_environment_steps"]) + accepted
                updates = states // int(config["training"]["update_every_budgeted_states"])
                if updates > trainer.update_count:
                    metrics = trainer.train_updates(
                        replay, updates=updates - trainer.update_count,
                        batch_size=int(config["training"]["batch_size"]),
                        sampling_mode=str(arm["sampling_mode"]),
                    )
                    progress.setdefault("training_history", []).append({
                        **metrics, "states_seen": states, "game_id": game["game_id"],
                        "reason": "reconcile_fsynced_game_after_restart",
                    })
                progress["budgeted_environment_steps"] = states
                progress["completed_games"] = len(logged)
                progress["optimizer_updates"] = trainer.update_count
                progress["replay_positions"] = replay.position_count
                progress["replay_games"] = replay.game_count
                progress["resumed_reconciliation"] = True
                digest = save_v3_state(
                    paths["state"], trainer, replay, arm="G0", seed=seed,
                    progress=progress, initial_bn_buffers={},
                )
                progress["training_state_sha256"] = digest
                _atomic_json(paths["progress"], progress)
            elif len(logged) != saved_count:
                raise ValueError("G V4.1 complete-game log is ahead of its train state")
        else:
            raise ValueError("G V4.1 train state has no durable game cursor")
    else:
        games, _ = load_game_log(paths["game_log"])
        if games:
            raise ValueError("G V4.1 results exist without a corresponding resumable checkpoint")
        progress = _initialize_state(root_path, config, arm_id, replay, trainer, paths)
        logged = []

    if not paths["state"].exists():
        raise RuntimeError("G V4.1 failed to persist its initial resumable state")
    durable_games, _ = load_game_log(paths["game_log"])
    _reconcile_training_cpu(root_path, arm_id, durable_games, progress)
    if paths["node10k"].is_file():
        if int(progress.get("budgeted_environment_steps", -1)) != total_states:
            raise ValueError("G V4.1 final checkpoint exists but its state quota is incomplete")
        return {**progress, "completed": True}
    games, seen = load_game_log(paths["game_log"])
    current = int(progress["budgeted_environment_steps"])
    if current > total_states:
        raise ValueError("G V4.1 arm exceeds its 10K state quota")
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    train_groups = schedule["train_openings_by_seed"]["741101"]
    started_games = len(games)
    # The source commit is frozen; a working-tree digest makes uncommitted execution reproducible.
    code_digest = hashlib.sha256()
    for name in ("src/research/alphazero_g/replay.py", "src/research/alphazero_g/training.py",
                 "src/research/alphazero_g/v4_1_config.py", "src/research/alphazero_g/v4_1_runner.py"):
        code_digest.update((root_path / name).read_bytes())
    this_code_sha = code_digest.hexdigest()
    invocation_game_cpu = 0.0
    invocation_optimizer_cpu = 0.0
    invocation_game_wall = 0.0
    invocation_optimizer_wall = 0.0
    while current < total_states and (max_games is None or len(games) - started_games < max_games):
        cpu_cap = float(config["compute"]["cpu_hour_cap"]) * 3600
        # Reserve room for one additional full game before starting it. The
        # reserve is the larger of 60 CPU seconds and 1.5x the slowest game
        # already observed; this makes the hard cap safe despite game-level
        # (rather than move-level) durable checkpoints.
        observed_game_costs = [float(game.get("runner_cpu_seconds", 0.0)) for game in games]
        reserve = max(60.0, 1.5 * max(observed_game_costs, default=0.0))
        if _cpu_used(root_path) + reserve >= cpu_cap:
            progress.update({"completed": False, "stop_reason": "six_cpu_hour_cap", "phase": "PAUSED"})
            _atomic_json(paths["progress"], progress)
            return progress
        game_index = len(games)
        group = train_groups[game_index % len(train_groups)]
        game_id = f"g-v4-1-{config['config_sha256'][:10]}-{arm_id}-game-{game_index:06d}"
        game_started_cpu, game_started_wall = time.process_time(), time.perf_counter()
        game = play_selfplay_game(
            trainer.model, group["opening"], arm_id=f"G0-v4-1-{arm_id}", game_id=game_id,
            opening_group_id=str(group["opening_group_id"]), simulations=64,
            seed=deterministic_game_seed(game_index, int(group["opening_seed"])),
            temperature_moves=int(config["training"]["temperature_moves"]),
            late_temperature=float(config["training"]["late_temperature"]), device="cpu",
        )
        accepted, replay_game = budgeted_game_prefix(game, remaining_states=total_states - current)
        game.update({
            "job_id": arm_id, "stage_index": 0,
            "training_opening_kind": group["kind"],
            "budgeted_environment_steps": len(accepted),
            "completion_overrun_steps": len(game["experience_rows"]) - len(accepted),
            "runner_cpu_seconds": time.process_time() - game_started_cpu,
            "runner_wall_seconds": time.perf_counter() - game_started_wall,
            "code_commit": BASE_COMMIT, "working_tree_code_sha256": this_code_sha,
            "config_sha256": config["config_sha256"],
        })
        if not append_complete_game(paths["game_log"], game, seen):
            raise ValueError(f"duplicate G V4.1 self-play game ID: {game_id}")
        games.append(game)
        if accepted:
            replay.add_complete_game(replay_game)
        target_states = current + len(accepted)
        target_updates = target_states // int(config["training"]["update_every_budgeted_states"])
        metrics = {"updates": 0, "cpu_seconds": 0.0, "wall_seconds": 0.0}
        if target_updates > trainer.update_count:
            metrics = trainer.train_updates(
                replay, updates=target_updates - trainer.update_count,
                batch_size=int(config["training"]["batch_size"]),
                sampling_mode=str(arm["sampling_mode"]),
            )
        if trainer.update_count != target_updates:
            raise ValueError("G V4.1 optimizer updates differ from the registered 16-state cadence")
        current = target_states
        game_cpu = float(game["runner_cpu_seconds"]) + float(metrics["cpu_seconds"])
        if _cpu_used(root_path) + game_cpu > cpu_cap:
            # This should be prevented by the reserve above. Raise visibly
            # instead of ever silently exceeding the registered hard limit.
            raise RuntimeError("G V4.1 six CPU-hour hard cap reached inside a full-game batch")
        invocation_game_cpu += float(game["runner_cpu_seconds"])
        invocation_optimizer_cpu += float(metrics["cpu_seconds"])
        invocation_game_wall += float(game["runner_wall_seconds"])
        invocation_optimizer_wall += float(metrics["wall_seconds"])
        progress.update({
            "phase": "TRAINING", "v4_1_arm_id": arm_id, "config_sha256": config["config_sha256"],
            "budgeted_environment_steps": current, "completed_games": len(games),
            "optimizer_updates": trainer.update_count, "replay_positions": replay.position_count,
            "replay_games": replay.game_count,
            "actual_simulations": int(progress.get("actual_simulations", 0)) + int(game["stats"]["simulations"]),
            "selfplay_cpu_seconds": float(progress.get("selfplay_cpu_seconds", 0.0)) + float(game["runner_cpu_seconds"]),
            "optimizer_cpu_seconds": float(progress.get("optimizer_cpu_seconds", 0.0)) + float(metrics["cpu_seconds"]),
            "training_history": list(progress.get("training_history", [])) + ([{
                **metrics, "states_seen": current, "game_id": game_id,
                "sampling_mode": arm["sampling_mode"],
            }] if int(metrics["updates"]) else []),
            "last_game_id": game_id,
        })
        state_sha = save_v3_state(
            paths["state"], trainer, replay, arm="G0", seed=seed,
            progress=progress, initial_bn_buffers={},
        )
        progress["training_state_sha256"] = state_sha
        _atomic_json(paths["progress"], progress)
        _record_cpu(root_path, f"train:{arm_id}:{game_id}", game_cpu,
                    float(game["runner_wall_seconds"]) + float(metrics["wall_seconds"]), "train")
    process_cpu = time.process_time() - cpu_start
    process_wall = time.perf_counter() - wall_start
    if process_cpu > 0.5:
        overhead = max(0.0, process_cpu - invocation_game_cpu - invocation_optimizer_cpu)
        overhead_wall = max(0.0, process_wall - invocation_game_wall - invocation_optimizer_wall)
        _record_cpu(root_path, f"runner_overhead:{arm_id}:{games[-1]['game_id'] if games else 'empty'}", overhead,
            overhead_wall, "orchestration")
    if current < total_states:
        progress["completed"] = False
        progress["phase"] = "PAUSED"
        _atomic_json(paths["progress"], progress)
        return progress
    sha = save_g_checkpoint(paths["node10k"], trainer.model, {
        "experiment": "g_v4_1", "arm_id": arm_id, "node": total_states,
        "new_budgeted_states": total_states, "optimizer_updates": trainer.update_count,
        "initial_checkpoint_id": config["initial_checkpoint_id"],
        "sampling_mode": arm["sampling_mode"], "learning_rate": arm["learning_rate"],
        "training_data_source": "own G0 PUCT-64 self-play only",
        "config_sha256": config["config_sha256"],
    })
    progress.update({
        "completed": True, "phase": "TRAINED", "checkpoint": str(paths["node10k"].relative_to(root_path)),
        "checkpoint_sha256": sha, "optimizer_updates": trainer.update_count,
    })
    _atomic_json(paths["progress"], progress)
    stage = {
        "checkpoint_id": f"{arm_id}@10000", "arm_id": arm_id,
        "new_states": total_states, "total_lineage_states": 30_000,
        "game_count": len(games), "optimizer_updates": trainer.update_count,
        "checkpoint": str(paths["node10k"].relative_to(root_path)), "checkpoint_sha256": sha,
        "selfplay_cpu_seconds": progress["selfplay_cpu_seconds"],
        "optimizer_cpu_seconds": progress["optimizer_cpu_seconds"],
        "actual_simulations": progress["actual_simulations"],
        "replay_positions": replay.position_count, "replay_games": replay.game_count,
        "sampling_mode": arm["sampling_mode"], "learning_rate": arm["learning_rate"],
        "config_sha256": config["config_sha256"],
    }
    _append_jsonl_unique(paths["stages"], stage, "checkpoint_id")
    return progress
