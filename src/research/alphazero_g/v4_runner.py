"""Resumable G V4 G0 self-play jobs, development gates, and blind evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .replay import ReplayBuffer, append_complete_game, budgeted_prefix, load_game_log
from .selfplay import play_selfplay_game
from .training import GTrainer, initialize_arm, load_g_checkpoint, save_g_checkpoint
from .v3_runner import _budgeted_game_prefix, _progress_from_games, load_v3_state, save_v3_state
from .v3_bn import capture_bn_buffers
from .v3_evaluation import (
    _group_seed, build_v3_tactical_puzzles, evaluate_v3_tactics, play_v3_match,
    promotion_decision, summarize_v3_matches,
)
from .v3_schedule import validate_v3_schedule
from .evaluation import append_evaluation_match, load_evaluation_log
from .v4_budget import V4BudgetExhausted, _load_rows, ensure_cpu_room, record_cpu_event
from .v4_config import (
    BRANCH, V3_CHAMPION_ID, V3_CHAMPION_REL, _atomic_json, load_v4_config,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _identity(root: Path) -> tuple[str, str]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                           capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise ValueError(f"G V4 runner requires {BRANCH}")
    return branch, commit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any], id_key: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_id = str(row[id_key])
    if path.exists():
        raw = path.read_bytes()
        complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
        if len(complete) != len(raw):
            with path.open("r+b") as handle:
                handle.truncate(len(complete))
                handle.flush()
                os.fsync(handle.fileno())
        for line in complete.decode("utf-8").splitlines():
            prior = json.loads(line)
            if str(prior.get(id_key)) == event_id:
                if prior != row:
                    raise ValueError(f"conflicting duplicate G V4 record {event_id}")
                return False
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("short append while writing G V4 record")
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    if complete != raw:
        raise ValueError(f"G V4 JSONL has an uncommitted tail: {path}")
    return [json.loads(line) for line in complete.splitlines()]


def _phase_nodes(phase: str, total: int) -> list[int]:
    if phase == "A":
        return [10_000, 20_000, 30_000]
    return list(range(10_000, int(total) + 1, 10_000))


def phase_a_job(seed: int) -> dict[str, Any]:
    return {
        "job_id": f"A-G0-s{int(seed)}", "phase": "A", "seed": int(seed),
        "opening_seed": int(seed), "new_states": 30_000, "nodes": [10_000, 20_000, 30_000],
        "source_model": None, "start_checkpoint_id": f"A-G0-s{int(seed)}@0",
        "source_replay_log": None, "replay_mode": "empty", "replay_source_states": 0,
        "optimizer_mode": "fresh_random_initialization",
    }


def phase_b_job() -> dict[str, Any]:
    return {
        "job_id": "B-G0-v3-champion-continue", "phase": "B", "seed": 540404,
        "opening_seed": 540404, "new_states": 100_000, "nodes": _phase_nodes("B", 100_000),
        "source_model": V3_CHAMPION_REL, "start_checkpoint_id": V3_CHAMPION_ID,
        "source_replay_log": "research/g_v3/results/selfplay/G0-seed-510901.jsonl",
        "replay_mode": "prefix", "replay_source_states": 20_000,
        "optimizer_mode": "reinitialized_adamw_scheduler_from_weight_only_20k_checkpoint",
    }


def phase_c_job(seed: int, source_checkpoint: str, source_checkpoint_id: str) -> dict[str, Any]:
    return {
        "job_id": f"C-G0-from-A-s{int(seed)}", "phase": "C", "seed": 540405,
        "source_seed": int(seed),
        "opening_seed": 540405, "new_states": 110_000, "nodes": _phase_nodes("C", 110_000),
        "source_model": source_checkpoint, "start_checkpoint_id": source_checkpoint_id,
        "source_replay_log": f"research/g_v4/results/selfplay/A-G0-s{int(seed)}.jsonl",
        "replay_mode": "all_complete_games_into_capacity_20k_replay",
        "replay_source_states": None,
        "optimizer_mode": "reinitialized_adamw_scheduler_for_second_tier_continuation",
    }


def _job_paths(root: Path, job: dict[str, Any]) -> dict[str, Path]:
    base = root / "research/g_v4"
    checkpoint = base / "checkpoints" / str(job["job_id"])
    return {
        "checkpoint_dir": checkpoint,
        "game_log": base / "results/selfplay" / f"{job['job_id']}.jsonl",
        "state": checkpoint / "training_state.pt",
        "progress": checkpoint / "progress.json",
        "selection": root / "research/g_v4/results/selection/jobs" / f"{job['job_id']}.json",
    }


def _training_seed(job: dict[str, Any], config: dict[str, Any]) -> int:
    token = f"{config['config_sha256']}:{job['job_id']}:trainer"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def _model_for_job(root: Path, job: dict[str, Any]) -> torch.nn.Module:
    if job.get("source_model"):
        path = root / str(job["source_model"])
        if not path.is_file():
            raise FileNotFoundError(path)
        model = load_g_checkpoint(path, device="cpu")
        model.checkpoint_metadata = {
            **getattr(model, "checkpoint_metadata", {}),
            "g_v4_job": job["job_id"], "g_v4_initialization": "weight_inheritance",
            "g_v4_source_checkpoint": job["source_model"],
            "g_v4_source_checkpoint_sha256": _sha(path),
        }
        return model
    model = initialize_arm("G0", seed=int(job["seed"]), device="cpu")
    model.checkpoint_metadata = {
        **getattr(model, "checkpoint_metadata", {}),
        "g_v4_job": job["job_id"], "g_v4_initialization": "seeded_random",
    }
    return model


def _prefix_from_game(game: dict[str, Any], count: int) -> dict[str, Any]:
    total = len(game["moves"]) - len(game["opening"])
    rows = copy.deepcopy(game["experience_rows"][:int(count)])
    if len(rows) != int(count):
        raise ValueError("G V4 replay reconstruction exceeded recorded self-play rows")
    return _budgeted_game_prefix(game, rows, total - int(count))


def _rebuild_initial_replay(root: Path, job: dict[str, Any], replay: ReplayBuffer) -> int:
    source = job.get("source_replay_log")
    if not source:
        return 0
    games, _ = load_game_log(root / str(source))
    remaining = job.get("replay_source_states")
    used = 0
    for game in games:
        available = int(game.get("budgeted_environment_steps", len(game["experience_rows"])))
        if remaining is not None:
            take = min(available, int(remaining) - used)
            if take <= 0:
                break
        else:
            take = available
        if take:
            replay.add_complete_game(_prefix_from_game(game, take))
            used += take
        if remaining is not None and used >= int(remaining):
            break
    if remaining is not None and used != int(remaining):
        raise ValueError(f"could not rebuild the requested {remaining} source replay states; found {used}")
    return used


def _rebuild_replay(root: Path, job: dict[str, Any], games: list[dict[str, Any]], replay: ReplayBuffer) -> int:
    initial = _rebuild_initial_replay(root, job, replay)
    added = 0
    for game in games:
        count = int(game.get("budgeted_environment_steps", -1))
        if count < 0:
            raise ValueError("V4 self-play game has no accepted state count")
        if count:
            replay.add_complete_game(_prefix_from_game(game, count))
            added += count
    return initial + added


def _job_selection(root: Path, job: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    paths = _job_paths(root, job)
    path = paths["selection"]
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config["config_sha256"] or state.get("job_id") != job["job_id"]:
            raise ValueError("G V4 job selection state does not match registration")
        return state
    start = paths["checkpoint_dir"] / "node_0.pt"
    if not start.is_file():
        raise FileNotFoundError("G V4 job node-0 checkpoint must be prepared before selection")
    state = {
        "format_version": 1, "config_sha256": config["config_sha256"],
        "job_id": job["job_id"], "phase": job["phase"], "seed": int(job["seed"]),
        "start_checkpoint_id": job["start_checkpoint_id"],
        "start_checkpoint": str(start.relative_to(root).as_posix()),
        "start_checkpoint_sha256": _sha(start),
        "champion_checkpoint_id": job["start_checkpoint_id"],
        "champion_checkpoint": str(start.relative_to(root).as_posix()),
        "champion_checkpoint_sha256": _sha(start),
        "last_decision_node": 0, "consecutive_below_50pct": 0,
        "completed_nodes": [], "decisions": [], "optimizer_mode": job["optimizer_mode"],
    }
    _atomic_json(path, state)
    return state


def prepare_v4_job(root: str | Path, job: dict[str, Any]) -> dict[str, Any]:
    root_path = Path(root).resolve()
    _identity(root_path)
    config = load_v4_config(root_path)
    if job.get("phase") not in {"A", "B", "C"}:
        raise ValueError("unknown G V4 training phase")
    paths = _job_paths(root_path, job)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    model = _model_for_job(root_path, job)
    total = int(job["new_states"])
    if total % 16:
        raise ValueError("G V4 new state quotas must align to the 16-state optimizer cadence")
    trainer = GTrainer(
        model, seed=_training_seed(job, config), learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]), expected_updates=total // 16,
        device="cpu", bn_mode=str(config["training"]["bn_policy"]),
    )
    replay = ReplayBuffer(int(config["training"]["replay_capacity_positions"]), seed=_training_seed(job, config) + 17)
    initial_replay_states = _rebuild_initial_replay(root_path, job, replay)
    if paths["state"].exists():
        payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
        if payload.get("arm") != "G0" or int(payload.get("seed", -1)) != int(job["seed"]):
            raise ValueError("G V4 training-state job identity mismatch")
        return {"job_id": job["job_id"], "prepared": True,
                "initial_replay_states": initial_replay_states, "resumed": True}
    if paths["game_log"].exists() and paths["game_log"].stat().st_size:
        raise ValueError("G V4 game log exists without a matching train-state checkpoint")
    initial_bn = capture_bn_buffers(model)
    start_path = paths["checkpoint_dir"] / "node_0.pt"
    sha = save_g_checkpoint(start_path, model, {
        "experiment": "g_v4", "job_id": job["job_id"], "phase": job["phase"],
        "seed": int(job["seed"]), "node": 0,
        "source_checkpoint": job.get("source_model"),
        "source_checkpoint_sha256": _sha(root_path / job["source_model"]) if job.get("source_model") else None,
        "optimizer_mode": job["optimizer_mode"], "config_sha256": config["config_sha256"],
        "training_data_source": "PUCT-64 self-play only",
    })
    progress = _progress_from_games([], list(map(int, job["nodes"])), [])
    progress.update({
        "arm": "G0", "seed": int(job["seed"]), "job_id": job["job_id"],
        "phase": job["phase"], "config_sha256": config["config_sha256"],
        "start_checkpoint": str(start_path.relative_to(root_path).as_posix()),
        "start_checkpoint_sha256": sha, "source_checkpoint": job.get("source_model"),
        "source_checkpoint_id": job.get("start_checkpoint_id"),
        "source_replay_log": job.get("source_replay_log"),
        "source_replay_states": initial_replay_states,
        "optimizer_mode": job["optimizer_mode"],
        "replay_positions": replay.position_count, "replay_games": replay.game_count,
    })
    state_sha = save_v3_state(paths["state"], trainer, replay, arm="G0", seed=int(job["seed"]),
                              progress=progress, initial_bn_buffers=initial_bn)
    progress["training_state_sha256"] = state_sha
    _atomic_json(paths["progress"], progress)
    selection = _job_selection(root_path, job, config)
    return {"job_id": job["job_id"], "prepared": True, "checkpoint_sha256": sha,
            "initial_replay_states": initial_replay_states, "resumed": False}


def _load_job_runtime(root: Path, job: dict[str, Any], config: dict[str, Any]):
    paths = _job_paths(root, job)
    games, seen = load_game_log(paths["game_log"])
    model = _model_for_job(root, job)
    trainer = GTrainer(
        model, seed=_training_seed(job, config), learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]), expected_updates=int(job["new_states"]) // 16,
        device="cpu", bn_mode=str(config["training"]["bn_policy"]),
    )
    replay = ReplayBuffer(int(config["training"]["replay_capacity_positions"]), seed=_training_seed(job, config) + 17)
    _rebuild_initial_replay(root, job, replay)
    state_payload = torch.load(paths["state"], map_location="cpu", weights_only=False)
    progress = state_payload.get("progress", {})
    saved_count = int(progress.get("game_count", -1))
    saved_ids = list(progress.get("completed_game_ids", []))
    if saved_count < 0 or saved_count > len(games) or saved_ids != [str(game["game_id"]) for game in games[:saved_count]]:
        raise ValueError("G V4 game cursor conflicts with its last atomic training state")
    if len(games) - saved_count > 1:
        raise ValueError("multiple G V4 games are ahead of the last durable optimizer state")
    # Self-play is fsynced before training state is replaced. Reconstruct the
    # checkpointed replay first, then replay at most one complete game that
    # crossed that durability boundary and finish its missing optimizer steps.
    for game in games[:saved_count]:
        count = int(game.get("budgeted_environment_steps", -1))
        if count < 0:
            raise ValueError("G V4 durable self-play record lacks a budgeted state count")
        if count:
            replay.add_complete_game(_prefix_from_game(game, count))
    saved_replay_ids = list(state_payload.get("replay", {}).get("game_ids", []))
    replay_ids = [str(game["game_id"]) for game in replay.state_dict().get("games", [])]
    if saved_replay_ids != replay_ids:
        raise ValueError("G V4 reconstructed experience replay differs from saved game IDs")
    initial_bn = state_payload.get("initial_bn_buffers")
    if not isinstance(initial_bn, dict):
        raise ValueError("G V4 resume state is missing BatchNorm reference buffers")
    loaded_progress = load_v3_state(
        paths["state"], trainer, replay, arm="G0", seed=int(job["seed"]),
        expected_config_sha256=config["config_sha256"],
    )
    if len(games) > saved_count:
        ahead = games[saved_count]
        count = int(ahead.get("budgeted_environment_steps", -1))
        if count < 0:
            raise ValueError("G V4 durable self-play record lacks a budgeted state count")
        if count:
            replay.add_complete_game(_prefix_from_game(ahead, count))
        reconciled = _progress_from_games(
            games, list(map(int, job["nodes"])), list(loaded_progress.get("training_history", [])),
        )
        target_updates = int(reconciled["budgeted_environment_steps"]) // 16
        if trainer.update_count > target_updates:
            raise ValueError("saved optimizer is ahead of the durable G V4 experience rows")
        if trainer.update_count < target_updates:
            metrics = trainer.train_updates(
                replay, updates=target_updates - trainer.update_count,
                batch_size=int(config["training"]["batch_size"]),
            )
            metrics.update({"states_seen": int(reconciled["budgeted_environment_steps"]),
                            "game_id": str(ahead["game_id"]), "reason": "reconcile_fsynced_game_after_restart"})
            reconciled["training_history"].append(metrics)
            record_cpu_event(
                root, f"optimizer-recovery:{job['job_id']}:{uuid.uuid4().hex}",
                float(metrics["cpu_seconds"]), wall_seconds=float(metrics["wall_seconds"]), category="optimizer",
            )
        if trainer.update_count != target_updates:
            raise ValueError("recovered G V4 optimizer updates do not match accepted state count")
        reconciled.update({
            "arm": "G0", "phase": job["phase"], "job_id": job["job_id"],
            "seed": int(job["seed"]), "config_sha256": config["config_sha256"],
            "source_checkpoint": job.get("source_model"), "source_checkpoint_id": job.get("start_checkpoint_id"),
            "source_replay_log": job.get("source_replay_log"), "optimizer_mode": job["optimizer_mode"],
            "optimizer_updates": trainer.update_count, "replay_positions": replay.position_count,
            "replay_games": replay.game_count,
        })
        state_sha = save_v3_state(paths["state"], trainer, replay, arm="G0", seed=int(job["seed"]),
                                  progress=reconciled, initial_bn_buffers=initial_bn)
        reconciled["training_state_sha256"] = state_sha
        _atomic_json(paths["progress"], reconciled)
        loaded_progress = reconciled
    progress = _progress_from_games(games, list(map(int, job["nodes"])),
                                    list(loaded_progress.get("training_history", [])))
    progress.update({key: loaded_progress[key] for key in (
        "job_id", "phase", "source_checkpoint", "source_checkpoint_id", "source_replay_log",
        "source_replay_states", "optimizer_mode", "start_checkpoint", "start_checkpoint_sha256",
    ) if key in loaded_progress})
    progress.update({
        "arm": "G0", "seed": int(job["seed"]), "config_sha256": config["config_sha256"],
        "optimizer_updates": trainer.update_count,
        "replay_positions": replay.position_count, "replay_games": replay.game_count,
        "training_state_sha256": _sha(paths["state"]),
    })
    expected_updates = int(progress["budgeted_environment_steps"]) // 16
    if trainer.update_count != expected_updates:
        raise ValueError("G V4 optimizer update count does not match the durable accepted state count")
    return paths, trainer, replay, games, seen, progress, initial_bn


def run_v4_job_to_node(root: str | Path, job: dict[str, Any], node: int) -> dict[str, Any]:
    root_path = Path(root).resolve()
    _identity(root_path)
    config = load_v4_config(root_path)
    nodes = list(map(int, job["nodes"]))
    if int(node) not in nodes:
        raise ValueError("requested G V4 node is not registered for this job")
    if int(config["training"]["simulation_budget"]) != 64 or int(config["training"]["update_every_budgeted_states"]) != 16:
        raise ValueError("G V4 requires the registered PUCT-64 and 16-state update schedule")
    torch.set_num_threads(1)
    prepare_v4_job(root_path, job)
    paths, trainer, replay, games, seen, progress, initial_bn = _load_job_runtime(root_path, job, config)
    current = int(progress["budgeted_environment_steps"])
    target = int(node)
    if current > target:
        raise ValueError("G V4 durable job state is already ahead of requested stage")
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    groups = schedule["train_openings_by_seed"][str(job["opening_seed"])]
    if not groups:
        raise ValueError("G V4 schedule has no training openings for job")
    game_path = paths["game_log"]
    cap_hours = float(config["hard_cpu_hour_cap"])
    _, commit = _identity(root_path)
    while current < target:
        try:
            ensure_cpu_room(root_path, cap_hours, reserve_seconds=10.0)
        except V4BudgetExhausted as exc:
            return {**progress, "completed": False, "target_node": target, "budget_exhausted": str(exc)}
        next_stage = next(value for value in nodes if current < value)
        remaining = min(next_stage - current, target - current)
        game_index = len(games)
        group = groups[game_index % len(groups)]
        game_id = f"g-v4-{config['config_sha256'][:10]}-{job['job_id']}-game-{game_index:06d}"
        game_seed = int(hashlib.sha256(f"{job['job_id']}:{game_index}:{group['opening_seed']}".encode()).hexdigest()[:8], 16)
        game_cpu_start, game_wall_start = time.process_time(), time.perf_counter()
        game = play_selfplay_game(
            trainer.model, group["opening"], arm_id=f"G0-v4-{job['job_id']}", game_id=game_id,
            opening_group_id=group["opening_group_id"], simulations=64, seed=game_seed,
            temperature_moves=int(config["training"]["temperature_moves"]),
            late_temperature=float(config["training"]["late_temperature"]), device="cpu",
        )
        accepted, overrun = budgeted_prefix(game["experience_rows"], remaining)
        game.update({
            "phase": job["phase"], "job_id": job["job_id"],
            "stage_index": nodes.index(next_stage), "training_opening_kind": group["kind"],
            "budgeted_environment_steps": len(accepted), "completion_overrun_steps": overrun,
            "runner_cpu_seconds": time.process_time() - game_cpu_start,
            "runner_wall_seconds": time.perf_counter() - game_wall_start,
            "code_commit": commit, "config_sha256": config["config_sha256"],
        })
        if not append_complete_game(game_path, game, seen):
            raise ValueError(f"duplicate G V4 self-play game {game_id}")
        games.append(game)
        record_cpu_event(root_path, f"selfplay:{game_id}", game["runner_cpu_seconds"],
                         wall_seconds=game["runner_wall_seconds"], category="selfplay")
        if accepted:
            replay.add_complete_game(_budgeted_game_prefix(game, accepted, overrun))
        progress = _progress_from_games(games, nodes, list(progress.get("training_history", [])))
        progress.update({
            "arm": "G0", "phase": job["phase"], "job_id": job["job_id"],
            "seed": int(job["seed"]), "config_sha256": config["config_sha256"],
            "source_checkpoint": job.get("source_model"), "source_checkpoint_id": job.get("start_checkpoint_id"),
            "source_replay_log": job.get("source_replay_log"),
            "optimizer_mode": job["optimizer_mode"],
        })
        target_updates = int(progress["budgeted_environment_steps"]) // 16
        if trainer.update_count > target_updates:
            raise ValueError("G V4 optimizer updates exceed the registered state cadence")
        if target_updates > trainer.update_count:
            ensure_cpu_room(root_path, cap_hours, reserve_seconds=1.0)
            metrics = trainer.train_updates(
                replay, updates=target_updates - trainer.update_count,
                batch_size=int(config["training"]["batch_size"]),
            )
            metrics.update({
                "states_seen": int(progress["budgeted_environment_steps"]),
                "game_id": game_id, "stage_index": nodes.index(next_stage),
            })
            progress["training_history"].append(metrics)
            record_cpu_event(
                root_path,
                f"optimizer:{job['job_id']}:states-{int(progress['budgeted_environment_steps'])}:{uuid.uuid4().hex}",
                float(metrics["cpu_seconds"]), wall_seconds=float(metrics["wall_seconds"]), category="optimizer",
            )
        if trainer.update_count != target_updates:
            raise ValueError("G V4 optimizer count differs from the registered 16-state cadence")
        progress.update({
            "optimizer_updates": trainer.update_count,
            "replay_positions": replay.position_count,
            "replay_games": replay.game_count,
            "simulations_per_move": 64,
            "bn_policy": trainer.bn_mode,
        })
        state_sha = save_v3_state(paths["state"], trainer, replay, arm="G0", seed=int(job["seed"]),
                                  progress=progress, initial_bn_buffers=initial_bn)
        progress["training_state_sha256"] = state_sha
        _atomic_json(paths["progress"], progress)
        current = int(progress["budgeted_environment_steps"])

    node_path = paths["checkpoint_dir"] / f"node_{target}.pt"
    if not node_path.is_file():
        sha = save_g_checkpoint(node_path, trainer.model, {
            "experiment": "g_v4", "job_id": job["job_id"], "phase": job["phase"],
            "seed": int(job["seed"]), "node": target, "new_budgeted_states": target,
            "optimizer_updates": trainer.update_count, "source_checkpoint": job.get("source_model"),
            "optimizer_mode": job["optimizer_mode"], "config_sha256": config["config_sha256"],
            "training_data_source": "own PUCT-64 self-play only",
        })
        stage_row = {
            "checkpoint_id": f"{job['job_id']}@{target}", "job_id": job["job_id"],
            "phase": job["phase"], "seed": int(job["seed"]), "new_states": target,
            "total_lineage_states": (20_000 + target if job["phase"] == "B" else target),
            "game_count": len(games), "optimizer_updates": trainer.update_count,
            "checkpoint": node_path.relative_to(root_path).as_posix(), "checkpoint_sha256": sha,
            "selfplay_cpu_seconds": progress["selfplay_cpu_seconds"],
            "optimizer_cpu_seconds": progress["optimizer_cpu_seconds"],
            "actual_simulations": progress["actual_simulations"],
            "replay_positions": replay.position_count, "replay_games": replay.game_count,
            "mean_policy_entropy": float(np.mean([
                row["policy_entropy"] for game in games for row in game["experience_rows"]
                if row.get("policy_entropy") is not None
            ])) if games else 0.0,
            "config_sha256": config["config_sha256"],
        }
        _append_jsonl(root_path / "research/g_v4/results/training/stages.jsonl", stage_row, "checkpoint_id")
    else:
        sha = _sha(node_path)
    progress["last_checkpoint"] = node_path.relative_to(root_path).as_posix()
    progress["last_checkpoint_sha256"] = sha
    state_sha = save_v3_state(paths["state"], trainer, replay, arm="G0", seed=int(job["seed"]),
                              progress=progress, initial_bn_buffers=initial_bn)
    progress["training_state_sha256"] = state_sha
    _atomic_json(paths["progress"], progress)
    return {**progress, "completed": True, "target_node": target, "checkpoint": progress["last_checkpoint"]}


def _evaluation_log(root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    return load_evaluation_log(root / "research/g_v4/results/evaluation/development.jsonl")


def _opponents(root: Path, job: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    base = root / "research/g_v4"
    rows: list[dict[str, Any]] = []
    paths = [
        ("own_start", selection["start_checkpoint"], selection["start_checkpoint_id"]),
        ("stage_champion", selection["champion_checkpoint"], selection["champion_checkpoint_id"]),
        ("G0-V3-20K", V3_CHAMPION_REL, V3_CHAMPION_ID),
    ]
    if job["phase"] == "B":
        random_reference = "research/g_v4/checkpoints/A-G0-s540401/node_0.pt"
        if (root / random_reference).is_file():
            paths.append(("G0-random-reference", random_reference, "G0-random-seed-540401@0"))
    elif job["phase"] == "C":
        random_reference = f"research/g_v4/checkpoints/A-G0-s{int(job['source_seed'])}/node_0.pt"
        if (root / random_reference).is_file():
            paths.append(("G0-random-reference", random_reference, f"A-G0-s{int(job['source_seed'])}@0"))
    seen: dict[str, dict[str, Any]] = {}
    for role, relative, checkpoint_id in paths:
        path = root / relative
        digest = _sha(path)
        if digest in seen:
            seen[digest]["aliases"].append(role)
            if role == "stage_champion":
                seen[digest]["role"] = role
                seen[digest]["checkpoint_id"] = checkpoint_id
        else:
            seen[digest] = {"role": role, "family": "G0", "path": relative,
                            "checkpoint_id": checkpoint_id, "sha256": digest,
                            "aliases": [role]}
    rows.extend(seen.values())
    for family in ("A", "B", "F2"):
        rows.append({"role": family, "family": family, "path": None,
                     "checkpoint_id": f"{family}@frozen", "sha256": None, "aliases": [family]})
    return rows


def evaluate_v4_checkpoint(
    root: str | Path,
    job: dict[str, Any],
    *,
    node: int,
    phase: str = "development",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    torch.set_num_threads(1)
    config = load_v4_config(root_path)
    if phase != "development":
        raise ValueError("only development grids are used for checkpoint selection")
    paths = _job_paths(root_path, job)
    candidate_path = paths["checkpoint_dir"] / f"node_{int(node)}.pt"
    candidate_id = f"{job['job_id']}@{int(node)}"
    candidate = load_g_checkpoint(candidate_path, device="cpu")
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    groups = schedule["development_openings"]
    selection = _job_selection(root_path, job, config)
    matches_path = root_path / "research/g_v4/results/evaluation/development.jsonl"
    records, seen = load_evaluation_log(matches_path)
    by_id = {str(row["game_id"]): row for row in records}
    opponents = _opponents(root_path, job, selection)
    loaded: dict[str, torch.nn.Module] = {}
    for item in opponents:
        if item["family"] == "G0":
            loaded[item["sha256"]] = load_g_checkpoint(root_path / item["path"], device="cpu")
    new_count = 0
    for item in opponents:
        if item["family"] == "G0" and item["sha256"] == _sha(candidate_path):
            continue
        opponent_model = loaded[item["sha256"]] if item["family"] == "G0" else None
        for mode in ("network", "puct"):
            for group in groups:
                for color in ("B", "W"):
                    role_key = "-".join(item["aliases"])
                    game_id = (
                        f"g-v4-dev-{config['config_sha256'][:8]}-{candidate_id}-"
                        f"{role_key}-{item['sha256'][:8] if item['sha256'] else item['family']}-"
                        f"{mode}-{group['opening_group_id']}-{color}"
                    )
                    if game_id in by_id:
                        continue
                    ensure_cpu_room(root_path, float(config["hard_cpu_hour_cap"]), reserve_seconds=30.0)
                    started_cpu, started_wall = time.process_time(), time.perf_counter()
                    match = play_v3_match(
                        candidate, opponent_model, game_id=game_id, opening_group=group,
                        agent_color=color, mode=mode, opponent=item["family"],
                        seed=_group_seed({
                            "seed": int(job["seed"]), "opening_group_id": group["opening_group_id"],
                            "arm": job["job_id"], "candidate_checkpoint_id": candidate_id,
                            "opponent": item["checkpoint_id"], "mode": mode, "game_id": game_id,
                        }),
                        simulations=64, opponent_checkpoint_id=item["checkpoint_id"],
                        candidate_checkpoint_id=candidate_id, phase="development",
                    )
                    wall = time.perf_counter() - started_wall
                    cpu = time.process_time() - started_cpu
                    match.update({
                        "config_sha256": config["config_sha256"],
                        "candidate_checkpoint_sha256": _sha(candidate_path),
                        "opponent_checkpoint_sha256": item["sha256"],
                        "opening_seed": int(group["opening_seed"]),
                        "v4_opponent_role": item["role"],
                        "v4_opponent_aliases": list(item["aliases"]),
                        "runner_cpu_seconds": cpu, "runner_wall_seconds": wall,
                    })
                    if not append_evaluation_match(matches_path, match, seen):
                        raise ValueError(f"duplicate G V4 evaluation ID {game_id}")
                    record_cpu_event(root_path, f"evaluation:{game_id}", cpu, wall_seconds=wall, category="evaluation")
                    by_id[game_id] = match
                    new_count += 1
    # Exact-rule tactic probes are evaluation-only. Their labels come from
    # legal five-in-a-row enumeration and never affect model decisions/training.
    tactic_path = root_path / "research/g_v4/results/evaluation/tactics.jsonl"
    tactic_ids = {str(row["record_id"]) for row in _read_jsonl_if_present(tactic_path)}
    for tactic_mode in ("network", "puct"):
        record_id = f"{candidate_id}:{tactic_mode}"
        if record_id in tactic_ids:
            continue
        ensure_cpu_room(root_path, float(config["hard_cpu_hour_cap"]), reserve_seconds=2.0)
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        tactical = evaluate_v3_tactics(candidate, mode=tactic_mode,
                                       puzzles=build_v3_tactical_puzzles(), simulations=64)
        cpu = time.process_time() - started_cpu
        wall = time.perf_counter() - started_wall
        tactical.update({"record_id": record_id, "candidate_checkpoint_id": candidate_id,
                         "checkpoint_sha256": _sha(candidate_path),
                         "config_sha256": config["config_sha256"], "cpu_seconds": cpu,
                         "wall_seconds": wall})
        _append_jsonl(tactic_path, tactical, "record_id")
        record_cpu_event(root_path, f"tactics:{record_id}", cpu,
                         wall_seconds=wall, category="evaluation")
    return {"candidate_id": candidate_id, "job_id": job["job_id"], "node": int(node),
            "new_games": new_count, "total_games": len(by_id)}


def _candidate_matches(
    rows: list[dict[str, Any]], candidate_id: str, opponent_id: str, *, mode: str = "puct",
    opponent_role: str | None = None,
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("candidate_checkpoint_id") == candidate_id
            and row.get("opponent_checkpoint_id") == opponent_id and row.get("mode") == mode
            and (opponent_role is None or row.get("v4_opponent_role") == opponent_role)]


def select_v4_node(root: str | Path, job: dict[str, Any], *, node: int) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_config(root_path)
    paths = _job_paths(root_path, job)
    selection = _job_selection(root_path, job, config)
    candidate_id = f"{job['job_id']}@{int(node)}"
    if candidate_id in {row.get("candidate_id") for row in selection.get("decisions", [])}:
        return next(row for row in selection["decisions"] if row.get("candidate_id") == candidate_id)
    records, _ = _evaluation_log(root_path)
    incumbent_id = str(selection["champion_checkpoint_id"])
    matches = _candidate_matches(
        records, candidate_id, incumbent_id, opponent_role="stage_champion",
    )
    decision = promotion_decision(
        matches, candidate_id=candidate_id, incumbent_id=incumbent_id,
        min_score_rate=float(config["evaluation"]["promotion"]["score_rate_at_least"]),
        lower_ci_minimum=float(config["evaluation"]["promotion"]["paired_ci95_lower_above"]),
        min_color_score_rate=float(config["evaluation"]["promotion"]["each_color_at_least"]),
        required_groups=int(config["evaluation"]["development_opening_groups"]),
        seed=int(job["seed"]) + int(node),
    )
    candidate_path = paths["checkpoint_dir"] / f"node_{int(node)}.pt"
    decision.update({
        "job_id": job["job_id"], "phase": job["phase"], "seed": int(job["seed"]),
        "node": int(node), "candidate_checkpoint_sha256": _sha(candidate_path),
        "incumbent_checkpoint_sha256": selection["champion_checkpoint_sha256"],
        "config_sha256": config["config_sha256"],
    })
    if decision["promote"]:
        selection.update({
            "champion_checkpoint_id": candidate_id,
            "champion_checkpoint": str(candidate_path.relative_to(root_path).as_posix()),
            "champion_checkpoint_sha256": _sha(candidate_path),
            "last_decision_node": int(node), "consecutive_below_50pct": 0,
        })
    else:
        selection["last_decision_node"] = int(node)
        selection["consecutive_below_50pct"] = (
            int(selection.get("consecutive_below_50pct", 0)) + 1
            if float(decision["score_rate"]) < 0.5 else 0
        )
    selection.setdefault("completed_nodes", []).append(int(node))
    selection.setdefault("decisions", []).append(decision)
    _atomic_json(paths["selection"], selection)
    _append_jsonl(root_path / "research/g_v4/results/selection/promotions.jsonl", decision, "candidate_id")
    return decision


def _gate_against_v3(root: Path, candidate_id: str, seed: int) -> dict[str, Any] | None:
    config = load_v4_config(root)
    records, _ = _evaluation_log(root)
    matches = _candidate_matches(
        records, candidate_id, V3_CHAMPION_ID, opponent_role="G0-V3-20K",
    )
    if not matches:
        return None
    decision = promotion_decision(
        matches, candidate_id=candidate_id, incumbent_id=V3_CHAMPION_ID,
        min_score_rate=float(config["evaluation"]["promotion"]["score_rate_at_least"]),
        lower_ci_minimum=float(config["evaluation"]["promotion"]["paired_ci95_lower_above"]),
        min_color_score_rate=float(config["evaluation"]["promotion"]["each_color_at_least"]),
        required_groups=int(config["evaluation"]["development_opening_groups"]),
        seed=int(seed) + 7321,
    )
    decision["promotion_basis"] = "fixed PUCT-64 development matches against frozen V3 G0@20K"
    return decision


def _experiment_state(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / "research/g_v4/results/selection/experiment_state.json"
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config["config_sha256"]:
            raise ValueError("G V4 experiment progress config hash mismatch")
        return state
    state = {
        "format_version": 1, "config_sha256": config["config_sha256"],
        "phase": "A", "phase_a_seed_index": 0, "phase_b_initialized": False,
        "phase_c_initialized": False, "phase_a_global_gate": None,
        "phase_c_decision": None, "final_candidate": None,
        "blind_started": False, "completed": False,
    }
    _atomic_json(path, state)
    return state


def _reconcile_logged_cpu_work(root: Path) -> int:
    """Backfill CPU entries for fsynced result rows whose event was missed."""
    added = 0
    base = root / "research/g_v4/results"
    event_path = base / "runtime/cpu_events.jsonl"
    known = {str(row["event_id"]) for row in _load_rows(event_path)}

    def record(event_id: str, cpu: float, wall: float, category: str) -> None:
        nonlocal added
        if event_id not in known:
            added += int(record_cpu_event(root, event_id, cpu, wall_seconds=wall, category=category))
            known.add(event_id)

    for path in sorted((base / "selfplay").glob("*.jsonl")):
        games, _ = load_game_log(path)
        for game in games:
            if "runner_cpu_seconds" in game:
                record(f"selfplay:{game['game_id']}", float(game["runner_cpu_seconds"]),
                       float(game.get("runner_wall_seconds", 0.0)), "selfplay")
    for path in (base / "evaluation/development.jsonl", base / "evaluation/blind.jsonl"):
        matches, _ = load_evaluation_log(path)
        for match in matches:
            if "runner_cpu_seconds" in match:
                record(f"evaluation:{match['game_id']}", float(match["runner_cpu_seconds"]),
                       float(match.get("runner_wall_seconds", 0.0)), "evaluation")
    for row in _read_jsonl_if_present(base / "evaluation/tactics.jsonl"):
        if "record_id" in row and "cpu_seconds" in row:
            record(f"tactics:{row['record_id']}", float(row["cpu_seconds"]),
                   float(row.get("wall_seconds", 0.0)), "evaluation")
    return added


def initialize_v4_jobs(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_config(root_path)
    jobs = [phase_a_job(seed) for seed in config["stages"]["A"]["new_random_seeds"]]
    jobs.append(phase_b_job())
    outputs = [prepare_v4_job(root_path, job) for job in jobs]
    state = _experiment_state(root_path, config)
    _atomic_json(root_path / "research/g_v4/results/selection/registered_jobs.json", {
        "config_sha256": config["config_sha256"], "jobs": jobs,
        "note": "C job is created only if an A-stage seed champion passes the registered V3 champion gate.",
    })
    return {"prepared_jobs": outputs, "state": state}


def _node0_evaluate(root: Path, job: dict[str, Any]) -> None:
    selection = _job_selection(root, job, load_v4_config(root))
    if selection.get("node0_evaluated"):
        return
    evaluate_v4_checkpoint(root, job, node=0)
    selection["node0_evaluated"] = True
    _atomic_json(_job_paths(root, job)["selection"], selection)


def _phase_a_result(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    per_seed = []
    for seed in config["stages"]["A"]["new_random_seeds"]:
        job = phase_a_job(int(seed))
        selection = _job_selection(root, job, config)
        champion_id = str(selection["champion_checkpoint_id"])
        gate = _gate_against_v3(root, champion_id, int(seed)) if champion_id != job["start_checkpoint_id"] else None
        row = {"seed": int(seed), "champion_checkpoint_id": champion_id,
               "checkpoint": selection["champion_checkpoint"], "promotion_vs_v3_champion": gate}
        per_seed.append(row)
        if gate and gate.get("promote"):
            candidates.append({**row, "score_rate": float(gate["score_rate"]),
                               "ci": gate["paired_opening_bootstrap_ci95"]})
    candidates.sort(key=lambda row: (-row["score_rate"], row["seed"], row["champion_checkpoint_id"]))
    result = {
        "completed": True, "per_seed": per_seed,
        "qualified_seed_count": len(candidates),
        "second_tier_candidate": candidates[0] if candidates else None,
        "rule": "A-stage per-seed champion must independently pass the registered 12-opening V3 G0@20K PUCT-64 gate; otherwise C is skipped.",
    }
    _atomic_json(root / "research/g_v4/results/selection/phase_a_candidates.json", result)
    return result


def _candidate_score_against_v3(root: Path, candidate_id: str) -> tuple[float, list[float]]:
    records, _ = _evaluation_log(root)
    matches = _candidate_matches(
        records, candidate_id, V3_CHAMPION_ID, opponent_role="G0-V3-20K",
    )
    if not matches:
        return -1.0, [-1.0, -1.0]
    summary = summarize_v3_matches(matches)
    return float(summary.get("score_rate", -1.0)), list(summary.get("paired_opening_bootstrap_ci95", [-1.0, -1.0]))


def choose_final_development_candidate(root: Path, config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    jobs = [phase_a_job(int(seed)) for seed in config["stages"]["A"]["new_random_seeds"]]
    jobs.append(phase_b_job())
    if state.get("phase_c_job"):
        jobs.append(state["phase_c_job"])
    champion_ids: set[str] = set()
    phase_a = state.get("phase_a_global_gate")
    if phase_a is None:
        phase_a_path = root / "research/g_v4/results/selection/phase_a_candidates.json"
        if phase_a_path.is_file():
            phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    for row in (phase_a or {}).get("per_seed", []):
        gate = row.get("promotion_vs_v3_champion")
        if gate and gate.get("promote"):
            champion_ids.add(str(row["champion_checkpoint_id"]))
    for job in jobs[3:]:
        selection_path = _job_paths(root, job)["selection"]
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            champion_id = str(selection["champion_checkpoint_id"])
            if champion_id != str(job["start_checkpoint_id"]):
                champion_ids.add(champion_id)
    # Only promoted stage champions may become the final V4 candidate. This
    # prevents development-set ranking from silently bypassing the registered
    # champion/challenger acceptance gate.
    for checkpoint_id in sorted(champion_ids):
        if checkpoint_id == V3_CHAMPION_ID:
            continue
        job_id, node_text = checkpoint_id.rsplit("@", maxsplit=1)
        path = root / "research/g_v4/checkpoints" / job_id / f"node_{int(node_text)}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"selected G V4 champion checkpoint is missing: {path}")
        score, ci = _candidate_score_against_v3(root, checkpoint_id)
        if score >= 0:
            candidates.append({"checkpoint_id": checkpoint_id,
                               "checkpoint": path.relative_to(root).as_posix(),
                               "score_rate_vs_v3_champion": score, "paired_ci95": ci,
                               "checkpoint_sha256": _sha(path)})
    if candidates:
        candidates.sort(key=lambda row: (-row["score_rate_vs_v3_champion"], -row["paired_ci95"][0],
                                         row["checkpoint_id"]))
        selected = candidates[0]
        selected["selection_rule"] = "best promoted stage champion by development PUCT-64 score against frozen V3 G0@20K; blind records unused"
    else:
        selected = {"checkpoint_id": V3_CHAMPION_ID, "checkpoint": V3_CHAMPION_REL,
                    "checkpoint_sha256": _sha(root / V3_CHAMPION_REL),
                    "score_rate_vs_v3_champion": 0.5, "paired_ci95": [0.5, 0.5],
                    "selection_rule": "no V4 checkpoint passed the pre-registered promotion gate; retain frozen V3 champion"}
    state["final_candidate"] = selected
    state["development_candidates"] = candidates
    _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
    _atomic_json(root / "research/g_v4/results/selection/final_candidate.json", selected)
    return selected


def _blind_opponents(root: Path, candidate: dict[str, Any], job: dict[str, Any] | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if job:
        sel_path = _job_paths(root, job)["selection"]
        if sel_path.is_file():
            selection = json.loads(sel_path.read_text(encoding="utf-8"))
            entries.extend([
                {"role": "own_start", "family": "G0", "path": selection["start_checkpoint"],
                 "checkpoint_id": selection["start_checkpoint_id"]},
                {"role": "stage_champion", "family": "G0", "path": selection["champion_checkpoint"],
                 "checkpoint_id": selection["champion_checkpoint_id"]},
            ])
        if str(job["phase"]) == "C":
            seed = int(job["source_seed"])
            entries.append({"role": "random_initial", "family": "G0",
                            "path": f"research/g_v4/checkpoints/A-G0-s{seed}/node_0.pt",
                            "checkpoint_id": f"A-G0-s{seed}@0"})
        elif str(job["phase"]) == "B":
            entries.append({"role": "G0-random-reference", "family": "G0",
                            "path": "research/g_v4/checkpoints/A-G0-s540401/node_0.pt",
                            "checkpoint_id": "G0-random-seed-540401@0"})
    entries.append({"role": "G0-V3-20K", "family": "G0", "path": V3_CHAMPION_REL,
                    "checkpoint_id": V3_CHAMPION_ID})
    unique: dict[str, dict[str, Any]] = {}
    for item in entries:
        path = root / item["path"]
        digest = _sha(path)
        if digest in unique:
            unique[digest]["aliases"].append(item["role"])
        else:
            unique[digest] = {**item, "sha256": digest, "aliases": [item["role"]]}
    result = list(unique.values())
    for family in ("A", "B", "F2"):
        result.append({"role": family, "family": family, "path": None,
                       "checkpoint_id": f"{family}@frozen", "sha256": None, "aliases": [family]})
    return result


def _run_blind_model(
    root: Path,
    *,
    candidate_id: str,
    candidate_path: str,
    candidate_sha: str,
    candidate_job: dict[str, Any] | None,
    group_count: int,
) -> int:
    config = load_v4_config(root)
    candidate = load_g_checkpoint(root / candidate_path, device="cpu")
    schedule = json.loads((root / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    groups = schedule["blind_openings"][:int(group_count)]
    if len(groups) != int(group_count):
        raise ValueError("V4 blind pool is smaller than registered")
    path = root / "research/g_v4/results/evaluation/blind.jsonl"
    existing, seen = load_evaluation_log(path)
    by_id = {str(row["game_id"]): row for row in existing}
    opponents = _blind_opponents(root, {"checkpoint_id": candidate_id}, candidate_job)
    loaded = {item["sha256"]: load_g_checkpoint(root / item["path"], device="cpu")
              for item in opponents if item["family"] == "G0"}
    new_count = 0
    for item in opponents:
        if item["family"] == "G0" and item["sha256"] == candidate_sha:
            continue
        opponent_model = loaded[item["sha256"]] if item["family"] == "G0" else None
        for mode in ("network", "puct"):
            for group in groups:
                for color in ("B", "W"):
                    role_key = "-".join(item["aliases"])
                    game_id = (
                        f"g-v4-blind-{config['config_sha256'][:8]}-{candidate_id}-"
                        f"{role_key}-{item['sha256'][:8] if item['sha256'] else item['family']}-"
                        f"{mode}-{group['opening_group_id']}-{color}"
                    )
                    if game_id in by_id:
                        continue
                    ensure_cpu_room(root, float(config["hard_cpu_hour_cap"]), reserve_seconds=30.0)
                    started_cpu, started_wall = time.process_time(), time.perf_counter()
                    match = play_v3_match(
                        candidate, opponent_model, game_id=game_id, opening_group=group,
                        agent_color=color, mode=mode, opponent=item["family"],
                        seed=_group_seed({"seed": 777701, "opening_group_id": group["opening_group_id"],
                                          "arm": candidate_id, "candidate_checkpoint_id": candidate_id,
                                          "opponent": item["checkpoint_id"], "mode": mode, "game_id": game_id}),
                        simulations=64, opponent_checkpoint_id=item["checkpoint_id"],
                        candidate_checkpoint_id=candidate_id, phase="blind",
                    )
                    wall, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
                    match.update({
                        "config_sha256": config["config_sha256"],
                        "candidate_checkpoint_sha256": candidate_sha,
                        "opponent_checkpoint_sha256": item["sha256"],
                        "opening_seed": int(group["opening_seed"]),
                        "v4_opponent_role": item["role"], "v4_opponent_aliases": item["aliases"],
                        "runner_cpu_seconds": cpu, "runner_wall_seconds": wall,
                    })
                    if not append_evaluation_match(path, match, seen):
                        raise ValueError(f"duplicate G V4 blind game ID {game_id}")
                    record_cpu_event(root, f"evaluation:{game_id}", cpu, wall_seconds=wall, category="evaluation")
                    by_id[game_id] = match
                    new_count += 1
    return new_count


def run_v4_blind(root: str | Path, *, candidate_job: dict[str, Any] | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    torch.set_num_threads(1)
    config = load_v4_config(root_path)
    state = _experiment_state(root_path, config)
    candidate = state.get("final_candidate") or choose_final_development_candidate(root_path, config, state)
    new_games = _run_blind_model(
        root_path, candidate_id=str(candidate["checkpoint_id"]), candidate_path=str(candidate["checkpoint"]),
        candidate_sha=str(candidate["checkpoint_sha256"]), candidate_job=candidate_job,
        group_count=int(config["evaluation"]["blind_opening_groups"]),
    )
    baseline_path = V3_CHAMPION_REL
    new_games += _run_blind_model(
        root_path, candidate_id=V3_CHAMPION_ID, candidate_path=baseline_path,
        candidate_sha=_sha(root_path / baseline_path), candidate_job=None,
        group_count=int(config["evaluation"]["blind_opening_groups"]),
    )
    records, _ = load_evaluation_log(root_path / "research/g_v4/results/evaluation/blind.jsonl")
    state["blind_completed"] = True
    state["completed"] = True
    _atomic_json(root_path / "research/g_v4/results/selection/experiment_state.json", state)
    return {"new_games": new_games, "total_blind_games": len(records),
            "final_candidate": candidate, "completed": True}


def _finish_phase_a(root: Path, config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    result = _phase_a_result(root, config)
    state["phase_a_global_gate"] = result
    state["phase"] = "B"
    _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
    return result


def _advance_one(root: Path, config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if state.get("phase") == "A":
        seeds = list(map(int, config["stages"]["A"]["new_random_seeds"]))
        index = int(state.get("phase_a_seed_index", 0))
        if index >= len(seeds):
            result = _finish_phase_a(root, config, state)
            return {"kind": "phase_a_selection", "result": result, "completed": False}
        job = phase_a_job(seeds[index])
        prepare_v4_job(root, job)
        selection = _job_selection(root, job, config)
        if not selection.get("node0_evaluated"):
            _node0_evaluate(root, job)
            return {"kind": "node0_evaluation", "job_id": job["job_id"], "completed": False}
        completed_nodes = set(map(int, selection.get("completed_nodes", [])))
        next_node = next((node for node in job["nodes"] if node not in completed_nodes), None)
        if next_node is None:
            state["phase_a_seed_index"] = index + 1
            _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
            return {"kind": "phase_a_seed_finished", "job_id": job["job_id"], "completed": False}
        result = run_v4_job_to_node(root, job, int(next_node))
        if result.get("completed") is not True:
            return {"kind": "training_incomplete", "job_id": job["job_id"], "result": result, "completed": False}
        evaluate_v4_checkpoint(root, job, node=int(next_node))
        decision = select_v4_node(root, job, node=int(next_node))
        return {"kind": "training_milestone", "job_id": job["job_id"],
                "node": int(next_node), "promotion": decision, "completed": False}

    if state.get("phase") == "B":
        job = phase_b_job()
        prepare_v4_job(root, job)
        selection = _job_selection(root, job, config)
        if not selection.get("node0_evaluated"):
            _node0_evaluate(root, job)
            return {"kind": "node0_evaluation", "job_id": job["job_id"], "completed": False}
        completed_nodes = set(map(int, selection.get("completed_nodes", [])))
        next_node = next((node for node in job["nodes"] if node not in completed_nodes), None)
        if next_node is None:
            phase_a = state.get("phase_a_global_gate") or _phase_a_result(root, config)
            candidate = phase_a.get("second_tier_candidate")
            if candidate:
                seed = int(candidate["seed"])
                state["phase_c_job"] = phase_c_job(seed, str(candidate["checkpoint"]),
                                                    str(candidate["champion_checkpoint_id"]))
                state["phase_c_decision"] = {"started": True, "candidate": candidate}
                state["phase"] = "C"
            else:
                state["phase_c_decision"] = {"started": False,
                                              "reason": "No A-stage seed champion passed the pre-registered G0-V3@20K gate."}
                state["phase"] = "FINALIZE"
            _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
            return {"kind": "phase_b_finished", "phase_c": state["phase_c_decision"], "completed": False}
        result = run_v4_job_to_node(root, job, int(next_node))
        if result.get("completed") is not True:
            return {"kind": "training_incomplete", "job_id": job["job_id"], "result": result, "completed": False}
        evaluate_v4_checkpoint(root, job, node=int(next_node))
        decision = select_v4_node(root, job, node=int(next_node))
        score, ci = _candidate_score_against_v3(root, f"{job['job_id']}@{int(next_node)}")
        return {"kind": "training_milestone", "job_id": job["job_id"],
                "node": int(next_node), "promotion": decision,
                "score_rate_vs_v3_champion": score, "ci_vs_v3_champion": ci,
                "completed": False}

    if state.get("phase") == "C":
        job = state["phase_c_job"]
        prepare_v4_job(root, job)
        selection = _job_selection(root, job, config)
        if not selection.get("node0_evaluated"):
            _node0_evaluate(root, job)
            return {"kind": "node0_evaluation", "job_id": job["job_id"], "completed": False}
        completed_nodes = set(map(int, selection.get("completed_nodes", [])))
        next_node = next((node for node in job["nodes"] if node not in completed_nodes), None)
        if next_node is None:
            state["phase"] = "FINALIZE"
            _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
            return {"kind": "phase_c_finished", "completed": False}
        result = run_v4_job_to_node(root, job, int(next_node))
        if result.get("completed") is not True:
            return {"kind": "training_incomplete", "job_id": job["job_id"], "result": result, "completed": False}
        evaluate_v4_checkpoint(root, job, node=int(next_node))
        decision = select_v4_node(root, job, node=int(next_node))
        if float(decision["score_rate"]) < 0.5:
            selection = _job_selection(root, job, config)
        failures = int(selection.get("consecutive_below_50pct", 0))
        early_stopped = failures >= 3
        if early_stopped:
            selection["early_stopped"] = True
            selection["early_stop_reason"] = "three consecutive challengers scored below 50% against the current stage champion"
            _atomic_json(_job_paths(root, job)["selection"], selection)
            state["phase_c_decision"]["early_stopped"] = True
            state["phase"] = "FINALIZE"
            _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
        return {"kind": "training_milestone", "job_id": job["job_id"], "node": int(next_node),
                "promotion": decision, "early_stopped": early_stopped, "completed": False}

    if state.get("phase") == "FINALIZE":
        candidate = choose_final_development_candidate(root, config, state)
        state = _experiment_state(root, config)
        state["phase"] = "BLIND"
        state["final_candidate"] = candidate
        _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", state)
        return {"kind": "final_model_selected_by_development", "candidate": candidate, "completed": False}

    if state.get("phase") == "BLIND":
        job = None
        for phase_job in [phase_a_job(seed) for seed in config["stages"]["A"]["new_random_seeds"]] + [phase_b_job()]:
            if phase_job["job_id"] in str(state.get("final_candidate", {}).get("checkpoint", "")):
                job = phase_job
                break
        if state.get("phase_c_job") and state["phase_c_job"]["job_id"] in str(state.get("final_candidate", {}).get("checkpoint", "")):
            job = state["phase_c_job"]
        blind = run_v4_blind(root, candidate_job=job)
        _atomic_json(root / "research/g_v4/results/selection/experiment_state.json", {
            **_experiment_state(root, config), "phase": "DONE", "completed": True,
            "final_candidate": state.get("final_candidate"), "blind_completed": True,
        })
        return {"kind": "blind_completed", **blind, "completed": True}
    if state.get("phase") == "DONE" or state.get("completed"):
        return {"kind": "already_completed", "completed": True}
    raise ValueError(f"unknown G V4 experiment phase: {state.get('phase')}")


def advance_v4_experiment(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    _identity(root_path)
    config = load_v4_config(root_path)
    budget_module = __import__("src.research.alphazero_g.v4_budget", fromlist=["cpu_seconds_committed", "active_wall_seconds_committed"])
    _reconcile_logged_cpu_work(root_path)
    ledger_before = budget_module.cpu_seconds_committed(root_path)
    ledger_wall_before = budget_module.active_wall_seconds_committed(root_path)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    initialize_v4_jobs(root_path)
    state = _experiment_state(root_path, config)
    result = _advance_one(root_path, config, state)
    elapsed_cpu, elapsed_wall = time.process_time() - started_cpu, time.perf_counter() - started_wall
    ledger_after = budget_module.cpu_seconds_committed(root_path)
    ledger_wall_after = budget_module.active_wall_seconds_committed(root_path)
    # Subtract the fine-grained self-play/update/evaluation records to account
    # for reconstruction, loading, serialization, audits, and scheduler work.
    record_cpu_event(root_path, f"runner-overhead:{uuid.uuid4().hex}",
                     max(0.0, elapsed_cpu - (ledger_after - ledger_before)),
                     wall_seconds=max(0.0, elapsed_wall - (ledger_wall_after - ledger_wall_before)),
                     category="orchestration")
    result["active_cpu_seconds"] = __import__("src.research.alphazero_g.v4_budget", fromlist=["cpu_seconds_committed"]).cpu_seconds_committed(root_path)
    result["active_wall_seconds"] = __import__("src.research.alphazero_g.v4_budget", fromlist=["active_wall_seconds_committed"]).active_wall_seconds_committed(root_path)
    result["next_snapshot_threshold_seconds"] = __import__("src.research.alphazero_g.v4_budget", fromlist=["next_snapshot_threshold"]).next_snapshot_threshold(result["active_wall_seconds"])
    return result

