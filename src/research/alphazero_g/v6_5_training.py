"""V6.5: continue the frozen V6.3 G5000 AlphaZero state to G8000.

The policy/value network, self-play, PUCT, target generation and optimizer step
are delegated unchanged to the V5 reference training implementation. This file
only adds a resumable V6.5 boundary, budget, milestone and provenance wrapper.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import torch

from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board
from .v5_training import load_training_checkpoint, train_reference_arm
from .v6_2_training import _replay_digest

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_5")
ARM = "reference400_empty"
SEED = 590502
SOURCE_COMMIT = "66b88d2270c799870aa81651fe3c07be18479cf0"
SOURCE_ARCHIVE_CODE_COMMIT = "b31f24acffee423b33beca19ccb9389964bd7a36"
SOURCE_FOLDER_ID = "1u3556ink-VFf9a7AbS701flr0V0spSzQ"
SOURCE_ARCHIVE_SHA256 = "c7a615c112324d955e32122f66f6930254510d111ff23606f7805e58f61af592"
SOURCE_STATE_SHA256 = "0fb9886068a014d000227ae7b275ff9043322a678d7f259e9730d44ac95398d2"
SOURCE_MODEL_SHA256 = "cc383f0667cfe02b8e8ccbaecf51eccc2c4f088d56c836a4e91e016cfae3ee97"
BASE_GAMES = 5000
BASE_STATES = 87999
BASE_UPDATES = 4999
TARGET_GAMES = 8000
MILESTONES = (6000, 7000, 8000)
SELFPLAY_SIMULATIONS = 400
MAX_NEW_CPU_SECONDS = 40 * 3600
SNAPSHOT_INTERVAL_GAMES = 250
SNAPSHOT_INTERVAL_CPU_SECONDS = 3 * 3600


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as src, temp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temp, destination)


def _numpy_rng_equal(left, right) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def validate_training_state(path: str | Path, *, root: str | Path = ROOT,
                            expected_games: int | None = None) -> dict[str, Any]:
    """Restore into a fresh network and compare model, Adam, replay and RNG state."""
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    config_path = root / ARTIFACT / "config/reference_training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config.get("config_sha256"):
        raise ValueError("checkpoint does not match the frozen V5/V6.3 reference configuration")
    if payload.get("arm_id") != ARM or int(payload.get("base_seed", -1)) != SEED:
        raise ValueError("checkpoint arm or training seed differs from the frozen G5000 run")
    if int(payload.get("board_size", -1)) != 9:
        raise ValueError("checkpoint board size is not 9x9")
    if expected_games is not None and int(payload.get("game_count", -1)) != int(expected_games):
        raise ValueError("checkpoint cumulative game count differs from the requested boundary")
    if len(payload.get("replay_buffer", [])) != int(config["training"]["buffer_size"]):
        raise ValueError("full replay buffer is missing or has the wrong capacity")
    if not payload.get("optimizer_state", {}).get("state"):
        raise ValueError("Adam optimizer state is missing")

    model = ReferencePolicyValueNet(9, seed=SEED)
    old_python, old_numpy, old_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        loaded = load_training_checkpoint(
            path, model=model, arm=ARM, config_hash=config["config_sha256"],
            buffer_size=int(config["training"]["buffer_size"]),
        )
        checks = {
            "model_restored": all(torch.equal(model.get_policy_param()[key], payload["model_state"][key])
                                  for key in payload["model_state"]),
            "adam_restored": _state_equal(model.optimizer.state_dict(), payload["optimizer_state"]),
            "replay_restored": len(loaded[4]) == len(payload["replay_buffer"])
                and _replay_digest(loaded[4]) == _replay_digest(payload["replay_buffer"]),
            "python_rng_restored": random.getstate() == payload["python_rng_state"],
            "numpy_rng_restored": _numpy_rng_equal(np.random.get_state(), payload["numpy_rng_state"]),
            "torch_rng_restored": torch.equal(torch.get_rng_state(), payload["torch_rng_state"]),
        }
    finally:
        random.setstate(old_python)
        np.random.set_state(old_numpy)
        torch.set_rng_state(old_torch)
    if not all(checks.values()):
        raise ValueError(f"full training-state restore failed: {checks}")
    result = {
        "path": str(path), "sha256": sha_file(path),
        "game_count": int(payload["game_count"]),
        "environment_states": int(payload["environment_states"]),
        "optimizer_updates": int(payload["update_count"]),
        "optimizer_slots": len(payload["optimizer_state"]["state"]),
        "replay_rows": len(loaded[4]), "replay_sha256": _replay_digest(loaded[4]),
        "cumulative_training_cpu_seconds": float(payload["cumulative_cpu_seconds"]),
        **checks, "valid": True,
    }
    return result


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _save_model_node(training_state_path: Path, node_path: Path, game_count: int) -> None:
    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    if int(state["game_count"]) != int(game_count):
        raise ValueError(f"checkpoint copy expected G{game_count}, found G{state['game_count']}")
    node = {
        "format_version": 1, "arm_id": ARM, "board_size": 9, "n_in_row": 5,
        "simulations_per_move": SELFPLAY_SIMULATIONS,
        "environment_states": int(state["environment_states"]),
        "game_count": int(state["game_count"]), "games_completed": int(state["game_count"]),
        "optimizer_updates": int(state["update_count"]), "update_count": int(state["update_count"]),
        "model_state": state["model_state"], "config_sha256": state["config_sha256"],
        "source_training_state_sha256": sha_file(training_state_path),
    }
    node_path.parent.mkdir(parents=True, exist_ok=True)
    temp = node_path.with_suffix(node_path.suffix + ".tmp")
    torch.save(node, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, node_path)


def _baseline_cpu_seconds(root: str | Path = ROOT) -> float:
    path = Path(root).resolve() / ARTIFACT / "checkpoints" / ARM / "training_state_games_5000.pt"
    if not path.is_file():
        raise FileNotFoundError("frozen G5000 full-state checkpoint is required to measure added CPU time")
    return float(torch.load(path, map_location="cpu", weights_only=False)["cumulative_cpu_seconds"])


def prepare_experiment(root: str | Path = ROOT, *, source_state_path: str | Path | None = None,
                       source_node_path: str | Path | None = None,
                       drive_folder_id: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    if source_state_path is None:
        source_state_path = root / "research/g_v6_3/checkpoints/reference400_empty/training_state_games_5000.pt"
    if source_node_path is None:
        source_node_path = root / "research/g_v6_3/checkpoints/reference400_empty/node_games_5000.pt"
    source_state_path, source_node_path = Path(source_state_path), Path(source_node_path)
    if not source_state_path.is_file() or not source_node_path.is_file():
        raise FileNotFoundError("the verified V6.3 G5000 state and model must be restored before preparation")
    source_state_sha, source_node_sha = sha_file(source_state_path), sha_file(source_node_path)
    if source_state_sha != SOURCE_STATE_SHA256 or source_node_sha != SOURCE_MODEL_SHA256:
        raise ValueError("Drive-restored G5000 checkpoint SHA-256 differs from the registered source")
    source_validation = validate_training_state(source_state_path, root=root, expected_games=BASE_GAMES)
    if (source_validation["environment_states"], source_validation["optimizer_updates"]) != (BASE_STATES, BASE_UPDATES):
        raise ValueError("G5000 checkpoint counters differ from the frozen V6.3 baseline")
    node = torch.load(source_node_path, map_location="cpu", weights_only=False)
    state = torch.load(source_state_path, map_location="cpu", weights_only=False)
    if not _state_equal(node["model_state"], state["model_state"]):
        raise ValueError("frozen G5000 node weights differ from the full training state")
    if (int(node["game_count"]), int(node["environment_states"]), int(node["update_count"])) != (BASE_GAMES, BASE_STATES, BASE_UPDATES):
        raise ValueError("G5000 node metadata differs from the frozen V6.3 baseline")

    artifact = root / ARTIFACT
    config_dir = artifact / "config"
    checkpoint_dir = artifact / "checkpoints" / ARM
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_config = root / "research/g_v6_3/config/reference_training_config.json"
    config_path = config_dir / "reference_training_config.json"
    if not source_config.is_file():
        raise FileNotFoundError("the committed V6.3 frozen reference configuration is missing")
    config_bytes = source_config.read_bytes()
    if config_path.exists() and config_path.read_bytes() != config_bytes:
        raise ValueError("V6.5 configuration differs from the frozen V6.3 reference")
    if not config_path.exists():
        config_path.write_bytes(config_bytes)
    config = json.loads(config_bytes)
    if config.get("config_sha256") != state["config_sha256"]:
        raise ValueError("G5000 checkpoint config hash differs from V6.3")

    baseline_state = checkpoint_dir / "training_state_games_5000.pt"
    baseline_node = checkpoint_dir / "node_games_5000.pt"
    current_state = checkpoint_dir / "training_state.pt"
    if current_state.exists():
        current = torch.load(current_state, map_location="cpu", weights_only=False)
        if int(current["game_count"]) < BASE_GAMES:
            raise ValueError("existing V6.5 state is behind the G5000 start point")
    else:
        atomic_copy(source_state_path, current_state)
    for source, destination in ((source_state_path, baseline_state), (source_node_path, baseline_node)):
        if destination.exists() and sha_file(destination) != sha_file(source):
            raise ValueError(f"frozen V6.5 baseline was modified: {destination.name}")
        if not destination.exists():
            atomic_copy(source, destination)
    current_validation = validate_training_state(current_state, root=root)
    if current_validation["game_count"] > TARGET_GAMES:
        raise ValueError("existing V6.5 state is past G8000")

    results = artifact / "results"
    results.mkdir(parents=True, exist_ok=True)
    registration_path = artifact / "config/registration.json"
    if registration_path.exists():
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        for key, expected in (("source_state_sha256", SOURCE_STATE_SHA256),
                              ("source_model_sha256", SOURCE_MODEL_SHA256),
                              ("target_games_cumulative", TARGET_GAMES)):
            if registration.get(key) != expected:
                raise ValueError(f"existing registration disagrees with frozen field {key}")
    else:
        registration = {
            "format_version": 1,
            "experiment_id": "wuziq-alphazero-g-v6-5-selfplay-scale",
            "branch": "research/alphazero-g-v6-5-selfplay-scale",
            "base_commit": SOURCE_COMMIT,
            "source_branch": "research/alphazero-g-v6-3-5k",
            "source_archive_code_commit": SOURCE_ARCHIVE_CODE_COMMIT,
            "source_v6_3_drive_folder_id": SOURCE_FOLDER_ID,
            "source_v6_3_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_state_sha256": source_state_sha,
            "source_model_sha256": source_node_sha,
            "source_games": BASE_GAMES, "source_states": BASE_STATES,
            "source_updates": BASE_UPDATES,
            "source_cumulative_training_cpu_seconds": source_validation["cumulative_training_cpu_seconds"],
            "seed": SEED, "board_size": 9, "n_in_row": 5,
            "training_simulations_per_move": SELFPLAY_SIMULATIONS,
            "reference_config_sha256": config["config_sha256"],
            "training_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "training_config_copied_unchanged_from_v6_3": True,
            "target_games_cumulative": TARGET_GAMES,
            "additional_games": TARGET_GAMES - BASE_GAMES,
            "training_milestones_games": list(MILESTONES),
            "snapshot_policy": {
                "trigger": "first of 250 completed games or 3 added CPU hours",
                "interval_games": SNAPSHOT_INTERVAL_GAMES,
                "interval_cpu_seconds": SNAPSHOT_INTERVAL_CPU_SECONDS,
                "retain_latest_two_rolling_and_G5000_source_plus_G6000_G7000_G8000_milestones": True,
            },
            "cpu_budget_new_experiment_hours": 40,
            "training_algorithm_modified": False,
            "training_sources": "current G0 self-play only; no teacher, tactical reward, or rule patch",
            "drive_folder_id": drive_folder_id,
            "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(registration_path, registration)
    progress_path = results / "v6_5_progress.json"
    if not progress_path.exists():
        atomic_json(progress_path, {
            "format_version": 1, "experiment_id": registration["experiment_id"],
            "games_completed": BASE_GAMES, "training_states_cumulative": BASE_STATES,
            "optimizer_updates_cumulative": BASE_UPDATES,
            "added_cpu_seconds": 0.0, "last_snapshot_games": BASE_GAMES,
            "phase": "ready_at_g5000", "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return {
        "source_validation": source_validation,
        "frozen_model_sha256": source_node_sha,
        "staged_state": current_validation,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "registration": registration,
    }


def run_training_batch(root: str | Path = ROOT, *, batch_games: int = SNAPSHOT_INTERVAL_GAMES,
                       batch_cpu_seconds: float = SNAPSHOT_INTERVAL_CPU_SECONDS) -> dict[str, Any]:
    root = Path(root).resolve()
    torch.set_num_threads(1)
    artifact = root / ARTIFACT
    config = json.loads((artifact / "config/reference_training_config.json").read_text(encoding="utf-8"))
    state_path = artifact / "checkpoints" / ARM / "training_state.pt"
    current_validation = validate_training_state(state_path, root=root)
    current_games = int(current_validation["game_count"])
    baseline_state_path = artifact / "checkpoints" / ARM / "training_state_games_5000.pt"
    baseline_cpu = float(torch.load(baseline_state_path, map_location="cpu", weights_only=False)["cumulative_cpu_seconds"])
    added_cpu = float(current_validation["cumulative_training_cpu_seconds"]) - baseline_cpu
    if added_cpu >= MAX_NEW_CPU_SECONDS:
        return {"stopped": "new_cpu_hard_cap", **current_validation}
    if current_games >= TARGET_GAMES:
        return {"stopped": "target_reached", **current_validation}
    remaining_budget = MAX_NEW_CPU_SECONDS - added_cpu
    stop_after = min(float(batch_cpu_seconds), remaining_budget)
    next_milestone = min((value for value in MILESTONES if value > current_games), default=TARGET_GAMES)
    chunk_target = min(current_games + int(batch_games), next_milestone, TARGET_GAMES)
    progress_path = artifact / "results/v6_5_progress.json"
    prior_progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    training_started_at = prior_progress.get("training_started_at_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not prior_progress.get("training_started_at_utc"):
        prior_progress["training_started_at_utc"] = training_started_at
        atomic_json(progress_path, prior_progress)
    started_cpu = time.process_time()
    result = train_reference_arm(
        root, arm=ARM, board_size=9, n_in_row=5, simulations=SELFPLAY_SIMULATIONS,
        target_states=10**12, target_games=chunk_target, milestones=[], seed=SEED,
        config=config, stop_at_cpu_seconds=started_cpu + stop_after,
        artifact_dir=ARTIFACT,
        progress_experiment_id="wuziq-alphazero-g-v6-5-selfplay-scale",
    )
    final_validation = validate_training_state(state_path, root=root)
    games_after = int(final_validation["game_count"])
    if games_after < current_games or games_after > chunk_target:
        raise ValueError("V6.5 batch did not advance monotonically within its registered boundary")
    if int(final_validation["optimizer_updates"]) != int(final_validation["game_count"]) - 1:
        raise ValueError("optimizer update count diverged from the one-update-per-game reference loop")
    milestone_saved = games_after in MILESTONES
    if milestone_saved:
        _save_model_node(state_path, artifact / "checkpoints" / ARM / f"node_games_{games_after}.pt", games_after)
        atomic_copy(state_path, artifact / "checkpoints" / ARM / f"training_state_games_{games_after}.pt")
        milestone_validation = validate_training_state(
            artifact / "checkpoints" / ARM / f"training_state_games_{games_after}.pt", root=root,
            expected_games=games_after,
        )
    else:
        milestone_validation = None
    added_cpu_after = float(final_validation["cumulative_training_cpu_seconds"]) - baseline_cpu
    batch_cpu = time.process_time() - started_cpu
    history_path = artifact / "results/training/reference400_empty.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    segment_path = artifact / "results/training_segments.jsonl"
    previous_segments = [json.loads(line) for line in segment_path.read_text(encoding="utf-8").splitlines() if line.strip()] if segment_path.exists() else []
    segment_start = int(previous_segments[-1]["end_games"]) if previous_segments else BASE_GAMES
    if segment_start > current_games:
        raise ValueError("training segment progress is ahead of the durable checkpoint")
    previous_segment_cpu = float(previous_segments[-1]["added_cpu_seconds_cumulative"]) if previous_segments else 0.0
    batch_rows = [row for row in history_rows
                  if segment_start <= int(row["game_id"].rsplit("-", 1)[-1]) < games_after]
    mean_metric = lambda key: float(np.mean([float(row.get(key, 0.0)) for row in batch_rows])) if batch_rows else 0.0
    progress = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-5-selfplay-scale",
        "games_completed": games_after,
        "training_states_cumulative": int(final_validation["environment_states"]),
        "optimizer_updates_cumulative": int(final_validation["optimizer_updates"]),
        "added_cpu_seconds": added_cpu_after,
        "last_batch_cpu_seconds": batch_cpu,
        "last_batch_completed_games": games_after - current_games,
        "last_batch_mean_loss": mean_metric("loss"),
        "last_batch_mean_policy_loss": mean_metric("policy_loss"),
        "last_batch_mean_value_loss": mean_metric("value_loss"),
        "last_batch_mean_policy_entropy": mean_metric("entropy"),
        "checkpoint_sha256": final_validation["sha256"],
        "training_started_at_utc": training_started_at,
        "phase": "training_complete" if games_after == TARGET_GAMES else "self_play_training",
        "milestone_saved": milestone_saved,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if games_after == TARGET_GAMES else None,
    }
    atomic_json(progress_path, progress)
    segment = {
        "start_games": segment_start, "end_games": games_after,
        "training_states_cumulative": int(final_validation["environment_states"]),
        "optimizer_updates_cumulative": int(final_validation["optimizer_updates"]),
        "added_cpu_seconds_cumulative": added_cpu_after,
        "batch_cpu_seconds": added_cpu_after - previous_segment_cpu,
        "last_executor_cpu_seconds": batch_cpu,
        "completed_games": games_after - segment_start,
        "mean_loss": mean_metric("loss"),
        "mean_policy_loss": mean_metric("policy_loss"),
        "mean_value_loss": mean_metric("value_loss"),
        "mean_policy_entropy": mean_metric("entropy"),
        "milestone_saved": milestone_saved,
        "timestamp_utc": progress["updated_at_utc"],
    }
    with segment_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(segment, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    result.update({
        "previous_games": current_games,
        "added_training_cpu_seconds": added_cpu_after - added_cpu,
        "added_training_cpu_total_seconds": added_cpu_after,
        "milestone_saved": milestone_saved,
        "milestone_validation": milestone_validation,
        "progress": progress,
    })
    return result


def benchmark_g5000(root: str | Path = ROOT) -> dict[str, Any]:
    """Measure one real PUCT-400 decision from the frozen G5000 policy."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    node_path = root / ARTIFACT / "checkpoints" / ARM / "node_games_5000.pt"
    node = torch.load(node_path, map_location="cpu", weights_only=False)
    model = ReferencePolicyValueNet(9, seed=SEED)
    model.load_policy_param(node["model_state"])
    model.network.eval()
    player = instrumented_player(model.policy_value_fn, c_puct=5.0, n_playout=400, is_selfplay=True)
    board = make_board(9, 5, 0)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    move, _ = player.get_action(board, temp=1e-3, return_prob=1)
    return {
        "board": "9x9 open board", "simulations_requested": 400,
        "upstream_configured_playouts": 400, "root_action_visits": int(player.last_search_stats.get("root_visits", -1)),
        "chosen_move": int(move), "measured_cpu_seconds": time.process_time() - started_cpu,
        "measured_wall_seconds": time.perf_counter() - started_wall,
        "search_stats": player.last_search_stats,
    }


def preflight(root: str | Path = ROOT) -> dict[str, Any]:
    """Forecast from measured V6.3 continuation throughput and current PUCT cost."""
    root = Path(root).resolve()
    bench = benchmark_g5000(root)
    source_5000 = torch.load(root / ARTIFACT / "checkpoints" / ARM / "training_state_games_5000.pt",
                             map_location="cpu", weights_only=False)
    source_2500_path = root / "research/g_v6_3/checkpoints/reference400_empty/training_state_games_2500.pt"
    source_2500 = torch.load(source_2500_path, map_location="cpu", weights_only=False)
    historical_games = BASE_GAMES - 2500
    historical_cpu = float(source_5000["cumulative_cpu_seconds"] - source_2500["cumulative_cpu_seconds"])
    seconds_per_game = historical_cpu / historical_games
    remaining_games = TARGET_GAMES - BASE_GAMES
    training_normal = seconds_per_game * remaining_games
    estimates = {
        "training_cpu_hours": {
            "optimistic": training_normal * 0.80 / 3600,
            "normal": training_normal / 3600,
            "conservative": training_normal * 1.30 / 3600,
        },
        "snapshot_cpu_hours": {"optimistic": 0.25, "normal": 0.75, "conservative": 1.50},
        "total_cpu_hours": {
            "optimistic": training_normal * 0.80 / 3600 + 0.25,
            "normal": training_normal / 3600 + 0.75,
            "conservative": training_normal * 1.30 / 3600 + 1.50,
        },
    }
    quota = None
    try:
        quota_tokens = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_tokens[0] != "max":
            quota = float(quota_tokens[0]) / float(quota_tokens[1])
    except Exception:
        pass
    memory_limit = None
    for memory_file in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            value = memory_file.read_text().strip()
            if value != "max":
                memory_limit = int(value)
                break
        except Exception:
            continue
    report = {
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logical_cpu_count": os.cpu_count(), "cpu_quota_cores": quota,
        "training_parallelism": 1, "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "memory_limit_bytes": memory_limit,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "single_decision_benchmark": bench,
        "historical_training_rate": {
            "source": "actual V6.3 G2500-to-G5000 continuation full-state counters",
            "completed_games": historical_games,
            "cpu_seconds": historical_cpu,
            "cpu_seconds_per_game": seconds_per_game,
        },
        "planned_new_games": remaining_games,
        "estimates": estimates,
        "hard_cap_cpu_hours": 40,
        "wall_estimate_basis": "serial self-play with one PyTorch thread; expected wall time tracks one-process CPU time, subject to host contention",
        "disk_projection": "V6.3 full recovery archive was 184 MB uncompressed and 37 MB compressed at G5000; V6.5 local plus Drive staging provisioned below 1.5 GB, with 17 GB presently free",
    }
    atomic_json(root / ARTIFACT / "results/preflight.json", report)
    return report
