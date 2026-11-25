"""V6.9: continue the frozen V6.7 G20,000 state to 50,000 self-play games.

Only the continuation boundary, milestone bookkeeping, and recovery packaging
are new here.  The neural network, self-play, PUCT-400, policy/value targets,
optimizer, replay buffer, and learning loop remain the V5/V6 reference code.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
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

from .v5_reference import ReferencePolicyValueNet
from .v5_training import load_training_checkpoint, train_reference_arm

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_9")
ARM = "reference400_empty"
BASE_GAMES = 20_000
TARGET_GAMES = 50_000
MILESTONES = (25_000, 30_000, 35_000, 40_000, 45_000, 50_000)
SEED = 590_502
SIMULATIONS = 400
SNAPSHOT_INTERVAL_GAMES = 250
SNAPSHOT_INTERVAL_CPU_SECONDS = 3 * 3600
SOURCE_COMMIT = "911242c10c8a374740db072f71a6b5f450a837e6"
V68_BASE_COMMIT = "ed7890e61034455c19b2c3bb37d6c6f3b7c8df6a"
SOURCE_STATE_SHA256 = "f932ec34eaf4da6ac63bc77b6636e0517235a8986bbcf981139afa4cae7b7719"
SOURCE_MODEL_SHA256 = "aa8b69920675c9cd43a9b8a960abf3a43b88ebf6ad92dff2d2fb0ed2f4fcecbc"
SOURCE_DRIVE_FOLDER_ID = "1bkakjo5fdhgELAzyuOohDE8wynVBY1fh"
DRIVE_FOLDER_ID = "1Tk9gkA0MwzMMB6rueZB-IpFWLVrVx6Qq"
SOURCE_CUMULATIVE_CPU_SECONDS = 397_075.726405704


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


def artifact_path(root: str | Path = ROOT) -> Path:
    return Path(root).resolve() / ARTIFACT


def state_path(root: str | Path = ROOT) -> Path:
    return artifact_path(root) / "checkpoints" / ARM / "training_state.pt"


def progress_path(root: str | Path = ROOT) -> Path:
    return artifact_path(root) / "results" / "progress.json"


def _numpy_rng_equal(left: tuple, right: tuple) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _replay_digest(rows) -> str:
    digest = hashlib.sha256()
    for state, policy, value in rows:
        for array in (state, policy):
            arr = np.asarray(array)
            digest.update(str(arr.shape).encode())
            digest.update(arr.dtype.str.encode())
            digest.update(arr.tobytes())
        digest.update(np.asarray([value], dtype=np.float32).tobytes())
    return digest.hexdigest()


def validate_training_state(path: str | Path | None = None, *, root: str | Path = ROOT,
                            expected_games: int | None = None) -> dict[str, Any]:
    """Validate weights, Adam, replay, counters and RNG by restoring a fresh network."""
    root = Path(root).resolve()
    path = Path(path or state_path(root))
    if not path.is_absolute():
        path = root / path
    config_path = root / ARTIFACT / "config" / "reference_training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config.get("config_sha256"):
        raise ValueError("training state differs from the frozen reference configuration")
    if payload.get("arm_id") != ARM or int(payload.get("base_seed", -1)) != SEED:
        raise ValueError("training state arm or seed differs from the registered continuation")
    if int(payload.get("board_size", -1)) != 9:
        raise ValueError("training state is not a 9x9 model")
    if expected_games is not None and int(payload.get("game_count", -1)) != int(expected_games):
        raise ValueError("training state game count differs from requested boundary")
    games = int(payload["game_count"])
    updates = int(payload["update_count"])
    if not BASE_GAMES <= games <= TARGET_GAMES or updates != games - 1:
        raise ValueError(f"invalid cumulative training counters: games={games}, updates={updates}")
    if len(payload.get("replay_buffer", [])) != int(config["training"]["buffer_size"]):
        raise ValueError("full replay buffer is missing or has the wrong capacity")
    if not payload.get("optimizer_state", {}).get("state"):
        raise ValueError("Adam optimizer state is missing")

    model = ReferencePolicyValueNet(9, seed=SEED)
    old_python, old_numpy, old_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        loaded = load_training_checkpoint(path, model=model, arm=ARM,
                                          config_hash=config["config_sha256"],
                                          buffer_size=int(config["training"]["buffer_size"]))
        checks = {
            "model_restored": _state_equal(model.get_policy_param(), payload["model_state"]),
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
        raise ValueError(f"full-state restore checks failed: {checks}")
    return {
        "path": str(path), "sha256": sha_file(path), "game_count": games,
        "environment_states": int(payload["environment_states"]),
        "optimizer_updates": updates, "optimizer_slots": len(payload["optimizer_state"]["state"]),
        "replay_rows": len(loaded[4]), "replay_sha256": _replay_digest(loaded[4]),
        "cumulative_training_cpu_seconds": float(payload["cumulative_cpu_seconds"]),
        **checks, "valid": True,
    }


@contextmanager
def exclusive_training_lock(lock_path: str | Path):
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a V6.9 training batch is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_experiment(root: str | Path = ROOT, *, source_state_path: str | Path,
                       source_node_path: str | Path, drive_folder_id: str = DRIVE_FOLDER_ID) -> dict[str, Any]:
    root = Path(root).resolve()
    source_state_path, source_node_path = Path(source_state_path), Path(source_node_path)
    if not source_state_path.is_file() or not source_node_path.is_file():
        raise FileNotFoundError("Drive-restored G20,000 full state and model node are required")
    if sha_file(source_state_path) != SOURCE_STATE_SHA256:
        raise ValueError("G20,000 training-state SHA-256 differs from the frozen source")
    if sha_file(source_node_path) != SOURCE_MODEL_SHA256:
        raise ValueError("G20,000 model-node SHA-256 differs from the frozen source")

    # Validate the state against the V6.7 source artifact before copying it into V6.9.
    source_config_path = root / "research/g_v6_7/config/reference_training_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(source_config_path)
    config_bytes = source_config_path.read_bytes()
    source_payload = torch.load(source_state_path, map_location="cpu", weights_only=False)
    node = torch.load(source_node_path, map_location="cpu", weights_only=False)
    if int(source_payload["game_count"]) != BASE_GAMES or int(source_payload["environment_states"]) != 348_216:
        raise ValueError("G20,000 counters do not match the verified V6.7 checkpoint")
    if int(source_payload["update_count"]) != BASE_GAMES - 1 or len(source_payload["replay_buffer"]) != 10_000:
        raise ValueError("G20,000 Adam/replay training state is incomplete")
    config = json.loads(config_bytes)
    if source_payload["config_sha256"] != config["config_sha256"]:
        raise ValueError("G20,000 state does not match the frozen V6.7 training configuration")
    if int(node["game_count"]) != BASE_GAMES or int(node["simulations_per_move"]) != SIMULATIONS:
        raise ValueError("G20,000 model node metadata differs from the frozen baseline")
    if not _state_equal(node["model_state"], source_payload["model_state"]):
        raise ValueError("G20,000 model node weights differ from full training state")

    artifact = artifact_path(root)
    for directory in (artifact / "config", artifact / "checkpoints" / ARM,
                      artifact / "models", artifact / "raw/selfplay" / ARM,
                      artifact / "results/training", artifact / "results/snapshots",
                      artifact / "recovery"):
        directory.mkdir(parents=True, exist_ok=True)
    config_dest = artifact / "config/reference_training_config.json"
    if config_dest.exists() and config_dest.read_bytes() != config_bytes:
        raise ValueError("V6.9 training configuration differs from frozen V6.7")
    if not config_dest.exists():
        config_dest.write_bytes(config_bytes)

    full_path = artifact / "checkpoints" / ARM / "training_state_games_20000.pt"
    live_path = state_path(root)
    if live_path.exists():
        current = validate_training_state(live_path, root=root)
        if current["game_count"] < BASE_GAMES:
            raise ValueError("existing V6.9 training state is behind G20,000")
    else:
        atomic_copy(source_state_path, live_path)
    if not full_path.exists():
        atomic_copy(source_state_path, full_path)
    model_dest = artifact / "models/G20000.pt"
    node_dest = artifact / "checkpoints" / ARM / "node_games_20000.pt"
    for source, dest in ((source_node_path, model_dest), (source_node_path, node_dest)):
        if dest.exists() and sha_file(dest) != SOURCE_MODEL_SHA256:
            raise ValueError(f"frozen G20,000 model was modified: {dest}")
        if not dest.exists():
            atomic_copy(source, dest)
    staged = validate_training_state(live_path, root=root)
    if staged["game_count"] < BASE_GAMES:
        raise ValueError("staged V6.9 state is behind G20,000")

    registration = artifact / "config/registration.json"
    expected_registration = {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-9-selfplay-50k",
        "branch": "research/alphazero-g-v6-9-train-50k", "base_commit": V68_BASE_COMMIT,
        "source_training_branch": "research/alphazero-g-v6-7-selfplay-20k",
        "source_training_commit": SOURCE_COMMIT,
        "source_drive_folder_id": SOURCE_DRIVE_FOLDER_ID,
        "source_drive_archive_sha256": "860218130788b7710df95441bac0439dbec53e4520ecc66b2f5ccc8d85e60911",
        "source_games": BASE_GAMES, "source_states": 348_216, "source_updates": 19_999,
        "source_training_state_sha256": SOURCE_STATE_SHA256, "source_model_sha256": SOURCE_MODEL_SHA256,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "reference_config_sha256": config["config_sha256"],
        "board_size": 9, "n_in_row": 5, "simulations_per_move": SIMULATIONS,
        "seed": SEED, "target_games_cumulative": TARGET_GAMES,
        "additional_games": TARGET_GAMES - BASE_GAMES,
        "milestones_games": list(MILESTONES),
        "training_algorithm_modified": False,
        "training_data": "current G0 self-play only; no teacher data, external labels, or shaped rewards",
        "snapshot_policy": {"trigger": "first of 250 completed games or 3 CPU hours",
                            "retain_latest_two_rolling": True,
                            "permanent_milestones": [BASE_GAMES, *MILESTONES],
                            "drive_folder_id": drive_folder_id},
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if registration.exists():
        previous = json.loads(registration.read_text(encoding="utf-8"))
        for field in ("experiment_id", "base_commit", "source_training_state_sha256", "source_model_sha256",
                      "config_sha256", "target_games_cumulative", "milestones_games"):
            if previous.get(field) != expected_registration.get(field):
                raise ValueError(f"existing V6.9 registration differs in frozen field {field}")
        expected_registration = previous
    else:
        atomic_json(registration, expected_registration)

    baseline_path = artifact / "results/baseline.json"
    if not baseline_path.exists():
        atomic_json(baseline_path, {
            "games": BASE_GAMES, "environment_states": 348_216, "optimizer_updates": 19_999,
            "cumulative_training_cpu_seconds": SOURCE_CUMULATIVE_CPU_SECONDS,
            "model_sha256": SOURCE_MODEL_SHA256, "training_state_sha256": SOURCE_STATE_SHA256,
            "source_snapshot_sha256": expected_registration["source_drive_archive_sha256"],
        })
    progress = progress_path(root)
    if not progress.exists():
        atomic_json(progress, {
            "experiment_id": expected_registration["experiment_id"], "games_completed": BASE_GAMES,
            "training_states_cumulative": 348_216, "optimizer_updates_cumulative": 19_999,
            "training_state_sha256": SOURCE_STATE_SHA256, "last_snapshot_games": BASE_GAMES,
            "last_snapshot_total_cpu_seconds": 0.0, "snapshot_due": True,
            "phase": "ready_at_g20000", "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    accounting = artifact / "results/cpu_accounting.json"
    if not accounting.exists():
        atomic_json(accounting, {"snapshot_packaging_cpu_seconds": 0.0, "audit_cpu_seconds": 0.0})
    return {"source_validation": {"game_count": BASE_GAMES, "environment_states": 348_216,
                                  "optimizer_updates": 19_999, "state_sha256": SOURCE_STATE_SHA256,
                                  "model_sha256": SOURCE_MODEL_SHA256},
            "staged_state": staged, "registration": expected_registration}


def _save_milestone(root: Path, games: int) -> dict[str, Any]:
    live = state_path(root)
    validation = validate_training_state(live, root=root, expected_games=games)
    checkpoint_dir = artifact_path(root) / "checkpoints" / ARM
    full_path = checkpoint_dir / f"training_state_games_{games}.pt"
    if not full_path.exists() or sha_file(full_path) != validation["sha256"]:
        atomic_copy(live, full_path)
    payload = torch.load(live, map_location="cpu", weights_only=False)
    node = {
        "format_version": 1, "arm_id": ARM, "board_size": 9, "n_in_row": 5,
        "simulations_per_move": SIMULATIONS, "environment_states": int(payload["environment_states"]),
        "game_count": games, "games_completed": games, "optimizer_updates": int(payload["update_count"]),
        "update_count": int(payload["update_count"]), "model_state": payload["model_state"],
        "config_sha256": payload["config_sha256"], "source_training_state_sha256": validation["sha256"],
    }
    node_path = checkpoint_dir / f"node_games_{games}.pt"
    temp = node_path.with_suffix(".pt.tmp")
    torch.save(node, temp)
    os.replace(temp, node_path)
    model_path = artifact_path(root) / "models" / f"G{games}.pt"
    atomic_copy(node_path, model_path)

    history = artifact_path(root) / "results/training" / f"{ARM}.jsonl"
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()] if history.exists() else []
    start = max((value for value in MILESTONES if value < games), default=BASE_GAMES)
    interval = [row for row in rows
                if start <= int(row["game_id"].rsplit("-", 1)[-1]) < games]
    stage = {
        "games": games, "training_states_cumulative": validation["environment_states"],
        "optimizer_updates_cumulative": validation["optimizer_updates"],
        "interval_start_games": start, "interval_games": len(interval),
        "interval_mean_loss": float(np.mean([r.get("loss", 0.0) for r in interval])) if interval else None,
        "interval_mean_policy_loss": float(np.mean([r.get("policy_loss", 0.0) for r in interval])) if interval else None,
        "interval_mean_value_loss": float(np.mean([r.get("value_loss", 0.0) for r in interval])) if interval else None,
        "interval_mean_policy_entropy": float(np.mean([r.get("entropy", 0.0) for r in interval])) if interval else None,
        "new_training_cpu_seconds": validation["cumulative_training_cpu_seconds"] - SOURCE_CUMULATIVE_CPU_SECONDS,
        "training_state_sha256": validation["sha256"], "model_node_sha256": sha_file(model_path),
        "full_state_restoration_valid": True,
    }
    stage_path = artifact_path(root) / "results/stage_metrics.json"
    existing = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else {"stages": []}
    existing["stages"] = [row for row in existing.get("stages", []) if int(row["games"]) != games] + [stage]
    existing["stages"].sort(key=lambda row: int(row["games"]))
    atomic_json(stage_path, existing)
    return stage


def _reconcile_progress(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    progress = json.loads(progress_path(root).read_text(encoding="utf-8"))
    games = int(state["game_count"])
    last_games = int(progress.get("last_snapshot_games", BASE_GAMES))
    baseline_cpu = float(json.loads((artifact_path(root) / "results/baseline.json").read_text())[
        "cumulative_training_cpu_seconds"])
    total_cpu = max(0.0, float(state["cumulative_training_cpu_seconds"]) - baseline_cpu)
    if int(progress.get("games_completed", -1)) != games or progress.get("training_state_sha256") != state["sha256"]:
        # The checkpoint is written atomically after every whole game. A process
        # restart can therefore be reconciled without replaying or double-counting.
        progress.update({"games_completed": games, "training_states_cumulative": state["environment_states"],
                         "optimizer_updates_cumulative": state["optimizer_updates"],
                         "training_state_sha256": state["sha256"], "recovered_complete_games_since_snapshot":
                         max(0, games - last_games), "snapshot_due": True,
                         "phase": "self_play_training" if games < TARGET_GAMES else "training_complete",
                         "completed": games == TARGET_GAMES,
                         "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        atomic_json(progress_path(root), progress)
    return progress


def run_training_batch(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    torch.set_num_threads(1)
    artifact = artifact_path(root)
    config = json.loads((artifact / "config/reference_training_config.json").read_text(encoding="utf-8"))
    current = validate_training_state(root=root)
    progress = _reconcile_progress(root, current)
    current_games = int(current["game_count"])
    if current_games >= TARGET_GAMES:
        return {"stopped": "target_reached", "state": current, "progress": progress}
    if bool(progress.get("snapshot_due", False)):
        return {"stopped": "snapshot_due", "state": current, "progress": progress}

    last_snapshot_games = int(progress.get("last_snapshot_games", BASE_GAMES))
    target = min(current_games + SNAPSHOT_INTERVAL_GAMES,
                 min((value for value in MILESTONES if value > current_games), default=TARGET_GAMES),
                 TARGET_GAMES)
    if target <= current_games or target - last_snapshot_games > SNAPSHOT_INTERVAL_GAMES:
        raise ValueError("invalid or stale V6.9 batch target; take a verified snapshot before continuing")
    start_cpu = time.process_time()
    result = train_reference_arm(
        root, arm=ARM, board_size=9, n_in_row=5, simulations=SIMULATIONS,
        target_states=10**12, target_games=target, milestones=[], seed=SEED,
        config=config, stop_at_cpu_seconds=start_cpu + SNAPSHOT_INTERVAL_CPU_SECONDS,
        artifact_dir=ARTIFACT,
        progress_experiment_id="wuziq-alphazero-g-v6-9-selfplay-50k",
    )
    updated = validate_training_state(root=root)
    games_after = int(updated["game_count"])
    if not current_games < games_after <= target:
        raise ValueError(f"V6.9 batch did not advance monotonically: G{current_games} -> G{games_after}, target G{target}")
    stage = _save_milestone(root, games_after) if games_after in MILESTONES else None
    baseline = json.loads((artifact / "results/baseline.json").read_text(encoding="utf-8"))
    training_cpu = float(updated["cumulative_training_cpu_seconds"]) - float(baseline["cumulative_training_cpu_seconds"])
    due = (games_after - last_snapshot_games >= SNAPSHOT_INTERVAL_GAMES
           or training_cpu - float(progress.get("last_snapshot_training_cpu_seconds", 0.0))
              >= SNAPSHOT_INTERVAL_CPU_SECONDS
           or stage is not None or games_after == TARGET_GAMES)
    progress.update({
        "games_completed": games_after, "training_states_cumulative": updated["environment_states"],
        "optimizer_updates_cumulative": updated["optimizer_updates"],
        "training_state_sha256": updated["sha256"], "new_training_cpu_seconds": training_cpu,
        "last_batch_completed_games": games_after - current_games,
        "last_batch_cpu_seconds": time.process_time() - start_cpu,
        "snapshot_due": due, "milestone": stage is not None,
        "phase": "training_complete" if games_after == TARGET_GAMES else "self_play_training",
        "completed": games_after == TARGET_GAMES,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(progress_path(root), progress)
    atomic_json(artifact / "results/training_progress.json",
                {**progress, "active_arm": ARM, "replay_rows": updated["replay_rows"]})
    return {"previous_games": current_games, "games_completed": games_after,
            "training_states_cumulative": updated["environment_states"],
            "optimizer_updates_cumulative": updated["optimizer_updates"],
            "new_training_cpu_seconds": training_cpu, "milestone": stage,
            "snapshot_due": due, "target_games": target,
            "batch_training_cpu_seconds": time.process_time() - start_cpu,
            "training_result": result, "training_state_sha256": updated["sha256"], "progress": progress}


def mark_snapshot_verified(root: str | Path = ROOT, *, games: int,
                           snapshot_record: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    progress = json.loads(progress_path(root).read_text(encoding="utf-8"))
    if int(progress["games_completed"]) != int(games):
        raise ValueError("verified Drive snapshot does not match local progress")
    if snapshot_record.get("sha256") != snapshot_record.get("downloaded_sha256"):
        raise ValueError("Drive archive hash was not verified")
    if not snapshot_record.get("empty_directory_restore"):
        raise ValueError("empty-directory recovery test did not pass")
    state = validate_training_state(root=root, expected_games=games)
    if state["sha256"] != snapshot_record.get("training_state_sha256"):
        raise ValueError("Drive snapshot contains a different full training state")
    baseline = json.loads((artifact_path(root) / "results/baseline.json").read_text())
    total_cpu = float(state["cumulative_training_cpu_seconds"]) - float(baseline["cumulative_training_cpu_seconds"])
    progress.update({"last_snapshot_games": int(games), "last_snapshot_training_cpu_seconds": total_cpu,
                     "last_verified_snapshot": snapshot_record, "snapshot_due": False,
                     "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    atomic_json(progress_path(root), progress)
    atomic_json(artifact_path(root) / "results/training_progress.json",
                {**progress, "active_arm": ARM, "replay_rows": state["replay_rows"]})
    snapshot_path = artifact_path(root) / "results/snapshots" / f"G{games}.json"
    atomic_json(snapshot_path, snapshot_record)
    return progress


def preflight_resources(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    disk = shutil.disk_usage(root)
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    memory_max = Path("/sys/fs/cgroup/memory.max")
    source = json.loads((artifact_path(root) / "results/baseline.json").read_text(encoding="utf-8"))
    validation = validate_training_state(root=root)
    historical_v67_baseline = json.loads(
        (root / "research/g_v6_7/results/baseline.json").read_text(encoding="utf-8"))
    historical_games = int(validation["game_count"]) - 8_000
    if historical_games <= 0:
        raise ValueError("G20,000 history is required to estimate continuation cost")
    history_rate = (float(validation["cumulative_training_cpu_seconds"])
                    - float(historical_v67_baseline["cumulative_training_cpu_seconds"])) / historical_games
    remaining = TARGET_GAMES - validation["game_count"]
    expected = history_rate * remaining
    return {
        "visible_cpus": os.cpu_count(), "cpu_quota": cpu_max.read_text().strip() if cpu_max.exists() else None,
        "memory_limit_bytes": memory_max.read_text().strip() if memory_max.exists() else None,
        "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
        "torch": torch.__version__, "torch_threads": 1,
        "current_games": validation["game_count"], "remaining_games": remaining,
        "historical_cpu_seconds_per_game": history_rate,
        "historical_training_cpu_hours_for_remaining": expected / 3600,
        "optimistic_training_cpu_hours": expected * 0.8 / 3600,
        "conservative_training_cpu_hours": expected * 1.3 / 3600,
        "cpu_estimate_basis": "observed V6.7 G8000-G20000 training CPU seconds, extrapolated to the remaining 30K games",
    }
