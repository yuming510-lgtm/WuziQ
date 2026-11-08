"""Exact-state continuation orchestration for the G V6.2 scale experiment."""
from __future__ import annotations

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
from .v6_registration import _same


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_2")
ARM = "reference400_empty"
V6_STATE = Path("research/g_v6/checkpoints/reference400_empty/training_state.pt")
V6_NODE = Path("research/g_v6/checkpoints/reference400_empty/node_20000.pt")
V5_CONFIG = Path("research/g_v5/config/experiment.json")
INITIAL_GAMES = 1_012
INITIAL_STATES = 20_001
INITIAL_UPDATES = 1_011
MILESTONES = (1_250, 1_500, 1_750, 2_000, 2_500, 3_000)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as src, temp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temp, destination)


def _numpy_rng_equal(left, right) -> bool:
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


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


def validate_full_training_state(root: str | Path, path: str | Path, *, baseline: bool = False) -> dict[str, Any]:
    """Load a real trainer checkpoint and verify every restorable component."""
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = json.loads((root / V5_CONFIG).read_text(encoding="utf-8"))
    config_hash = config["config_sha256"]
    if (checkpoint.get("arm_id") != ARM or checkpoint.get("config_sha256") != config_hash
            or int(checkpoint.get("board_size", 0)) != 9
            or int(checkpoint.get("base_seed", -1)) != 590502):
        raise ValueError("V6.2 checkpoint identity/configuration is invalid")
    if baseline and (int(checkpoint["game_count"]) != INITIAL_GAMES
                     or int(checkpoint["environment_states"]) != INITIAL_STATES
                     or int(checkpoint["update_count"]) != INITIAL_UPDATES):
        raise ValueError("V6.2 baseline counters do not match the frozen V6 checkpoint")
    if (int(checkpoint["game_count"]) < INITIAL_GAMES
            or int(checkpoint["environment_states"]) < INITIAL_STATES
            or int(checkpoint["update_count"]) < INITIAL_UPDATES
            or len(checkpoint.get("replay_buffer", [])) != int(config["training"]["buffer_size"])
            or not checkpoint.get("optimizer_state", {}).get("state")):
        raise ValueError("V6.2 full checkpoint is missing counters, Adam state, or replay rows")

    model = ReferencePolicyValueNet(9, seed=int(checkpoint["base_seed"]))
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    previous_torch = torch.get_rng_state().clone()
    try:
        loaded = load_training_checkpoint(
            path, model=model, arm=ARM, config_hash=config_hash,
            buffer_size=int(config["training"]["buffer_size"]),
        )
        python_ok = random.getstate() == checkpoint["python_rng_state"]
        numpy_ok = _numpy_rng_equal(np.random.get_state(), checkpoint["numpy_rng_state"])
        torch_ok = torch.equal(torch.get_rng_state(), checkpoint["torch_rng_state"])
        model_ok = _same(model.get_policy_param(), checkpoint["model_state"])
        optimizer_ok = _same(model.optimizer.state_dict(), checkpoint["optimizer_state"])
        replay_ok = (len(loaded[4]) == len(checkpoint["replay_buffer"])
                     and _replay_digest(loaded[4]) == _replay_digest(checkpoint["replay_buffer"]))
    finally:
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.set_rng_state(previous_torch)
    checks = {
        "model_restored": model_ok,
        "adam_restored": optimizer_ok,
        "replay_restored": replay_ok,
        "python_rng_restored": python_ok,
        "numpy_rng_restored": numpy_ok,
        "torch_rng_restored": torch_ok,
    }
    if not all(checks.values()):
        raise ValueError(f"V6.2 full-state restore validation failed: {checks}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha(path),
        "game_count": int(checkpoint["game_count"]),
        "environment_states": int(checkpoint["environment_states"]),
        "optimizer_updates": int(checkpoint["update_count"]),
        "optimizer_slots": len(checkpoint["optimizer_state"]["state"]),
        "replay_rows": len(loaded[4]),
        "replay_sha256": _replay_digest(loaded[4]),
        "cumulative_training_cpu_seconds": float(checkpoint["cumulative_cpu_seconds"]),
        **checks,
        "valid": True,
    }


def prepare_v6_2(root: str | Path = ROOT) -> dict[str, Any]:
    """Stage the exact V6 full checkpoint into a separate V6.2 namespace once."""
    root = Path(root).resolve()
    registration_path = root / ARTIFACT / "config/registration.json"
    if not registration_path.is_file():
        raise FileNotFoundError(f"V6.2 registration is missing: {registration_path}")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source_state = root / V6_STATE
    source_node = root / V6_NODE
    source_sha = _sha(source_state)
    if source_sha != registration["resume_state_sha256"]:
        raise ValueError("V6.2 source checkpoint SHA differs from the frozen registration")
    source_validation = validate_full_training_state(root, source_state, baseline=True)
    source_payload = torch.load(source_state, map_location="cpu", weights_only=False)
    node_payload = torch.load(source_node, map_location="cpu", weights_only=False)
    if not _same(source_payload["model_state"], node_payload["model_state"]):
        raise ValueError("V6 G20K full state model differs from its frozen node_20000 checkpoint")

    target_dir = root / ARTIFACT / "checkpoints" / ARM
    current = target_dir / "training_state.pt"
    if not current.exists():
        if target_dir.exists() and any(target_dir.iterdir()):
            raise ValueError("V6.2 checkpoint directory is nonempty but has no full state")
        _atomic_copy(source_state, current)
    if _sha(current) == source_sha:
        current_validation = validate_full_training_state(root, current, baseline=True)
    else:
        current_validation = validate_full_training_state(root, current)
        if int(current_validation["game_count"]) < INITIAL_GAMES:
            raise ValueError("V6.2 current state is behind the registered V6 baseline")

    baseline_node = target_dir / "node_20000.pt"
    if not baseline_node.exists():
        _atomic_copy(source_node, baseline_node)
    elif _sha(baseline_node) != _sha(source_node):
        raise ValueError("V6.2 frozen baseline node was changed")
    result = {
        "format_version": 1,
        "source_checkpoint": source_validation,
        "staged_checkpoint": current_validation,
        "source_model_matches_node_20000": True,
        "v6_source_unchanged": _sha(source_state) == source_sha,
    }
    _atomic_json(root / ARTIFACT / "results/resume_validation.json", result)
    return result


def _history_row(root: Path, game_count: int) -> dict[str, Any]:
    game_id = f"{ARM}-game-{game_count - 1:06d}"
    path = root / ARTIFACT / f"results/training/{ARM}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if row.get("game_id") == game_id:
                return row
    raise ValueError(f"training metric missing for milestone game {game_id}")


def save_milestone(root: str | Path, target_games: int, previous_games: int) -> dict[str, Any]:
    root = Path(root).resolve()
    if int(target_games) not in MILESTONES:
        raise ValueError("V6.2 milestone is not registered")
    checkpoint_path = root / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(state["game_count"]) != int(target_games):
        raise ValueError("milestone checkpoint does not land on the registered game count")
    full_path = checkpoint_path.with_name(f"training_state_games_{target_games}.pt")
    _atomic_copy(checkpoint_path, full_path)
    model_path = checkpoint_path.with_name(f"node_games_{target_games}.pt")
    temp_model_path = model_path.with_suffix(".pt.tmp")
    torch.save({
        "format_version": 1, "arm_id": ARM, "board_size": 9, "n_in_row": 5,
        "simulations_per_move": 400, "environment_states": int(state["environment_states"]),
        "game_count": int(state["game_count"]), "games_completed": int(state["game_count"]),
        "optimizer_updates": int(state["update_count"]), "update_count": int(state["update_count"]),
        "model_state": state["model_state"], "config_sha256": state["config_sha256"],
        "source_training_state_sha256": _sha(full_path),
    }, temp_model_path)
    os.replace(temp_model_path, model_path)

    raw_dir = root / ARTIFACT / f"raw/selfplay/{ARM}"
    games = []
    for path in sorted(raw_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        numeric_id = int(str(row["game_id"]).rsplit("-", 1)[1])
        if int(previous_games) <= numeric_id < int(target_games):
            games.append(row)
    if len(games) != int(target_games) - int(previous_games):
        raise ValueError("the current V6.2 training interval has missing or duplicate game records")
    history_path = root / ARTIFACT / f"results/training/{ARM}.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    selected_history = [row for row in history
                        if int(str(row["game_id"]).rsplit("-", 1)[1]) >= int(previous_games)
                        and int(str(row["game_id"]).rsplit("-", 1)[1]) < int(target_games)]
    if len(selected_history) != len(games):
        raise ValueError("training metric rows do not match completed self-play games")
    prior_state_path = root / ARTIFACT / "results/milestones.json"
    prior = json.loads(prior_state_path.read_text(encoding="utf-8")) if prior_state_path.is_file() else {"milestones": []}
    prior_cumulative = (float(prior["milestones"][-1]["cumulative_training_cpu_seconds"])
                        if prior["milestones"] else 0.0)
    cumulative_new_cpu = float(state["cumulative_cpu_seconds"]) - float(
        registration_value(root, "resume_cumulative_cpu_seconds")
    )
    metric = {
        "games_completed": int(state["game_count"]),
        "training_states_cumulative": int(state["environment_states"]),
        "optimizer_updates_cumulative": int(state["update_count"]),
        "training_states_added_in_interval": sum(int(row["training_state_count"]) for row in games),
        "mean_selfplay_steps_in_interval": float(np.mean([row["move_count"] for row in games])),
        "mean_loss_in_interval": float(np.mean([row["loss"] for row in selected_history])),
        "mean_policy_loss_in_interval": float(np.mean([row.get("policy_loss", 0.0) for row in selected_history])),
        "mean_value_loss_in_interval": float(np.mean([row.get("value_loss", 0.0) for row in selected_history])),
        "mean_policy_entropy_in_interval": float(np.mean([row["entropy"] for row in selected_history])),
        "last_game_loss": float(selected_history[-1]["loss"]),
        "last_game_policy_loss": float(selected_history[-1].get("policy_loss", 0.0)),
        "last_game_value_loss": float(selected_history[-1].get("value_loss", 0.0)),
        "last_game_policy_entropy": float(selected_history[-1]["entropy"]),
        "learning_rate_multiplier": float(state["lr_multiplier"]),
        "cumulative_training_cpu_seconds": cumulative_new_cpu,
        "interval_training_cpu_seconds": _training_cpu_interval(cumulative_new_cpu, prior_cumulative),
        "full_checkpoint_path": str(full_path.relative_to(root)),
        "full_checkpoint_sha256": _sha(full_path),
        "model_checkpoint_path": str(model_path.relative_to(root)),
        "model_checkpoint_sha256": _sha(model_path),
    }
    if not any(int(row["games_completed"]) == int(target_games) for row in prior["milestones"]):
        prior["milestones"].append(metric)
    prior["milestones"].sort(key=lambda item: int(item["games_completed"]))
    _atomic_json(prior_state_path, prior)
    validate_full_training_state(root, full_path)
    return metric


def registration_value(root: Path, key: str) -> Any:
    registration = json.loads((root / ARTIFACT / "config/registration.json").read_text(encoding="utf-8"))
    if key in registration:
        return registration[key]
    return registration["compute"][key]


def train_to_games(
    root: str | Path = ROOT,
    *,
    target_games: int,
    session_cpu_seconds: float | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    if int(target_games) not in MILESTONES:
        raise ValueError("target game count is not a registered V6.2 checkpoint")
    if int(target_games) > 2_000:
        gate_path = root / ARTIFACT / "results/gate_2000.json"
        if not gate_path.is_file() or not json.loads(gate_path.read_text(encoding="utf-8")).get("passed"):
            raise ValueError("training beyond 2,000 games requires the pre-registered PUCT-400 advancement gate")
        if int(target_games) == 3_000:
            stage_path = root / ARTIFACT / "results/development/stage_games_2500.json"
            if not stage_path.is_file() or not json.loads(stage_path.read_text(encoding="utf-8")).get("complete"):
                raise ValueError("the 2,500-game checkpoint evaluation must complete before training to 3,000")
    state_path = root / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
    if not state_path.exists():
        prepare_v6_2(root)
    validate_full_training_state(root, state_path)
    state_before = torch.load(state_path, map_location="cpu", weights_only=False)
    before_games = int(state_before["game_count"])
    if before_games > int(target_games):
        raise ValueError("V6.2 training state is already beyond this requested target")
    if before_games == int(target_games):
        return save_milestone(root, int(target_games), _previous_milestone_game(root, int(target_games)))
    config = json.loads((root / V5_CONFIG).read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    used_cpu = total_cpu_used_seconds(root)
    total_cap = float(registration_value(root, "cpu_hard_cap_total_new_hours")) * 3600.0
    remaining_total_cpu = total_cap - used_cpu
    if remaining_total_cpu <= 0:
        raise RuntimeError("V6.2 cumulative new-CPU hard cap has been reached")
    allowed_cpu = remaining_total_cpu if session_cpu_seconds is None else min(
        remaining_total_cpu, float(session_cpu_seconds)
    )
    deadline = time.process_time() + allowed_cpu
    started = time.process_time()
    settings = train_reference_arm(
        root,
        arm=ARM,
        board_size=9,
        n_in_row=5,
        simulations=400,
        target_states=2**60,
        target_games=int(target_games),
        milestones=[],
        seed=590502,
        config=config,
        stop_at_cpu_seconds=deadline,
        artifact_dir=ARTIFACT,
        progress_experiment_id=registration_value(root, "experiment_id"),
    )
    session_cpu = time.process_time() - started
    state_after = torch.load(state_path, map_location="cpu", weights_only=False)
    progress = {
        "format_version": 1,
        "experiment_id": registration_value(root, "experiment_id"),
        "target_games": int(target_games),
        "games_completed": int(state_after["game_count"]),
        "training_states_cumulative": int(state_after["environment_states"]),
        "optimizer_updates_cumulative": int(state_after["update_count"]),
        "session_cpu_seconds": session_cpu,
        "cpu_used_before_session_seconds": used_cpu,
        "cpu_allowed_for_session_seconds": allowed_cpu,
        "total_cpu_cap_seconds": total_cap,
        "training_cpu_seconds_since_v6": float(state_after["cumulative_cpu_seconds"])
                                              - float(registration_value(root, "resume_cumulative_cpu_seconds")),
        "completed": bool(settings["completed"]),
        "checkpoint_sha256": _sha(state_path),
        "checkpoint_path": str(state_path.relative_to(root)),
    }
    _atomic_json(root / ARTIFACT / "results/training_progress.json", progress)
    experiment_progress = {
        "experiment_id": progress["experiment_id"],
        "phase": "self_play_training" if not progress["completed"] else "training_milestone_complete",
        "baseline_games": INITIAL_GAMES,
        "baseline_states": INITIAL_STATES,
        "baseline_updates": INITIAL_UPDATES,
        "games_completed": int(state_after["game_count"]),
        "training_states_cumulative": int(state_after["environment_states"]),
        "optimizer_updates_cumulative": int(state_after["update_count"]),
        "last_game_id": f"{ARM}-game-{int(state_after['game_count']) - 1:06d}",
        "next_game_id": f"{ARM}-game-{int(state_after['game_count']):06d}",
        "next_milestone_games": int(target_games),
        "training_state_sha256": _sha(state_path),
        "training_cpu_seconds_since_v6": progress["training_cpu_seconds_since_v6"],
        "cpu_used_before_session_seconds": used_cpu,
        "cpu_allowed_for_session_seconds": allowed_cpu,
        "session_cpu_seconds": session_cpu,
        "completed": bool(settings["completed"]),
        "resume_requires_full_state": True,
    }
    _atomic_json(root / ARTIFACT / "results/progress.json", experiment_progress)
    if int(state_after["game_count"]) == int(target_games):
        previous_games = _previous_milestone_game(root, int(target_games))
        progress["milestone"] = save_milestone(root, int(target_games), previous_games)
        _atomic_json(root / ARTIFACT / "results/training_progress.json", progress)
    return progress


def _previous_milestone_game(root: Path, target_games: int) -> int:
    prior = json.loads((root / ARTIFACT / "results/milestones.json").read_text(encoding="utf-8")) if (
        root / ARTIFACT / "results/milestones.json").is_file() else {"milestones": []}
    completed = [int(row["games_completed"]) for row in prior["milestones"]
                 if int(row["games_completed"]) < int(target_games)]
    return max(completed) if completed else INITIAL_GAMES


def _training_cpu_interval(cumulative_new_cpu_seconds: float,
                           previous_cumulative_new_cpu_seconds: float) -> float:
    """Return a milestone interval from values measured from the same V6 baseline."""
    current = float(cumulative_new_cpu_seconds)
    previous = float(previous_cumulative_new_cpu_seconds)
    if current < previous:
        raise ValueError("cumulative training CPU time cannot move backwards")
    return current - previous


def total_cpu_used_seconds(root: str | Path = ROOT) -> float:
    """Count V6.2-only training and persisted evaluation CPU toward the hard cap."""
    root = Path(root).resolve()
    state_path = root / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
    training_seconds = 0.0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        training_seconds = max(0.0, float(state["cumulative_cpu_seconds"])
                               - float(registration_value(root, "resume_cumulative_cpu_seconds")))
    match_seconds = 0.0
    match_root = root / ARTIFACT / "results/matches"
    if match_root.exists():
        seen: set[str] = set()
        for path in match_root.rglob("v6m-*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            game_id = str(row["game_id"])
            if game_id in seen:
                raise ValueError(f"duplicate match ID across V6.2 evaluation sets: {game_id}")
            seen.add(game_id)
            match_seconds += float(row.get("total_cpu_seconds", 0.0))
    tactic_seconds = 0.0
    tactic_root = root / ARTIFACT / "results/tactics"
    if tactic_root.exists():
        for path in tactic_root.glob("*.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            tactic_seconds += sum(float(row.get("search_cpu_seconds", 0.0))
                                  for row in result.get("rows", []))
    return training_seconds + match_seconds + tactic_seconds
