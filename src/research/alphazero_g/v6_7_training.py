"""V6.7 orchestration for an unchanged V6.5 AlphaZero training loop."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from .v5_reference import ReferencePolicyValueNet
from .v5_training import train_reference_arm
from . import v6_5_training as v65

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_7")
ARM = "reference400_empty"
BASE_GAMES = 8_000
TARGET_GAMES = 20_000
MILESTONES = (10_000, 12_000, 16_000, 20_000)
EVAL_OPPONENTS = ("G8000", "A", "C@50ms", "C@200ms")
SEED = 590_502
SIMULATIONS = 400
SNAPSHOT_INTERVAL_GAMES = 250
SNAPSHOT_INTERVAL_CPU_SECONDS = 3 * 3600
TOTAL_CPU_BUDGET_SECONDS = 110 * 3600
TRAINING_CPU_BUDGET_SECONDS = 100 * 3600
SOURCE_MODEL_SHA256 = "490b70eabb75de1b076e1eb9d13a9d82d2ccc213f3b769bfe6c98eddcb679272"
SOURCE_STATE_SHA256 = "6b56dee19e66365fd3623a80817e397eb71875861e1f28be8809447daef88a82"
SOURCE_ARCHIVE_SHA256 = "0b80829fef0b62c7bef653c5c6a9afa0ac0bbd946c8491dce6297cf091d260e0"
SOURCE_COMMIT = "ba41fa266ef725815cdef44ebf753aedfaecba3a"
SOURCE_DRIVE_FOLDER_ID = "1u3556ink-VFf9a7AbS701flr0V0spSzQ"
SOURCE_DRIVE_ARCHIVE_ID = "1koTkXhGJyrvk09RC5gwhk0G1DSXyGrSA"
V67_DRIVE_FOLDER_ID = "1bkakjo5fdhgELAzyuOohDE8wynVBY1fh"


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


def registration_path(root: str | Path = ROOT) -> Path:
    return artifact_path(root) / "config" / "registration.json"


def progress_path(root: str | Path = ROOT) -> Path:
    return artifact_path(root) / "results" / "progress.json"


def accounting_path(root: str | Path = ROOT) -> Path:
    return artifact_path(root) / "results" / "cpu_accounting.json"


def validate_training_state(path: str | Path | None = None, *, root: str | Path = ROOT,
                            expected_games: int | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    path = Path(path or state_path(root))
    if not path.is_absolute():
        path = root / path
    result = v65.validate_training_state(path, root=root, expected_games=expected_games)
    games = int(result["game_count"])
    updates = int(result["optimizer_updates"])
    if not BASE_GAMES <= games <= TARGET_GAMES:
        raise ValueError(f"V6.7 checkpoint is outside G{BASE_GAMES}..G{TARGET_GAMES}: G{games}")
    if updates != games - 1:
        raise ValueError(f"optimizer updates do not reconcile with complete games: {updates} vs {games - 1}")
    if result["replay_rows"] != 10_000:
        raise ValueError("the original full 10,000-row replay buffer was not restored")
    return result


def next_batch_target(games: int) -> int:
    games = int(games)
    if games >= TARGET_GAMES:
        return TARGET_GAMES
    milestone = min((value for value in MILESTONES if value > games), default=TARGET_GAMES)
    return min(games + SNAPSHOT_INTERVAL_GAMES, milestone, TARGET_GAMES)


def snapshot_due(games: int, last_snapshot_games: int, total_cpu_seconds: float,
                 last_snapshot_total_cpu_seconds: float) -> bool:
    return (
        int(games) - int(last_snapshot_games) >= SNAPSHOT_INTERVAL_GAMES
        or float(total_cpu_seconds) - float(last_snapshot_total_cpu_seconds) >= SNAPSHOT_INTERVAL_CPU_SECONDS
    )


def _load_accounting(root: Path) -> dict[str, Any]:
    path = accounting_path(root)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"evaluation_cpu_seconds": 0.0, "audit_cpu_seconds": 0.0,
            "snapshot_packaging_cpu_seconds": 0.0, "evaluation_game_cpu": {}}


def record_aux_cpu(root: str | Path, category: str, seconds: float, *, unique_id: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    if category not in {"evaluation_cpu_seconds", "audit_cpu_seconds", "snapshot_packaging_cpu_seconds"}:
        raise ValueError("unsupported auxiliary CPU category")
    accounting = _load_accounting(root)
    if unique_id is not None:
        seen = accounting.setdefault("evaluation_game_cpu", {})
        if unique_id in seen:
            return accounting
        seen[unique_id] = float(seconds)
    accounting[category] = float(accounting.get(category, 0.0)) + float(seconds)
    atomic_json(accounting_path(root), accounting)
    return accounting


def total_new_cpu_seconds(root: str | Path = ROOT) -> float:
    root = Path(root).resolve()
    state = validate_training_state(state_path(root), root=root)
    baseline = json.loads((artifact_path(root) / "results" / "baseline.json").read_text(encoding="utf-8"))
    training_seconds = float(state["cumulative_training_cpu_seconds"]) - float(baseline["cumulative_training_cpu_seconds"])
    extra = _load_accounting(root)
    return training_seconds + sum(float(extra.get(key, 0.0)) for key in (
        "evaluation_cpu_seconds", "audit_cpu_seconds", "snapshot_packaging_cpu_seconds"))


def prepare_experiment(root: str | Path = ROOT, *, source_state_path: str | Path,
                       source_node_path: str | Path, drive_folder_id: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    source_state_path, source_node_path = Path(source_state_path), Path(source_node_path)
    if not source_state_path.is_file() or not source_node_path.is_file():
        raise FileNotFoundError("the downloaded V6.5 G8000 state and node are required")
    source_state_sha, source_node_sha = sha_file(source_state_path), sha_file(source_node_path)
    if source_state_sha != SOURCE_STATE_SHA256 or source_node_sha != SOURCE_MODEL_SHA256:
        raise ValueError("G8000 full-state or node SHA-256 does not match the registered V6.5 source")
    source_validation = v65.validate_training_state(source_state_path, root=root, expected_games=BASE_GAMES)
    if (source_validation["environment_states"], source_validation["optimizer_updates"],
            source_validation["replay_rows"]) != (139_884, 7_999, 10_000):
        raise ValueError("G8000 counts/replay do not match the registered continuation point")

    source_node = torch.load(source_node_path, map_location="cpu", weights_only=False)
    source_state = torch.load(source_state_path, map_location="cpu", weights_only=False)
    if not v65._state_equal(source_node["model_state"], source_state["model_state"]):
        raise ValueError("G8000 node weights differ from the full training-state weights")
    if int(source_node.get("simulations_per_move", -1)) != SIMULATIONS:
        raise ValueError("G8000 node search budget differs from frozen PUCT-400")

    artifact = artifact_path(root)
    config_dir = artifact / "config"
    checkpoint_dir = artifact / "checkpoints" / ARM
    models_dir = artifact / "models"
    for directory in (config_dir, checkpoint_dir, models_dir,
                      artifact / "raw/selfplay" / ARM,
                      artifact / "results/training", artifact / "results/evaluations/games",
                      artifact / "results/evaluations/summaries", artifact / "recovery"):
        directory.mkdir(parents=True, exist_ok=True)
    config_source = root / "research/g_v6_5/config/reference_training_config.json"
    config_dest = config_dir / "reference_training_config.json"
    config_bytes = config_source.read_bytes()
    if config_dest.exists() and config_dest.read_bytes() != config_bytes:
        raise ValueError("V6.7 config differs from the frozen V6.5 reference config")
    if not config_dest.exists():
        config_dest.write_bytes(config_bytes)
    config = json.loads(config_bytes)
    if config.get("config_sha256") != source_state["config_sha256"]:
        raise ValueError("G8000 state does not match the frozen V6.5 training configuration")

    baseline_full = checkpoint_dir / "training_state_games_8000.pt"
    current = state_path(root)
    if current.exists():
        existing = validate_training_state(current, root=root)
        if existing["game_count"] < BASE_GAMES:
            raise ValueError("existing V6.7 state is behind the G8000 source")
    else:
        atomic_copy(source_state_path, current)
    if not baseline_full.exists():
        atomic_copy(source_state_path, baseline_full)
    baseline_node_dest = models_dir / "G8000.pt"
    if baseline_node_dest.exists() and sha_file(baseline_node_dest) != SOURCE_MODEL_SHA256:
        raise ValueError("the frozen G8000 model was modified")
    if not baseline_node_dest.exists():
        atomic_copy(source_node_path, baseline_node_dest)
    state_validation = validate_training_state(current, root=root)

    registration = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
        "branch": "research/alphazero-g-v6-7-selfplay-20k",
        "base_commit": SOURCE_COMMIT,
        "source_branch": "research/alphazero-g-v6-5-selfplay-scale",
        "source_games": BASE_GAMES,
        "source_states": 139_884,
        "source_updates": 7_999,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_training_state_sha256": SOURCE_STATE_SHA256,
        "source_v6_5_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_drive_folder_id": SOURCE_DRIVE_FOLDER_ID,
        "source_drive_archive_id": SOURCE_DRIVE_ARCHIVE_ID,
        "source_validation": {key: value for key, value in source_validation.items() if key != "path"},
        "source_restore_relative_path": "research/g_v6_7/checkpoints/reference400_empty/training_state_games_8000.pt",
        "training_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "reference_config_sha256": config["config_sha256"],
        "board_size": 9, "n_in_row": 5, "simulations_per_move": SIMULATIONS,
        "seed": SEED,
        "target_games_cumulative": TARGET_GAMES,
        "additional_games": TARGET_GAMES - BASE_GAMES,
        "milestones_games": list(MILESTONES),
        "training_algorithm_changed": False,
        "training_input": "same-network AlphaZero self-play only; no teacher data, handcrafted reward, or inference rule patch",
        "training_limits": {"total_new_cpu_hours": 110, "training_cpu_reserve_hours": 100},
        "snapshot_policy": {"trigger": "first of 250 games or 3 CPU hours",
                            "rolling_to_retain": 2,
                            "permanent_milestones": list(MILESTONES),
                            "drive_folder_id": drive_folder_id or V67_DRIVE_FOLDER_ID},
        "evaluation_protocol": {
            "development_opening_seed": 670_701,
            "blind_opening_seed": 670_702,
            "opening_pairs_per_condition": 10,
            "opening_plies": 6,
            "opponents": ["G8000", "A", "C@50ms", "C@200ms"],
            "c_protocol": {"algorithm": "Alpha-Beta", "max_depth": 69,
                           "candidate_radius": 2, "timeout_returns_last_complete_depth": True},
            "g_protocol": "same original 9x9 PUCT-400 reference engine",
            "blind_only_at_g20000_after_development_evaluation": True,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    reg_path = registration_path(root)
    if reg_path.exists():
        previous = json.loads(reg_path.read_text(encoding="utf-8"))
        for key in ("experiment_id", "base_commit", "source_model_sha256", "source_training_state_sha256",
                    "training_config_sha256", "target_games_cumulative", "milestones_games"):
            if previous.get(key) != registration.get(key):
                raise ValueError(f"existing V6.7 registration differs in frozen field {key}")
        registration = previous
    else:
        atomic_json(reg_path, registration)

    baseline_path = artifact / "results/baseline.json"
    if not baseline_path.exists():
        atomic_json(baseline_path, {
            "games": BASE_GAMES,
            "environment_states": int(source_validation["environment_states"]),
            "optimizer_updates": int(source_validation["optimizer_updates"]),
            "cumulative_training_cpu_seconds": float(source_validation["cumulative_training_cpu_seconds"]),
            "model_sha256": SOURCE_MODEL_SHA256,
            "training_state_sha256": SOURCE_STATE_SHA256,
        })
    historical_path = artifact / "results/historical_g5000_cpu_baseline.json"
    if not historical_path.exists():
        g5000_state_path = source_state_path.parent / "training_state_games_5000.pt"
        if g5000_state_path.is_file():
            g5000_state = v65.validate_training_state(g5000_state_path, root=root, expected_games=5_000)
            atomic_json(historical_path, {
                "games": 5_000, "environment_states": g5000_state["environment_states"],
                "optimizer_updates": g5000_state["optimizer_updates"],
                "cumulative_training_cpu_seconds": g5000_state["cumulative_training_cpu_seconds"],
                "training_state_sha256": g5000_state["sha256"],
                "verified_from_same_v6_5_source_archive": True,
            })
    progress = progress_path(root)
    if progress.exists():
        prior = json.loads(progress.read_text(encoding="utf-8"))
        if int(prior.get("games_completed", -1)) > int(state_validation["game_count"]):
            raise ValueError("V6.7 progress is ahead of the verified training checkpoint")
    else:
        atomic_json(progress, {
            "experiment_id": registration["experiment_id"],
            "games_completed": BASE_GAMES,
            "training_states_cumulative": int(state_validation["environment_states"]),
            "optimizer_updates_cumulative": int(state_validation["optimizer_updates"]),
            "training_state_sha256": state_validation["sha256"],
            "last_snapshot_games": BASE_GAMES,
            "last_snapshot_total_cpu_seconds": 0.0,
            "phase": "ready_at_g8000",
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    accounting = accounting_path(root)
    if not accounting.exists():
        atomic_json(accounting, {"evaluation_cpu_seconds": 0.0, "audit_cpu_seconds": 0.0,
                                 "snapshot_packaging_cpu_seconds": 0.0, "evaluation_game_cpu": {}})
    return {"source_validation": source_validation, "staged_state": state_validation,
            "registration": registration, "source_model_sha256": SOURCE_MODEL_SHA256,
            "source_training_state_sha256": SOURCE_STATE_SHA256}


def _save_milestone(root: Path, game_count: int) -> dict[str, Any]:
    path = state_path(root)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state["game_count"]) != int(game_count):
        raise ValueError("milestone save requested away from an exact game boundary")
    checkpoint_dir = artifact_path(root) / "checkpoints" / ARM
    full_path = checkpoint_dir / f"training_state_games_{game_count}.pt"
    if not full_path.exists() or sha_file(full_path) != sha_file(path):
        atomic_copy(path, full_path)
    node = {
        "format_version": 1, "arm_id": ARM, "board_size": 9, "n_in_row": 5,
        "simulations_per_move": SIMULATIONS,
        "environment_states": int(state["environment_states"]),
        "game_count": int(game_count), "games_completed": int(game_count),
        "optimizer_updates": int(state["update_count"]), "update_count": int(state["update_count"]),
        "model_state": state["model_state"], "config_sha256": state["config_sha256"],
        "source_training_state_sha256": sha_file(path),
    }
    node_path = checkpoint_dir / f"node_games_{game_count}.pt"
    if not node_path.exists():
        temp = node_path.with_suffix(".pt.tmp")
        torch.save(node, temp)
        os.replace(temp, node_path)
    model_path = artifact_path(root) / "models" / f"G{game_count}.pt"
    if not model_path.exists():
        atomic_copy(node_path, model_path)
    validation = validate_training_state(full_path, root=root, expected_games=game_count)
    if validation["sha256"] != sha_file(path):
        raise ValueError("milestone full-state copy differs from the live checkpoint")
    history_path = artifact_path(root) / "results/training" / f"{ARM}.jsonl"
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lo = BASE_GAMES
    if game_count == BASE_GAMES:
        selected = []
    else:
        selected = [row for row in rows if lo <= int(row["game_id"].rsplit("-", 1)[-1]) < game_count]
    previous_boundary = max((value for value in MILESTONES if value < game_count), default=BASE_GAMES)
    interval_rows = [row for row in rows
                     if previous_boundary <= int(row["game_id"].rsplit("-", 1)[-1]) < game_count]
    stage = {
        "games": int(game_count), "training_states_cumulative": int(validation["environment_states"]),
        "optimizer_updates_cumulative": int(validation["optimizer_updates"]),
        "mean_loss_since_g8000": float(np.mean([r.get("loss", 0.0) for r in selected])) if selected else None,
        "mean_policy_loss_since_g8000": float(np.mean([r.get("policy_loss", 0.0) for r in selected])) if selected else None,
        "mean_value_loss_since_g8000": float(np.mean([r.get("value_loss", 0.0) for r in selected])) if selected else None,
        "mean_policy_entropy_since_g8000": float(np.mean([r.get("entropy", 0.0) for r in selected])) if selected else None,
        "interval_start_games": previous_boundary,
        "interval_games": len(interval_rows),
        "interval_mean_loss": float(np.mean([r.get("loss", 0.0) for r in interval_rows])) if interval_rows else None,
        "interval_mean_policy_loss": float(np.mean([r.get("policy_loss", 0.0) for r in interval_rows])) if interval_rows else None,
        "interval_mean_value_loss": float(np.mean([r.get("value_loss", 0.0) for r in interval_rows])) if interval_rows else None,
        "interval_mean_policy_entropy": float(np.mean([r.get("entropy", 0.0) for r in interval_rows])) if interval_rows else None,
        "new_training_cpu_seconds": float(validation["cumulative_training_cpu_seconds"]
                                           - json.loads((artifact_path(root) / "results/baseline.json").read_text())["cumulative_training_cpu_seconds"]),
        "training_state_sha256": validation["sha256"], "model_node_sha256": sha_file(model_path),
        "full_state_restoration_valid": True,
    }
    stages_path = artifact_path(root) / "results/stage_metrics.json"
    existing = json.loads(stages_path.read_text(encoding="utf-8")) if stages_path.exists() else {"stages": []}
    existing["stages"] = [r for r in existing.get("stages", []) if int(r["games"]) != int(game_count)] + [stage]
    existing["stages"].sort(key=lambda r: int(r["games"]))
    atomic_json(stages_path, existing)
    return stage


def run_training_batch(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    torch.set_num_threads(1)
    artifact = artifact_path(root)
    config = json.loads((artifact / "config/reference_training_config.json").read_text(encoding="utf-8"))
    current_validation = validate_training_state(state_path(root), root=root)
    current_games = int(current_validation["game_count"])
    progress = json.loads(progress_path(root).read_text(encoding="utf-8"))
    accounting = _load_accounting(root)
    baseline = json.loads((artifact / "results/baseline.json").read_text(encoding="utf-8"))
    training_added = float(current_validation["cumulative_training_cpu_seconds"]) - float(baseline["cumulative_training_cpu_seconds"])
    other_cpu = sum(float(accounting.get(key, 0.0)) for key in (
        "evaluation_cpu_seconds", "audit_cpu_seconds", "snapshot_packaging_cpu_seconds"))
    total_added = training_added + other_cpu
    if current_games >= TARGET_GAMES:
        return {"stopped": "target_reached", "state": current_validation,
                "total_new_cpu_seconds": total_added}
    remaining_total = TOTAL_CPU_BUDGET_SECONDS - total_added
    remaining_training = TRAINING_CPU_BUDGET_SECONDS - training_added
    last_snapshot_cpu = float(progress.get("last_snapshot_total_cpu_seconds", 0.0))
    since_snapshot_cpu = total_added - last_snapshot_cpu
    until_snapshot = SNAPSHOT_INTERVAL_CPU_SECONDS - since_snapshot_cpu
    if until_snapshot <= 0:
        return {"stopped": "snapshot_due", "snapshot_due": True,
                "state": current_validation, "training_cpu_seconds": training_added,
                "total_new_cpu_seconds": total_added}
    allowed = min(remaining_total, remaining_training, until_snapshot)
    if allowed <= 1:
        return {"stopped": "cpu_budget_exhausted", "state": current_validation,
                "training_cpu_seconds": training_added, "total_new_cpu_seconds": total_added}
    target = next_batch_target(current_games)
    started_cpu = time.process_time()
    result = train_reference_arm(
        root, arm=ARM, board_size=9, n_in_row=5, simulations=SIMULATIONS,
        target_states=10**12, target_games=target, milestones=[], seed=SEED,
        config=config, stop_at_cpu_seconds=started_cpu + min(SNAPSHOT_INTERVAL_CPU_SECONDS, allowed),
        artifact_dir=ARTIFACT,
        progress_experiment_id="wuziq-alphazero-g-v6-7-selfplay-20k",
    )
    updated = validate_training_state(state_path(root), root=root)
    games_after = int(updated["game_count"])
    if not current_games < games_after <= target:
        raise ValueError(f"V6.7 batch failed to advance monotonically: G{current_games} -> G{games_after}, target G{target}")
    milestone = games_after in MILESTONES
    stage = _save_milestone(root, games_after) if milestone else None
    training_added_after = float(updated["cumulative_training_cpu_seconds"]) - float(baseline["cumulative_training_cpu_seconds"])
    other_cpu_after = sum(float(accounting.get(key, 0.0)) for key in (
        "evaluation_cpu_seconds", "audit_cpu_seconds", "snapshot_packaging_cpu_seconds"))
    last_snapshot_games = int(progress.get("last_snapshot_games", BASE_GAMES))
    last_snapshot_cpu = float(progress.get("last_snapshot_total_cpu_seconds", 0.0))
    due = snapshot_due(games_after, last_snapshot_games,
                       training_added_after + other_cpu_after, last_snapshot_cpu)
    updated_progress = {
        **progress,
        "games_completed": games_after,
        "training_states_cumulative": int(updated["environment_states"]),
        "optimizer_updates_cumulative": int(updated["optimizer_updates"]),
        "training_state_sha256": updated["sha256"],
        "new_training_cpu_seconds": training_added_after,
        "total_new_cpu_seconds": training_added_after + other_cpu_after,
        "last_batch_completed_games": games_after - current_games,
        "last_batch_cpu_seconds": time.process_time() - started_cpu,
        "snapshot_due": due,
        "milestone": milestone,
        "phase": "training_complete" if games_after == TARGET_GAMES else "self_play_training",
        "completed": games_after == TARGET_GAMES,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(progress_path(root), updated_progress)
    progress_file = artifact / "results" / "training_progress.json"
    atomic_json(progress_file, {**updated_progress, "active_arm": ARM, "replay_rows": updated["replay_rows"]})
    return {"previous_games": current_games, "games_completed": games_after,
            "training_states_cumulative": updated["environment_states"],
            "optimizer_updates_cumulative": updated["optimizer_updates"],
            "new_training_cpu_seconds": training_added_after,
            "total_new_cpu_seconds": training_added_after + other_cpu_after,
            "milestone": stage, "snapshot_due": due,
            "target_games": target, "training_result": result,
            "training_state_sha256": updated["sha256"], "progress": updated_progress}


def mark_snapshot_verified(root: str | Path, *, games: int, total_cpu_seconds: float,
                           snapshot_record: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    progress = json.loads(progress_path(root).read_text(encoding="utf-8"))
    if int(progress["games_completed"]) != int(games):
        raise ValueError("cannot mark a different checkpoint as remotely saved")
    if not snapshot_record.get("empty_directory_restore") or snapshot_record.get("sha256") != snapshot_record.get("downloaded_sha256"):
        raise ValueError("unverified Drive snapshot cannot advance progress")
    updated = {**progress, "last_snapshot_games": int(games),
               "last_snapshot_total_cpu_seconds": float(total_cpu_seconds),
               "last_verified_snapshot": snapshot_record,
               "snapshot_due": False,
               "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(progress_path(root), updated)
    return updated


def preflight_resources(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    import subprocess

    def command(args: list[str]) -> str:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()

    cpu_quota = Path("/sys/fs/cgroup/cpu.max")
    memory_max = Path("/sys/fs/cgroup/memory.max")
    disk = shutil.disk_usage(root)
    return {
        "cpu_count_visible": os.cpu_count(),
        "cpu_quota": cpu_quota.read_text().strip() if cpu_quota.exists() else None,
        "memory_limit_bytes": memory_max.read_text().strip() if memory_max.exists() else None,
        "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
        "python": command([os.sys.executable, "--version"]),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "training_budget_cpu_hours": TRAINING_CPU_BUDGET_SECONDS / 3600,
        "total_budget_cpu_hours": TOTAL_CPU_BUDGET_SECONDS / 3600,
    }


def historical_training_forecast(root: str | Path = ROOT) -> dict[str, float]:
    root = Path(root).resolve()
    baseline_path = artifact_path(root) / "results/historical_g5000_cpu_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source = torch.load(state_path(root).parent / "training_state_games_8000.pt",
                        map_location="cpu", weights_only=False)
    delta = float(source["cumulative_cpu_seconds"]) - float(baseline["cumulative_training_cpu_seconds"])
    rate = delta / 3_000
    expected = rate * (TARGET_GAMES - BASE_GAMES)
    return {"historical_games": 3_000, "historical_cpu_seconds": delta,
            "historical_cpu_seconds_per_game": rate,
            "remaining_games": TARGET_GAMES - BASE_GAMES,
            "expected_training_cpu_hours": expected / 3600,
            "optimistic_training_cpu_hours": expected * 0.8 / 3600,
            "conservative_training_cpu_hours": expected * 1.3 / 3600}

