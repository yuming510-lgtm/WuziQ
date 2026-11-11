"""G V6.3: exact-state continuation from the frozen G2500 checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board
from .v5_training import load_training_checkpoint, train_reference_arm
from .v5_evaluation import generate_opening_pool, opening_key
from .v6_registration import _same
from .v6_2_training import _replay_digest, validate_full_training_state


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_3")
ARM = "reference400_empty"
SEED = 590502
BASE_COMMIT = "3630283639363a24c10f84e8473c3b7798261d32"
BASE_GAMES = 2500
BASE_STATES = 45284
BASE_UPDATES = 2499
TARGET_GAMES = 5000
MILESTONES = (3000, 3500, 4000, 4500, 5000)
SELFPLAY_SIMULATIONS = 400
MAX_NEW_CPU_SECONDS = 50 * 3600
TRAINING_CPU_RESERVE_SECONDS = 45 * 3600
SNAPSHOT_INTERVAL_GAMES = 250
SNAPSHOT_INTERVAL_CPU_SECONDS = 3 * 3600


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
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


def validate_training_state(path: str | Path, *, expected_games: int | None = None) -> dict[str, Any]:
    """Actually load model, Adam, replay, and all saved RNG states."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = json.loads((ROOT / ARTIFACT / "config/reference_training_config.json").read_text(encoding="utf-8"))
    if payload.get("config_sha256") != config.get("config_sha256"):
        raise ValueError("training state does not match the frozen V5 training configuration")
    if payload.get("arm_id") != ARM or int(payload.get("base_seed", -1)) != SEED:
        raise ValueError("training state arm or random seed differs from the registered V6.3 continuation")
    if expected_games is not None and int(payload.get("game_count", -1)) != int(expected_games):
        raise ValueError("training state game count differs from the registered boundary")
    if len(payload.get("replay_buffer", [])) != int(config["training"]["buffer_size"]):
        raise ValueError("full replay buffer is missing")
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
            "model_restored": _same(model.get_policy_param(), payload["model_state"]),
            "adam_restored": _same(model.optimizer.state_dict(), payload["optimizer_state"]),
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
        raise ValueError(f"training state restore validation failed: {checks}")
    return {
        "path": str(path), "sha256": sha_file(path),
        "game_count": int(payload["game_count"]),
        "environment_states": int(payload["environment_states"]),
        "optimizer_updates": int(payload["update_count"]),
        "optimizer_slots": len(payload["optimizer_state"]["state"]),
        "replay_rows": len(loaded[4]),
        "replay_sha256": _replay_digest(loaded[4]),
        "cumulative_training_cpu_seconds": float(payload["cumulative_cpu_seconds"]),
        **checks, "valid": True,
    }


def _all_opening_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.is_file():
        return keys
    data = json.loads(path.read_text(encoding="utf-8"))
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            moves = value.get("moves")
            if isinstance(moves, list) and moves and all(
                isinstance(m, dict) and {"row", "col", "player"} <= set(m) for m in moves
            ):
                keys.add(opening_key(moves, 9))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(data)
    return keys


def _historical_opening_keys(root: Path) -> set[str]:
    paths = [
        root / "research/g_v5/config/opening_schedule.json",
        root / "research/g_v6/config/opening_schedule.json",
        root / ARTIFACT / "../g_v6_2/config/opening_schedule.json",
        root / ARTIFACT / "../g_v6_2/config/v6_1_openings_reference.json",
    ]
    keys: set[str] = set()
    for path in paths:
        keys |= _all_opening_keys(path.resolve())
    return keys


def build_opening_schedule(root: str | Path = ROOT, *, seed: int = 630603, groups: int = 40) -> dict[str, Any]:
    root = Path(root).resolve()
    historical = _historical_opening_keys(root)
    used_seed = int(seed)
    while True:
        rows = generate_opening_pool(size=9, n_in_row=5, groups=int(groups), plies=6, seed=used_seed)
        keys = [opening_key(row["moves"], 9) for row in rows]
        if len(set(keys)) == len(rows) and not (set(keys) & historical):
            break
        used_seed += 1
        if used_seed - int(seed) > 100:
            raise RuntimeError("could not make a fresh opening pool outside all V5/V6/V6.2 openings")
    for index, row in enumerate(rows):
        row["opening_group_id"] = f"v6-3-blind-{used_seed}-{index:03d}"
        row["seed"] = used_seed + index
    schedule = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
        "board_size": 9, "n_in_row": 5,
        "candidate_search": {"type": "AlphaZero reference PUCT", "simulations": 400, "c_puct": 5.0},
        "opening_seed": used_seed, "opening_plies": 6, "paired_opening_groups": len(rows),
        "planned_opponents": ["A-rule", "B-minimax-50ms", "C-alpha-beta-50ms"],
        "models": ["G2500-frozen", "G5000-final"],
        "games_per_model_opponent": 80,
        "openings": rows,
        "validation": {
            "all_openings_legal_and_nonterminal": True,
            "unique_d4_openings": len(set(keys)),
            "historical_d4_overlap": 0,
            "same_openings_for_all_models_and_opponents": True,
        },
    }
    return schedule


def _save_model_node(training_state_path: Path, node_path: Path, game_count: int) -> None:
    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    if int(state["game_count"]) != int(game_count):
        raise ValueError(f"checkpoint copy expected G{game_count}, got G{state['game_count']}")
    node = {
        "format_version": 1, "arm_id": ARM, "board_size": 9, "n_in_row": 5,
        "simulations_per_move": 400, "environment_states": int(state["environment_states"]),
        "game_count": int(state["game_count"]), "games_completed": int(state["game_count"]),
        "optimizer_updates": int(state["update_count"]), "update_count": int(state["update_count"]),
        "model_state": state["model_state"], "config_sha256": state["config_sha256"],
        "source_training_state_sha256": sha_file(training_state_path),
    }
    node_path.parent.mkdir(parents=True, exist_ok=True)
    temp = node_path.with_suffix(node_path.suffix + ".tmp")
    torch.save(node, temp)
    os.replace(temp, node_path)


def prepare_experiment(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    base_state = root / "research/g_v6_2/checkpoints/reference400_empty/training_state_games_2500.pt"
    base_node = root / "research/g_v6_2/checkpoints/reference400_empty/node_games_2500.pt"
    if not base_state.is_file() or not base_node.is_file():
        raise FileNotFoundError("the exact V6.2 G2500 full checkpoint/model is missing")
    baseline_validation = validate_full_training_state(root, base_state)
    if (baseline_validation["game_count"], baseline_validation["environment_states"], baseline_validation["optimizer_updates"]) != (BASE_GAMES, BASE_STATES, BASE_UPDATES):
        raise ValueError("G2500 checkpoint counters differ from the user-specified continuation point")
    base_node_sha = sha_file(base_node)
    current_model = torch.load(base_state, map_location="cpu", weights_only=False)["model_state"]
    node_model = torch.load(base_node, map_location="cpu", weights_only=False)["model_state"]
    if not _same(current_model, node_model):
        raise ValueError("G2500 full-state checkpoint weights differ from the frozen model checkpoint")

    artifact = root / ARTIFACT
    config_dir = artifact / "config"
    checkpoint_dir = artifact / "checkpoints" / ARM
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_config = root / "research/g_v5/config/experiment.json"
    config = json.loads(source_config.read_text(encoding="utf-8"))
    target_config = config_dir / "reference_training_config.json"
    if target_config.exists() and target_config.read_bytes() != source_config.read_bytes():
        raise ValueError("V6.3 must not change the frozen V5 training configuration")
    if not target_config.exists():
        atomic_copy(source_config, target_config)

    current_path = checkpoint_dir / "training_state.pt"
    baseline_state_copy = checkpoint_dir / "training_state_games_2500.pt"
    baseline_model_copy = checkpoint_dir / "node_games_2500.pt"
    if current_path.exists():
        current = torch.load(current_path, map_location="cpu", weights_only=False)
        if int(current["game_count"]) < BASE_GAMES:
            raise ValueError("existing V6.3 state is behind its registered starting point")
    else:
        atomic_copy(base_state, current_path)
    if not baseline_state_copy.exists():
        atomic_copy(base_state, baseline_state_copy)
    elif sha_file(baseline_state_copy) != sha_file(base_state):
        raise ValueError("frozen G2500 full state copy has changed")
    if not baseline_model_copy.exists():
        atomic_copy(base_node, baseline_model_copy)
    elif sha_file(baseline_model_copy) != base_node_sha:
        raise ValueError("frozen G2500 model copy has changed")
    current_validation = validate_training_state(current_path)
    schedule_path = config_dir / "opening_schedule.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    else:
        schedule = build_opening_schedule(root)
        atomic_json(schedule_path, schedule)
    if int(schedule["paired_opening_groups"]) != 40 or schedule["validation"]["historical_d4_overlap"] != 0:
        raise ValueError("V6.3 challenge opening schedule is not the preregistered fresh 40-pair pool")
    progress_path = artifact / "results/v6_3_progress.json"
    if not progress_path.exists():
        atomic_json(progress_path, {
            "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
            "games_completed": BASE_GAMES, "training_states_cumulative": BASE_STATES,
            "optimizer_updates_cumulative": BASE_UPDATES,
            "cumulative_new_training_cpu_seconds": 0.0,
            "phase": "ready_at_g2500", "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    snapshot_status = artifact / "results/snapshot_status.json"
    if not snapshot_status.exists():
        atomic_json(snapshot_status, {
            "format_version": 1, "drive_folder_name": "WuziQ G V6.3 Recovery",
            "drive_folder_id": None, "verified_snapshots": [],
            "policy": "retain latest two verified V6.3 snapshots only",
        })
    reg_path = config_dir / "registration.json"
    if reg_path.exists():
        registration = json.loads(reg_path.read_text(encoding="utf-8"))
    else:
        registration = {
            "format_version": 1,
            "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
            "branch": "research/alphazero-g-v6-3-5k",
            "base_commit": BASE_COMMIT,
            "source_branch": "research/alphazero-g-v6-2-selfplay-scale",
            "source_checkpoint": str(base_state.relative_to(root)),
            "source_checkpoint_sha256": sha_file(base_state),
            "source_model_sha256": base_node_sha,
            "source_games": BASE_GAMES,
            "source_states": BASE_STATES,
            "source_updates": BASE_UPDATES,
            "source_cumulative_training_cpu_seconds": float(torch.load(base_state, map_location="cpu", weights_only=False)["cumulative_cpu_seconds"]),
            "seed": SEED,
            "board_size": 9, "n_in_row": 5,
            "training_simulations_per_move": 400,
            "reference_config_sha256": config["config_sha256"],
            "training_config_is_unmodified_v5_reference": True,
            "selfplay_opening": "empty board; identical to V6.2 source",
            "target_games_cumulative": TARGET_GAMES,
            "training_milestones_games": list(MILESTONES),
            "snapshot_policy": {
                "interval_games": SNAPSHOT_INTERVAL_GAMES,
                "interval_cpu_seconds": SNAPSHOT_INTERVAL_CPU_SECONDS,
                "trigger": "first of 250 completed games or 3 cumulative new training CPU hours",
                "keep_latest_verified": 2,
            },
            "cpu_budget_new_experiment_hours": 50,
            "training_cpu_budget_hours": 45,
            "challenge": {
                "models": ["G2500-frozen", "G5000-final"],
                "opponents": ["A-rule", "B-pure-minimax-50ms", "C-alpha-beta-50ms"],
                "puct_simulations": 400,
                "paired_openings": 40,
                "games_per_model_opponent": 80,
                "total_games": 480,
                "opening_schedule_sha256": sha_bytes(json.dumps(schedule, ensure_ascii=False, sort_keys=True).encode()),
                "opponent_configs": {
                    "A": {"engine": "src.ai.choose_move model A", "time_limit_seconds": 1e9, "note": "rules AI natural decision time; no artificial wait"},
                    "B": {"engine": "src.ai.choose_move model B (pure Minimax)", "time_limit_seconds": 0.05},
                    "C": {"engine": "src.ai.choose_move model C (Alpha-Beta)", "time_limit_seconds": 0.05},
                },
            },
            "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(reg_path, registration)
    return {
        "baseline_full_state": baseline_validation,
        "frozen_model_sha256": base_node_sha,
        "staged_current_state": current_validation,
        "opening_schedule_sha256": sha_bytes(json.dumps(schedule, ensure_ascii=False, sort_keys=True).encode()),
        "openings": schedule["validation"],
        "registration": registration,
    }


def run_training_batch(root: str | Path = ROOT, *, batch_games: int = SNAPSHOT_INTERVAL_GAMES,
                       batch_cpu_seconds: float = SNAPSHOT_INTERVAL_CPU_SECONDS) -> dict[str, Any]:
    root = Path(root).resolve()
    torch.set_num_threads(1)
    artifact = root / ARTIFACT
    config = json.loads((artifact / "config/reference_training_config.json").read_text(encoding="utf-8"))
    state_path = artifact / "checkpoints" / ARM / "training_state.pt"
    current_validation = validate_training_state(state_path)
    current_games = int(current_validation["game_count"])
    prior_progress = json.loads((artifact / "results/v6_3_progress.json").read_text(encoding="utf-8"))
    training_started_at = prior_progress.get("training_started_at_utc")
    if current_games >= TARGET_GAMES:
        return {"stopped": "target_reached", **current_validation}
    baseline = float(torch.load(artifact / "checkpoints" / ARM / "training_state_games_2500.pt",
                                map_location="cpu", weights_only=False)["cumulative_cpu_seconds"])
    added_cpu = float(current_validation["cumulative_training_cpu_seconds"]) - baseline
    remaining_train_budget = TRAINING_CPU_RESERVE_SECONDS - added_cpu
    if remaining_train_budget <= 0:
        return {"stopped": "training_cpu_budget_reserve", **current_validation}
    stop_after = min(float(batch_cpu_seconds), remaining_train_budget)
    next_milestone = min((value for value in MILESTONES if value > current_games), default=TARGET_GAMES)
    chunk_target = min(current_games + int(batch_games), next_milestone, TARGET_GAMES)
    start_cpu = time.process_time()
    result = train_reference_arm(
        root, arm=ARM, board_size=9, n_in_row=5, simulations=SELFPLAY_SIMULATIONS,
        target_states=10**12, target_games=chunk_target, milestones=[], seed=SEED,
        config=config, stop_at_cpu_seconds=start_cpu + stop_after,
        artifact_dir=ARTIFACT,
        progress_experiment_id="wuziq-alphazero-g-v6-3-5k-continuation",
    )
    final_validation = validate_training_state(state_path)
    game_count = int(final_validation["game_count"])
    if game_count < current_games or game_count > chunk_target:
        raise ValueError("V6.3 training batch did not advance monotonically within its registered boundary")
    if game_count in MILESTONES:
        _save_model_node(state_path, artifact / "checkpoints" / ARM / f"node_games_{game_count}.pt", game_count)
        atomic_copy(state_path, artifact / "checkpoints" / ARM / f"training_state_games_{game_count}.pt")
    added_cpu_after = float(final_validation["cumulative_training_cpu_seconds"]) - baseline
    progress = {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
        "games_completed": game_count, "training_states_cumulative": final_validation["environment_states"],
        "optimizer_updates_cumulative": final_validation["optimizer_updates"],
        "cumulative_new_training_cpu_seconds": added_cpu_after,
        "last_batch_cpu_seconds": time.process_time() - start_cpu,
        "last_batch_completed_games": game_count - current_games,
        "checkpoint_sha256": final_validation["sha256"],
        "training_started_at_utc": training_started_at,
        "phase": "self_play_training" if game_count < TARGET_GAMES else "training_complete",
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(artifact / "results/v6_3_progress.json", progress)
    result.update({"previous_games": current_games, "added_training_cpu_seconds": added_cpu_after - added_cpu,
                   "milestone_saved": game_count in MILESTONES, "progress": progress})
    return result


def benchmark(root: str | Path = ROOT) -> dict[str, Any]:
    """Measure the frozen reference PUCT-400 move on a real opening position."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    node_path = root / ARTIFACT / "checkpoints" / ARM / "node_games_2500.pt"
    node = torch.load(node_path, map_location="cpu", weights_only=False)
    model = ReferencePolicyValueNet(9, seed=SEED)
    model.load_policy_param(node["model_state"])
    model.network.eval()
    player = instrumented_player(model.policy_value_fn, c_puct=5.0, n_playout=401, is_selfplay=False)
    board = make_board(9, 5, 0)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    move, _ = player.get_action(board, temp=1e-3, return_prob=1)
    elapsed_cpu, elapsed_wall = time.process_time() - started_cpu, time.perf_counter() - started_wall
    return {
        "board": "9x9 open board", "simulations_requested": 400,
        "upstream_configured_playouts": 401, "chosen_move": int(move),
        "measured_cpu_seconds": elapsed_cpu, "measured_wall_seconds": elapsed_wall,
        "search_stats": player.last_search_stats,
        "historical_latest_500_games_cpu_seconds": 22416.035950849,
        "historical_latest_500_games_rate_seconds_per_game": 22416.035950849 / 500,
    }


def preflight(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    bench = benchmark(root)
    rate = float(bench["historical_latest_500_games_rate_seconds_per_game"])
    training_normal = rate * (TARGET_GAMES - BASE_GAMES)
    training_optimistic = training_normal * 0.80
    training_conservative = training_normal * 1.30
    match_baseline = json.loads((root / "research/g_v6_2/results/final_blind.json").read_text(encoding="utf-8"))
    match400 = match_baseline["matches"]["400"]
    avg_a = float(np.mean([row["candidate_cpu_seconds"] for row in match400["A"]["game_rows"]]))
    avg_b = float(np.mean([row["candidate_cpu_seconds"] for row in match400["B-50ms"]["game_rows"]]))
    match_estimate = 160 * (avg_a + 2 * avg_b)
    estimates = {
        "training_cpu_hours": {
            "optimistic": training_optimistic / 3600,
            "normal": training_normal / 3600,
            "conservative": training_conservative / 3600,
        },
        "final_480_match_cpu_hours": {
            "optimistic": match_estimate * 0.75 / 3600,
            "normal": match_estimate / 3600,
            "conservative": match_estimate * 1.50 / 3600,
        },
        "total_cpu_hours": {
            "optimistic": (training_optimistic + match_estimate * 0.75) / 3600,
            "normal": (training_normal + match_estimate) / 3600,
            "conservative": (training_conservative + match_estimate * 1.50) / 3600,
        },
    }
    try:
        cpu_max = (root / "/sys/fs/cgroup/cpu.max").read_text().split()
        cpu_quota = float(cpu_max[0]) / float(cpu_max[1])
    except Exception:
        cpu_quota = float(os.cpu_count() or 1)
    report = {
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_logical_count": os.cpu_count(), "cpu_quota_cores": cpu_quota,
        "effective_training_parallelism": 1.0,
        "wall_estimate_basis": "the reference self-play/training loop is one serial worker with PyTorch threads fixed at 1; CPU quota does not halve its wall time",
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "memory_limit_bytes": 8 * 1024**3, "workspace_disk_free_bytes": shutil.disk_usage(root).free,
        "single_decision_benchmark": bench,
        "actual_training_rate_basis": "V6.2 cumulative trainer CPU from G2000 to G2500: 22,416.035950849 CPU seconds / 500 games; includes self-play, per-game optimizer update, and checkpoint time.",
        "match_cost_basis": {
            "A_candidate_cpu_seconds_per_game": avg_a,
            "B_50ms_candidate_cpu_seconds_per_game": avg_b,
            "C_estimated_equal_to_B": True,
            "total_matches": 480,
        },
        "estimates": estimates,
        "cpu_hard_cap_hours": 50,
        "planned_stage_caps": {"training_hours": 45, "evaluation_plus_audit_hours": 5},
        "wall_clock_hours_at_quota_if_cpu_bound": {
            key: value / 1.0 for key, value in estimates["total_cpu_hours"].items()
        },
    }
    atomic_json(root / ARTIFACT / "results/preflight.json", report)
    return report

