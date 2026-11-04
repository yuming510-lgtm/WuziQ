"""Frozen V6 configuration, opening schedule, and V5 continuation verification."""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .v5_evaluation import generate_opening_pool, make_board, opening_key
from .v5_reference import ReferencePolicyValueNet
from .v5_training import load_training_checkpoint


EXPERIMENT_ID = "wuziq-alphazero-g-v6-9x9-400-equalized-training"
V5_COMMIT = "73aff7b2adb6d91b5890d6fef12ee7b0a43acc11"
V6_BRANCH = "research/alphazero-g-v6"
DEVELOPMENT_SEED = 696103
BLIND_SEED = 696104
DEVELOPMENT_PAIRS = 8
BLIND_PAIRS = 20
MILESTONES = [2_000, 5_000, 10_000, 15_000, 20_000]
TARGET_STATES = 20_000
SIMULATIONS_PER_MOVE = 400
EVALUATION_BUDGETS = [64, 400]
CPU_HARD_CAP_SECONDS = 10 * 3600


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _replay_digest(rows: list | deque) -> str:
    digest = hashlib.sha256()
    for state, policy, value in rows:
        digest.update(np.asarray(state).tobytes(order="C"))
        digest.update(np.asarray(policy).tobytes(order="C"))
        digest.update(float(value).hex().encode("ascii"))
    return digest.hexdigest()


def build_opening_schedule() -> dict[str, Any]:
    schedule: dict[str, Any] = {
        "format_version": 1,
        "development_seed": DEVELOPMENT_SEED,
        "blind_seed": BLIND_SEED,
        "opening_plies": 6,
        "development_pairs": DEVELOPMENT_PAIRS,
        "blind_pairs": BLIND_PAIRS,
        "development": generate_opening_pool(
            size=9, n_in_row=5, groups=DEVELOPMENT_PAIRS,
            plies=6, seed=DEVELOPMENT_SEED,
        ),
        "blind": generate_opening_pool(
            size=9, n_in_row=5, groups=BLIND_PAIRS,
            plies=6, seed=BLIND_SEED,
        ),
    }
    for label in ("development", "blind"):
        for index, opening in enumerate(schedule[label]):
            opening["opening_group_id"] = f"v6-{label}-{opening['seed']}-{index:03d}"
    validate_opening_schedule(schedule)
    return schedule


def validate_opening_schedule(schedule: dict[str, Any], root: str | Path | None = None) -> None:
    root = Path(root or Path(__file__).resolve().parents[3])
    if schedule.get("development_seed") != DEVELOPMENT_SEED or schedule.get("blind_seed") != BLIND_SEED:
        raise ValueError("V6 opening seeds differ from the frozen registration")
    if len(schedule.get("development", [])) != DEVELOPMENT_PAIRS:
        raise ValueError("V6 development opening count mismatch")
    if len(schedule.get("blind", [])) != BLIND_PAIRS:
        raise ValueError("V6 blind opening count mismatch")

    seen: set[str] = set()
    for label in ("development", "blind"):
        rows = schedule[label]
        keys: set[str] = set()
        for opening in rows:
            moves = opening.get("moves", [])
            if len(moves) != 6 or not opening.get("opening_group_id", "").startswith("v6-"):
                raise ValueError(f"malformed V6 {label} opening")
            board = make_board(9, 5, 0)
            expected = "B"
            for move in moves:
                color = str(move.get("player"))
                row, col = int(move["row"]), int(move["col"])
                if color != expected:
                    raise ValueError("V6 opening colors must alternate from black")
                action = row * 9 + col
                if action not in board.availables:
                    raise ValueError("V6 opening contains an illegal or repeated move")
                board.do_move(action)
                if board.game_end()[0]:
                    raise ValueError("V6 opening is terminal")
                expected = "W" if expected == "B" else "B"
            key = opening_key(moves, 9)
            if key in keys or key in seen:
                raise ValueError("V6 development/blind openings are not D4-distinct")
            keys.add(key)
        seen.update(keys)

    old_schedule_path = root / "research/g_v5/config/opening_schedule.json"
    if old_schedule_path.is_file():
        old = json.loads(old_schedule_path.read_text(encoding="utf-8"))["nine_by_nine"]
        old_keys = {
            opening_key(row["moves"], 9)
            for group in old.values() for row in group
        }
        if old_keys.intersection(seen):
            raise ValueError("V6 opening duplicates a V5 development/blind/historical opening")


def validate_v5_resume_checkpoint(root: str | Path) -> dict[str, Any]:
    """Load V5's full state through its loader and verify every restored component."""
    root = Path(root)
    v5_root = root / "research/g_v5"
    source_dir = v5_root / "checkpoints/reference400_empty"
    state_path = source_dir / "training_state.pt"
    milestone_path = source_dir / "node_1500.pt"
    config_path = v5_root / "config/experiment.json"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    milestone = torch.load(milestone_path, map_location="cpu", weights_only=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    canonical = dict(config)
    declared_config_hash = canonical.pop("config_sha256", None)
    computed_config_hash = hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if declared_config_hash != computed_config_hash:
        raise ValueError("V5 source config hash is invalid")
    if state.get("config_sha256") != declared_config_hash:
        raise ValueError("V5 training state is registered to another config")
    if state.get("arm_id") != "reference400_empty" or state.get("board_size") != 9:
        raise ValueError("V5 training state identity or board size mismatch")
    if int(state.get("environment_states", 0)) != int(milestone.get("environment_states", -1)):
        raise ValueError("V5 state and node_1500 milestone counts differ")
    if int(state.get("environment_states", 0)) < 1_500:
        raise ValueError("V5 400-simulation state does not reach its 1.5K milestone")
    if state.get("model_state", {}).keys() != milestone.get("model_state", {}).keys():
        raise ValueError("V5 milestone and complete state model keys differ")
    if not _same(state["model_state"], milestone["model_state"]):
        raise ValueError("V5 complete state model differs from node_1500")
    if not state.get("optimizer_state", {}).get("state"):
        raise ValueError("V5 Adam optimizer slots are missing")
    replay = state.get("replay_buffer")
    buffer_size = int(config["training"]["buffer_size"])
    if not isinstance(replay, list) or len(replay) != buffer_size:
        raise ValueError("V5 replay buffer is not complete")
    for row in (replay[0], replay[len(replay) // 2], replay[-1]):
        if len(row) != 3:
            raise ValueError("V5 replay item schema mismatch")
        sample_state, policy, value = row
        if np.asarray(sample_state).shape != (4, 9, 9) or np.asarray(policy).shape != (81,):
            raise ValueError("V5 replay sample dimensions mismatch")
        if (not np.isfinite(sample_state).all() or not np.isfinite(policy).all()
                or np.any(policy < -1e-6) or not np.isclose(policy.sum(), 1.0, atol=2e-3)
                or float(value) not in (-1.0, 0.0, 1.0)):
            raise ValueError("V5 replay sample data is invalid")

    model = ReferencePolicyValueNet(9, seed=int(state["base_seed"]))
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    previous_torch = torch.get_rng_state().clone()
    try:
        loaded = load_training_checkpoint(
            state_path, model=model, arm="reference400_empty",
            config_hash=declared_config_hash, buffer_size=buffer_size,
        )
        python_ok = random.getstate() == state["python_rng_state"]
        numpy_now = np.random.get_state()
        numpy_ok = (
            numpy_now[0] == state["numpy_rng_state"][0]
            and np.array_equal(numpy_now[1], state["numpy_rng_state"][1])
            and numpy_now[2:] == state["numpy_rng_state"][2:]
        )
        torch_ok = torch.equal(torch.get_rng_state(), state["torch_rng_state"])
        model_ok = _same(model.get_policy_param(), state["model_state"])
        optimizer_ok = _same(model.optimizer.state_dict(), state["optimizer_state"])
        replay_ok = _replay_digest(loaded[4]) == _replay_digest(replay)
        if not all((python_ok, numpy_ok, torch_ok, model_ok, optimizer_ok, replay_ok)):
            raise ValueError("V5 checkpoint failed exact state restore validation")
    finally:
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.set_rng_state(previous_torch)

    return {
        "valid": True,
        "arm_id": state["arm_id"],
        "board_size": int(state["board_size"]),
        "simulations_per_move": int(milestone["simulations_per_move"]),
        "environment_states": int(state["environment_states"]),
        "game_count": int(state["game_count"]),
        "optimizer_updates": int(state["update_count"]),
        "optimizer_slots": len(state["optimizer_state"]["state"]),
        "replay_rows": len(replay),
        "replay_sha256": _replay_digest(replay),
        "python_rng_restored": python_ok,
        "numpy_rng_restored": numpy_ok,
        "torch_rng_restored": torch_ok,
        "model_restored": model_ok,
        "optimizer_restored": optimizer_ok,
        "replay_restored": replay_ok,
        "model_matches_node_1500": model_ok and _same(state["model_state"], milestone["model_state"]),
        "training_state_sha256": _sha(state_path),
        "node_1500_sha256": _sha(milestone_path),
        "v5_config_sha256": declared_config_hash,
        "v5_cumulative_training_cpu_seconds": float(state["cumulative_cpu_seconds"]),
    }


def make_registration(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    config = json.loads((root / "research/g_v5/config/experiment.json").read_text(encoding="utf-8"))
    schedule = build_opening_schedule()
    validate_opening_schedule(schedule, root)
    source = root / "research/g_v5/checkpoints/reference400_empty/training_state.pt"
    source_state = torch.load(source, map_location="cpu", weights_only=False)
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "branch": V6_BRANCH,
        "registration_code_commit": current_commit,
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v5_baseline_commit": V5_COMMIT,
        "board_size": 9,
        "n_in_row": 5,
        "simulations_per_move": SIMULATIONS_PER_MOVE,
        "target_cumulative_states": TARGET_STATES,
        "milestones": MILESTONES,
        "resume_source": "research/g_v5/checkpoints/reference400_empty/training_state.pt",
        "resume_state_sha256": _sha(source),
        "resume_initial_states": int(source_state["environment_states"]),
        "resume_initial_games": int(source_state["game_count"]),
        "resume_initial_updates": int(source_state["update_count"]),
        "resume_initial_seed": int(source_state["base_seed"]),
        "v5_config_sha256": config["config_sha256"],
        "v5_training_parameters": config["training"],
        "evaluation_budgets_root_visits": EVALUATION_BUDGETS,
        "evaluation_opponents": [
            "random initial AlphaZero reference", "G0-V3-510901-5K",
            "G0-V3-510901-10K", "G0-V3-510901-20K", "A", "B-50ms",
        ],
        "development_pairs": DEVELOPMENT_PAIRS,
        "blind_pairs_per_opponent_and_budget": BLIND_PAIRS,
        "development_seed": DEVELOPMENT_SEED,
        "blind_seed": BLIND_SEED,
        "opening_plies": 6,
        "cpu_hour_hard_cap": CPU_HARD_CAP_SECONDS / 3600,
        "selection_rule": "highest PUCT-64 paired score against frozen G0-V3-510901-10K on the fixed development pool; ties choose the earlier milestone; blind data is never used for selection",
        "training_source": "V5 reference400_empty self-play only; complete model/Adam/replay/RNG state resumed",
        "forbidden_training_inputs": [
            "A/B/C/E/F models or games", "human/teacher action labels",
            "handcrafted rewards", "inference-time tactical rules", "external games",
        ],
    }

