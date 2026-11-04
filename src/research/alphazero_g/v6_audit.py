"""Integrity, checkpoint, and game-replay audit for G V6."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .v5_audit import audit_evaluation_game, audit_selfplay_game
from .v5_training import load_training_checkpoint
from .v5_reference import ReferencePolicyValueNet
from .v6_experiment import OPPONENTS
from .v6_registration import (
    EVALUATION_BUDGETS,
    MILESTONES,
    SIMULATIONS_PER_MOVE,
    TARGET_STATES,
    _same,
    validate_opening_schedule,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: dict[str, Any], key: str) -> str:
    body = dict(value)
    declared = body.pop(key, None)
    actual = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    if actual != declared:
        raise ValueError(f"V6 registered configuration hash mismatch: {key}")
    return actual


def _verify_checkpoint(root: Path, registration: dict[str, Any]) -> dict[str, Any]:
    path = root / "research/g_v6/checkpoints/reference400_empty/training_state.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    v5_config = json.loads((root / "research/g_v5/config/experiment.json").read_text(encoding="utf-8"))
    if (checkpoint.get("arm_id") != "reference400_empty"
            or checkpoint.get("config_sha256") != registration["v5_config_sha256"]
            or int(checkpoint.get("board_size", 0)) != 9
            or int(checkpoint.get("environment_states", 0)) < int(registration["resume_initial_states"])):
        raise ValueError("V6 resumable checkpoint identity or counters are invalid")
    model = ReferencePolicyValueNet(9, seed=int(checkpoint["base_seed"]))
    previous_python, previous_numpy, previous_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        loaded = load_training_checkpoint(
            path, model=model, arm="reference400_empty",
            config_hash=registration["v5_config_sha256"],
            buffer_size=int(v5_config["training"]["buffer_size"]),
        )
        python_ok = random.getstate() == checkpoint["python_rng_state"]
        current_numpy = np.random.get_state()
        numpy_ok = (current_numpy[0] == checkpoint["numpy_rng_state"][0]
                    and np.array_equal(current_numpy[1], checkpoint["numpy_rng_state"][1])
                    and current_numpy[2:] == checkpoint["numpy_rng_state"][2:])
        torch_ok = torch.equal(torch.get_rng_state(), checkpoint["torch_rng_state"])
        model_ok = _same(model.get_policy_param(), checkpoint["model_state"])
        optimizer_ok = _same(model.optimizer.state_dict(), checkpoint["optimizer_state"])
    finally:
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.set_rng_state(previous_torch)
    if (not checkpoint.get("optimizer_state", {}).get("state")
            or len(checkpoint.get("replay_buffer", [])) != int(v5_config["training"]["buffer_size"])
            or len(loaded[4]) != len(checkpoint["replay_buffer"])
            or not all((python_ok, numpy_ok, torch_ok, model_ok, optimizer_ok))):
        raise ValueError("V6 training state is missing optimizer or replay contents")
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha(path),
        "board_size": 9,
        "simulations_per_move": SIMULATIONS_PER_MOVE,
        "environment_states": int(checkpoint["environment_states"]),
        "games_completed": int(checkpoint["game_count"]),
        "optimizer_updates": int(checkpoint["update_count"]),
        "optimizer_slots": len(checkpoint["optimizer_state"]["state"]),
        "python_rng_restored": python_ok,
        "numpy_rng_restored": numpy_ok,
        "torch_rng_restored": torch_ok,
        "model_restored": model_ok,
        "optimizer_restored": optimizer_ok,
        "replay_rows": len(checkpoint["replay_buffer"]),
        "v5_origin_states": int(registration["resume_initial_states"]),
        "v6_added_states": int(checkpoint["environment_states"]) - int(registration["resume_initial_states"]),
        "v6_added_games": int(checkpoint["game_count"]) - int(registration["resume_initial_games"]),
    }


def audit_v6(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    base = root / "research/g_v6"
    registration = json.loads((base / "config/registration.json").read_text(encoding="utf-8"))
    registration_sha = _canonical_sha(registration, "registration_sha256")
    schedule_path = base / "config/opening_schedule.json"
    tactics_path = base / "config/tactics.json"
    if _sha(schedule_path) != registration["opening_schedule_sha256"]:
        raise ValueError("V6 opening schedule hash mismatch")
    if _sha(tactics_path) != registration["tactics_sha256"]:
        raise ValueError("V6 tactic set hash mismatch")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_opening_schedule(schedule, root)

    v5_training = root / registration["resume_source"]
    if _sha(v5_training) != registration["resume_state_sha256"]:
        raise ValueError("frozen V5 continuation source was modified")
    checkpoints = _verify_checkpoint(root, registration)

    selfplay_rows = []
    selfplay_ids: set[str] = set()
    for path in sorted((base / "raw/selfplay/reference400_empty").glob("*.json")):
        row = audit_selfplay_game(path)
        if row["game_id"] in selfplay_ids:
            raise ValueError(f"duplicate V6 self-play game ID: {row['game_id']}")
        selfplay_ids.add(row["game_id"])
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("simulations_configured_per_move", -1)) != SIMULATIONS_PER_MOVE:
            raise ValueError("V6 self-play game used a different configured simulation budget")
        for move in record["moves"]:
            if not move.get("opening") and int(move.get("configured_playouts", -1)) != SIMULATIONS_PER_MOVE:
                raise ValueError("V6 self-play move did not use 400 configured playouts")
        selfplay_rows.append(row)
    orphan_samples = [str(path) for path in (base / "raw/selfplay/reference400_empty").glob("*.npz")
                      if not path.with_suffix(".json").is_file()]
    if orphan_samples:
        raise ValueError(f"V6 self-play sample archives are orphaned: {orphan_samples[:5]}")
    sample_states = sum(int(row["samples"]) for row in selfplay_rows)
    if checkpoints["v6_added_states"] != sample_states:
        raise ValueError("V6 accumulated state counter differs from replayed V6 sample count")
    if checkpoints["v6_added_games"] != len(selfplay_rows):
        raise ValueError("V6 game counter differs from unique persisted self-play games")

    evaluation_rows = []
    evaluation_ids: set[str] = set()
    group_colors: dict[tuple[str, str, str, int, str], set[str]] = {}
    for path in sorted((base / "results/matches").rglob("v6m-*.json")):
        row = audit_evaluation_game(path)
        if row["game_id"] in evaluation_ids:
            raise ValueError(f"duplicate V6 evaluation game ID: {row['game_id']}")
        evaluation_ids.add(row["game_id"])
        game = json.loads(path.read_text(encoding="utf-8"))
        budget = int(game.get("simulations_per_reference_turn", -1))
        if budget not in EVALUATION_BUDGETS or int(game.get("configured_search_budget_root_visits", -1)) != budget:
            raise ValueError("V6 evaluation game budget metadata mismatch")
        if int(game.get("reference_configured_playouts", -1)) != budget + 1:
            raise ValueError("reference evaluator did not configure exact root action visits")
        key = (str(game["phase"]), str(game["candidate"]), str(game["opponent"]),
               budget, str(game["opening_group_id"]))
        colors = group_colors.setdefault(key, set())
        color = str(game["candidate_color"])
        if color in colors:
            raise ValueError(f"duplicate color within V6 paired opening: {key}")
        colors.add(color)
        evaluation_rows.append(row)

    stage_results = {}
    for milestone in MILESTONES:
        checkpoint_path = base / f"checkpoints/reference400_empty/node_{milestone}.pt"
        stage_path = base / f"results/development/stage_{milestone}.json"
        if not checkpoint_path.exists():
            stage_results[str(milestone)] = {"checkpoint_present": False, "complete": False}
            continue
        node = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (int(node.get("board_size", 0)) != 9 or int(node.get("n_in_row", 0)) != 5
                or int(node.get("simulations_per_move", 0)) != SIMULATIONS_PER_MOVE
                or int(node.get("environment_states", 0)) < milestone):
            raise ValueError(f"V6 milestone checkpoint metadata mismatch: {milestone}")
        stage = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.is_file() else {}
        stage_results[str(milestone)] = {
            "checkpoint_present": True,
            "actual_states": int(node["environment_states"]),
            "complete": bool(stage.get("complete")),
            "comparisons": stage.get("comparisons", {}),
            "tactics": stage.get("tactics", {}),
        }

    blind = json.loads((base / "results/final_blind.json").read_text(encoding="utf-8")) if (base / "results/final_blind.json").exists() else {}
    if blind:
        selected = blind.get("selection", {}).get("selected", {})
        if not blind.get("selection", {}).get("frozen_before_blind") or blind.get("selection", {}).get("blind_results_used_for_selection"):
            raise ValueError("V6 final blind model selection was not locked to development-only data")
        if selected.get("checkpoint_sha256"):
            # Selection metadata stores a repository-relative path (as do the
            # runner outputs). Joining it to ``base`` duplicates
            # ``research/g_v6`` and makes a valid frozen candidate appear
            # missing during the final audit.
            frozen = root / selected["checkpoint_path"]
            if _sha(frozen) != selected["checkpoint_sha256"]:
                raise ValueError("V6 blind candidate checkpoint differs from frozen selection")

    result = {
        "format_version": 1,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration_sha,
        "v5_resume_source_unchanged": True,
        "resumable_checkpoint": checkpoints,
        "v6_selfplay_games_verified": len(selfplay_rows),
        "v6_training_states_verified": sample_states,
        "v6_total_cumulative_states": checkpoints["environment_states"],
        "v6_unique_selfplay_ids": len(selfplay_ids),
        "v6_evaluation_games_verified": len(evaluation_rows),
        "v6_unique_evaluation_ids": len(evaluation_ids),
        "paired_evaluation_openings_complete": sum(colors == {"B", "W"} for colors in group_colors.values()),
        "paired_evaluation_openings_incomplete": sum(colors != {"B", "W"} for colors in group_colors.values()),
        "all_completed_games_legal_and_terminal": True,
        "milestones": stage_results,
        "final_blind_complete": bool(blind.get("complete")),
        "final_blind_matches": blind.get("matches", {}),
        "projected_full_milestone_count": TARGET_STATES,
    }
    return result

