"""Checkpoint, self-play, match, and selection audit for G V6.2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .v5_audit import audit_evaluation_game, audit_selfplay_game
from .v6_2_registration import validate_schedule
from .v6_2_training import ARTIFACT, ARM, INITIAL_GAMES, INITIAL_STATES, INITIAL_UPDATES
from .v6_2_training import _same, _sha, total_cpu_used_seconds, validate_full_training_state
from .v6_audit import audit_v6


def _canonical_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    actual = hashlib.sha256(json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if actual != declared:
        raise ValueError(f"V6.2 registration digest mismatch: {field}")
    return actual


def audit_v6_2(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    registration_hash = _canonical_hash(registration, "registration_sha256")
    schedule_path = artifact / "config/opening_schedule.json"
    if _sha(schedule_path) != registration["opening_schedule_sha256"]:
        raise ValueError("V6.2 opening schedule SHA does not match registration")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    opening_check = validate_schedule(schedule, root=root)
    puzzles = {}
    for label, key in (("historical_v6_64", "historical_64_sha256"), ("new_independent_128", "new_128_sha256")):
        path = artifact / ("config/tactics_historical_v6_64.json" if label == "historical_v6_64"
                           else "config/tactics_new_128.json")
        expected = registration["tactics"][key]
        if _sha(path) != expected:
            raise ValueError(f"V6.2 frozen puzzle set changed: {label}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != (64 if label == "historical_v6_64" else 128):
            raise ValueError(f"V6.2 puzzle-set size mismatch: {label}")
        puzzles[label] = {"count": len(rows), "sha256": expected}

    current_path = artifact / f"checkpoints/{ARM}/training_state.pt"
    current = validate_full_training_state(root, current_path)
    if int(current["game_count"]) < INITIAL_GAMES:
        raise ValueError("V6.2 current checkpoint fell behind the frozen baseline")
    new_game_count = int(current["game_count"]) - INITIAL_GAMES
    selfplay_dir = artifact / f"raw/selfplay/{ARM}"
    game_rows = []
    sample_ids: set[str] = set()
    sample_count = 0
    game_ids: list[int] = []
    for path in sorted(selfplay_dir.glob("*.json")):
        result = audit_selfplay_game(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("simulations_configured_per_move", -1)) != 400:
            raise ValueError("V6.2 self-play did not use the registered 400 simulations")
        for move in record.get("moves", []):
            if not move.get("opening") and int(move.get("configured_playouts", -1)) != 400:
                raise ValueError("V6.2 self-play move has a different playout count")
        game_id = str(result["game_id"])
        if game_id in sample_ids:
            raise ValueError(f"duplicate V6.2 self-play ID: {game_id}")
        sample_ids.add(game_id)
        numeric_id = int(game_id.rsplit("-", 1)[1])
        game_ids.append(numeric_id)
        sample_count += int(result["samples"])
        game_rows.append(result)
    orphan_samples = [path.name for path in selfplay_dir.glob("*.npz")
                      if not path.with_suffix(".json").is_file()]
    if orphan_samples:
        raise ValueError(f"V6.2 has unpaired self-play NPZ files: {orphan_samples[:5]}")
    if len(game_rows) != new_game_count:
        raise ValueError("V6.2 self-play game count differs from the full checkpoint")
    if sorted(game_ids) != list(range(INITIAL_GAMES, int(current["game_count"]))):
        raise ValueError("V6.2 self-play IDs are not a gap-free continuation from game 1,012")
    if int(current["environment_states"]) != INITIAL_STATES + sample_count:
        raise ValueError("V6.2 cumulative state counter differs from persisted self-play samples")
    if int(current["optimizer_updates"]) != INITIAL_UPDATES + new_game_count:
        raise ValueError("V6.2 optimizer update counter differs from the complete-game increments")
    history_path = artifact / f"results/training/{ARM}.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    if len(history) != new_game_count or len({row["game_id"] for row in history}) != len(history):
        raise ValueError("V6.2 per-game training metrics are incomplete or duplicated")
    if {row["game_id"] for row in history} != sample_ids:
        raise ValueError("V6.2 training metric IDs differ from persisted self-play games")
    for row in history:
        for field in ("loss", "policy_loss", "value_loss", "entropy", "update_count"):
            if field not in row:
                raise ValueError(f"V6.2 training history lacks metric: {field}")

    milestone_rows = []
    milestones_path = artifact / "results/milestones.json"
    milestones = json.loads(milestones_path.read_text(encoding="utf-8")) if milestones_path.is_file() else {"milestones": []}
    for entry in milestones.get("milestones", []):
        games = int(entry["games_completed"])
        full_path = artifact / f"checkpoints/{ARM}/training_state_games_{games}.pt"
        node_path = artifact / f"checkpoints/{ARM}/node_games_{games}.pt"
        if _sha(full_path) != entry["full_checkpoint_sha256"] or _sha(node_path) != entry["model_checkpoint_sha256"]:
            raise ValueError(f"V6.2 milestone checkpoint hash mismatch: {games}")
        full_validation = validate_full_training_state(root, full_path)
        full = torch.load(full_path, map_location="cpu", weights_only=False)
        node = torch.load(node_path, map_location="cpu", weights_only=False)
        if int(full["game_count"]) != games or not _same(full["model_state"], node["model_state"]):
            raise ValueError(f"V6.2 milestone model/counter mismatch: {games}")
        milestone_rows.append({"games": games, "validation": full_validation})

    evaluation_rows = []
    match_ids: set[str] = set()
    pairs: dict[tuple[str, str, str, int, str], set[str]] = {}
    match_root = artifact / "results/matches"
    for path in sorted(match_root.rglob("v6m-*.json")) if match_root.exists() else []:
        audit = audit_evaluation_game(path)
        game = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(audit["game_id"])
        if game_id in match_ids:
            raise ValueError(f"duplicate V6.2 match ID: {game_id}")
        match_ids.add(game_id)
        budget = int(game["simulations_per_reference_turn"])
        if budget not in (64, 400) or int(game.get("reference_configured_playouts", -1)) != budget + 1:
            raise ValueError("V6.2 evaluation budget metadata mismatch")
        key = (str(game.get("phase")), str(game.get("candidate")),
               str(game.get("opponent")), budget, str(game.get("opening_group_id")))
        colors = pairs.setdefault(key, set())
        color = str(game.get("candidate_color"))
        if color in colors:
            raise ValueError(f"duplicate color in a V6.2 paired match: {key}")
        colors.add(color)
        evaluation_rows.append(audit)
    complete_pairs = sum(colors == {"B", "W"} for colors in pairs.values())
    incomplete_pairs = sum(colors != {"B", "W"} for colors in pairs.values())

    gate = json.loads((artifact / "results/gate_2000.json").read_text(encoding="utf-8")) if (
        artifact / "results/gate_2000.json").is_file() else {}
    if gate:
        stage_path = artifact / "results/development/stage_games_2000.json"
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        primary = stage["comparisons"]["400"]["G20K"]
        recomputed_pass = bool(primary.get("complete") and int(primary.get("paired_openings", 0)) >= 80
                               and float(primary["score_rate"]) > 0.55
                               and float(primary["paired_opening_bootstrap_ci95"][0]) > 0.50)
        if gate.get("passed") != recomputed_pass:
            raise ValueError("V6.2 2,000-game gate differs from registered primary metrics")
        if not recomputed_pass and int(current["game_count"]) > 2_000:
            raise ValueError("V6.2 continued beyond 2,000 after a failed gate")
    blind = json.loads((artifact / "results/final_blind.json").read_text(encoding="utf-8")) if (
        artifact / "results/final_blind.json").is_file() else {}
    if blind:
        selection = json.loads((artifact / "results/selection.json").read_text(encoding="utf-8"))
        if (not selection.get("frozen_before_blind") or selection.get("blind_results_used_for_selection")
                or blind.get("selection") != selection):
            raise ValueError("V6.2 blind testing did not use the frozen development-only selection")

    v6_integrity = audit_v6(root)
    result = {
        "format_version": 1,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration_hash,
        "base_v6_commit": registration["base_v6_commit"],
        "v6_integrity_revalidated": True,
        "baseline_v6_integrity": {
            "training_state_sha256": v6_integrity["resumable_checkpoint"]["sha256"],
            "selfplay_games_verified": v6_integrity["v6_selfplay_games_verified"],
            "evaluation_games_verified": v6_integrity["v6_evaluation_games_verified"],
        },
        "opening_validation": opening_check,
        "puzzle_sets": puzzles,
        "current_checkpoint": current,
        "v6_2_added_games": new_game_count,
        "v6_2_added_states": sample_count,
        "v6_2_selfplay_games_verified": len(game_rows),
        "v6_2_unique_selfplay_ids": len(sample_ids),
        "v6_2_evaluation_games_verified": len(evaluation_rows),
        "v6_2_unique_evaluation_ids": len(match_ids),
        "paired_openings_complete": complete_pairs,
        "paired_openings_incomplete": incomplete_pairs,
        "milestones_verified": milestone_rows,
        "advancement_gate": gate,
        "final_blind_complete": bool(blind.get("complete")),
        "cumulative_new_cpu_seconds_accounted": total_cpu_used_seconds(root),
        "all_completed_games_legal_and_terminal": True,
    }
    return result
