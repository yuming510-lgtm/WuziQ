"""Full replay, budget, provenance and pairing audit for V4.2 artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .evaluation import validate_match_record
from .replay import _validate_game
from .training import load_g_checkpoint
from .v4_2_config import load_v4_2_config
from .v4_2_diagnostics import puzzle_symmetry_key, validate_v4_2_puzzle
from .v4_2_schedule import validate_v4_2_schedule
from .v4_2_storage import atomic_write_json, iter_games, load_json, sha256_file
from .v4_2_training import validate_history_pool_game


_MATCH_NAMESPACES = (
    "history", "frozen_search_matches", "training_eval", "arm_internal",
    "blind_puct64", "blind_reference_puct64", "blind_network",
)


def audit_v4_2_results(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_2_config(root_path)
    schedule = load_json(root_path / config["opening_schedule_path"])
    validate_v4_2_schedule(schedule)
    matches_by_namespace: dict[str, list[dict[str, Any]]] = {}
    all_match_ids: set[str] = set()
    for namespace in _MATCH_NAMESPACES:
        rows = iter_games(root_path, namespace)
        matches_by_namespace[namespace] = rows
        for row in rows:
            validate_match_record(row)
            if row["game_id"] in all_match_ids:
                raise ValueError(f"duplicate match ID across result stores: {row['game_id']}")
            all_match_ids.add(row["game_id"])
            for move in row["moves"]:
                if move.get("opening"):
                    continue
                if row.get("mode") == "puct" and move.get("actor") == "agent" and int(move.get("simulations", 0)) != 64:
                    raise ValueError("formal evaluation candidate did not use fixed PUCT-64")
                if row.get("opponent_mode") == "puct" and move.get("actor") == "MODEL" and int(move.get("simulations", 0)) != 64:
                    raise ValueError("formal evaluation opponent did not use fixed PUCT-64")
    games_root = root_path / "research/g_v4_2/results/games"
    selfplay_namespaces = [
        path.relative_to(games_root).as_posix()
        for path in games_root.glob("selfplay/*") if path.is_dir()
    ]
    selfplay_by_arm: dict[str, list[dict[str, Any]]] = {}
    sample_ids_by_arm: dict[str, set[str]] = {}
    states_by_arm: dict[str, int] = {}
    for namespace in selfplay_namespaces:
        arm_id = namespace.split("/", 1)[1] if "/" in namespace else namespace
        games = iter_games(root_path, namespace)
        sample_ids: set[str] = set()
        state_total = 0
        for game in games:
            if int(game.get("simulations_per_move", 0)) != 64:
                raise ValueError(f"training arm {arm_id} did not use PUCT-64")
            rows = game.get("experience_rows", [])
            budgeted = int(game.get("budgeted_training_state_count", -1))
            state_ids = game.get("training_state_ids", [])
            if budgeted != len(state_ids) or budgeted < 1 or budgeted > len(rows):
                raise ValueError(f"training-state count does not match its self-play log: {game.get('game_id')}")
            if state_ids != [row.get("sample_id") for row in rows[:budgeted]]:
                raise ValueError(f"training row IDs do not match the recorded state prefix: {game.get('game_id')}")
            for state_id in state_ids:
                if state_id in sample_ids:
                    raise ValueError(f"duplicate training state id: {state_id}")
                sample_ids.add(state_id)
            state_total += budgeted
            if game.get("opponent_pool") == "historical_g0":
                validate_history_pool_game(game, game["learner_color"])
            else:
                _validate_game(game)
            for move in game["moves"]:
                if not move.get("opening") and int(move.get("simulations", 0)) != 64:
                    raise ValueError(f"self-play decision did not use 64 simulations: {game.get('game_id')}")
        selfplay_by_arm[arm_id] = games
        sample_ids_by_arm[arm_id] = sample_ids
        states_by_arm[arm_id] = state_total
    if sum(states_by_arm.values()) > 40_000:
        raise ValueError("V4.2 exceeded its 40K maximum state budget")

    puzzle_path = root_path / "research/g_v4_2/config/tactical_puzzles.json"
    puzzles = load_json(puzzle_path)
    for puzzle in puzzles:
        validate_v4_2_puzzle(puzzle)
    puzzle_keys = [puzzle_symmetry_key(puzzle) for puzzle in puzzles]
    if len(puzzles) != 64 or len(puzzle_keys) != len(set(puzzle_keys)):
        raise ValueError("V4.2 tactical dataset is incomplete or contains symmetric duplicates")
    for candidate in config["historical_candidates"]:
        checkpoint = root_path / candidate["checkpoint_path"]
        if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError(f"historical baseline changed: {candidate['candidate_id']}")
        if candidate["training_family"] != "G0":
            raise ValueError("V4.2 historical training pool contains a non-G0 model")
    training_summary_path = root_path / "research/g_v4_2/results/training/summary.json"
    training_summary = load_json(training_summary_path) if training_summary_path.exists() else {}
    for arm_id, expected_states in training_summary.get("states_by_arm", {}).items():
        if int(expected_states) != states_by_arm.get(arm_id, 0):
            raise ValueError(f"training summary state count differs from raw game rows for {arm_id}")

    blind_summary_path = root_path / "research/g_v4_2/results/blind/summary.json"
    blind_summary = load_json(blind_summary_path) if blind_summary_path.exists() else {}
    if blind_summary and int(blind_summary.get("blind_groups", 0)) < 20:
        raise ValueError("V4.2 final blind set is smaller than 20 paired openings")
    audit = {
        "format_version": 1, "config_sha256": config["config_sha256"],
        "opening_schedule_sha256": schedule["schedule_sha256"],
        "tactical_puzzles": len(puzzles), "unique_tactical_mother_positions": len(set(puzzle_keys)),
        "matches_by_namespace": {name: len(rows) for name, rows in matches_by_namespace.items()},
        "selfplay_games_by_arm": {arm: len(rows) for arm, rows in selfplay_by_arm.items()},
        "training_states_by_arm": states_by_arm,
        "training_states_total": sum(states_by_arm.values()),
        "unique_match_ids": len(all_match_ids),
        "unique_training_state_ids_by_arm": {arm: len(rows) for arm, rows in sample_ids_by_arm.items()},
        "all_match_legal_replay_passed": True,
        "all_training_games_legal_replay_passed": True,
        "fixed_puct64_budget_verified": True,
        "blind_summary_present": bool(blind_summary),
    }
    atomic_write_json(root_path / "research/g_v4_2/results/integrity_audit.json", audit)
    return audit
