"""Frozen data and match schedules plus factual F1-vs-A loss analysis."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from ...gomoku import GomokuBoard
from .dataset import validate_game_record
from .evaluation import find_immediate_wins
from .generation import build_evaluation_openings


def build_experiment_plan(
    seed: int,
    train_count: int,
    development_count: int,
    blind_count: int,
    excluded_groups: Iterable[str] = (),
) -> dict[str, Any]:
    """Build deterministic, pairwise-disjoint four-ply opening schedules."""
    if min(int(train_count), int(development_count), int(blind_count)) <= 0:
        raise ValueError("schedule counts must be positive")
    used = {str(value) for value in excluded_groups}
    schedules: dict[str, list[dict[str, Any]]] = {}
    for offset, (name, count) in enumerate((
        ("training_openings", int(train_count)),
        ("development_openings", int(development_count)),
        ("blind_openings", int(blind_count)),
    )):
        rows = build_evaluation_openings(int(seed) + offset, count, tuple(sorted(used)))
        group_ids = [str(row["opening_group_id"]) for row in rows]
        if len(set(group_ids)) != count or used.intersection(group_ids):
            raise AssertionError("opening generator returned a reused group")
        used.update(group_ids)
        schedules[name] = rows
    encoded = json.dumps(schedules, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "seed": int(seed),
        **schedules,
        "excluded_group_ids": sorted(str(value) for value in excluded_groups),
        "schedule_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def analyze_f1_a_losses(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count F1 misses of exact immediate wins and unique mandatory blocks."""
    losses = [
        record for record in records
        if set(record.get("pair", ())) == {"F", "A"} and record.get("winner_model") == "A"
    ]
    examples: list[dict[str, Any]] = []
    missed_wins = 0
    missed_blocks = 0
    for record in sorted(losses, key=lambda row: str(row.get("game_id", ""))):
        validate_game_record(record)
        board = GomokuBoard(9)
        mistakes: list[dict[str, Any]] = []
        for move in record["moves"]:
            player = str(move["player"])
            action = int(move["row"]), int(move["col"])
            if move.get("model") == "F":
                own_wins = find_immediate_wins(board, player)
                opponent = "W" if player == "B" else "B"
                opponent_wins = [] if own_wins else find_immediate_wins(board, opponent)
                if own_wins and action not in own_wins:
                    missed_wins += 1
                    mistakes.append({
                        "kind": "missed_immediate_win", "ply": int(move["ply"]),
                        "chosen": list(action), "winning_moves": [list(coord) for coord in own_wins],
                    })
                elif len(opponent_wins) == 1 and action != opponent_wins[0]:
                    missed_blocks += 1
                    mistakes.append({
                        "kind": "missed_single_point_block", "ply": int(move["ply"]),
                        "chosen": list(action), "required_block": list(opponent_wins[0]),
                    })
            board.place_stone(player, action)
        examples.append({
            "game_id": str(record["game_id"]),
            "f_color": "B" if record["black_model"] == "F" else "W",
            "plies": len(record["moves"]),
            "mistakes": mistakes,
        })
    return {
        "games": len(losses),
        "missed_immediate_wins": missed_wins,
        "missed_single_point_blocks": missed_blocks,
        "examples": examples,
    }


def analyze_network_losses(
    records: Iterable[dict[str, Any]], model_name: str, opponent_name: str = "A"
) -> dict[str, Any]:
    """Find immediate wins and unique blocks missed in a frozen network's losses."""
    losses = [
        record for record in records
        if record.get("challenger_model") == model_name and record.get("winner_model") == opponent_name
    ]
    examples = []
    missed_wins = missed_blocks = 0
    for record in sorted(losses, key=lambda row: str(row.get("game_id", ""))):
        validate_game_record(record)
        board = GomokuBoard(9)
        mistakes = []
        for move in record["moves"]:
            player = str(move["player"])
            action = int(move["row"]), int(move["col"])
            if move.get("model") == model_name:
                own_wins = find_immediate_wins(board, player)
                opponent = "W" if player == "B" else "B"
                opponent_wins = [] if own_wins else find_immediate_wins(board, opponent)
                if own_wins and action not in own_wins:
                    missed_wins += 1
                    mistakes.append({"kind": "missed_immediate_win", "ply": int(move["ply"]),
                                     "chosen": list(action), "winning_moves": [list(coord) for coord in own_wins]})
                elif len(opponent_wins) == 1 and action != opponent_wins[0]:
                    missed_blocks += 1
                    mistakes.append({"kind": "missed_single_point_block", "ply": int(move["ply"]),
                                     "chosen": list(action), "required_block": list(opponent_wins[0])})
            board.place_stone(player, action)
        examples.append({
            "game_id": str(record["game_id"]), "group_id": str(record.get("group_id", "")),
            "f_color": "B" if record.get("black_model") == model_name else "W",
            "plies": len(record["moves"]), "mistakes": mistakes,
        })
    return {
        "model": model_name, "opponent": opponent_name, "games": len(losses),
        "missed_immediate_wins": missed_wins, "missed_single_point_blocks": missed_blocks,
        "examples": examples,
    }


def estimate_run_seconds(
    *,
    train_games: int,
    mean_game_seconds: float,
    mean_f_turns_per_game: float,
    mean_annotation_seconds: float,
    train_rows: int,
    seconds_per_update: float,
    expected_epochs: int,
    maximum_epochs: int,
    maximum_training_runs: int,
    evaluation_games: int,
    mean_evaluation_game_seconds: float,
    overhead_seconds: float,
    batch_size: int = 64,
) -> dict[str, float | int]:
    """Estimate full task time from measured pilot throughput and fixed caps."""
    rates = (
        mean_game_seconds, mean_f_turns_per_game, mean_annotation_seconds,
        seconds_per_update, mean_evaluation_game_seconds, overhead_seconds,
    )
    if min(train_games, train_rows, expected_epochs, maximum_epochs, maximum_training_runs,
           evaluation_games, batch_size) <= 0 or any(value < 0 for value in rates):
        raise ValueError("counts must be positive and measured rates nonnegative")
    if expected_epochs > maximum_epochs:
        raise ValueError("expected epochs cannot exceed maximum epochs")
    steps_per_epoch = math.ceil(train_rows / batch_size)
    data_seconds = train_games * (mean_game_seconds + mean_f_turns_per_game * mean_annotation_seconds)
    evaluation_seconds = evaluation_games * mean_evaluation_game_seconds
    central = (
        data_seconds
        + expected_epochs * steps_per_epoch * seconds_per_update
        + evaluation_seconds
        + overhead_seconds
    )
    optimistic = (
        0.7 * data_seconds
        + max(30, expected_epochs) * steps_per_epoch * seconds_per_update
        + 0.7 * evaluation_seconds
        + 0.5 * overhead_seconds
    )
    conservative = (
        1.5 * data_seconds
        + maximum_training_runs * maximum_epochs * steps_per_epoch * seconds_per_update * 1.35
        + 1.5 * evaluation_seconds
        + 2 * overhead_seconds
    )
    return {
        "data_seconds": data_seconds,
        "evaluation_seconds": evaluation_seconds,
        "steps_per_epoch": steps_per_epoch,
        "optimistic_seconds": optimistic,
        "central_seconds": central,
        "conservative_seconds": conservative,
    }
