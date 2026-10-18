from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.mcts_v2 import _other, build_schedule, paired_opening
from src.research.mcts_v2_validation import validate_records


def _config() -> dict:
    return {
        "experiment_id": "validation-test",
        "base_seed": 57,
        "opening_plies": 4,
        "phases": [
            {
                "name": "paired",
                "track": "fair",
                "board_size": 9,
                "games_per_matchup": 2,
                "matchups": [["C", "D2"]],
                "budget_seconds_by_model": {"default": 0.05},
            }
        ],
    }


def _game(spec: dict) -> dict:
    size = int(spec["board_size"])
    opening = paired_opening(
        size,
        int(spec["opening_seed"]),
        opening_plies=4,
        color_swap=bool(spec["color_swap"]),
    )
    models = {"B": spec["black_model"], "W": spec["white_model"]}
    moves = [
        {
            "ply": index,
            "player": player,
            "model": None,
            "row": coord[0],
            "col": coord[1],
            "opening": True,
            "timed_out": False,
            "nodes": 0,
            "simulations": 0,
            "rollout_plies": 0,
            "elapsed_seconds": 0.0,
            "tree_depth": 0,
            "completed_depth": 0,
            "terminal_simulations": 0,
            "truncated_simulations": 0,
        }
        for index, (player, coord) in enumerate(opening)
    ]
    current = _other(opening[-1][0])
    opponent = _other(current)
    winner_model = None
    winner_color = None
    for col in range(5):
        for player, coord in (
            (current, (0, col)),
            *(([(opponent, (8, col))]) if col < 4 else []),
        ):
            model = models[player]
            moves.append(
                {
                    "ply": len(moves),
                    "player": player,
                    "model": model,
                    "row": coord[0],
                    "col": coord[1],
                    "opening": False,
                    "timed_out": False,
                    "nodes": 0,
                    "simulations": 0,
                    "rollout_plies": 0,
                    "elapsed_seconds": 0.0,
                    "tree_depth": 0,
                    "completed_depth": 0,
                    "terminal_simulations": 0,
                    "truncated_simulations": 0,
                }
            )
        winner_model, winner_color = models[current], current

    return {
        **spec,
        "opening": [
            {"player": player, "row": coord[0], "col": coord[1]}
            for player, coord in opening
        ],
        "moves": moves,
        "move_count": len(moves),
        "winner_model": winner_model,
        "winner_color": winner_color,
        "outcome": f"{winner_color.lower()}_win",
        "termination": "five_in_row",
        "timeout_count": 0,
        "search_nodes_total": 0,
        "simulations_total": 0,
        "rollout_plies_total": 0,
        "terminal_simulations_total": 0,
        "truncated_simulations_total": 0,
        "tree_depth_max": 0,
        "thinking_seconds_total": 0,
    }


def test_v2_validation_checks_legal_replay_and_color_swapped_pair() -> None:
    config = _config()
    records = [_game(spec) for spec in build_schedule(config)]

    result = validate_records(records, config)

    assert result["game_count"] == 2
    assert result["unique_game_ids"] == 2
    assert result["paired_opening_groups"] == 1
    assert result["illegal_moves"] == 0
    assert records[0]["opening"] == records[1]["opening"]
    assert records[0]["moves"][4]["model"] != records[1]["moves"][4]["model"]


def test_v2_validation_rejects_duplicate_ids() -> None:
    config = _config()
    game = _game(build_schedule(config)[0])
    with pytest.raises(ValueError, match="Duplicate game_id"):
        validate_records([game, dict(game)], config)
