"""Checks for reproducible V3 benchmark positions and runtime estimates."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.mcts_v3_preflight import (
    estimate_official_runtime,
    historical_turn_averages,
    representative_position,
    resource_snapshot,
)


def test_preflight_position_is_reproducible_and_nonterminal() -> None:
    first_board, first_player = representative_position()
    second_board, second_player = representative_position()

    assert first_board.size == 9
    assert first_player == second_player == "B"
    assert len(first_board.history) == 12
    assert first_board.history == second_board.history
    assert not first_board.is_full()
    assert all(
        not first_board.check_victory(coord, player)
        for player, coord in first_board.history
    )


def test_runtime_estimate_counts_all_three_budgets_and_opponents() -> None:
    historical = {
        "A": {"D2_moves_per_game": 2.0, "opponent_moves_per_game": 3.0},
        "B": {"D2_moves_per_game": 4.0, "opponent_moves_per_game": 5.0},
        "C": {"D2_moves_per_game": 6.0, "opponent_moves_per_game": 7.0},
    }
    measured = {10.0: 0.4, 20.0: 0.8, 50.0: 2.0}

    estimate = estimate_official_runtime(historical, measured)

    assert estimate["official_games"] == 54
    assert estimate["condition_count"] == 9
    assert estimate["estimated_d3_moves_by_budget"] == {
        "10": 72.0,
        "20": 72.0,
        "50": 72.0,
    }
    assert estimate["estimated_d3_compute_seconds"] == 230.4
    assert estimate["estimated_opponent_compute_seconds"] == 13.5
    assert estimate["estimated_total_seconds"] == 243.9


def test_historical_turn_averages_counts_real_model_moves() -> None:
    records = [
        {
            "board_size": 9,
            "matchup": ["A", "D2"],
            "moves": [
                {"model": "D2"},
                {"model": "A"},
                {"model": "D2"},
                {"model": "A"},
                {"model": "D2"},
            ],
        },
        {
            "board_size": 9,
            "matchup": ["A", "D2"],
            "moves": [{"model": "D2"}, {"model": "A"}, {"model": "A"}],
        },
        *[
            {
                "board_size": 9,
                "matchup": [opponent, "D2"],
                "moves": [{"model": "D2"}, {"model": opponent}],
            }
            for opponent in ("B", "C")
        ],
    ]

    averages = historical_turn_averages(records)

    assert averages["A"]["calibration_games"] == 2
    assert averages["A"]["D2_moves_per_game"] == 2.0
    assert averages["A"]["opponent_moves_per_game"] == 2.0
    assert averages["B"]["D2_moves_per_game"] == 1.0
    assert averages["C"]["opponent_moves_per_game"] == 1.0


def test_resource_snapshot_reports_available_workspace_resources(tmp_path: Path) -> None:
    resources = resource_snapshot(tmp_path)

    assert resources["logical_cpu_count"] is not None
    assert resources["memory_total_bytes"] is not None
    assert resources["workspace_disk_free_bytes"] > 0


def test_preflight_uses_configured_or_current_full_code_commit() -> None:
    from src.research.mcts_v3_preflight import code_commit_for_config

    configured = code_commit_for_config({"code_commit": "a" * 40})
    current = code_commit_for_config({})

    assert configured == "a" * 40
    assert len(current) == 40
    assert all(character in "0123456789abcdef" for character in current)
