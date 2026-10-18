"""Correctness checks for the V2 MCTS research variants."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import SearchConfig, choose_move
from src.gomoku import GomokuBoard


def test_d1_matches_original_uct_when_the_rollout_horizon_is_already_full() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    config = SearchConfig(
        time_limit_seconds=10,
        mcts_simulations=16,
        mcts_rollout_max_plies=24,
    )

    baseline = choose_move(board, "B", "D", config, seed=731)
    high_budget = choose_move(board, "B", "D1", config, seed=731)

    assert high_budget.move == baseline.move
    assert high_budget.root_value == baseline.root_value
    assert high_budget.nodes == baseline.nodes
    assert high_budget.simulations == baseline.simulations


def test_d1_reports_completed_terminal_rollouts_separately() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    result = choose_move(
        board,
        "B",
        "D1",
        SearchConfig(time_limit_seconds=10, mcts_simulations=8),
        seed=219,
    )

    assert result.simulations == 8
    assert result.terminal_simulations == result.simulations
    assert result.truncated_simulations == 0
    assert result.rollout_plies >= result.simulations
