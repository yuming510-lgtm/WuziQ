"""Correctness checks for the V2 MCTS research variants."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import SearchConfig, choose_move, generate_candidates
from src.gomoku import GomokuBoard
from src.research.mcts_d2 import MCTSAgent, _Node, _uct_child


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


def test_d2_takes_an_immediate_win_without_minimax_search() -> None:
    board = GomokuBoard(5)
    for col in range(4):
        board.place_stone("B", (2, col))
    original_history = list(board.history)

    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(time_limit_seconds=0.05, mcts_simulations=2),
        seed=1,
    )

    assert result.move == (2, 4)
    assert board.history == original_history


def test_d2_blocks_a_unique_immediate_opponent_win() -> None:
    board = GomokuBoard(5)
    for col in range(4):
        board.place_stone("W", (2, col))

    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(time_limit_seconds=0.05, mcts_simulations=2),
        seed=2,
    )

    assert result.move == (2, 4)


def test_d2_covers_each_root_candidate_before_revisiting_one() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    candidates = generate_candidates(board, "B", "W", radius=1)
    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(
            time_limit_seconds=10,
            mcts_simulations=len(candidates),
            candidate_radius=1,
        ),
        seed=33,
    )

    assert result.root_candidates == len(candidates)
    assert result.root_min_visits >= 1


def test_d2_terminal_rollout_counts_are_explicit_and_board_is_restored() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    initial_history = list(board.history)
    initial_grid = [row[:] for row in board.grid]
    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(time_limit_seconds=10, mcts_simulations=8),
        seed=219,
    )

    assert result.simulations == 8
    assert result.terminal_simulations == result.simulations
    assert result.truncated_simulations == 0
    assert result.rollout_plies >= result.simulations
    assert board.history == initial_history
    assert board.grid == initial_grid


def test_d2_is_deterministic_for_a_fixed_simulation_budget_and_seed() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    config = SearchConfig(time_limit_seconds=10, mcts_simulations=20)

    first = choose_move(board, "B", "D2", config, seed=999)
    second = choose_move(board, "B", "D2", config, seed=999)

    assert first.move == second.move
    assert first.simulations == second.simulations == 20


def test_d2_reuses_matching_subtree_after_its_move_and_opponent_reply() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    config = SearchConfig(time_limit_seconds=10, mcts_simulations=40)
    agent = MCTSAgent()

    first = agent.choose_move(board, "B", config, seed=51)
    board.place_stone("B", first.move)
    selected = next(child for child in agent._root.children if child.move == first.move)
    assert selected.children
    opponent_reply = selected.children[0].move
    assert opponent_reply is not None
    board.place_stone("W", opponent_reply)

    second = agent.choose_move(board, "B", config, seed=52)

    assert board.get(second.move) is None
    assert second.reused_nodes >= 1


def test_d2_zero_budget_returns_a_legal_fallback() -> None:
    board = GomokuBoard(7)
    board.place_stone("B", (3, 3))
    result = choose_move(
        board,
        "W",
        "D2",
        SearchConfig(time_limit_seconds=0, mcts_simulations=2),
        seed=4,
    )

    assert result.timed_out
    assert board.get(result.move) is None


def test_d2_uct_selects_unvisited_nodes_without_dividing_by_zero() -> None:
    parent = _Node(None, "W", visits=1)
    first = _Node((0, 0), "B", parent=parent)
    second = _Node((0, 1), "B", parent=parent)
    parent.children = [first, second]

    assert _uct_child(parent, 1.4142135623730951) is first
