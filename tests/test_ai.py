"""Correctness checks for the four independent Gomoku engines."""
from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gomoku import GomokuBoard


def _engine():
    assert importlib.util.find_spec("src.ai") is not None, "src.ai should expose the four-model engine"
    return importlib.import_module("src.ai")


def _tactical_board(player: str = "B") -> GomokuBoard:
    board = GomokuBoard(5)
    row = 2
    for col in range(4):
        board.place_stone(player, (row, col))
    return board


def test_engine_module_available() -> None:
    assert importlib.util.find_spec("src.ai") is not None


def test_all_models_return_legal_moves() -> None:
    engine = _engine()
    board = GomokuBoard(7)
    board.place_stone("B", (3, 3))
    board.place_stone("W", (3, 4))
    config = engine.SearchConfig(max_depth=2, time_limit_seconds=2, mcts_simulations=12)

    for model in ("A", "B", "C", "D"):
        result = engine.choose_move(board, "B", model, config, seed=510)
        assert board.in_bounds(*result.move)
        assert board.get(result.move) is None


def test_empty_board_uses_center_for_search_models() -> None:
    engine = _engine()
    board = GomokuBoard(9)
    config = engine.SearchConfig(max_depth=1, time_limit_seconds=1, mcts_simulations=2)
    for model in ("B", "C", "D"):
        assert engine.choose_move(board, "B", model, config, seed=1).move == (4, 4)


def test_immediate_win_is_selected() -> None:
    engine = _engine()
    board = _tactical_board("B")
    config = engine.SearchConfig(max_depth=1, time_limit_seconds=2)
    for model in ("B", "C"):
        result = engine.choose_move(board, "B", model, config, seed=2)
        assert result.move == (2, 4)
        assert result.root_value == engine.WIN_SCORE


def test_immediate_opponent_win_is_blocked() -> None:
    engine = _engine()
    board = _tactical_board("W")
    config = engine.SearchConfig(max_depth=2, time_limit_seconds=2)
    result = engine.choose_move(board, "B", "C", config, seed=3)
    assert result.move == (2, 4)


def test_minimax_and_alpha_beta_agree_at_same_fixed_depth() -> None:
    engine = _engine()
    board = GomokuBoard(5)
    for player, coord in [
        ("B", (2, 2)),
        ("W", (2, 3)),
        ("B", (3, 2)),
        ("W", (1, 3)),
    ]:
        board.place_stone(player, coord)
    config = engine.SearchConfig(max_depth=2, time_limit_seconds=10)
    minimax = engine.choose_move(board, "B", "B", config, seed=19)
    alpha_beta = engine.choose_move(board, "B", "C", config, seed=19)
    assert minimax.completed_depth == alpha_beta.completed_depth == 2
    assert minimax.root_value == alpha_beta.root_value
    assert minimax.move in minimax.optimal_moves
    assert alpha_beta.move in minimax.optimal_moves
    assert alpha_beta.nodes <= minimax.nodes


def test_board_evaluator_does_not_rescan_each_cell_through_getter() -> None:
    engine = _engine()

    class CountingBoard(GomokuBoard):
        reads = 0

        def get(self, coord):
            self.reads += 1
            return super().get(coord)

    board = CountingBoard(15)
    board.place_stone("B", (7, 7))
    board.place_stone("W", (7, 8))
    engine.evaluate_board(board, "B")
    assert board.reads <= board.size * board.size * 2


def test_alpha_beta_records_pruning() -> None:
    engine = _engine()
    board = GomokuBoard(5)
    for player, coord in [
        ("B", (2, 2)),
        ("W", (2, 3)),
        ("B", (3, 2)),
        ("W", (1, 3)),
    ]:
        board.place_stone(player, coord)
    config = engine.SearchConfig(max_depth=3, time_limit_seconds=10)
    minimax = engine.choose_move(board, "B", "B", config, seed=20)
    alpha_beta = engine.choose_move(board, "B", "C", config, seed=20)
    assert alpha_beta.cutoffs > 0
    assert alpha_beta.nodes < minimax.nodes
    assert alpha_beta.root_value == minimax.root_value


def test_mcts_obeys_simulation_limit() -> None:
    engine = _engine()
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    config = engine.SearchConfig(time_limit_seconds=10, mcts_simulations=24)
    result = engine.choose_move(board, "W", "D", config, seed=2026)
    assert result.simulations == 24
    assert result.nodes > 0


def test_mcts_completes_rollouts_on_sparse_15_by_15_with_shared_budget() -> None:
    engine = _engine()
    board = GomokuBoard(15)
    for player, coord in [
        ("B", (7, 7)),
        ("W", (7, 8)),
        ("B", (8, 7)),
        ("W", (6, 8)),
    ]:
        board.place_stone(player, coord)
    result = engine.choose_move(
        board,
        "B",
        "D",
        engine.SearchConfig(time_limit_seconds=0.05, mcts_simulations=10_000),
        seed=1001,
    )
    assert result.simulations >= 1
    assert board.get(result.move) is None


@pytest.mark.parametrize("model", ["B", "C", "D"])
def test_search_deadline_stays_within_the_shared_budget(model: str) -> None:
    engine = _engine()
    board = GomokuBoard(15)
    for player, coord in [
        ("B", (7, 7)),
        ("W", (7, 8)),
        ("B", (8, 7)),
        ("W", (6, 8)),
    ]:
        board.place_stone(player, coord)
    budget = 0.01
    result = engine.choose_move(
        board,
        "B",
        model,
        engine.SearchConfig(time_limit_seconds=budget),
        seed=2002,
    )
    # The search uses a soft wall-clock deadline; allow a small scheduler jitter.
    assert result.elapsed_seconds <= budget + 0.005
    assert result.move[0] in range(board.size) and result.move[1] in range(board.size)
    assert board.get(result.move) is None


def test_randomized_models_are_repeatable_for_same_seed() -> None:
    engine = _engine()
    board = GomokuBoard(7)
    board.place_stone("B", (3, 3))
    board.place_stone("W", (3, 4))
    config = engine.SearchConfig(max_depth=1, time_limit_seconds=1, mcts_simulations=16)
    for model in ("A", "D"):
        first = engine.choose_move(board, "B", model, config, seed=999)
        second = engine.choose_move(board, "B", model, config, seed=999)
        assert first.move == second.move
        assert first.simulations == second.simulations


def test_zero_time_budget_returns_legal_fallback_and_marks_timeout() -> None:
    engine = _engine()
    board = GomokuBoard(7)
    board.place_stone("B", (3, 3))
    result = engine.choose_move(
        board,
        "W",
        "B",
        engine.SearchConfig(max_depth=5, time_limit_seconds=0),
        seed=4,
    )
    assert board.get(result.move) is None
    assert result.timed_out
    assert result.completed_depth == 0


def test_search_rejects_full_board() -> None:
    engine = _engine()
    board = GomokuBoard(5)
    for row in range(5):
        for col in range(5):
            board.place_stone("B" if (row + col) % 2 == 0 else "W", (row, col))
    with pytest.raises(ValueError, match="no legal move"):
        engine.choose_move(board, "B", "C", engine.SearchConfig(), seed=0)


def test_existing_heuristic_accepts_injected_random_generator() -> None:
    board = GomokuBoard(7)
    board.place_stone("B", (3, 3))
    first = _engine().select_ai_move(board, "W", "B", rng=random.Random(82))
    second = _engine().select_ai_move(board, "W", "B", rng=random.Random(82))
    assert first == second
