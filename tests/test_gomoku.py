"""Tests for the core Gomoku logic."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gomoku import (
    GomokuBoard,
    deserialize_state,
    parse_move,
    select_ai_move,
    serialize_state,
)


@pytest.fixture
def board() -> GomokuBoard:
    return GomokuBoard(15)


def test_horizontal_victory(board: GomokuBoard) -> None:
    player = "B"
    moves = [(7, col) for col in range(3, 8)]
    for coord in moves[:-1]:
        board.place_stone(player, coord)
    board.place_stone(player, moves[-1])
    assert board.check_victory(moves[-1], player)


def test_move_out_of_bounds(board: GomokuBoard) -> None:
    with pytest.raises(ValueError):
        board.place_stone("B", (board.size, board.size))


def test_cannot_overwrite_existing_stone(board: GomokuBoard) -> None:
    coord = (5, 5)
    board.place_stone("B", coord)
    with pytest.raises(ValueError):
        board.place_stone("W", coord)


def test_parse_move_letter_and_number(board: GomokuBoard) -> None:
    assert parse_move("H8", board.size) == (7, 7)


def test_parse_move_space_separated(board: GomokuBoard) -> None:
    assert parse_move("8 7", board.size) == (7, 6)


def test_parse_move_out_of_bounds(board: GomokuBoard) -> None:
    with pytest.raises(ValueError):
        parse_move("Z30", board.size)


def test_undo_restores_previous_state(board: GomokuBoard) -> None:
    board.place_stone("B", (0, 0))
    board.place_stone("W", (0, 1))

    undone = board.undo()
    assert undone == ("W", (0, 1))
    assert board.get((0, 1)) is None
    assert board.get((0, 0)) == "B"

    second = board.undo()
    assert second == ("B", (0, 0))
    assert all(cell is None for row in board.grid for cell in row)


def test_serialise_and_deserialise_are_equivalent() -> None:
    board = GomokuBoard(10)
    moves = [
        ("B", (0, 0)),
        ("W", (1, 0)),
        ("B", (0, 1)),
        ("W", (1, 1)),
        ("B", (0, 2)),
    ]
    for player, coord in moves:
        board.place_stone(player, coord)

    serialized = serialize_state(board, "W", "white")
    loaded_board, next_player, ai_mode = deserialize_state(serialized)

    assert next_player == "W"
    assert ai_mode == "white"
    assert loaded_board.size == board.size
    assert loaded_board.history == board.history

    last_move = moves[-1][1]
    assert loaded_board.check_victory(last_move, "B") == board.check_victory(last_move, "B")


def test_ai_move_is_legal() -> None:
    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    board.place_stone("W", (3, 3))

    for _ in range(10):
        coord = select_ai_move(board, "W", "B")
        assert board.in_bounds(*coord)
        assert board.get(coord) is None
