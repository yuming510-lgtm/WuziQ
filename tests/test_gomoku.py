"""Tests for the core Gomoku logic."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gomoku import GomokuBoard, parse_move


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
