"""Command-line interface for the minimalist Gomoku game."""
from __future__ import annotations

import argparse
from itertools import cycle
from typing import Iterable

from .gomoku import GomokuBoard, format_board, parse_move


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Gomoku CLI."""

    parser = argparse.ArgumentParser(description="Play a minimalist Gomoku game.")
    parser.add_argument(
        "--size",
        type=int,
        default=15,
        help="Board dimension (default: 15). Must be at least 5.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for running the Gomoku command-line interface."""

    args = parse_args(argv)
    board = GomokuBoard(args.size)
    players = cycle(("B", "W"))
    current_player = next(players)

    print("Minimalist Gomoku")
    print(f"Board size: {board.size}x{board.size}\n")
    print(format_board(board))

    while True:
        try:
            move_input = input(f"Player {current_player}, enter your move: ")
        except EOFError:
            print("\nInput ended. Exiting the game.")
            return

        if move_input.strip().lower() in {"quit", "exit"}:
            print("Game aborted.")
            return

        try:
            coord = parse_move(move_input, board.size)
            board.place_stone(current_player, coord)
        except ValueError as exc:
            print(f"Invalid move: {exc}")
            continue

        print()
        print(format_board(board))

        if board.check_victory(coord, current_player):
            print(f"Player {current_player} wins!")
            return
        if board.is_full():
            print("The board is full. It's a draw!")
            return

        current_player = next(players)


if __name__ == "__main__":
    main()
