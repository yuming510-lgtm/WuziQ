"""Command-line interface for the minimalist Gomoku game."""
from __future__ import annotations

import argparse
from typing import Iterable

from .gomoku import (
    GomokuBoard,
    deserialize_state,
    format_board,
    parse_move,
    select_ai_move,
    serialize_state,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Gomoku CLI."""

    parser = argparse.ArgumentParser(description="Play a minimalist Gomoku game.")
    parser.add_argument(
        "--size",
        type=int,
        default=15,
        help="Board dimension (default: 15). Must be at least 5.",
    )
    parser.add_argument(
        "--ai",
        choices=("off", "black", "white"),
        default="off",
        help="Enable a simple AI opponent controlling the given colour.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for running the Gomoku command-line interface."""

    args = parse_args(argv)
    board = GomokuBoard(args.size)
    current_player: str = "B"
    ai_mode: str = args.ai
    ai_player = _ai_player_from_mode(ai_mode)

    print("Minimalist Gomoku")
    print(f"Board size: {board.size}x{board.size}\n")
    print(format_board(board))

    while True:
        if ai_player and current_player == ai_player:
            opponent = "W" if current_player == "B" else "B"
            coord = select_ai_move(board, current_player, opponent)
            board.place_stone(current_player, coord)
            move_label = _format_coord(coord)
            print(f"\nAI ({current_player}) plays {move_label}.")
        else:
            try:
                move_input = input(f"Player {current_player}, enter your move: ")
            except EOFError:
                print("\nInput ended. Exiting the game.")
                return

            stripped = move_input.strip()
            if not stripped:
                print("Please enter a coordinate or a command.")
                continue

            if stripped.lower() in {":quit", ":exit", "quit", "exit"}:
                print("Game aborted.")
                return

            if stripped.startswith(":"):
                command, *rest = stripped[1:].split()
                if command.lower() == "undo":
                    undone = board.undo()
                    if undone is None:
                        print("No moves to undo.")
                    else:
                        undone_player, _ = undone
                        current_player = undone_player
                        if ai_player and undone_player == ai_player and board.history:
                            second = board.undo()
                            if second:
                                current_player = second[0]
                                print(
                                    "Undid the latest AI and player moves to maintain turn order."
                                )
                            else:
                                print("AI move undone.")
                        else:
                            print("Last move undone.")
                    print()
                    print(format_board(board))
                    continue
                if command.lower() == "save":
                    if not rest:
                        print("Usage: :save filename.json")
                        continue
                    filename = rest[0]
                    try:
                        serialized = serialize_state(board, current_player, ai_mode)
                        with open(filename, "w", encoding="utf-8") as handle:
                            handle.write(serialized)
                    except OSError as exc:
                        print(f"Failed to save game: {exc}")
                        continue
                    print(f"Game saved to {filename}.")
                    continue
                if command.lower() == "load":
                    if not rest:
                        print("Usage: :load filename.json")
                        continue
                    filename = rest[0]
                    try:
                        with open(filename, "r", encoding="utf-8") as handle:
                            serialized = handle.read()
                        board, current_player, ai_mode = deserialize_state(serialized)
                        ai_player = _ai_player_from_mode(ai_mode)
                    except (OSError, ValueError) as exc:
                        print(f"Failed to load game: {exc}")
                        continue
                    print(f"Game loaded from {filename}.")
                    print()
                    print(format_board(board))
                    continue

                print(f"Unknown command: {stripped}")
                continue

            try:
                coord = parse_move(stripped, board.size)
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

        current_player = "W" if current_player == "B" else "B"


def _ai_player_from_mode(mode: str) -> str | None:
    """Return the player symbol controlled by the AI for ``mode``."""

    if mode == "black":
        return "B"
    if mode == "white":
        return "W"
    return None


def _format_coord(coord: tuple[int, int]) -> str:
    """Return a human-readable coordinate label (e.g. ``H8``)."""

    row, col = coord
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"{letters[col]}{row + 1}"


if __name__ == "__main__":
    main()
