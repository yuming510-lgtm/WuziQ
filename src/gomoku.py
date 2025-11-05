"""Core game logic for a minimalist Gomoku implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Player = str
Coordinate = Tuple[int, int]


@dataclass
class GomokuBoard:
    """Represent a Gomoku board and provide move validation and win checks."""

    size: int = 15
    grid: List[List[Optional[Player]]] = field(init=False)

    def __post_init__(self) -> None:
        if self.size < 5:
            raise ValueError("Board size must be at least 5 to play Gomoku.")
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]

    def in_bounds(self, row: int, col: int) -> bool:
        """Return whether the provided coordinate lies inside the board."""

        return 0 <= row < self.size and 0 <= col < self.size

    def get(self, coord: Coordinate) -> Optional[Player]:
        """Return the stone at the coordinate if present."""

        row, col = coord
        return self.grid[row][col]

    def place_stone(self, player: Player, coord: Coordinate) -> None:
        """Place ``player``'s stone at ``coord`` if the move is legal.

        Raises:
            ValueError: If the coordinate is outside the board or already occupied.
        """

        row, col = coord
        if not self.in_bounds(row, col):
            raise ValueError("Move is out of bounds.")
        if self.grid[row][col] is not None:
            raise ValueError("Cell is already occupied.")
        self.grid[row][col] = player

    def is_full(self) -> bool:
        """Return ``True`` if no more moves are possible."""

        return all(cell is not None for row in self.grid for cell in row)

    def check_victory(self, coord: Coordinate, player: Player) -> bool:
        """Return ``True`` if ``player`` has five in a row through ``coord``."""

        row, col = coord
        directions = (
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal down-right
            (1, -1),  # diagonal down-left
        )
        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= 5:
                return True
        return False

    def _count_direction(
        self, row: int, col: int, dr: int, dc: int, player: Player
    ) -> int:
        """Count consecutive stones for ``player`` from ``(row, col)`` toward ``dr, dc``."""

        total = 0
        r, c = row + dr, col + dc
        while self.in_bounds(r, c) and self.grid[r][c] == player:
            total += 1
            r += dr
            c += dc
        return total


def parse_move(raw: str, size: int) -> Coordinate:
    """Parse user input into a zero-based coordinate ``(row, col)``.

    The function accepts chess-style inputs such as ``"H8"`` (column letter then row
    number) and whitespace-separated coordinates like ``"8 7"``.

    Args:
        raw: The raw input string provided by the player.
        size: Current board dimension, used to validate bounds.

    Raises:
        ValueError: If the input cannot be interpreted or lies outside the board.
    """

    text = raw.strip().upper()
    if not text:
        raise ValueError("Empty input is not a valid move.")

    if " " in text:
        parts = text.split()
        if len(parts) != 2:
            raise ValueError("Provide row and column as two numbers or e.g. 'H8'.")
        try:
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("Row and column must be numbers.") from exc
    else:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if text[0] not in letters:
            raise ValueError("Coordinate must start with a letter column.")
        col = letters.index(text[0])
        try:
            row = int(text[1:]) - 1
        except ValueError as exc:
            raise ValueError("Row index must be numeric after the column letter.") from exc

    if row < 0 or col < 0 or row >= size or col >= size:
        raise ValueError("Move is out of bounds for the current board size.")

    return (row, col)


def format_board(board: GomokuBoard) -> str:
    """Return a human-readable ASCII representation of ``board``."""

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    header_letters = letters[: board.size]
    header = "   " + " ".join(header_letters)
    rows = [header]
    for idx, row in enumerate(board.grid, start=1):
        label = f"{idx:2d}"
        cells = [cell if cell is not None else "." for cell in row]
        rows.append(f"{label} " + " ".join(cells))
    return "\n".join(rows)
