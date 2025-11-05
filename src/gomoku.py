"""Core game logic for a minimalist Gomoku implementation."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Player = str
Coordinate = Tuple[int, int]


@dataclass
class GomokuBoard:
    """Represent a Gomoku board and provide move validation and win checks."""

    size: int = 15
    grid: List[List[Optional[Player]]] = field(init=False)
    history: List[Tuple[Player, Coordinate]] = field(default_factory=list)

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
        self.history.append((player, coord))

    def is_full(self) -> bool:
        """Return ``True`` if no more moves are possible."""

        return all(cell is not None for row in self.grid for cell in row)

    def undo(self) -> Optional[Tuple[Player, Coordinate]]:
        """Undo the latest move and return it if present."""

        if not self.history:
            return None
        player, coord = self.history.pop()
        row, col = coord
        self.grid[row][col] = None
        return player, coord

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

    def potential_line_length(self, coord: Coordinate, player: Player) -> int:
        """Return the maximum line length ``player`` could achieve through ``coord``."""

        row, col = coord
        directions = (
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        )
        max_length = 0
        for dr, dc in directions:
            length = 1
            length += self._count_direction(row, col, dr, dc, player)
            length += self._count_direction(row, col, -dr, -dc, player)
            max_length = max(max_length, length)
        return max_length

    def line_details(
        self, coord: Coordinate, player: Player
    ) -> List[Tuple[int, bool, bool]]:
        """Return length and openness for each direction through ``coord``."""

        row, col = coord
        details: List[Tuple[int, bool, bool]] = []
        directions = (
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        )
        for dr, dc in directions:
            forward = self._count_direction(row, col, dr, dc, player)
            backward = self._count_direction(row, col, -dr, -dc, player)
            length = forward + backward + 1
            forward_open = self._is_open_end(row, col, dr, dc, forward)
            backward_open = self._is_open_end(row, col, -dr, -dc, backward)
            details.append((length, forward_open, backward_open))
        return details

    def _is_open_end(
        self, row: int, col: int, dr: int, dc: int, steps: int
    ) -> bool:
        """Return ``True`` if the end of a run is open (on-board and empty)."""

        target_row = row + (steps + 1) * dr
        target_col = col + (steps + 1) * dc
        return self.in_bounds(target_row, target_col) and self.grid[target_row][target_col] is None

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

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable mapping of the current board state."""

        return {
            "size": self.size,
            "moves": [
                {"player": player, "row": coord[0], "col": coord[1]}
                for player, coord in self.history
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GomokuBoard":
        """Create a board from a serialised mapping."""

        if "size" not in data:
            raise ValueError("Serialized board is missing the 'size' field.")
        size = int(data["size"])
        board = cls(size)
        moves = data.get("moves", [])
        for entry in moves:  # type: ignore[assignment]
            player = entry["player"]  # type: ignore[index]
            row = int(entry["row"])  # type: ignore[index]
            col = int(entry["col"])  # type: ignore[index]
            board.place_stone(player, (row, col))
        return board


def serialize_state(
    board: GomokuBoard, next_player: Player, ai_mode: str = "off"
) -> str:
    """Serialise a game state, including metadata, into a JSON string."""

    payload = {
        "board": board.to_dict(),
        "next_player": next_player,
        "ai_mode": ai_mode,
    }
    return json.dumps(payload)


def deserialize_state(serialized: str) -> Tuple[GomokuBoard, Player, str]:
    """Return the game components stored in ``serialized`` JSON text."""

    payload = json.loads(serialized)
    board = GomokuBoard.from_dict(payload["board"])
    next_player = str(payload.get("next_player", "B"))
    ai_mode = str(payload.get("ai_mode", "off"))
    return board, next_player, ai_mode


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


def select_ai_move(board: GomokuBoard, ai_player: Player, opponent: Player) -> Coordinate:
    """Return a coordinate for the AI player using a lightweight heuristic."""

    if board.is_full():
        raise ValueError("Board is full; no AI move is possible.")

    center = (board.size // 2, board.size // 2)
    if not board.history and board.get(center) is None:
        return center

    candidates: List[Tuple[int, int]] = []
    scores: List[float] = []
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for row in range(board.size):
        for col in range(board.size):
            if board.grid[row][col] is not None:
                continue
            coord = (row, col)
            ai_length = board.potential_line_length(coord, ai_player)
            opponent_length = board.potential_line_length(coord, opponent)

            score = 0.0
            if ai_length >= 5:
                score += 1_000_000
            if opponent_length >= 5:
                score += 900_000

            opponent_details = board.line_details(coord, opponent)
            for length, open_forward, open_backward in opponent_details:
                open_count = int(open_forward) + int(open_backward)
                if length >= 4 and open_count:
                    score += 50_000
                elif length == 3 and open_count == 2:
                    score += 25_000

            # Encourage adjacency to any stones to avoid isolated random plays.
            adjacency = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if board.in_bounds(nr, nc) and board.grid[nr][nc] is not None:
                    adjacency += 1
            score += adjacency * 100

            # Mild random factor to diversify otherwise identical choices.
            score += random.random()

            candidates.append(coord)
            scores.append(score)

    max_score = max(scores)
    best_indices = [idx for idx, value in enumerate(scores) if value == max_score]
    choice_index = random.choice(best_indices)
    return candidates[choice_index]
