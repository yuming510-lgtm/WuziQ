"""Independent, rule-labelled tactical positions and frozen-network search probes."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from ...gomoku import GomokuBoard, Player

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_CATEGORIES = ("direct_win", "unique_block", "false_threat", "edge_win")


def _inside(coord: tuple[int, int]) -> bool:
    return 0 <= coord[0] < 9 and 0 <= coord[1] < 9


def _random_line(rng: np.random.Generator, *, edge: bool = False) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    starts = []
    for dr, dc in _DIRECTIONS:
        for row in range(9):
            for col in range(9):
                end = (row + 4 * dr, col + 4 * dc)
                if not _inside(end):
                    continue
                coords = [(row + i * dr, col + i * dc) for i in range(5)]
                if edge and not any(r in {0, 8} or c in {0, 8} for r, c in (coords[0], coords[-1])):
                    continue
                starts.append((coords, (dr, dc)))
    if not starts:
        raise RuntimeError("no legal five-cell line was generated")
    return starts[int(rng.integers(0, len(starts)))]


def _board_from_position(position: list[dict[str, Any]]) -> tuple[GomokuBoard, Player]:
    board = GomokuBoard(9)
    expected: Player = "B"
    for move in position:
        if not isinstance(move, dict) or move.get("player") != expected:
            raise ValueError("tactical position must alternate B/W from black")
        try:
            coord = (int(move["row"]), int(move["col"]))
            board.place_stone(expected, coord)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("tactical position has an illegal move") from exc
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("tactical position must be nonterminal")
        expected = "W" if expected == "B" else "B"
    return board, expected


def _winning_actions(board: GomokuBoard, player: Player) -> list[int]:
    wins: list[int] = []
    for row in range(9):
        for col in range(9):
            if board.get((row, col)) is not None:
                continue
            clone = GomokuBoard(9)
            for existing_player, existing_coord in board.history:
                clone.place_stone(existing_player, existing_coord)
            clone.place_stone(player, (row, col))
            if clone.check_victory((row, col), player):
                wins.append(row * 9 + col)
    return wins


def _consecutive_stones(board: GomokuBoard, player: Player, minimum: int) -> bool:
    for row in range(9):
        for col in range(9):
            for dr, dc in _DIRECTIONS:
                end = (row + (minimum - 1) * dr, col + (minimum - 1) * dc)
                if not _inside(end):
                    continue
                if all(board.get((row + i * dr, col + i * dc)) == player for i in range(minimum)):
                    return True
    return False


def exact_tactical_actions(puzzle: dict[str, Any]) -> list[int]:
    to_move = puzzle.get("to_move")
    if to_move not in {"B", "W"}:
        raise ValueError("tactical side-to-move must be B or W")
    board, actual_to_move = _board_from_position(puzzle.get("position", []))
    if actual_to_move != to_move:
        raise ValueError("tactical side-to-move differs from its legal move sequence")
    category = puzzle.get("category")
    if category in {"direct_win", "edge_win"}:
        actions = _winning_actions(board, to_move)
        if not actions:
            raise ValueError("direct-win puzzle has no immediate rule-legal win")
        if category == "edge_win" and not any(
            row in {0, 8} or col in {0, 8} for row, col in (divmod(action, 9) for action in actions)
        ):
            raise ValueError("edge-win target is not on the board edge")
        return actions
    if category not in {"unique_block", "false_threat"}:
        raise ValueError("unknown tactical category")
    opponent: Player = "W" if to_move == "B" else "B"
    opponent_wins = _winning_actions(board, opponent)
    own_wins = _winning_actions(board, to_move)
    if not opponent_wins or own_wins:
        raise ValueError("blocking puzzle must have an opponent win and no current-side win")
    safe: list[int] = []
    for row in range(9):
        for col in range(9):
            if board.get((row, col)) is not None:
                continue
            clone = GomokuBoard(9)
            for existing_player, existing_coord in board.history:
                clone.place_stone(existing_player, existing_coord)
            clone.place_stone(to_move, (row, col))
            if not _winning_actions(clone, opponent):
                safe.append(row * 9 + col)
    if len(safe) != 1:
        raise ValueError("blocking puzzle must have one exact safe move")
    if category == "false_threat" and not _consecutive_stones(board, to_move, 3):
        raise ValueError("false-threat puzzle must include a nonwinning three-stone decoy")
    return safe


def _transformed_key(position: list[dict[str, Any]]) -> str:
    cells = [["." for _ in range(9)] for _ in range(9)]
    for move in position:
        cells[int(move["row"])][int(move["col"])] = str(move["player"])
    variants = []
    for reflect in (False, True):
        for turns in range(4):
            board = [row[:] for row in cells]
            if reflect:
                board = [list(reversed(row)) for row in board]
            for _ in range(turns):
                board = [list(row) for row in zip(*board[::-1])]
            variants.append("".join(cell for row in board for cell in row))
    return min(variants)


def puzzle_symmetry_key(puzzle: dict[str, Any]) -> str:
    return _transformed_key(puzzle["position"])


def validate_v4_2_puzzle(puzzle: dict[str, Any]) -> None:
    if puzzle.get("format_version") != 1 or puzzle.get("category") not in _CATEGORIES:
        raise ValueError("unsupported V4.2 tactical puzzle")
    if not puzzle.get("puzzle_id") or puzzle.get("label_source") != "exact_legal_five_in_a_row_rules":
        raise ValueError("tactical puzzle must have an exact rule label source")
    expected = exact_tactical_actions(puzzle)
    registered = puzzle.get("optimal_actions")
    if not isinstance(registered, list) or sorted(map(int, registered)) != sorted(expected):
        raise ValueError("tactical puzzle differs from exact rule labels")


def _place_random_fillers(
    rng: np.random.Generator,
    occupied: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(9) for c in range(9) if (r, c) not in occupied]
    if len(candidates) < count:
        raise ValueError("not enough empty filler cells")
    indexes = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indexes]


def _interleave(black: list[tuple[int, int]], white: list[tuple[int, int]]) -> list[dict[str, Any]]:
    if len(black) != len(white):
        raise ValueError("tactical opening must have equal black and white stone counts")
    moves = []
    for black_coord, white_coord in zip(black, white):
        moves.append({"player": "B", "row": black_coord[0], "col": black_coord[1]})
        moves.append({"player": "W", "row": white_coord[0], "col": white_coord[1]})
    return moves


def _make_direct_puzzle(rng: np.random.Generator, puzzle_id: str, *, edge: bool) -> dict[str, Any]:
    for _ in range(100_000):
        line, _direction = _random_line(rng, edge=edge)
        target_candidates = [i for i, (row, col) in enumerate(line)
                             if not edge or row in {0, 8} or col in {0, 8}]
        target_index = target_candidates[int(rng.integers(0, len(target_candidates)))]
        black = [coord for i, coord in enumerate(line) if i != target_index]
        white = _place_random_fillers(rng, set(line), 4)
        position = _interleave(black, white)
        puzzle = {"format_version": 1, "puzzle_id": puzzle_id,
                  "category": "edge_win" if edge else "direct_win", "to_move": "B",
                  "position": position, "label_source": "exact_legal_five_in_a_row_rules"}
        try:
            puzzle["optimal_actions"] = exact_tactical_actions(puzzle)
            board, _ = _board_from_position(position)
            if _winning_actions(board, "W"):
                continue
            validate_v4_2_puzzle(puzzle)
            return puzzle
        except ValueError:
            continue
    raise RuntimeError("could not generate a direct-win tactical puzzle")


def _make_block_puzzle(
    rng: np.random.Generator,
    puzzle_id: str,
    *,
    false_threat: bool,
) -> dict[str, Any]:
    category = "false_threat" if false_threat else "unique_block"
    for _ in range(100_000):
        line, direction = _random_line(rng)
        white = list(line[:4])
        target = line[4]
        before = (line[0][0] - direction[0], line[0][1] - direction[1])
        blocker = [before] if _inside(before) else []
        occupied = set(white + blocker)
        if false_threat:
            threat_line, _ = _random_line(rng)
            threat = list(threat_line[:3])
            if set(threat).intersection(occupied | {target}) or not _inside(threat_line[3]):
                continue
            occupied.update(threat)
            remaining_black = 4 - len(blocker) - len(threat)
            if remaining_black < 0:
                continue
            filler = _place_random_fillers(rng, occupied | {target}, remaining_black)
            black = blocker + threat + filler
        else:
            black = blocker + _place_random_fillers(rng, occupied | {target}, 4 - len(blocker))
        if len(set(black)) != 4 or len(set(black).intersection(white + [target])):
            continue
        position = _interleave(black, white)
        puzzle = {"format_version": 1, "puzzle_id": puzzle_id, "category": category,
                  "to_move": "B", "position": position,
                  "label_source": "exact_legal_five_in_a_row_rules"}
        try:
            puzzle["optimal_actions"] = exact_tactical_actions(puzzle)
            if len(puzzle["optimal_actions"]) != 1 or int(puzzle["optimal_actions"][0]) != target[0] * 9 + target[1]:
                continue
            validate_v4_2_puzzle(puzzle)
            return puzzle
        except ValueError:
            continue
    raise RuntimeError(f"could not generate a {category} tactical puzzle")


def build_v4_2_tactical_set(*, seed: int = 742_202, per_category: int = 16) -> list[dict[str, Any]]:
    if isinstance(per_category, bool) or int(per_category) != per_category or per_category < 1:
        raise ValueError("per_category must be a positive integer")
    rng = np.random.default_rng(int(seed))
    puzzles: list[dict[str, Any]] = []
    seen: set[str] = set()
    builders = {
        "direct_win": lambda puzzle_id: _make_direct_puzzle(rng, puzzle_id, edge=False),
        "unique_block": lambda puzzle_id: _make_block_puzzle(rng, puzzle_id, false_threat=False),
        "false_threat": lambda puzzle_id: _make_block_puzzle(rng, puzzle_id, false_threat=True),
        "edge_win": lambda puzzle_id: _make_direct_puzzle(rng, puzzle_id, edge=True),
    }
    for category in _CATEGORIES:
        accepted = 0
        attempts = 0
        while accepted < int(per_category):
            attempts += 1
            if attempts > int(per_category) * 2_000:
                raise RuntimeError(f"could not produce enough unique {category} positions")
            puzzle = builders[category](f"g-v4-2-{category}-{accepted:04d}")
            key = puzzle_symmetry_key(puzzle)
            if key in seen:
                continue
            seen.add(key)
            puzzles.append(puzzle)
            accepted += 1
    for puzzle in puzzles:
        validate_v4_2_puzzle(puzzle)
    return puzzles
