"""Post-game exact one-ply threat audit for V2 tournament records."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ..gomoku import GomokuBoard
from .mcts_d2 import _immediate_tactics


_MODELS = ("A", "B", "C", "D1", "D2")


def audit_tactics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Replay games and flag missed immediate wins or a unique forced block.

    This is an exact board-state audit, not a search used by any agent. It scans
    every empty square before each recorded move with the same four-ray line
    count used by D2's one-ply tactical detector.
    """

    counts: dict[str, dict[str, int]] = {
        model: {
            "decisions": 0,
            "immediate_win_available_decisions": 0,
            "missed_immediate_wins": 0,
            "unique_forced_block_decisions": 0,
            "missed_unique_forced_blocks": 0,
        }
        for model in _MODELS
    }
    examples: list[dict[str, Any]] = []
    game_count = 0

    for record in records:
        game_count += 1
        size = int(record["board_size"])
        board = GomokuBoard(size)
        for move in record.get("moves", ()):
            model = move.get("model")
            coordinate = move.get("row"), move.get("col")
            if (
                model in counts
                and type(coordinate[0]) is int
                and type(coordinate[1]) is int
                and 0 <= coordinate[0] < size
                and 0 <= coordinate[1] < size
                and board.get(coordinate) is None
            ):
                player = str(move["player"])
                candidates = [
                    (row, col)
                    for row in range(size)
                    for col in range(size)
                    if board.get((row, col)) is None
                ]
                wins, blocks = _immediate_tactics(board, candidates, player)
                current = counts[str(model)]
                current["decisions"] += 1
                tactic: str | None = None
                required: list[tuple[int, int]] = []
                if wins:
                    current["immediate_win_available_decisions"] += 1
                    if coordinate not in wins:
                        current["missed_immediate_wins"] += 1
                        tactic, required = "missed_immediate_win", wins
                elif len(blocks) == 1:
                    current["unique_forced_block_decisions"] += 1
                    if coordinate != blocks[0]:
                        current["missed_unique_forced_blocks"] += 1
                        tactic, required = "missed_unique_forced_block", blocks

                if tactic and len(examples) < 200:
                    examples.append(
                        {
                            "game_id": str(record["game_id"]),
                            "phase_name": record.get("phase_name"),
                            "board_size": size,
                            "matchup": list(record.get("matchup", ())),
                            "model": str(model),
                            "ply": int(move.get("ply", 0)),
                            "tactic": tactic,
                            "required_moves": [list(coord) for coord in required],
                            "chosen_move": list(coordinate),
                            "winner_model": record.get("winner_model"),
                            "termination": record.get("termination"),
                        }
                    )

            row, col = move.get("row"), move.get("col")
            if type(row) is int and type(col) is int and board.in_bounds(row, col):
                player = str(move["player"])
                if board.get((row, col)) is not None:
                    raise ValueError(f"Illegal replay move in {record['game_id']} at {(row, col)}")
                board.place_stone(player, (row, col))

    return {
        "games_audited": game_count,
        "exact_scan_scope": "all empty squares before each legal model move",
        "by_model": counts,
        "miss_count": sum(
            row["missed_immediate_wins"] + row["missed_unique_forced_blocks"]
            for row in counts.values()
        ),
        "examples_truncated": len(examples) == 200,
        "examples": examples,
    }
