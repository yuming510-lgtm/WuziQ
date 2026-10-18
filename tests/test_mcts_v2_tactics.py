from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gomoku import GomokuBoard
from src.research.mcts_d2 import _immediate_tactics
from src.research.mcts_v2_tactics import audit_tactics


def test_crossing_three_stone_lines_are_not_an_immediate_win_or_block() -> None:
    coordinates = ((4, 3), (4, 5), (3, 4), (5, 4))
    own_cross = GomokuBoard(9)
    for coord in coordinates:
        own_cross.place_stone("B", coord)
    wins, _ = _immediate_tactics(own_cross, [(4, 4)], "B")
    assert wins == []

    opponent_cross = GomokuBoard(9)
    for coord in coordinates:
        opponent_cross.place_stone("W", coord)
    _, blocks = _immediate_tactics(opponent_cross, [(4, 4)], "B")
    assert blocks == []


def _forced_edge_four(d2_move: tuple[int, int]) -> dict:
    moves = []
    final_black_move = (3, 4) if d2_move == (4, 4) else (4, 4)
    for player, model, coord in [
        ("B", "C", (4, 0)),
        ("W", "D2", (0, 0)),
        ("B", "C", (4, 1)),
        ("W", "D2", (0, 1)),
        ("B", "C", (4, 2)),
        ("W", "D2", (0, 2)),
        ("B", "C", (4, 3)),
        ("W", "D2", d2_move),
        ("B", "C", final_black_move),
    ]:
        moves.append(
            {
                "ply": len(moves),
                "player": player,
                "model": model,
                "row": coord[0],
                "col": coord[1],
                "opening": False,
            }
        )
    return {
        "game_id": "forced-edge-four",
        "board_size": 9,
        "matchup": ["C", "D2"],
        "moves": moves,
        "winner_model": "C",
    }


def test_tactical_audit_finds_a_missed_unique_edge_block() -> None:
    result = audit_tactics([_forced_edge_four((3, 3))])

    assert result["by_model"]["D2"]["missed_unique_forced_blocks"] == 1
    assert result["examples"][0]["required_moves"] == [[4, 4]]
    assert result["examples"][0]["chosen_move"] == [3, 3]


def test_tactical_audit_accepts_the_unique_forced_block() -> None:
    result = audit_tactics([_forced_edge_four((4, 4))])

    assert result["by_model"]["D2"]["missed_unique_forced_blocks"] == 0
    assert result["examples"] == []
