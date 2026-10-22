from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6.puzzles import board_for_position, load_tactical_positions
from src.research.v6.engine import V6Agent


def test_frozen_tactical_positions_are_legal_alternating_nonterminal_states() -> None:
    positions = load_tactical_positions()
    assert len(positions) == 12
    assert {position["category"] for position in positions} >= {
        "opening", "attack", "defense", "continuous_attack", "counterattack",
        "false_threat", "edge_attack", "edge_defense", "multi_threat_attack",
    }
    for position in positions:
        board = board_for_position(position)
        assert len(board.history) == len(position["moves"])
        assert not board.is_full()


def test_e5_takes_immediate_win_before_timed_search() -> None:
    positions = load_tactical_positions()
    position = next(row for row in positions if row["position_id"] == "immediate_win")
    board = board_for_position(position)

    result = V6Agent("E5").choose_move(board, "B", time_limit_seconds=0.05, seed=510)

    assert result.move in {(4, 0), (4, 5)}
    assert result.completed_depth == 1
    assert result.stats["root_tactical"] == "immediate_win"


def test_e5_blocks_unique_one_ply_win_before_timed_search() -> None:
    positions = load_tactical_positions()
    position = next(row for row in positions if row["position_id"] == "immediate_block")
    board = board_for_position(position)

    result = V6Agent("E5").choose_move(board, "B", time_limit_seconds=0.05, seed=510)

    assert result.move == (2, 4)
    assert result.stats["root_tactical"] == "forced_block"


def test_proof_evaluation_marks_counterwin_refutation_and_proofs() -> None:
    assert importlib.util.find_spec("src.research.v6") is not None
    from src.research.v6.puzzles import evaluate_tactical_result

    positions = load_tactical_positions()
    position = next(row for row in positions if row["position_id"] == "counterwin_refutes_vcf")
    board = board_for_position(position)
    result = V6Agent("E1").choose_move(board, "B", time_limit_seconds=0.1, seed=9)
    evaluation = evaluate_tactical_result(position, "E1", result, board)
    assert evaluation["correct"] is True
    assert evaluation["proof_claimed"] is False
