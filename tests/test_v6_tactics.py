from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gomoku import GomokuBoard


def _v6():
    assert importlib.util.find_spec("src.research.v6") is not None
    from src.research.v6 import V6Agent

    return V6Agent


def _position(moves: list[tuple[str, tuple[int, int]]], size: int = 9) -> GomokuBoard:
    board = GomokuBoard(size)
    for player, coord in moves:
        board.place_stone(player, coord)
    return board


@pytest.mark.parametrize("model", ["E1", "E2", "E3", "E4"])
def test_tactical_models_prove_an_open_four_without_mutating_board(model: str) -> None:
    V6Agent = _v6()
    board = _position([("B", (4, col)) for col in (3, 4, 5)])
    before = deepcopy((board.grid, board.history))

    result = V6Agent(model).choose_move(board, "B", time_limit_seconds=1.0, seed=73)

    assert result.proof_status == "proven"
    assert result.proof_kind in {"VCF", "VCT", "TSS", "PNS"}
    assert board.in_bounds(*result.move) and board.get(result.move) is None
    board.place_stone("B", result.move)
    assert board.check_victory(result.move, "B") or result.proof_certificate
    board.undo()
    assert (board.grid, board.history) == before


@pytest.mark.parametrize("model", ["E1", "E2", "E3", "E4"])
def test_tactical_fallback_blocks_one_ply_win_and_zero_budget_is_legal(model: str) -> None:
    V6Agent = _v6()
    board = _position([("W", (2, col)) for col in range(4)], size=5)

    result = V6Agent(model).choose_move(board, "B", time_limit_seconds=0.0, seed=9)

    assert board.get(result.move) is None
    assert result.timed_out
    assert result.proof_status == "unknown"


@pytest.mark.parametrize("model", ["E1", "E2", "E3", "E4"])
def test_tactical_shared_fallback_blocks_one_ply_win(model: str) -> None:
    V6Agent = _v6()
    board = _position([("W", (2, col)) for col in range(4)], size=5)

    result = V6Agent(model).choose_move(board, "B", time_limit_seconds=0.5, seed=9)

    assert result.move == (2, 4)


def test_vcf_certificate_fails_if_a_critical_defense_is_omitted() -> None:
    V6Agent = _v6()
    board = _position([("B", (4, col)) for col in (3, 4, 5)])
    result = V6Agent("E1").choose_move(board, "B", time_limit_seconds=1.0, seed=73)

    assert result.proof_certificate
    assert V6Agent.verify_proof_certificate(board, "B", result.proof_certificate)
    broken = deepcopy(result.proof_certificate)
    broken["root"]["defense_coverage"] -= 1
    assert not V6Agent.verify_proof_certificate(board, "B", broken)


def test_tactical_result_is_deterministic_and_board_restores_after_timeout() -> None:
    V6Agent = _v6()
    board = _position([("B", (4, 4)), ("W", (4, 5)), ("B", (5, 4)), ("W", (3, 5))])
    before = deepcopy((board.grid, board.history))
    agent = V6Agent("E2")

    first = agent.choose_move(board, "B", time_limit_seconds=0.0, seed=510)
    second = agent.choose_move(board, "B", time_limit_seconds=0.0, seed=510)

    assert first.move == second.move
    assert first.proof_status == second.proof_status == "unknown"
    assert (board.grid, board.history) == before


def test_vct_proves_a_cross_open_three_and_tss_reports_dependencies() -> None:
    V6Agent = _v6()
    board = _position([
        ("B", (4, 3)), ("B", (4, 5)), ("B", (3, 4)), ("B", (5, 4)),
    ])

    result = V6Agent("E2", max_attack_depth=4).choose_move(
        board, "B", time_limit_seconds=2.0, seed=101,
    )
    tss_result = V6Agent("E3", max_attack_depth=4).choose_move(
        board, "B", time_limit_seconds=2.0, seed=101,
    )

    assert result.proof_status == "proven"
    assert result.proof_kind == "VCT"
    assert V6Agent.verify_proof_certificate(board, "B", result.proof_certificate)
    assert tss_result.stats["threat_nodes"] > 0
    assert tss_result.stats["dependency_edges"] >= 0


def test_tactical_search_does_not_claim_win_when_defender_has_counter_win() -> None:
    V6Agent = _v6()
    board = _position(
        [("B", (4, col)) for col in (3, 4, 5)]
        + [("W", (0, col)) for col in range(4)]
    )

    for model in ("E1", "E2", "E3", "E4"):
        result = V6Agent(model, max_attack_depth=3).choose_move(
            board, "B", time_limit_seconds=0.5, seed=13,
        )
        assert result.proof_status != "proven"
        assert result.proof_certificate is None
