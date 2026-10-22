from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import SearchConfig, choose_move
from src.gomoku import GomokuBoard
from src.research.v6 import E5Options, E5Search, V6Agent, run_ablation
from src.research.v6.e5 import Zobrist


def _board() -> GomokuBoard:
    board = GomokuBoard(5)
    for player, coord in [
        ("B", (2, 2)),
        ("W", (2, 3)),
        ("B", (3, 2)),
        ("W", (1, 3)),
    ]:
        board.place_stone(player, coord)
    return board


def test_e5_matches_c_root_value_at_same_complete_shallow_depth() -> None:
    board = _board()
    before = deepcopy((board.grid, board.history))
    config = SearchConfig(max_depth=2, time_limit_seconds=5.0, candidate_radius=2)
    alpha_beta = choose_move(board, "B", "C", config, seed=61)
    baseline = E5Search(E5Options(False, False, False, False)).choose_move(
        board, "B", config, seed=61, max_depth=2,
    )

    assert alpha_beta.completed_depth == baseline.completed_depth == 2
    assert alpha_beta.root_value == baseline.root_value
    assert board.get(baseline.move) is None
    assert (board.grid, board.history) == before


def test_zobrist_hash_is_deterministic_side_sensitive_and_incremental() -> None:
    board = _board()
    zobrist = Zobrist(board.size, seed=91)
    black = zobrist.position(board, "B")
    assert black == Zobrist(board.size, seed=91).position(board, "B")
    assert black != zobrist.position(board, "W")

    move = (2, 1)
    child = zobrist.after_move(black, "B", move)
    board.place_stone("B", move)
    assert child == zobrist.position(board, "W")
    board.undo()


def test_e5_returns_legal_move_at_short_budget_and_exposes_component_metrics() -> None:
    board = _board()
    result = V6Agent("E5", e5_options=E5Options()).choose_move(
        board, "B", time_limit_seconds=0.05, seed=88, max_depth=24,
    )
    assert board.get(result.move) is None
    assert result.elapsed_seconds < 0.5
    assert "tt_hits" in result.stats
    assert "extensions" in result.stats


def test_component_ablation_runs_each_variant_on_same_position_at_fixed_depth() -> None:
    rows = run_ablation([("fixture", _board(), "B")], depth=2, time_limit_seconds=5.0, seed=19)
    assert len(rows) == 6
    assert {row["completed_depth"] for row in rows} == {2}
    assert {row["variant"] for row in rows} == {
        "E5_base", "E5_PVS_only", "E5_TT_only",
        "E5_ordering_only", "E5_extensions_only", "E5_all",
    }
    assert len({tuple(row["move"]) for row in rows}) >= 1
