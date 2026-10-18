from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.mcts_v2_metrics import summarize_phases


def _game(game_id: str, winner: str | None, color_swap: bool) -> dict:
    return {
        "game_id": game_id,
        "experiment_id": "statistics-test",
        "phase_name": "fair-9x9-50ms",
        "track": "fair",
        "board_size": 9,
        "matchup": ["C", "D2"],
        "budget_seconds_by_model": {"C": 0.05, "D2": 0.05},
        "pair_id": "fair-9x9-50ms-C-D2-o0000",
        "opening_index": 0,
        "color_swap": color_swap,
        "black_model": "D2" if color_swap else "C",
        "white_model": "C" if color_swap else "D2",
        "winner_model": winner,
        "winner_color": None if winner is None else ("B" if winner == ("D2" if color_swap else "C") else "W"),
        "moves": [
            {
                "model": "D2",
                "elapsed_seconds": 0.05,
                "nodes": 100,
                "simulations": 20,
                "tree_depth": 3,
                "timed_out": False,
            }
        ],
        "opening": [
            {"player": "B" if not color_swap else "W", "row": 4, "col": 4},
            {"player": "W" if not color_swap else "B", "row": 5, "col": 4},
        ],
    }


def test_v2_summary_builds_win_matrix_and_clusters_color_swapped_games() -> None:
    records = [
        _game("g0", "C", False),
        _game("g1", "D2", True),
    ]

    summary = summarize_phases(records, bootstrap_samples=200, seed=31)
    phase = summary["phases"]["fair-9x9-50ms"]

    assert summary["game_count"] == 2
    assert phase["ranking"][0]["score_rate"] == 0.5
    assert phase["win_matrix"]["D2"]["C"] == {
        "games": 2,
        "wins": 1,
        "losses": 1,
        "draws": 0,
        "score_rate": 0.5,
    }
    d2 = next(row for row in phase["ranking"] if row["model"] == "D2")
    assert d2["score_rate_ci95"] == [0.5, 0.5]
    assert d2["average_simulations_per_move"] == 20
    assert d2["model_simulations_total"] == 40
    assert d2["score_points_per_model_thinking_second"] == 10


def test_v2_summary_rejects_duplicate_game_ids() -> None:
    game = _game("duplicate", "C", False)
    with pytest.raises(ValueError, match="Duplicate game_id"):
        summarize_phases([game, dict(game)], bootstrap_samples=10)


def test_paired_wilson_interval_stays_non_degenerate_after_all_losses() -> None:
    records = [
        _game("loss-black", "C", False),
        _game("loss-white", "C", True),
    ]

    phase = summarize_phases(records, bootstrap_samples=50)["phases"][
        "fair-9x9-50ms"
    ]
    d2 = next(row for row in phase["ranking"] if row["model"] == "D2")

    assert d2["score_rate_ci95"] == [0.0, 0.0]
    assert d2["score_rate_paired_wilson_ci95"][0] == 0.0
    assert d2["score_rate_paired_wilson_ci95"][1] > 0.0
