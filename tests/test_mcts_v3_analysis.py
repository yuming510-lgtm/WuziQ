"""Statistical aggregation and diagnostics for V3 high-compute results."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _analysis():
    return importlib.import_module("src.research.mcts_v3_analysis")


def _condition_records(opponent: str = "A", budget: float = 10.0) -> list[dict]:
    records = []
    winners = ("D3", "A", None, None, "D3", "D3")
    for opening_index in range(3):
        for color_swap in (False, True):
            index = opening_index * 2 + int(color_swap)
            black_model = "D3" if color_swap else opponent
            white_model = opponent if color_swap else "D3"
            winner = opponent if winners[index] == "A" else winners[index]
            winner_color = None
            if winner is not None:
                winner_color = "B" if winner == black_model else "W"
            phase_name = f"d3-{int(budget)}s-vs-{opponent.lower()}"
            pair_id = f"{phase_name}-{opponent}-D3-o{opening_index:04d}"
            records.append(
                {
                    "game_id": f"{phase_name}-game-{index}",
                    "phase_name": phase_name,
                    "track": "high-compute",
                    "board_size": 9,
                    "matchup": sorted([opponent, "D3"]),
                    "budget_seconds_by_model": {opponent: 0.05, "D3": budget},
                    "pair_id": pair_id,
                    "opening_index": opening_index,
                    "color_swap": color_swap,
                    "black_model": black_model,
                    "white_model": white_model,
                    "winner_model": winner,
                    "winner_color": winner_color,
                    "moves": [
                        {
                            "model": "D3",
                            "elapsed_seconds": budget,
                            "nodes": 80,
                            "simulations": 11 + index,
                            "completed_depth": 2 + opening_index,
                            "timed_out": True,
                        },
                        {
                            "model": opponent,
                            "elapsed_seconds": 0.06,
                            "nodes": 40,
                            "simulations": 0,
                            "completed_depth": 0,
                            "timed_out": True,
                        },
                    ],
                    "opening": [{"player": "B", "row": 4, "col": 4}],
                }
            )
    return records


def test_v3_summary_uses_actual_simulations_and_reports_paired_uncertainty() -> None:
    summary = _analysis().summarize_v3_records(_condition_records(), bootstrap_samples=50)

    phase = summary["phases"]["d3-10s-vs-a"]
    assert phase["games"] == 6
    assert phase["opponent_model"] == "A"
    assert "approximate" in phase["interval_method"]
    assert (phase["D3"]["wins"], phase["D3"]["losses"], phase["D3"]["draws"]) == (
        3,
        1,
        2,
    )
    assert phase["D3"]["score_rate"] == pytest.approx(4 / 6)
    assert phase["D3"]["simulations_total"] == sum(11 + index for index in range(6))
    assert phase["D3"]["moves"] == 6
    assert phase["D3"]["max_tree_depth"] == 4
    assert phase["D3"]["search_timeout_flags"] == 6
    assert phase["D3"]["actual_over_budget_moves"] == 0
    assert len(phase["D3"]["score_rate_paired_wilson_ci95"]) == 2
    assert phase["D3_color_split"]["black"]["games"] == 3
    assert phase["D3_color_split"]["white"]["games"] == 3
    assert phase["win_matrix"]["D3"]["A"]["games"] == 6


def test_v3_summary_rejects_duplicate_game_ids() -> None:
    record = _condition_records()[0]
    with pytest.raises(ValueError, match="Duplicate game_id"):
        _analysis().summarize_v3_records([record, dict(record)])


def test_v3_trend_svg_contains_all_opponents_and_budget_values(tmp_path: Path) -> None:
    module = _analysis()
    records = [
        record
        for opponent in ("A", "B", "C")
        for budget in (10.0, 20.0, 50.0)
        for record in _condition_records(opponent, budget)
    ]
    summary = module.summarize_v3_records(records, bootstrap_samples=50)
    path = tmp_path / "trend.svg"

    module.write_d3_budget_trend_svg(summary, path)

    svg = path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "A" in svg and "B" in svg and "C" in svg
    assert "10" in svg and "20" in svg and "50" in svg
    assert "95%" in svg


def test_v3_validator_can_require_complete_registered_schedule() -> None:
    config = {
        "experiment_id": "wuziq-mcts-high-compute-v3-analysis-test",
        "base_seed": 20261018,
        "opening_seeds": [111, 222, 333],
        "opening_plies": 4,
        "search": {"time_limit_seconds": 0.05, "mcts_simulations": 1000},
        "phases": [
            {
                "name": "d3-10s-vs-a",
                "track": "high-compute",
                "board_size": 9,
                "games_per_matchup": 6,
                "matchups": [["D3", "A"]],
                "budget_seconds_by_model": {"D3": 10.0, "A": 0.05},
            }
        ],
    }

    partial = _analysis().validate_v3_records([], config, require_complete=False)
    assert partial["game_count"] == 0
    assert partial["scheduled_game_count"] == 6
    assert partial["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        _analysis().validate_v3_records([], config, require_complete=True)
