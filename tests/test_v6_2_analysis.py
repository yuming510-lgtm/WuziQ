from __future__ import annotations

from src.research.v6_2.analysis import aggregate_compute, render_champion_svg


def test_compute_aggregation_uses_real_decision_rows_and_excludes_opening_moves() -> None:
    records = [{
        "game_id": "fixture",
        "moves": [
            {"model": None, "opening": True, "elapsed_seconds": 0, "nodes": 0},
            {"model": "C", "opening": False, "elapsed_seconds": 2.0, "nodes": 100,
             "cutoffs": 20, "simulations": 0, "completed_depth": 5,
             "tree_depth": 5, "timed_out": True},
            {"model": "D", "opening": False, "elapsed_seconds": 200.0, "nodes": 900,
             "cutoffs": 0, "simulations": 50_000, "completed_depth": 4,
             "tree_depth": 4, "timed_out": True},
        ],
    }]

    result = aggregate_compute(records)

    assert result["C"]["moves"] == 1
    assert result["C"]["thinking_seconds"] == 2.0
    assert result["C"]["nodes"] == 100
    assert result["C"]["average_seconds_per_move"] == 2.0
    assert result["D"]["simulations"] == 50_000
    assert result["D"]["timeouts"] == 1
    assert result["D"]["max_tree_depth"] == 4


def test_champion_svg_renders_score_and_compute_comparisons() -> None:
    standings = {
        "C": {"points": 3.0},
        "D": {"points": 2.0},
        "E5": {"points": 1.0},
    }
    compute = {
        "C": {"thinking_seconds": 100.0},
        "D": {"thinking_seconds": 400.0},
        "E5": {"thinking_seconds": 200.0},
    }

    svg = render_champion_svg(standings, compute)

    assert svg.startswith("<svg")
    assert all(model in svg for model in ("C", "D", "E5"))
    assert "积分" in svg and "计算秒数" in svg
    assert "viewBox" in svg
