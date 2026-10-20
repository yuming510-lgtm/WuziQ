from __future__ import annotations

from pathlib import Path

import pytest

from src.research.abcd_ladder_v5 import build_schedule, load_config
from src.research.abcd_ladder_v5_analysis import summarize_records, validate_game_record, wilson_fractional
from src.research.abcd_ladder_v5_report import build_report


CONFIG = Path(__file__).parents[1] / "configs" / "abcd_ladder_v5.json"


def _valid_black_win(spec: dict) -> dict:
    moves = []
    for index, stone in enumerate(spec["opening"]):
        moves.append({
            "ply": index, "player": stone["player"], "model": None,
            "row": stone["row"], "col": stone["col"], "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 0, "tree_depth": 0,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
            "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": True,
        })
    continuation = [("B", 4, 0), ("W", 0, 0), ("B", 4, 1), ("W", 0, 1),
                    ("B", 4, 2), ("W", 0, 2), ("B", 4, 3)]
    elapsed_total = 0.0
    for player, row, col in continuation:
        model = spec["black_model"] if player == "B" else spec["white_model"]
        elapsed = 0.001
        elapsed_total += elapsed
        moves.append({
            "ply": len(moves), "player": player, "model": model,
            "row": row, "col": col, "elapsed_seconds": elapsed,
            "root_value": 0, "nodes": 0, "completed_depth": 0, "tree_depth": 0,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
            "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False,
        })
    return {
        **spec,
        "implementation_commit": "test-commit",
        "moves": moves,
        "move_count": len(moves),
        "winner_color": "B",
        "winner_model": spec["black_model"],
        "outcome": "b_win",
        "termination": "five_in_row",
        "timeout_count": 0,
        "search_nodes_total": 0,
        "simulations_total": 0,
        "tree_depth_max": 0,
        "thinking_seconds_total": elapsed_total,
    }


def test_valid_game_replays_and_reconstructs_actual_winner():
    config = load_config(CONFIG)
    first_group = build_schedule(config)[0]
    record = _valid_black_win(first_group["games"][0])

    result = validate_game_record(record, config)

    assert result["winner_model"] == "A"
    assert result["move_count"] == 11
    assert result["termination"] == "five_in_row"


def test_replay_rejects_repeated_move_even_if_winner_metadata_looks_valid():
    config = load_config(CONFIG)
    record = _valid_black_win(build_schedule(config)[0]["games"][0])
    record["moves"][-1]["row"] = 4
    record["moves"][-1]["col"] = 4

    with pytest.raises(ValueError, match="repeats or misplaces"):
        validate_game_record(record, config)


def test_incomplete_color_pair_is_not_counted_as_a_group_or_promotion():
    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    record = _valid_black_win(group["games"][0])

    summary = summarize_records([record], config)

    assert summary["actual_games"] == 1
    assert summary["actual_groups"] == 0
    assert summary["route_state"]["routes"]["A>B"]["levels"][0]["status"] == "incomplete_pair"
    assert summary["route_state"]["routes"]["A>B"]["highest_passed_budget"] is None


def test_fractional_score_interval_is_bounded_and_requires_data():
    assert wilson_fractional([]) is None
    interval = wilson_fractional([2.0, 0.0])
    assert interval is not None
    assert 0.0 <= interval[0] < 0.5 < interval[1] <= 1.0


def test_group_intervals_use_fractional_group_scores_and_report_renders(tmp_path):
    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    records = [_valid_black_win(spec) for spec in group["games"]]
    summary = summarize_records(records, config)
    pair = summary["pairwise"][0]

    assert pair["first_score_rate"] == 0.5
    assert pair["second_score_rate"] == 0.5
    assert pair["first_group_score_ci95"][0] < 0.5 < pair["first_group_score_ci95"][1]
    report_path = build_report(summary, tmp_path / "results", tmp_path / "v5")
    assert report_path.is_file()
    assert (tmp_path / "v5" / "figures" / "ladder_gradient.svg").is_file()
    assert "未运行" in report_path.read_text(encoding="utf-8")
