from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.research.abcd_ladder_v5_1 import build_schedule, load_config
from src.research.abcd_ladder_v5_1_analysis import (
    audit_v5_report_scores,
    model_result_points,
    summarize_records,
    validate_records,
)
from src.research.abcd_ladder_v5_1_runner import _validate_v51_record
from tests.test_abcd_ladder_v5_1_runner import _complete_legal_record


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "abcd_ladder_v5_1.json"
V5_RESULTS = ROOT / "research" / "v5" / "results"


def test_score_points_are_wins_plus_half_draws_and_rate_uses_actual_games():
    rows = [
        {"winner_model": "A"}, {"winner_model": None}, {"winner_model": "B"},
    ]
    assert model_result_points(rows, "A") == {"wins": 1, "draws": 1, "losses": 1, "games": 3,
                                               "points": 1.5, "score_rate": 0.5}
    assert model_result_points(rows, "B")["points"] == 1.5


def test_v5_report_first_listed_points_are_recomputed_from_raw_games():
    result = audit_v5_report_scores(V5_RESULTS, ROOT / "research" / "v5" / "REPORT.md")
    actual = {row["pair"]: row for row in result["pairs"]}

    assert result["raw_games_valid"] is True
    assert actual["A-C"]["expected_first_points"] == 0.5
    assert actual["A-C"]["reported_first_points"] == 0.2
    assert actual["A-D"]["expected_first_points"] == 7.5
    assert actual["A-D"]["reported_first_points"] == 3.8
    assert actual["B-D"]["expected_first_points"] == 14.0
    assert actual["B-D"]["reported_first_points"] == 7.0
    assert actual["C-D"]["expected_first_points"] == 14.0
    assert result["mismatched_report_pairs"] == ["A-C", "A-D", "B-D", "C-D"]


def test_v51_analysis_uses_the_two_swapped_games_and_exact_points():
    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    # The runner replay adapter accepts the frozen engine object used in the V5 configuration.
    from src.research.abcd_ladder_v5_1 import engine_config
    engine = engine_config(config)
    records = [_complete_legal_record(game, config) for game in group["games"]]
    for record, spec in zip(records, group["games"], strict=True):
        _validate_v51_record(record, spec, engine)

    check = validate_records(records, config, schedule=build_schedule(config), require_complete=False)
    summary = summarize_records(records, config, schedule=build_schedule(config))
    row = next(item for item in summary["pairwise"] if item["pair"] == "A-B")
    group_row = summary["groups"][0]

    assert check["valid"] is True
    assert row["games"] == 2
    assert row["first_points"] == row["second_points"] == 1.0
    assert row["first_score_rate"] == row["second_score_rate"] == 0.5
    assert group_row["first_points"] == group_row["wins_first"] + group_row["draws"] / 2
    assert group_row["second_points"] == group_row["wins_second"] + group_row["draws"] / 2


def test_analysis_rejects_non_swapped_or_duplicate_games():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    group = schedule[0]
    record = _complete_legal_record(group["games"][0], config)
    duplicate = deepcopy(record)
    with pytest.raises(ValueError, match="Duplicate V5.1 game ID"):
        validate_records([record, duplicate], config, schedule=schedule)
