from __future__ import annotations

from pathlib import Path

import pytest

from src.research.abcd_ladder_v5_1 import build_schedule, load_config
from src.research.abcd_ladder_v5_1_reuse import derive_state


CONFIG = Path(__file__).parents[1] / "configs" / "abcd_ladder_v5_1.json"


def _records(group: dict, winner_by_game: list[str | None]) -> list[dict]:
    return [
        {**game, "winner_model": winner, "outcome": "draw" if winner is None else "model_win"}
        for game, winner in zip(group["games"], winner_by_game, strict=True)
    ]


def test_shared_grid_result_after_one_route_fails_advances_only_live_route():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    earlier = [group for group in schedule if group["pair"][0] == "A"]
    records = [row for group in earlier for row in _records(group, ["A", "A"])]
    bc_baseline = next(
        group for group in schedule
        if group["pair"] == ["B", "C"]
        and group["model_versions"] == {"B": "50ms", "C": "50ms"}
    )
    bc_second_column = next(
        group for group in schedule
        if group["pair"] == ["B", "C"]
        and group["model_versions"] == {"B": "50ms", "C": "200ms"}
    )
    records.extend(_records(bc_baseline, ["C", "C"]))
    records.extend(_records(bc_second_column, ["C", "C"]))

    state = derive_state(records, config, schedule)
    routes = state["routes"]

    assert routes["B@50ms>C"]["status"] == "failed"
    assert routes["B@50ms>C"]["failed_budget"] == "50ms"
    assert routes["B@50ms>C"]["levels"][1]["status"] == "shared_result_after_failure"
    assert routes["C@200ms>B"]["levels"][0]["status"] == "passed"
    assert routes["B@200ms>C"]["status"] == "active"
    assert routes["B@200ms>C"]["levels"][0]["status"] == "pending"


def test_one_win_one_loss_and_two_draws_both_meet_one_point_threshold():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    a_first = schedule[0]
    a_second = schedule[1]
    records = _records(a_first, ["A", "B"])
    records.extend(_records(a_second, [None, None]))

    state = derive_state(records, config, schedule)

    assert state["routes"]["A>B"]["levels"][0]["score"] == 1.0
    assert state["routes"]["A>B"]["levels"][1]["score"] == 1.0
    assert state["routes"]["A>B"]["levels"][1]["status"] == "passed"
    assert state["routes"]["B@50ms>A"]["status"] == "completed"
    assert state["routes"]["B@50ms>A"]["levels"][0]["status"] == "passed"


def test_results_after_first_unfinished_active_group_are_rejected():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    later_records = _records(schedule[1], ["A", "A"])

    with pytest.raises(ValueError, match="after an unfinished active group"):
        derive_state(later_records, config, schedule)


def test_low_opponent_loss_infers_higher_opponent_versions_without_games():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    ab50 = next(
        group for group in schedule
        if group["pair"] == ["A", "B"] and group["model_versions"]["B"] == "50ms"
    )

    state = derive_state(_records(ab50, ["B", "B"]), config, schedule)

    assert state["routes"]["A>B"]["status"] == "failed"
    assert state["routes"]["A>B"]["levels"][1]["status"] == "not_run_after_failure"
    assert state["routes"]["B@200ms>A"]["levels"][0]["status"] == "inferred_pass"
    assert state["routes"]["B@200ms>A"]["levels"][0]["games"] == 0
    assert state["groups"]["v5-1-9x9-a-natural-b-200ms-o731102"]["status"] == "inferred_dominated"


def test_search_family_low_budget_loss_infers_higher_budget_reverse_matchups():
    config = load_config(CONFIG)
    schedule = [
        group for group in build_schedule(config)
        if group["pair"] == ["B", "C"]
    ]
    bc50 = next(
        group for group in schedule
        if group["model_versions"] == {"B": "50ms", "C": "50ms"}
    )
    bc200 = next(
        group for group in schedule
        if group["model_versions"] == {"B": "50ms", "C": "200ms"}
    )
    records = _records(bc50, ["C", "C"])

    state = derive_state(records, config, schedule)

    assert state["routes"]["B@50ms>C"]["status"] == "failed"
    assert state["routes"]["C@200ms>B"]["levels"][0]["status"] == "inferred_pass"
    assert state["routes"]["C@200ms>B"]["levels"][0]["games"] == 0
    assert state["groups"][bc200["group_id"]]["status"] == "inferred_dominated"
    assert state["pending_group_id"] != bc200["group_id"]


def test_partial_game_in_an_inferred_group_is_preserved_but_not_scored():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    ab50 = next(
        group for group in schedule
        if group["pair"] == ["A", "B"] and group["model_versions"]["B"] == "50ms"
    )
    ab200 = next(
        group for group in schedule
        if group["pair"] == ["A", "B"] and group["model_versions"]["B"] == "200ms"
    )
    records = _records(ab50, ["B", "B"]) + _records(ab200, ["B", "A"])
    records = records[:-1]

    state = derive_state(records, config, schedule)

    assert state["groups"][ab200["group_id"]]["status"] == "inferred_dominated_partial_observation"
    assert state["pending_group_id"] == "v5-1-9x9-a-natural-c-50ms-o731104"
    assert state["routes"]["B@200ms>A"]["levels"][0]["status"] == "inferred_pass"
