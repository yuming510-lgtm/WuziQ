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
