from __future__ import annotations

from src.research.v6.schedule import build_schedule
from src.research.v6.state import derive_state


def _group(schedule, pair, versions):
    return next(
        group for group in schedule
        if group["pair"] == list(pair) and group["model_versions"] == versions
    )


def _outcome_rows(group, winners):
    return [
        {
            "game_id": game["game_id"],
            "group_id": group["group_id"],
            "winner_model": winner,
        }
        for game, winner in zip(group["games"], winners)
    ]


def test_failure_stops_only_that_route_and_infers_reciprocal_higher_version() -> None:
    schedule, routes = build_schedule()
    first = _group(schedule, ("A", "B"), {"A": "natural", "B": "50ms"})
    rows = _outcome_rows(first, ["B", "B"])

    state = derive_state(rows, schedule, routes)

    assert state["routes"]["A>B"]["status"] == "failed"
    assert state["routes"]["A>B"]["failed_budget"] == "50ms"
    assert state["routes"]["A>B"]["levels"][1]["status"] == "inferred_fail"
    assert state["routes"]["B@50ms>A"]["status"] == "completed"
    assert state["routes"]["A>C"]["status"] == "active"
    reciprocal = state["routes"]["B@200ms>A"]["levels"][0]
    assert reciprocal["status"] == "inferred_pass"
    assert reciprocal["games"] == 0
    assert reciprocal["score"] is None


def test_shared_actual_result_overrides_an_inference_and_is_not_double_counted() -> None:
    schedule, routes = build_schedule()
    first = _group(schedule, ("A", "B"), {"A": "natural", "B": "50ms"})
    second = _group(schedule, ("A", "B"), {"A": "natural", "B": "200ms"})
    rows = _outcome_rows(first, ["B", "B"]) + _outcome_rows(second, ["A", "A"])

    state = derive_state(rows, schedule, routes)

    reverse = state["routes"]["B@200ms>A"]["levels"][0]
    assert reverse["status"] == "failed"
    assert reverse["games"] == 2
    assert reverse["score"] == 0
    assert state["groups"][second["group_id"]]["status"] == "played"


def test_unplayed_versions_are_not_given_empirical_scores() -> None:
    schedule, routes = build_schedule()
    first = _group(schedule, ("A", "B"), {"A": "natural", "B": "50ms"})
    state = derive_state(_outcome_rows(first, ["A", "B"]), schedule, routes)
    assert state["routes"]["A>B"]["levels"][1]["status"] == "pending"
    assert state["routes"]["A>B"]["levels"][1]["score"] is None
    assert state["routes"]["A>B"]["levels"][1]["games"] == 0
