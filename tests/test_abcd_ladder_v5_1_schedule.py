from __future__ import annotations

from pathlib import Path

from src.research.abcd_ladder_v5_1 import build_routes, build_schedule, load_config


CONFIG = Path(__file__).parents[1] / "configs" / "abcd_ladder_v5_1.json"


def test_schedule_has_66_independent_routes_and_168_unique_paired_groups():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    routes = build_routes(config)
    group_ids = [group["group_id"] for group in schedule]
    games = [game for group in schedule for game in group["games"]]

    assert len(routes) == 66
    assert len(schedule) == 168
    assert len(group_ids) == len(set(group_ids))
    assert len(games) == 336
    assert len({game["game_id"] for game in games}) == 336
    assert all(len(group["route_refs"]) == 2 for group in schedule)


def test_pairwise_grid_has_fixed_challenger_versions_and_independent_opponent_ladders():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    routes = build_routes(config)

    assert routes["D@20s>B"]["challenger_budget_label"] == "20s"
    assert [level["opponent_budget_label"] for level in routes["D@20s>B"]["levels"]] == [
        "50ms", "200ms", "1s", "5s", "10s", "20s", "50s",
    ]
    assert routes["D@50s>C"]["challenger_budget_label"] == "50s"
    assert len(routes["B@200ms>C"]["levels"]) == 7

    d20_b_rungs = [
        group for group in schedule
        if any(ref["route_id"] == "D@20s>B" for ref in group["route_refs"])
    ]
    assert [group["model_versions"]["B"] for group in d20_b_rungs] == [
        "50ms", "200ms", "1s", "5s", "10s", "20s", "50s",
    ]
    assert all(group["model_versions"]["D"] == "20s" for group in d20_b_rungs)


def test_schedule_is_topological_diverse_and_color_paired():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    routes = build_routes(config)
    positions = {group["group_id"]: i for i, group in enumerate(schedule)}

    for route in routes.values():
        levels = route["levels"]
        assert [level["level_index"] for level in levels] == list(range(len(levels)))
        assert [positions[level["group_id"]] for level in levels] == sorted(
            positions[level["group_id"]] for level in levels
        )

    assert {group["opening_seed"] for group in schedule} == set(config["opening_seeds"])
    for group in schedule:
        first, second = group["games"]
        assert first["opening"] == second["opening"]
        assert first["opening_seed"] == second["opening_seed"] == group["opening_seed"]
        assert first["black_model"] == second["white_model"]
        assert first["white_model"] == second["black_model"]
        assert first["pair"][0] != first["pair"][1]


def test_each_group_references_two_distinct_opposite_family_routes():
    schedule = build_schedule(load_config(CONFIG))

    for group in schedule:
        left, right = group["route_refs"]
        assert left["route_id"] != right["route_id"]
        assert ">" in left["route_id"] and ">" in right["route_id"]
        assert left["level_index"] >= 0 and right["level_index"] >= 0
        assert group["pair"][0] != group["pair"][1]
