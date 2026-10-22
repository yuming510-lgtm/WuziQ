from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.research.v6.config import FAMILIES, SEARCH_FAMILIES, load_config
from src.research.v6.schedule import build_schedule, validate_schedule
from src.research.abcd_ladder_v5_1 import build_schedule as build_v51_schedule


def test_v6_materializes_every_independent_budget_route_once() -> None:
    schedule, routes = build_schedule()
    assert len(schedule) == 1428
    assert len(routes) == 456
    assert sum(len(group["games"]) for group in schedule) == 2856
    assert len({game["game_id"] for group in schedule for game in group["games"]}) == 2856
    assert all(group["pair"][0] != group["pair"][1] for group in schedule)

    for family in SEARCH_FAMILIES:
        for budget in ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s"):
            for opponent in FAMILIES:
                if opponent != family:
                    assert f"{family}@{budget}>{opponent}" in routes
    assert set(routes["A>E1"]["levels"][i]["opponent_budget_label"] for i in range(7)) == {
        "50ms", "200ms", "1s", "5s", "10s", "20s", "50s",
    }


def test_every_group_pairs_one_opening_and_swaps_colors() -> None:
    schedule, routes = build_schedule()
    assert validate_schedule(schedule, routes)["same_family_groups"] == 0
    for group in schedule:
        first, second = group["games"]
        assert first["opening"] == second["opening"]
        assert first["opening_seed"] == second["opening_seed"]
        assert first["black_model"] == second["white_model"]
        assert first["white_model"] == second["black_model"]
        assert first["game_seed"] != second["game_seed"]


def test_abcd_seed_assignment_matches_v5_1_for_exact_reuse_candidates() -> None:
    schedule, _routes = build_schedule()
    v51_config = json.loads((Path(__file__).resolve().parents[1] / "configs/abcd_ladder_v5_1.json").read_text())
    v51_schedule = build_v51_schedule(v51_config)
    v51_groups = {
        (tuple(group["pair"]), tuple(group["model_versions"][family] for family in group["pair"])): group
        for group in v51_schedule
    }
    v6_groups = {
        (tuple(group["pair"]), tuple(group["model_versions"][family] for family in group["pair"])): group
        for group in schedule if set(group["pair"]).issubset({"A", "B", "C", "D"})
    }
    assert len(v6_groups) == len(v51_groups) == 168
    for group in schedule:
        if not set(group["pair"]).issubset({"A", "B", "C", "D"}):
            continue
        key = (tuple(group["pair"]), tuple(group["model_versions"][family] for family in group["pair"]))
        assert group["opening_seed"] == v51_groups[key]["opening_seed"]
        assert group["opening"] == v51_groups[key]["opening"]


def test_config_rejects_changed_budget_and_source_engine_hashes() -> None:
    config = load_config()
    changed = deepcopy(config)
    changed["budget_ladder"][0]["seconds"] = 0.1
    with pytest.raises(ValueError, match="budget seconds"):
        load_config(changed)

    changed = deepcopy(config)
    changed["engine_sha256"]["src/ai.py"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(changed)

    changed = deepcopy(config)
    changed["tactical_positions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="puzzle fixture hash mismatch"):
        load_config(changed)
