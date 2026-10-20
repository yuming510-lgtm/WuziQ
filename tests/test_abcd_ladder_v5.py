from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.research.abcd_ladder_v5 import (
    build_schedule,
    derive_routes,
    load_config,
    run_experiment,
    settings_for_model,
)


CONFIG = Path(__file__).parents[1] / "configs" / "abcd_ladder_v5.json"


def _game_result(spec: dict, winner: str | None) -> dict:
    return {
        **spec,
        "winner_model": winner,
        "winner_color": None,
        "outcome": "draw" if winner is None else "model_win",
        "moves": [],
    }


def _records_before(groups: list[dict], group_id: str) -> list[dict]:
    records = []
    for group in groups:
        if group["group_id"] == group_id:
            break
        records.extend(_game_result(game, None) for game in group["games"])
    return records


def test_frozen_schedule_has_sixty_unique_groups_and_120_games():
    config = load_config(CONFIG)
    groups = build_schedule(config)
    game_ids = [game["game_id"] for group in groups for game in group["games"]]

    assert len(groups) == 60
    assert len({group["group_id"] for group in groups}) == 60
    assert len(game_ids) == 120
    assert len(set(game_ids)) == 120
    assert len(config["opening_catalog"]) == 12


def test_config_requires_hash_pins_for_all_decision_engines():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["engine_sha256"].pop("src/ai.py")

    with pytest.raises(ValueError, match="engine_sha256 must pin exactly"):
        load_config(config)


def test_every_group_uses_one_opening_and_swaps_model_colors():
    groups = build_schedule(load_config(CONFIG))

    for group in groups:
        first, second = group["games"]
        assert first["opening"] == second["opening"]
        assert first["opening_seed"] == second["opening_seed"]
        assert first["black_model"] == second["white_model"]
        assert first["white_model"] == second["black_model"]


def test_a_stays_natural_while_opponent_budget_climbs_and_internal_routes_hold_baseline():
    groups = build_schedule(load_config(CONFIG))
    ab = [g for g in groups if g["pair"] == ["A", "B"]]
    assert ab[0]["games"][0]["budget_seconds_by_model"] == {"A": None, "B": 0.05}
    assert ab[-1]["games"][0]["budget_seconds_by_model"] == {"A": None, "B": 50.0}
    bc_up = next(g for g in groups if g["route_ids"] == ["B>C"] and g["challenge_level"] == 3)
    cb_up = next(g for g in groups if g["route_ids"] == ["C>B"] and g["challenge_level"] == 3)
    assert bc_up["games"][0]["budget_seconds_by_model"] == {"B": 5.0, "C": 0.05}
    assert cb_up["games"][0]["budget_seconds_by_model"] == {"C": 5.0, "B": 0.05}


def test_shared_baseline_advances_only_the_winning_route():
    config = load_config(CONFIG)
    groups = build_schedule(config)
    baseline = next(g for g in groups if g["pair"] == ["B", "C"] and g["shared_baseline"])
    records = _records_before(groups, baseline["group_id"])
    records.extend([_game_result(baseline["games"][0], "B"), _game_result(baseline["games"][1], "B")])

    routes = derive_routes(records, config)["routes"]

    assert routes["B>C"]["highest_passed_budget"] == "50ms"
    assert routes["B>C"]["status"] == "active"
    assert routes["C>B"]["status"] == "failed"
    assert routes["C>B"]["failed_budget"] == "50ms"


def test_shared_draw_advances_both_directions():
    config = load_config(CONFIG)
    groups = build_schedule(config)
    baseline = next(g for g in groups if g["pair"] == ["B", "C"] and g["shared_baseline"])
    records = _records_before(groups, baseline["group_id"])
    records.extend(_game_result(game, None) for game in baseline["games"])
    routes = derive_routes(records, config)["routes"]

    assert routes["B>C"]["status"] == "active"
    assert routes["C>B"]["status"] == "active"
    assert routes["B>C"]["levels"][0]["score"] == 1.0
    assert routes["C>B"]["levels"][0]["score"] == 1.0


def test_one_win_one_loss_reaches_exactly_one_point_and_both_routes_advance():
    config = load_config(CONFIG)
    groups = build_schedule(config)
    baseline = next(g for g in groups if g["pair"] == ["B", "C"] and g["shared_baseline"])
    records = _records_before(groups, baseline["group_id"])
    records.extend([_game_result(baseline["games"][0], "B"), _game_result(baseline["games"][1], "C")])

    routes = derive_routes(records, config)["routes"]

    assert routes["B>C"]["levels"][0]["score"] == 1.0
    assert routes["C>B"]["levels"][0]["score"] == 1.0
    assert routes["B>C"]["status"] == "active"
    assert routes["C>B"]["status"] == "active"


def test_one_game_of_a_pair_does_not_change_route_state():
    config = load_config(CONFIG)
    groups = build_schedule(config)
    baseline = next(g for g in groups if g["pair"] == ["B", "C"] and g["shared_baseline"])
    records = _records_before(groups, baseline["group_id"])
    records.append(_game_result(baseline["games"][0], "B"))
    routes = derive_routes(records, config)["routes"]

    assert routes["B>C"]["levels"][0]["status"] == "incomplete_pair"
    assert routes["B>C"]["status"] == "active"
    assert routes["C>B"]["status"] == "active"


def test_failed_route_cannot_be_reactivated_by_a_later_result():
    config = load_config(CONFIG)
    groups = build_schedule(config)
    baseline = groups[0]
    first_failed = [_game_result(baseline["games"][0], "B"), _game_result(baseline["games"][1], "B")]
    later_group = next(g for g in groups if g["route_ids"] == ["A>B"] and g["challenge_level"] == 1)

    with pytest.raises(ValueError, match="after every route"):
        derive_routes(first_failed + [_game_result(game, "A") for game in later_group["games"]], config)


def test_a_route_failure_records_the_opponents_actual_budget_rung():
    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    records = [_game_result(game, "B") for game in group["games"]]

    route = derive_routes(records, config)["routes"]["A>B"]

    assert route["status"] == "failed"
    assert route["failed_budget"] == "50ms"


def test_all_drawn_maximum_schedule_completes_every_route():
    config = load_config(CONFIG)
    records = [
        _game_result(game, None)
        for group in build_schedule(config)
        for game in group["games"]
    ]

    result = derive_routes(records, config)

    assert result["completed_games"] == 120
    assert result["completed_groups"] == 60
    assert result["complete"] is True
    assert {route["status"] for route in result["routes"].values()} == {"completed"}


def test_a_has_no_artificial_deadline_or_budget():
    config = load_config(CONFIG)
    settings = settings_for_model(config, "A", occupied=4)

    assert settings.time_limit_seconds == float("inf")
    assert config["budgets_seconds_by_model"]["A"] is None


def test_resume_runs_only_missing_game_from_half_complete_group(tmp_path, monkeypatch):
    import src.research.abcd_ladder_v5 as runner
    import src.research.abcd_ladder_v5_analysis as analysis

    config = load_config(CONFIG)
    schedule = build_schedule(config)
    calls: list[str] = []

    def fake_game(spec, _config, implementation_commit="test"):
        calls.append(spec["game_id"])
        return _game_result(spec, None)

    monkeypatch.setattr(runner, "run_game", fake_game)
    monkeypatch.setattr(analysis, "validate_game_record", lambda record, _config, _spec=None: {})
    monkeypatch.setattr(analysis, "validate_records", lambda records, _config, require_complete=False: {"valid": True})

    first = run_experiment(config, tmp_path, max_games=1, implementation_commit="test")
    assert first["completed_games"] == 1
    assert first["completed_groups"] == 0
    first_id = calls[0]

    second = run_experiment(config, tmp_path, max_games=1, implementation_commit="test")
    rows = [json.loads(line) for line in (tmp_path / "9x9_games.jsonl").read_text().splitlines()]
    assert second["completed_games"] == 2
    assert len(rows) == 2
    assert len(set(row["game_id"] for row in rows)) == 2
    assert first_id in {row["game_id"] for row in rows}
    assert calls == [schedule[0]["games"][0]["game_id"], schedule[0]["games"][1]["game_id"]]
