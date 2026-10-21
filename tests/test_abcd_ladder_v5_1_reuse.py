from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.research.abcd_ladder_v5_1 import build_schedule, load_config
from src.research.abcd_ladder_v5_1_reuse import prepare_v5_reuse


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "abcd_ladder_v5_1.json"
V5_RESULTS = ROOT / "research" / "v5" / "results"


def test_all_28_v5_groups_reuse_only_after_exact_condition_validation():
    config = load_config(CONFIG)
    schedule, manifest = prepare_v5_reuse(build_schedule(config), V5_RESULTS, config)

    assert manifest["reused_groups"] == 28
    assert manifest["reused_games"] == 56
    assert len(manifest["groups"]) == 28
    assert all("winner_model" not in game for group in manifest["groups"] for game in group["games"])
    reused = [group for group in schedule if group.get("reuse_source")]
    assert len(reused) == 28
    assert all(len(group["reuse_source"]["games"]) == 2 for group in reused)
    assert all(game["source_type"] == "reused_v5" for group in reused for game in group["games"])
    assert all(game["source_game_id"] and game["game_seed"] == game["source_game_seed"]
               for group in reused for game in group["games"])


def test_a_v5_group_with_a_different_opening_is_not_reused():
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    schedule[0]["opening_seed"] = 731112
    schedule[0]["opening"] = [
        {"player": player, "row": row, "col": col}
        for player, row, col in config["opening_catalog"]["731112"]
    ]
    for game in schedule[0]["games"]:
        game["opening_seed"] = 731112
        game["opening"] = deepcopy(schedule[0]["opening"])

    updated, manifest = prepare_v5_reuse(schedule, V5_RESULTS, config)

    assert manifest["reused_groups"] == 27
    assert manifest["reused_games"] == 54
    assert updated[0].get("reuse_source") is None


def test_v5_reuse_manifest_is_deterministic_and_never_uses_winner_as_a_selector():
    config = load_config(CONFIG)
    first = prepare_v5_reuse(build_schedule(config), V5_RESULTS, config)[1]
    second = prepare_v5_reuse(build_schedule(config), V5_RESULTS, config)[1]

    assert first == second
    assert set(first) == {"source_experiment_id", "source_commit", "source_config_sha256",
                          "engine_sha256", "reused_groups", "reused_games", "groups", "rejections",
                          "source_validation"}
    assert first["source_validation"] == {"valid": True, "complete": True, "games": 56, "groups": 28}
