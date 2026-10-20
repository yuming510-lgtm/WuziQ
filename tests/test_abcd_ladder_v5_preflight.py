from pathlib import Path

from src.research.abcd_ladder_v5 import load_config
from src.research.abcd_ladder_v5_preflight import estimate_full_schedule


ROOT = Path(__file__).parents[1]


def test_preflight_estimate_respects_schedule_budget_and_long_game_ceiling():
    config = load_config(ROOT / "configs" / "abcd_ladder_v5.json")
    benchmarks = [
        {
            "model": model,
            "budget_label": rung["label"],
            "elapsed_seconds_wall": float(rung["seconds"]),
        }
        for model in ("B", "C", "D")
        for rung in config["budget_ladder"]
    ]

    estimate = estimate_full_schedule(config, benchmarks, ROOT)

    assert estimate["maximum_groups"] == 60
    assert estimate["maximum_games"] == 120
    assert estimate["groups_by_family_pair"] == {
        "A-B": 7, "A-C": 7, "A-D": 7, "B-C": 13, "B-D": 13, "C-D": 13,
    }
    assert estimate["longest_single_game_budget_ceiling_minutes"] == 32.5325
    assert estimate["longest_paired_group_budget_ceiling_minutes"] == 65.065
    assert estimate["configured_budget_sum_full_board_ceiling_hours"] < 24
