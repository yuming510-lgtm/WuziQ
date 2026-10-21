from __future__ import annotations

from pathlib import Path

from src.research.abcd_ladder_v5_1 import build_schedule, load_config
from src.research.abcd_ladder_v5_1_preflight import estimate_runtime, _resource_snapshot, run_preflight
from src.research.abcd_ladder_v5_1_reuse import prepare_v5_reuse


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "abcd_ladder_v5_1.json"
V5_RESULTS = ROOT / "research" / "v5" / "results"


def test_estimate_subtracts_exact_v5_reuse_and_uses_budget_specific_trials():
    config = load_config(CONFIG)
    schedule, reuse = prepare_v5_reuse(build_schedule(config), V5_RESULTS, config)
    cases = [{"model": "A", "budget_label": "natural", "budget_seconds": None, "measured_seconds": 0.001}]
    for model in ("B", "C", "D"):
        for index, rung in enumerate(config["budget_ladder"], start=1):
            cases.append({"model": model, "budget_label": rung["label"],
                          "budget_seconds": rung["seconds"], "measured_seconds": float(index)})
    estimate = estimate_runtime(
        schedule, reuse, {"cases": cases}, {"A-B": 10.0, "A-C": 10.0, "A-D": 10.0,
                                             "B-C": 10.0, "B-D": 10.0, "C-D": 10.0},
        storage_bytes_per_game=1000,
    )

    assert estimate["eligible_v5_reused_groups"] == 28
    assert estimate["estimated_new_groups_if_all_routes_reach_50s"] == 140
    assert estimate["estimated_new_games_if_all_routes_reach_50s"] == 280
    assert estimate["estimated_storage_bytes_for_all_336_games"] == 336000
    assert estimate["estimated_full_schedule_wall_seconds_serial"] > 0
    assert estimate["configured_full_board_budget_ceiling_hours"] > estimate["estimated_full_schedule_wall_hours_serial"]


def test_resource_snapshot_exposes_cpu_memory_and_storage(tmp_path):
    result = _resource_snapshot(tmp_path)
    assert result["cpu_count_visible"] >= 1
    assert result["memory"]["memtotal_kib"] > 0
    assert result["memory"]["memavailable_kib"] > 0
    assert result["disk"]["free_bytes"] > 0


def test_preflight_writes_schedule_audit_and_runtime_estimate_without_running_games(tmp_path):
    config = load_config(CONFIG)
    cases = [{"model": "A", "budget_label": "natural", "budget_seconds": None,
              "measured_seconds": 0.001, "legal_move": True}]
    for model in ("B", "C", "D"):
        for index, rung in enumerate(config["budget_ladder"], start=1):
            cases.append({"model": model, "budget_label": rung["label"],
                          "budget_seconds": rung["seconds"], "measured_seconds": index * 0.01,
                          "legal_move": True})
    output = tmp_path / "preflight.json"
    result = run_preflight(CONFIG, output, benchmarker=lambda _config: {
        "cases": cases, "benchmark_wall_seconds": 1.0,
    })

    assert output.is_file()
    assert result["schedule_audit"]["independent_routes"] == 66
    assert result["schedule_audit"]["groups"] == 168
    assert result["schedule_audit"]["game_ids"] == 336
    assert result["estimate"]["estimated_new_groups_if_all_routes_reach_50s"] == 140
