from __future__ import annotations

import importlib.util

import pytest


def _preflight_module():
    name = "scripts.preflight_g_v6"
    if importlib.util.find_spec(name) is None:
        pytest.fail("V6 measured workload estimator is not implemented")
    from scripts import preflight_g_v6

    return preflight_g_v6


def test_workload_estimator_uses_20000_cumulative_states_and_full_registered_evaluations() -> None:
    module = _preflight_module()
    result = module.estimate_workload(
        measured_game_seconds=20.0,
        measured_game_states=40,
        observed_mean_states_per_game=32.0,
        optimizer_update_seconds=0.2,
        evaluation_seconds_per_game={64: 1.0, 400: 4.0},
        tactical_seconds_per_position={64: 0.01, 400: 0.02},
        preflight_seconds=30.0,
    )

    assert result["remaining_training_states"] == 18_486
    assert result["evaluation_games_by_budget"] == {"64": 560, "400": 560}
    assert result["planned_development_games"] == 960
    assert result["planned_blind_games"] == 160
    assert result["estimated_total_cpu_hours"]["normal"] > 0

