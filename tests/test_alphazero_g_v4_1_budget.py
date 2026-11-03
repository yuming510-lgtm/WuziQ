from __future__ import annotations

from src.research.alphazero_g.v4_1_budget import estimate_runtime


def test_v4_1_runtime_estimate_accounts_for_training_and_evaluation() -> None:
    measured = {
        "selfplay_states_per_cpu_second": 10.0,
        "optimizer_cpu_seconds_per_update": 0.01,
        "evaluation_cpu_seconds_per_network_game": 0.1,
        "evaluation_cpu_seconds_per_puct_game": 1.0,
        "diagnostic_cpu_seconds": 100.0,
    }

    estimate = estimate_runtime(measured)

    assert estimate["training_selfplay_cpu_seconds"]["central"] == 4000.0
    assert estimate["optimizer_cpu_seconds"]["central"] == 25.0
    assert estimate["evaluation_games"] == {"network": 496, "puct": 680, "total": 1176}
    assert estimate["evaluation_cpu_seconds"]["central"] == 729.6
    assert estimate["total_cpu_seconds"]["central"] == 4854.6
    assert estimate["total_cpu_seconds"]["conservative"] > estimate["total_cpu_seconds"]["central"]


def test_v4_1_runtime_estimate_rejects_nonpositive_or_overbudget_rates() -> None:
    import pytest

    for value in [0.0, -1.0]:
        with pytest.raises(ValueError):
            estimate_runtime({
                "selfplay_states_per_cpu_second": value,
                "optimizer_cpu_seconds_per_update": 0.01,
                "evaluation_cpu_seconds_per_network_game": 0.1,
                "evaluation_cpu_seconds_per_puct_game": 1.0,
                "diagnostic_cpu_seconds": 0,
            })
