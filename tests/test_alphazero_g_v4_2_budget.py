from __future__ import annotations

import pytest


def test_v4_2_runtime_estimate_counts_training_evaluation_search_and_overhead() -> None:
    from src.research.alphazero_g.v4_2_budget import estimate_v4_2_runtime

    estimate = estimate_v4_2_runtime(
        states_per_arm=1_000, arms=2, training_states_per_cpu_second=10.0,
        optimizer_updates_per_cpu_second=2.0, evaluation_games=10,
        evaluation_cpu_seconds_per_game=2.0, diagnostic_search_calls=20,
        diagnostic_cpu_seconds_per_search=0.5, fixed_overhead_cpu_seconds=5.0,
    )
    assert estimate["total_cpu_seconds"] == pytest.approx(297.5)
    assert estimate["total_cpu_hours"] == pytest.approx(297.5 / 3600)


def test_v4_2_runtime_estimate_rejects_zero_throughput() -> None:
    from src.research.alphazero_g.v4_2_budget import estimate_v4_2_runtime

    with pytest.raises(ValueError, match="throughput"):
        estimate_v4_2_runtime(
            states_per_arm=100, arms=2, training_states_per_cpu_second=0.0,
            optimizer_updates_per_cpu_second=1.0, evaluation_games=1,
            evaluation_cpu_seconds_per_game=1.0, diagnostic_search_calls=0,
            diagnostic_cpu_seconds_per_search=0.0,
        )
