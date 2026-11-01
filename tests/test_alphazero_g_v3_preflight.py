import pytest

from src.research.alphazero_g.v3_preflight import estimate_v3_runtime


def test_runtime_estimate_counts_both_routes_updates_and_evaluation_games():
    estimate = estimate_v3_runtime(
        states_per_cpu_second=100.0,
        update_cpu_seconds=0.01,
        evaluation_cpu_seconds_per_game=1.0,
        states_per_seed=1000,
        seed_count=2,
        route_count=2,
        total_evaluation_games=20,
    )

    assert estimate["selfplay_cpu_seconds"] == pytest.approx(40.0)
    assert estimate["optimizer_cpu_seconds"] == pytest.approx(2.5)
    assert estimate["evaluation_cpu_seconds"] == pytest.approx(20.0)
    assert estimate["central_cpu_seconds"] == pytest.approx(62.5)
    assert estimate["optimistic_cpu_seconds"] < estimate["central_cpu_seconds"]
    assert estimate["central_cpu_seconds"] < estimate["conservative_cpu_seconds"]


def test_runtime_estimate_scales_with_seed_count_and_rejects_zero_throughput():
    common = dict(
        states_per_cpu_second=50.0,
        update_cpu_seconds=0.02,
        evaluation_cpu_seconds_per_game=0.5,
        states_per_seed=1000,
        route_count=2,
        total_evaluation_games=10,
    )
    one = estimate_v3_runtime(seed_count=1, **common)
    two = estimate_v3_runtime(seed_count=2, **common)
    assert two["central_cpu_seconds"] > one["central_cpu_seconds"]

    with pytest.raises(ValueError, match="throughput"):
        estimate_v3_runtime(seed_count=1, **{**common, "states_per_cpu_second": 0.0})
