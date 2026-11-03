"""Transparent V4.2 CPU cost estimator using measured local throughput."""
from __future__ import annotations

import math
from typing import Any


def estimate_v4_2_runtime(
    *,
    states_per_arm: int,
    arms: int,
    training_states_per_cpu_second: float,
    optimizer_updates_per_cpu_second: float,
    evaluation_games: int,
    evaluation_cpu_seconds_per_game: float,
    diagnostic_search_calls: int,
    diagnostic_cpu_seconds_per_search: float,
    fixed_overhead_cpu_seconds: float = 0.0,
    update_every_states: int = 16,
) -> dict[str, Any]:
    if states_per_arm < 1 or arms not in (2, 3):
        raise ValueError("states_per_arm must be positive and V4.2 uses two or three arms")
    if evaluation_games < 0 or diagnostic_search_calls < 0 or fixed_overhead_cpu_seconds < 0:
        raise ValueError("game, search, and overhead counts must be nonnegative")
    if update_every_states < 1:
        raise ValueError("update_every_states must be positive")
    rates = (training_states_per_cpu_second, optimizer_updates_per_cpu_second)
    costs = (evaluation_cpu_seconds_per_game, diagnostic_cpu_seconds_per_search)
    if any(not math.isfinite(float(rate)) or float(rate) <= 0 for rate in rates):
        raise ValueError("measured training and optimizer throughput must be positive")
    if any(not math.isfinite(float(cost)) or float(cost) < 0 for cost in costs):
        raise ValueError("measured evaluation/search costs must be finite and nonnegative")
    total_states = int(states_per_arm) * int(arms)
    updates = math.ceil(total_states / int(update_every_states))
    components = {
        "selfplay_cpu_seconds": total_states / float(training_states_per_cpu_second),
        "optimizer_cpu_seconds": updates / float(optimizer_updates_per_cpu_second),
        "evaluation_cpu_seconds": int(evaluation_games) * float(evaluation_cpu_seconds_per_game),
        "diagnostic_search_cpu_seconds": int(diagnostic_search_calls) * float(diagnostic_cpu_seconds_per_search),
        "fixed_overhead_cpu_seconds": float(fixed_overhead_cpu_seconds),
    }
    total = sum(components.values())
    return {
        "total_training_states": total_states,
        "optimizer_updates": updates,
        "components_cpu_seconds": components,
        "total_cpu_seconds": total,
        "total_cpu_hours": total / 3600.0,
    }
