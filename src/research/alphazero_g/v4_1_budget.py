"""Pre-run wall/CPU estimates for the registered G V4.1 workload."""
from __future__ import annotations

import math
from typing import Any


def _positive(measured: dict[str, Any], name: str, *, allow_zero: bool = False) -> float:
    value = float(measured[name])
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} value")
    return value


def estimate_runtime(measured: dict[str, Any]) -> dict[str, Any]:
    """Estimate registered 40K training states plus 1,176 evaluation games.

    The central estimate uses measured rates. Optimistic/conservative bounds
    apply fixed throughput multipliers so the assumptions are auditable.
    """
    states_per_cpu = _positive(measured, "selfplay_states_per_cpu_second")
    update_cpu = _positive(measured, "optimizer_cpu_seconds_per_update")
    network_game = _positive(measured, "evaluation_cpu_seconds_per_network_game", allow_zero=True)
    puct_game = _positive(measured, "evaluation_cpu_seconds_per_puct_game", allow_zero=True)
    diagnostic_cpu = _positive(measured, "diagnostic_cpu_seconds", allow_zero=True)

    total_states = int(measured.get("training_states", 40_000))
    update_cadence = int(measured.get("update_every_states", 16))
    if total_states <= 0 or update_cadence <= 0:
        raise ValueError("training states and update cadence must be positive")
    updates = math.ceil(total_states / update_cadence)

    # Registered matrix: 496 network games and 680 PUCT games. Counts include
    # both players/modes, development checkpoints, and one final blind round.
    evaluation_games = {"network": 496, "puct": 680, "total": 1176}
    selfplay = total_states / states_per_cpu
    optimizer = updates * update_cpu
    evaluation = evaluation_games["network"] * network_game + evaluation_games["puct"] * puct_game
    central = selfplay + optimizer + evaluation + diagnostic_cpu

    def envelope(speed_factor: float, overhead_factor: float) -> dict[str, float]:
        # speed_factor is throughput relative to measurement; less than one is slower.
        training = (selfplay + optimizer) / speed_factor
        eval_cost = evaluation / speed_factor
        total = (training + eval_cost) * overhead_factor + diagnostic_cpu
        return {
            "training_cpu_seconds": training,
            "evaluation_cpu_seconds": eval_cost,
            "total_cpu_seconds": total,
            "total_wall_minutes_sequential": total / 60,
        }

    return {
        "training_states": total_states,
        "optimizer_updates": updates,
        "evaluation_games": evaluation_games,
        "training_selfplay_cpu_seconds": {
            "optimistic": selfplay * 1.25,
            "central": selfplay,
            "conservative": selfplay * 2.0,
        },
        "optimizer_cpu_seconds": {
            "optimistic": optimizer * 0.8,
            "central": optimizer,
            "conservative": optimizer * 1.5,
        },
        "evaluation_cpu_seconds": {
            "optimistic": evaluation * 0.8,
            "central": evaluation,
            "conservative": evaluation * 2.0,
        },
        "diagnostic_cpu_seconds": diagnostic_cpu,
        "total_cpu_seconds": {
            "optimistic": (selfplay + optimizer + evaluation) * 0.75 + diagnostic_cpu,
            "central": central,
            "conservative": (selfplay + optimizer + evaluation) * 2.25 + diagnostic_cpu,
        },
        "total_wall_minutes_sequential": {
            "optimistic": ((selfplay + optimizer + evaluation) * 0.75 + diagnostic_cpu) / 60,
            "central": central / 60,
            "conservative": ((selfplay + optimizer + evaluation) * 2.25 + diagnostic_cpu) / 60,
        },
        "resource_envelopes": {
            "optimistic": envelope(1.25, 1.15),
            "central": envelope(1.0, 1.25),
            "conservative": envelope(0.5, 1.4),
        },
        "cpu_hour_cap": 6.0,
    }
