from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.alphazero_g.v2_budget import (
    active_wall_seconds_committed,
    BudgetExhausted,
    cpu_seconds_committed,
    ensure_cpu_room,
)


def test_cpu_budget_counts_training_and_durable_evaluation_only(tmp_path):
    progress = tmp_path / "research/g_v2/checkpoints/lr_1e4/progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({"active_cpu_seconds": 125.0}))
    evaluation = tmp_path / "research/g_v2/results/evaluation/development.jsonl"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text(json.dumps({"stats": {"total_cpu_seconds": 75.0, "total_wall_seconds": 80.0}}) + "\n")
    preflight = tmp_path / "research/g_v2/results/preflight/preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"measurements_cpu_seconds": 9000.0}))

    assert cpu_seconds_committed(tmp_path) == 200.0
    assert active_wall_seconds_committed(tmp_path) == 80.0
    assert ensure_cpu_room(tmp_path, 300.0, reserve_seconds=10.0) == 200.0

    with pytest.raises(BudgetExhausted, match="registered CPU-hour cap"):
        ensure_cpu_room(tmp_path, 205.0, reserve_seconds=10.0)
