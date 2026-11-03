"""Shared CPU cap and atomic progress bookkeeping for V4.2 phases."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .v4_2_storage import atomic_write_json, load_json


class BudgetExhausted(RuntimeError):
    pass


class V4_2Runtime:
    def __init__(self, root: str | Path, *, preflight_cpu_seconds: float = 0.0) -> None:
        self.root = Path(root).resolve()
        self.path = self.root / "research/g_v4_2/results/progress.json"
        if self.path.exists():
            self.state = load_json(self.path)
        else:
            self.state = {
                "format_version": 1, "experiment_id": "wuziq-alphazero-g-v4-2",
                "preflight_cpu_seconds": float(preflight_cpu_seconds),
                "formal_cpu_seconds": 0.0, "phase": "registered", "completed_game_ids": {},
            }
        self.previous_formal_cpu = float(self.state.get("formal_cpu_seconds", 0.0))
        self.process_started_cpu = time.process_time()

    @property
    def total_cpu_seconds(self) -> float:
        return (float(self.state.get("preflight_cpu_seconds", 0.0))
                + self.previous_formal_cpu + time.process_time() - self.process_started_cpu)

    def check(self, *, reserve_seconds: float = 0.0) -> None:
        if self.total_cpu_seconds + float(reserve_seconds) >= 5.0 * 3600.0:
            self.save("budget_limit")
            raise BudgetExhausted("V4.2 reached its registered 5 CPU-hour cap")

    def save(self, phase: str, **fields: Any) -> None:
        self.state["formal_cpu_seconds"] = self.previous_formal_cpu + time.process_time() - self.process_started_cpu
        self.state["total_cpu_seconds_including_preflight"] = self.total_cpu_seconds
        self.state["phase"] = str(phase)
        self.state.update(fields)
        atomic_write_json(self.path, self.state)
