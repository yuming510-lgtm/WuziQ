"""Stable V6 move result shared by tactical and adversarial search players."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...gomoku import Coordinate


@dataclass(frozen=True)
class V6MoveResult:
    move: Coordinate
    root_value: int
    nodes: int
    completed_depth: int
    cutoffs: int
    simulations: int
    elapsed_seconds: float
    timed_out: bool
    proof_status: str = "not_applicable"
    proof_kind: str | None = None
    proof_certificate: dict[str, Any] | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> int:
        return self.root_value
