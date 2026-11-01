"""Finite CPU-time accounting for a resumable G V2 run."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

LR_IDS = ("lr_1e4", "lr_3e5", "lr_1e3")
EVALUATION_LOGS = ("development.jsonl", "champion.jsonl", "blind.jsonl")


class BudgetExhausted(RuntimeError):
    """Raised before starting work that cannot safely fit in the registered CPU budget."""


def cpu_seconds_committed(root: str | Path) -> float:
    """Return durable training plus complete evaluation CPU seconds; preflight is excluded."""
    base = Path(root) / "research/g_v2"
    total = 0.0
    for candidate_id in LR_IDS:
        progress_path = base / "checkpoints" / candidate_id / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            total += float(progress.get("active_cpu_seconds", 0.0))
    evaluation_dir = base / "results/evaluation"
    for name in EVALUATION_LOGS:
        path = evaluation_dir / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        complete_end = raw.rfind(b"\n") + 1
        for line_number, line in enumerate(raw[:complete_end].splitlines(), start=1):
            try:
                row: dict[str, Any] = json.loads(line)
                total += float(row.get("stats", {}).get("total_cpu_seconds", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid CPU accounting row in {path}:{line_number}") from exc
    if not math.isfinite(total) or total < 0:
        raise ValueError("durable CPU accounting total is invalid")
    return total


def active_wall_seconds_committed(root: str | Path) -> float:
    """Return durable effective work wall seconds, excluding idle time between resumes."""
    base = Path(root) / "research/g_v2"
    total = 0.0
    for candidate_id in LR_IDS:
        progress_path = base / "checkpoints" / candidate_id / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            total += float(progress.get("active_wall_seconds", 0.0))
    evaluation_dir = base / "results/evaluation"
    for name in EVALUATION_LOGS:
        path = evaluation_dir / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        complete_end = raw.rfind(b"\n") + 1
        for line_number, line in enumerate(raw[:complete_end].splitlines(), start=1):
            try:
                row: dict[str, Any] = json.loads(line)
                total += float(row.get("stats", {}).get("total_wall_seconds", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid wall-time accounting row in {path}:{line_number}") from exc
    if not math.isfinite(total) or total < 0:
        raise ValueError("durable wall-time accounting total is invalid")
    return total


def ensure_cpu_room(
    root: str | Path,
    hard_cap_seconds: float,
    *,
    reserve_seconds: float = 10.0,
) -> float:
    """Fail closed before another unit of work when the finite budget reserve is reached."""
    cap, reserve = float(hard_cap_seconds), float(reserve_seconds)
    if not math.isfinite(cap) or cap <= 0 or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("CPU cap and safety reserve must be finite nonnegative values")
    used = cpu_seconds_committed(root)
    if used + reserve >= cap:
        raise BudgetExhausted(
            f"registered CPU-hour cap reached: {used:.2f}s durable, "
            f"{reserve:.2f}s reserve, {cap:.2f}s cap"
        )
    return used
