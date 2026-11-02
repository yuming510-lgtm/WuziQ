"""Idempotent CPU accounting and restart-safe budget guards for G V4."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any

_CPU_CACHE: dict[str, tuple[tuple[int, int, int], list[dict[str, Any]], float, float]] = {}
_CPU_EVENT_INDEX: dict[str, dict[str, dict[str, Any]]] = {}


class V4BudgetExhausted(RuntimeError):
    pass


def _paths(root: Path) -> tuple[Path, Path]:
    base = root / "research/g_v4/results/runtime"
    base.mkdir(parents=True, exist_ok=True)
    return base / "cpu_events.jsonl", base / "cpu_events.lock"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    rows = []
    seen: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(complete.splitlines(), 1):
        row = json.loads(line)
        event_id = str(row.get("event_id", ""))
        if not event_id:
            raise ValueError(f"V4 CPU event missing ID at line {number}")
        if event_id in seen and seen[event_id] != row:
            raise ValueError(f"conflicting V4 CPU event ID {event_id}")
        if event_id not in seen:
            rows.append(row)
            seen[event_id] = row
    if len(complete) != len(raw):
        with path.open("r+b") as handle:
            handle.truncate(len(complete))
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def _stamp(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return (0, 0, 0)
    value = path.stat()
    return (int(value.st_ino), int(value.st_size), int(value.st_mtime_ns))


def _cached_rows(root_key: str, path: Path) -> tuple[list[dict[str, Any]], float, float]:
    stamp = _stamp(path)
    cached = _CPU_CACHE.get(root_key)
    if cached and cached[0] == stamp:
        return cached[1], cached[2], cached[3]
    rows = _load_rows(path)
    cpu = float(sum(float(row["cpu_seconds"]) for row in rows))
    wall = float(sum(float(row.get("wall_seconds", 0.0)) for row in rows))
    _CPU_CACHE[root_key] = (_stamp(path), rows, cpu, wall)
    _CPU_EVENT_INDEX[root_key] = {str(row["event_id"]): row for row in rows}
    return rows, cpu, wall


def cpu_seconds_committed(root: str | Path) -> float:
    path, lock_path = _paths(Path(root).resolve())
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        return _cached_rows(str(Path(root).resolve()), path)[1]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def active_wall_seconds_committed(root: str | Path) -> float:
    path, lock_path = _paths(Path(root).resolve())
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        return _cached_rows(str(Path(root).resolve()), path)[2]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def record_cpu_event(
    root: str | Path,
    event_id: str,
    cpu_seconds: float,
    *,
    wall_seconds: float,
    category: str,
) -> bool:
    path, lock_path = _paths(Path(root).resolve())
    cpu, wall = float(cpu_seconds), float(wall_seconds)
    if (not event_id or not category or not math.isfinite(cpu) or cpu < 0
            or not math.isfinite(wall) or wall < 0):
        raise ValueError("V4 CPU event fields are invalid")
    row = {"event_id": str(event_id), "category": str(category),
           "cpu_seconds": cpu, "wall_seconds": wall}
    payload = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        root_key = str(Path(root).resolve())
        existing, old_cpu, old_wall = _cached_rows(root_key, path)
        index = _CPU_EVENT_INDEX[root_key]
        prior = index.get(event_id)
        if prior is not None:
            if prior != row:
                raise ValueError(f"conflicting duplicate V4 CPU event {event_id}")
            return False
        out = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            count = os.write(out, payload)
            if count != len(payload):
                raise OSError("short append while recording V4 CPU event")
            os.fsync(out)
        finally:
            os.close(out)
        new_rows = existing + [row]
        _CPU_CACHE[root_key] = (
            _stamp(path), new_rows,
            old_cpu + cpu, old_wall + wall,
        )
        index[event_id] = row
        return True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def ensure_cpu_room(root: str | Path, hard_cap_hours: float, *, reserve_seconds: float = 10.0) -> float:
    cap, reserve = float(hard_cap_hours) * 3600, float(reserve_seconds)
    if not math.isfinite(cap) or cap <= 0 or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("V4 CPU cap/reserve must be finite and valid")
    root_path = Path(root).resolve()
    used = cpu_seconds_committed(root_path)
    preflight_path = root_path / "research/g_v4/results/preflight/preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        used += float(preflight.get("measurement_cpu_seconds", 0.0))
    if used + reserve >= cap:
        raise V4BudgetExhausted(f"V4 CPU cap reached: {used:.2f}s used including preflight, {reserve:.2f}s reserved, {cap:.2f}s cap")
    return used


def next_snapshot_threshold(active_seconds: float) -> int:
    value = float(active_seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError("active runtime must be nonnegative and finite")
    thresholds = [3600, 10800, 25200, 46800, 82800, 140400]
    for threshold in thresholds:
        if value < threshold:
            return threshold
    return thresholds[-1] + (int((value - thresholds[-1]) // (26 * 3600)) + 1) * 26 * 3600

