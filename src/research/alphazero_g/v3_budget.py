"""Finite CPU accounting for the registered G V3 work only."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any


_CPU_CACHES: dict[str, dict[str, Any]] = {}
_RECONCILED_ROOTS: set[str] = set()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _budget_paths(root: Path) -> tuple[Path, Path]:
    directory = root / "research/g_v3/results/runtime"
    return directory / "cpu_events.jsonl", directory / "cpu_budget.json"


def _load_cpu_cache(root: Path) -> dict[str, Any]:
    key = str(root.resolve())
    if key in _CPU_CACHES:
        return _CPU_CACHES[key]
    event_path, state_path = _budget_paths(root)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    raw = event_path.read_bytes() if event_path.exists() else b""
    end = raw.rfind(b"\n") + 1
    if end != len(raw):
        with event_path.open("r+b") as handle:
            handle.truncate(end)
            handle.flush()
            os.fsync(handle.fileno())
        raw = raw[:end]
    rows = []
    ids: set[str] = set()
    total = 0.0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        row = json.loads(line)
        event_id = str(row.get("event_id", ""))
        seconds = float(row.get("cpu_seconds", -1))
        if (not event_id or event_id in ids or not math.isfinite(seconds) or seconds < 0):
            raise ValueError(f"invalid or duplicate CPU event at {event_path}:{line_number}")
        ids.add(event_id)
        rows.append(row)
        total += seconds
    cache = {"ids": ids, "total": total, "event_count": len(rows), "log_size": len(raw)}
    state = {key: value for key, value in cache.items() if key != "ids"}
    if state_path.exists():
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        if (persisted.get("event_count") != cache["event_count"]
                or persisted.get("log_size") != cache["log_size"]
                or not math.isclose(float(persisted.get("total_cpu_seconds", -1)), total, rel_tol=0, abs_tol=1e-8)):
            _atomic_json(state_path, {**state, "total_cpu_seconds": total, "rebuilt_after_inconsistency": True})
    else:
        _atomic_json(state_path, {**state, "total_cpu_seconds": total})
    _CPU_CACHES[key] = cache
    return cache


def record_v3_cpu_event(root: str | Path, event_id: str, cpu_seconds: float, *, category: str) -> bool:
    """Durably add one idempotent CPU event and update the O(1) budget state."""
    root_path = Path(root).resolve()
    event_key = str(event_id)
    seconds = float(cpu_seconds)
    if not event_key or not math.isfinite(seconds) or seconds < 0 or not category:
        raise ValueError("CPU event identity, category and seconds must be valid")
    cache = _load_cpu_cache(root_path)
    if event_key in cache["ids"]:
        return False
    event_path, state_path = _budget_paths(root_path)
    payload = (json.dumps({"event_id": event_key, "category": category,
                           "cpu_seconds": seconds}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("failed to append G V3 CPU budget event")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    cache["ids"].add(event_key)
    cache["total"] += seconds
    cache["event_count"] += 1
    cache["log_size"] += len(payload)
    _atomic_json(state_path, {
        "event_count": cache["event_count"], "log_size": cache["log_size"],
        "total_cpu_seconds": cache["total"], "updated_unix_time": time.time(),
    })
    return True


def reconcile_v3_cpu_events(root: str | Path) -> float:
    """Repair missing ledger events once at process start/resume from durable logs."""
    root_path = Path(root).resolve()
    key = str(root_path)
    cache = _load_cpu_cache(root_path)
    if key in _RECONCILED_ROOTS:
        return float(cache["total"])
    base = root_path / "research/g_v3"
    selfplay = base / "results/selfplay"
    if selfplay.exists():
        for path in selfplay.glob("*.jsonl"):
            raw = path.read_bytes()
            end = raw.rfind(b"\n") + 1
            for line in raw[:end].splitlines():
                row = json.loads(line)
                game_id = str(row["game_id"])
                record_v3_cpu_event(root_path, f"selfplay:{game_id}",
                                    float(row.get("runner_cpu_seconds", row.get("stats", {}).get("cpu_seconds", 0.0))),
                                    category="selfplay")
    checkpoints = base / "checkpoints"
    if checkpoints.exists():
        for path in checkpoints.glob("*/seed-*/progress.json"):
            progress = json.loads(path.read_text(encoding="utf-8"))
            arm = str(progress.get("arm", path.parts[-3]))
            seed = int(progress.get("seed", path.parts[-2].removeprefix("seed-")))
            for row in progress.get("training_history", []):
                state = int(row.get("states_seen", -1))
                event_id = f"optimizer:{arm}:seed-{seed}:states-{state}"
                record_v3_cpu_event(root_path, event_id, float(row.get("cpu_seconds", 0.0)), category="optimizer")
    evaluations = base / "results/evaluation"
    if evaluations.exists():
        for path in evaluations.rglob("*.jsonl"):
            raw = path.read_bytes()
            end = raw.rfind(b"\n") + 1
            for line in raw[:end].splitlines():
                row = json.loads(line)
                if path.name == "tactics.jsonl":
                    record_v3_cpu_event(root_path, f"tactics:{row['run_id']}",
                                        float(row.get("stats", {}).get("total_cpu_seconds", 0.0)),
                                        category="tactics")
                elif row.get("source") == "evaluation":
                    record_v3_cpu_event(root_path, f"evaluation:{row['game_id']}",
                                        float(row.get("stats", {}).get("total_cpu_seconds", 0.0)),
                                        category="evaluation")
    _RECONCILED_ROOTS.add(key)
    return float(cache["total"])


class BudgetExhausted(RuntimeError):
    """The registered CPU cap is reached before the next resumable unit of work."""


def cpu_seconds_committed(root: str | Path) -> float:
    """Return the durable O(1) ledger total, reconciling append sources once."""
    return reconcile_v3_cpu_events(root)


def active_wall_seconds_committed(root: str | Path) -> float:
    """Sum measured self-play, optimizer, and evaluation wall time."""
    base = Path(root) / "research/g_v3"
    total = 0.0
    selfplay_root = base / "results/selfplay"
    if selfplay_root.exists():
        for log_path in selfplay_root.glob("*.jsonl"):
            raw = log_path.read_bytes()
            end = raw.rfind(b"\n") + 1
            for line_number, line in enumerate(raw[:end].splitlines(), start=1):
                try:
                    row: dict[str, Any] = json.loads(line)
                    total += float(row.get("runner_wall_seconds", row.get("stats", {}).get("wall_seconds", 0.0)))
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(f"invalid G V3 self-play wall record {log_path}:{line_number}") from exc
    checkpoint_root = base / "checkpoints"
    if checkpoint_root.exists():
        for progress_path in checkpoint_root.glob("*/seed-*/progress.json"):
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            total += sum(float(row.get("wall_seconds", 0.0)) for row in progress.get("training_history", []))
    evaluation_root = base / "results/evaluation"
    if evaluation_root.exists():
        for log_path in evaluation_root.rglob("*.jsonl"):
            raw = log_path.read_bytes()
            end = raw.rfind(b"\n") + 1
            for line_number, line in enumerate(raw[:end].splitlines(), start=1):
                try:
                    row: dict[str, Any] = json.loads(line)
                    total += float(row.get("stats", {}).get("total_wall_seconds", 0.0))
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(f"invalid G V3 evaluation wall record {log_path}:{line_number}") from exc
    if not math.isfinite(total) or total < 0:
        raise ValueError("G V3 committed wall total is invalid")
    return total


def ensure_v3_cpu_room(
    root: str | Path,
    hard_cap_seconds: float,
    *,
    reserve_seconds: float = 10.0,
) -> float:
    cap, reserve = float(hard_cap_seconds), float(reserve_seconds)
    if not math.isfinite(cap) or cap <= 0 or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("CPU cap and reserve must be finite and valid")
    used = cpu_seconds_committed(root)
    if used + reserve >= cap:
        raise BudgetExhausted(
            f"registered G V3 CPU cap reached: {used:.2f}s durable + {reserve:.2f}s reserve of {cap:.2f}s"
        )
    return used
