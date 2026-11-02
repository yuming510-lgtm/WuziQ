"""Finite CPU accounting for the registered G V3 work only."""
from __future__ import annotations

import json
import fcntl
import math
import os
from pathlib import Path
import tempfile
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _event_lock(root: Path):
    root_path = root.resolve()
    lock_path = root_path / "research/g_v3/results/runtime/cpu_events.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _release_event_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _serialize_event(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _parse_events(raw: bytes, event_path: Path) -> tuple[list[dict[str, Any]], bytes, bool]:
    end = raw.rfind(b"\n") + 1
    complete = raw[:end]
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_repaired = end != len(raw)
    for line_number, line in enumerate(complete.splitlines(), start=1):
        row = json.loads(line)
        event_id = str(row.get("event_id", ""))
        seconds = float(row.get("cpu_seconds", -1))
        category = row.get("category")
        if not event_id or not isinstance(category, str) or not category or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid CPU event at {event_path}:{line_number}")
        canonical = {"event_id": event_id, "category": category, "cpu_seconds": seconds}
        previous = by_id.get(event_id)
        if previous is not None:
            if previous != canonical:
                raise ValueError(f"conflicting duplicate CPU event at {event_path}:{line_number}")
            duplicate_repaired = True
            continue
        by_id[event_id] = canonical
        rows.append(canonical)
    canonical_raw = b"".join(_serialize_event(row) for row in rows)
    if canonical_raw != complete:
        duplicate_repaired = True
    return rows, canonical_raw, duplicate_repaired


def _build_cpu_cache(rows: list[dict[str, Any]], raw: bytes) -> dict[str, Any]:
    return {
        "ids": {str(row["event_id"]) for row in rows},
        "events": {str(row["event_id"]): row for row in rows},
        "rows": rows,
        "total": sum(float(row["cpu_seconds"]) for row in rows),
        "event_count": len(rows),
        "log_size": len(raw),
    }


def _sync_cpu_state(root: Path, cache: dict[str, Any], *, rebuilt: bool = False) -> None:
    _, state_path = _budget_paths(root)
    payload = {
        "event_count": cache["event_count"], "log_size": cache["log_size"],
        "total_cpu_seconds": cache["total"], "updated_unix_time": time.time(),
    }
    if rebuilt:
        payload["rebuilt_after_inconsistency"] = True
    _atomic_json(state_path, payload)


def _refresh_cpu_cache_locked(root: Path) -> dict[str, Any]:
    root_path = root.resolve()
    event_path, state_path = _budget_paths(root_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(root_path)
    cached = _CPU_CACHES.get(key)
    raw_size = event_path.stat().st_size if event_path.exists() else 0
    if cached is not None and raw_size == int(cached["log_size"]):
        return cached

    if cached is not None and raw_size > int(cached["log_size"]):
        with event_path.open("rb") as handle:
            handle.seek(int(cached["log_size"]))
            tail = handle.read()
        complete_end = tail.rfind(b"\n") + 1
        complete_tail = tail[:complete_end]
        new_rows: list[dict[str, Any]] = []
        duplicate = complete_end != len(tail)
        for line_number, line in enumerate(complete_tail.splitlines(), start=1):
            row = json.loads(line)
            event_id = str(row.get("event_id", ""))
            seconds = float(row.get("cpu_seconds", -1))
            category = row.get("category")
            if not event_id or not isinstance(category, str) or not category or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"invalid CPU event in appended ledger tail {event_path}:{line_number}")
            canonical = {"event_id": event_id, "category": category, "cpu_seconds": seconds}
            previous = cached["events"].get(event_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError(f"conflicting duplicate CPU event in appended ledger tail: {event_id}")
                duplicate = True
                continue
            if event_id in {item["event_id"] for item in new_rows}:
                previous_new = next(item for item in new_rows if item["event_id"] == event_id)
                if previous_new != canonical:
                    raise ValueError(f"conflicting duplicate CPU event in appended ledger tail: {event_id}")
                duplicate = True
                continue
            new_rows.append(canonical)
        for row in new_rows:
            cached["rows"].append(row)
            cached["events"][row["event_id"]] = row
            cached["ids"].add(row["event_id"])
            cached["total"] += float(row["cpu_seconds"])
            cached["event_count"] += 1
        if duplicate:
            canonical_raw = b"".join(_serialize_event(row) for row in cached["rows"])
            _atomic_bytes(event_path, canonical_raw)
            cached["log_size"] = len(canonical_raw)
        else:
            cached["log_size"] += len(complete_tail)
            if complete_end != len(tail):
                with event_path.open("r+b") as handle:
                    handle.truncate(int(cached["log_size"]))
                    handle.flush()
                    os.fsync(handle.fileno())
        _sync_cpu_state(root_path, cached, rebuilt=duplicate)
        return cached

    raw = event_path.read_bytes() if event_path.exists() else b""
    rows, canonical_raw, repaired = _parse_events(raw, event_path)
    if repaired:
        _atomic_bytes(event_path, canonical_raw)
    cache = _build_cpu_cache(rows, canonical_raw)
    persisted = {}
    if state_path.exists():
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
    inconsistent = (
        persisted.get("event_count") != cache["event_count"]
        or persisted.get("log_size") != cache["log_size"]
        or not math.isclose(float(persisted.get("total_cpu_seconds", -1)), cache["total"], rel_tol=0, abs_tol=1e-8)
    )
    _CPU_CACHES[key] = cache
    if not state_path.exists() or inconsistent or repaired:
        _sync_cpu_state(root_path, cache, rebuilt=inconsistent or repaired)
    return cache


def _load_cpu_cache(root: Path) -> dict[str, Any]:
    descriptor = _event_lock(root)
    try:
        return _refresh_cpu_cache_locked(root)
    finally:
        _release_event_lock(descriptor)


def record_v3_cpu_event(root: str | Path, event_id: str, cpu_seconds: float, *, category: str) -> bool:
    """Durably add one idempotent CPU event and update the O(1) budget state."""
    root_path = Path(root).resolve()
    event_key = str(event_id)
    seconds = float(cpu_seconds)
    if not event_key or not math.isfinite(seconds) or seconds < 0 or not category:
        raise ValueError("CPU event identity, category and seconds must be valid")
    event_path, state_path = _budget_paths(root_path)
    lock_descriptor = _event_lock(root_path)
    try:
        cache = _refresh_cpu_cache_locked(root_path)
        if event_key in cache["ids"]:
            return False
        row = {"event_id": event_key, "category": category, "cpu_seconds": seconds}
        payload = _serialize_event(row)
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
        cache["events"][event_key] = row
        cache["rows"].append(row)
        cache["total"] += seconds
        cache["event_count"] += 1
        cache["log_size"] += len(payload)
        _sync_cpu_state(root_path, cache)
    finally:
        _release_event_lock(lock_descriptor)
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
