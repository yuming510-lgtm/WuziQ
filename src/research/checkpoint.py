"""Atomic, resumable storage for tournament records and recovery snapshots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable
import zipfile
from contextlib import contextmanager
import fcntl


def load_game_records(path: str | Path) -> list[dict[str, Any]]:
    """Load unique JSONL game records and discard only a corrupt final tail."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    raw = file_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    valid_offset = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_offset += len(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                with file_path.open("r+b") as handle:
                    handle.truncate(valid_offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f"Invalid JSONL record before the final line in {file_path}") from exc
        if not isinstance(record, dict) or not record.get("game_id"):
            raise ValueError(f"Game record in {file_path} is missing game_id")
        game_id = str(record["game_id"])
        if game_id in seen:
            raise ValueError(f"Duplicate game_id in {file_path}: {game_id}")
        seen.add(game_id)
        records.append(record)
        valid_offset += len(line)
    return records


@contextmanager
def artifact_lock(path: str | Path, *, shared: bool = False):
    """Coordinate a live snapshot reader with the runner's append/progress transaction."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_game(
    path: str | Path, record: dict[str, Any], completed_ids: set[str]
) -> None:
    """Append, flush, and fsync one complete game after checking its ID."""

    game_id = record.get("game_id")
    if not game_id:
        raise ValueError("Game record is missing game_id")
    game_id = str(game_id)
    if game_id in completed_ids:
        raise ValueError(f"Duplicate game_id: {game_id}")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    with file_path.open("a+b") as handle:
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    completed_ids.add(game_id)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Replace one JSON file atomically after the bytes have reached disk."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.", suffix=".tmp", dir=file_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, file_path)
        _fsync_directory(file_path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_snapshot_name(name: str) -> bool:
    if name in {"config.json", "progress.json"}:
        return True
    path = PurePosixPath(name)
    return (
        len(path.parts) == 1
        and path.name == name
        and path.name.endswith("_games.jsonl")
        and path.name[0].isdigit()
    )


def create_snapshot(
    config_path: str | Path,
    progress_path: str | Path,
    game_paths: Iterable[str | Path],
    destination: str | Path,
) -> dict[str, Any]:
    """Create a minimal ZIP containing only config, progress, games, and hashes."""

    sources = [(Path(config_path), "config.json"), (Path(progress_path), "progress.json")]
    sources.extend((Path(path), Path(path).name) for path in game_paths if Path(path).exists())
    if not sources[0][0].is_file() or not sources[1][0].is_file():
        raise FileNotFoundError("Snapshot requires config.json and progress.json")
    names = [name for _, name in sources]
    if len(names) != len(set(names)) or any(not _valid_snapshot_name(name) for name in names):
        raise ValueError("Snapshot contains an unsupported or duplicate file name")

    contents = {name: source.read_bytes() for source, name in sources}
    manifest = {
        "schema_version": 1,
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(contents.items())
        },
    }
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(name, data)
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            )
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return manifest


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> list[str]:
    """Verify and restore a minimal snapshot, returning its completed game IDs."""

    with zipfile.ZipFile(snapshot_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name != "manifest.json" and not _valid_snapshot_name(name) for name in names
        ):
            raise ValueError("Snapshot contains an unsupported path")
        if "manifest.json" not in names or "config.json" not in names or "progress.json" not in names:
            raise ValueError("Snapshot is missing required recovery files")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported snapshot schema version")
        files = {name: archive.read(name) for name in names if name != "manifest.json"}

    expected = manifest.get("files", {})
    if set(expected) != set(files):
        raise ValueError("Snapshot manifest file list does not match archive")
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = output / name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=output
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    completed: list[str] = []
    for game_file in sorted(output.glob("*_games.jsonl")):
        completed.extend(str(record["game_id"]) for record in load_game_records(game_file))
    if len(completed) != len(set(completed)):
        raise ValueError("Duplicate game_id across snapshot files")
    _fsync_directory(output)
    return sorted(completed)
