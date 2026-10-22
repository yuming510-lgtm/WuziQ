"""Minimal V6 recovery archives with checksummed empty-directory restoration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from ..checkpoint import load_game_records
from .config import canonical


SNAPSHOT_NAMES = (
    "config.json",
    "schedule.json",
    "reuse_manifest.json",
    "progress.json",
    "9x9_games.jsonl",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_v6_snapshot(output_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    source = Path(output_dir)
    contents: dict[str, bytes] = {}
    for name in SNAPSHOT_NAMES:
        path = source / name
        if not path.is_file():
            if name == "9x9_games.jsonl":
                contents[name] = b""
                continue
            raise FileNotFoundError(f"V6 snapshot is missing {name}")
        contents[name] = path.read_bytes()
    progress = json.loads(contents["progress.json"])
    manifest = {
        "schema_version": 1,
        "experiment_id": progress["experiment_id"],
        "implementation_commit": progress["implementation_commit"],
        "created_at_utc": progress["updated_at_utc"],
        "files": {name: _sha(data) for name, data in sorted(contents.items())},
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
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    data = output.read_bytes()
    return {
        "path": str(output),
        "sha256": _sha(data),
        "size_bytes": len(data),
        "manifest": manifest,
    }


def restore_v6_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    source = Path(snapshot_path)
    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        expected_names = {*SNAPSHOT_NAMES, "manifest.json"}
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("V6 snapshot has missing, duplicate, or unsupported files")
        contents = {name: archive.read(name) for name in SNAPSHOT_NAMES}
        manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(SNAPSHOT_NAMES):
        raise ValueError("V6 snapshot manifest is malformed")
    for name, data in contents.items():
        if _sha(data) != manifest["files"][name]:
            raise ValueError(f"V6 snapshot checksum mismatch for {name}")
    config = json.loads(contents["config.json"])
    schedule = json.loads(contents["schedule.json"])
    reuse_manifest = json.loads(contents["reuse_manifest.json"])
    progress = json.loads(contents["progress.json"])
    if (config.get("experiment_id") != manifest.get("experiment_id")
            or progress.get("experiment_id") != manifest.get("experiment_id")):
        raise ValueError("V6 snapshot experiment identity mismatch")
    if progress.get("implementation_commit") != manifest.get("implementation_commit"):
        raise ValueError("V6 snapshot source commit mismatch")
    if progress.get("config_sha256") != _sha(canonical(config)):
        raise ValueError("V6 snapshot config identity mismatch")
    if progress.get("schedule_sha256") != _sha(canonical(schedule)):
        raise ValueError("V6 snapshot schedule identity mismatch")
    if progress.get("reuse_manifest_sha256") != _sha(canonical(reuse_manifest)):
        raise ValueError("V6 snapshot reuse manifest identity mismatch")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("V6 snapshot restoration requires an empty destination directory")
    for name, data in contents.items():
        target = destination / name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=destination
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
    records = load_game_records(destination / "9x9_games.jsonl")
    record_ids = {str(row["game_id"]) for row in records}
    progress_ids = set(map(str, progress.get("completed_ids", ())))
    if not progress_ids.issubset(record_ids):
        raise ValueError("V6 snapshot progress claims missing games")
    if len(record_ids) != len(records):
        raise ValueError("V6 snapshot has duplicate game IDs")
    return {
        "restored_files": list(SNAPSHOT_NAMES),
        "game_count": len(records),
        "progress_game_count": len(progress_ids),
        "empty_restore": True,
    }
