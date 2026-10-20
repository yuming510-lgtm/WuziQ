"""Minimal SHA-256 checked recovery archives for the V4C tournament."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile

from .bcd_high_compute_v4c import load_config
from .bcd_high_compute_v4c_analysis import validate_records
from .checkpoint import load_game_records

_ALLOWED = {"config.json", "progress.json", "9x9_games.jsonl", "code_version.txt"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ids_hash(ids: list[str]) -> str:
    return _sha256("\n".join(sorted(ids)).encode("utf-8"))


def create_snapshot(
    results_dir: str | Path,
    destination: str | Path,
    *,
    code_version: str,
) -> dict[str, Any]:
    root = Path(results_dir)
    required = ("config.json", "progress.json", "9x9_games.jsonl")
    if any(not (root / name).is_file() for name in required):
        raise FileNotFoundError("V4C snapshot requires config, progress, and games JSONL")
    config = load_config(root / "config.json")
    records = load_game_records(root / "9x9_games.jsonl")
    validation = validate_records(records, config, require_complete=False)
    ids = [str(record["game_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot snapshot duplicate V4C game IDs")
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if progress.get("experiment_id") != config["experiment_id"]:
        raise ValueError("V4C snapshot progress belongs to a different experiment")
    if int(progress.get("completed_games", -1)) != len(records) or set(progress.get("completed_ids", ())) != set(ids):
        raise ValueError("V4C snapshot progress differs from completed game records")
    if progress.get("implementation_commit") != code_version:
        raise ValueError("V4C code version differs from progress metadata")
    contents = {name: (root / name).read_bytes() for name in required}
    contents["code_version.txt"] = (str(code_version).strip() + "\n").encode("utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "completed_games": len(records),
        "completed_game_ids_sha256": _ids_hash(ids),
        "files": {name: _sha256(data) for name, data in sorted(contents.items())},
    }
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(name, data)
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {**manifest, "archive_sha256": _sha256(output.read_bytes()), "validation": validation}


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify the whole archive, then atomically restore into an empty directory."""
    with zipfile.ZipFile(snapshot_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("V4C snapshot contains duplicate archive names")
        if "manifest.json" not in names or not _ALLOWED.issubset(set(names)):
            raise ValueError("V4C snapshot is missing required recovery files")
        if any(name not in _ALLOWED | {"manifest.json"} or PurePosixPath(name).name != name for name in names):
            raise ValueError("V4C snapshot contains an unsupported path")
        manifest = json.loads(archive.read("manifest.json"))
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported V4C snapshot schema")
        contents = {name: archive.read(name) for name in names if name != "manifest.json"}
    if set(contents) != set(manifest.get("files", {})):
        raise ValueError("V4C snapshot manifest file set mismatch")
    for name, data in contents.items():
        if _sha256(data) != manifest["files"][name]:
            raise ValueError(f"V4C snapshot checksum mismatch for {name}")

    output = Path(destination)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("V4C restore destination must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        target = output / name
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=output)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    _fsync_directory(output)
    config = load_config(output / "config.json")
    records = load_game_records(output / "9x9_games.jsonl")
    validation = validate_records(records, config, require_complete=False)
    ids = sorted(str(record["game_id"]) for record in records)
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    if len(ids) != len(set(ids)):
        raise ValueError("Restored V4C records have duplicate IDs")
    if int(progress.get("completed_games", -1)) != len(ids) or set(progress.get("completed_ids", ())) != set(ids):
        raise ValueError("Restored V4C progress and record IDs disagree")
    code_version = (output / "code_version.txt").read_text(encoding="utf-8").strip()
    if progress.get("implementation_commit") != code_version:
        raise ValueError("Restored V4C code version differs from progress")
    if len(ids) != int(manifest.get("completed_games", -1)) or _ids_hash(ids) != manifest.get("completed_game_ids_sha256"):
        raise ValueError("Restored V4C IDs differ from archive manifest")
    return {
        "archive_sha256": _sha256(Path(snapshot_path).read_bytes()),
        "experiment_id": config["experiment_id"], "completed_games": len(ids),
        "game_ids": ids, "code_version": code_version, "validation": validation,
        "restored_files": sorted(contents),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
