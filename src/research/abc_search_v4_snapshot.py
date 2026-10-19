"""Small SHA-256-verified recovery archives for the V4 benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile

from .abc_search_v4 import load_run_records
from .checkpoint import _fsync_directory


_RECOVERY_FILES = {"config.json", "progress.json", "search_runs.jsonl", "code_version.txt"}


def create_snapshot(
    config_path: str | Path,
    progress_path: str | Path,
    records_path: str | Path,
    code_commit: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Create a minimal archive containing only recoverable benchmark state."""

    required = {"config.json": Path(config_path), "progress.json": Path(progress_path)}
    if any(not path.is_file() for path in required.values()):
        raise FileNotFoundError("V4 snapshot requires config.json and progress.json")
    sources = dict(required)
    if Path(records_path).is_file():
        sources["search_runs.jsonl"] = Path(records_path)
    contents = {name: path.read_bytes() for name, path in sources.items()}
    contents["code_version.txt"] = (str(code_commit).strip() + "\n").encode("utf-8")
    manifest = {
        "schema_version": 1,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())},
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
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True) + "\n")
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


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify the archive manifest, restore atomically, and reject duplicate run IDs."""

    with zipfile.ZipFile(snapshot_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Snapshot contains duplicate paths")
        if "manifest.json" not in names or not {"config.json", "progress.json"}.issubset(names):
            raise ValueError("Snapshot is missing required recovery files")
        for name in names:
            if name != "manifest.json" and (
                name not in _RECOVERY_FILES or PurePosixPath(name).name != name
            ):
                raise ValueError(f"Snapshot contains unsupported path: {name}")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported V4 snapshot schema")
        contents = {name: archive.read(name) for name in names if name != "manifest.json"}
    if set(manifest.get("files", {})) != set(contents):
        raise ValueError("Snapshot manifest file list does not match archive")
    for name, data in contents.items():
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
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
    _fsync_directory(output)
    runs_path = output / "search_runs.jsonl"
    records = load_run_records(runs_path) if runs_path.exists() else []
    return {
        "restored_files": sorted(contents),
        "run_ids": [str(row["run_id"]) for row in records],
        "manifest": manifest,
    }
