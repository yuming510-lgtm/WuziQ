"""Integrity-checked V2 recovery archive extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
from typing import Any


MANIFEST_NAME = "recovery_manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_snapshot(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Extract into a fresh directory and verify every archived file hash."""
    archive_path, destination = Path(archive_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("snapshot destination must be empty")
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or MANIFEST_NAME not in names:
            raise ValueError("snapshot archive has duplicate paths or no recovery manifest")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("snapshot archive contains an unsafe path")
            info = archive.getinfo(name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("snapshot archive must not contain symbolic links")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        expected = manifest.get("files", {})
        if set(names) != set(expected) | {MANIFEST_NAME}:
            raise ValueError("snapshot file list differs from its recovery manifest")
        total_bytes = 0
        for name, metadata in sorted(expected.items()):
            path = PurePosixPath(name)
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            if target.stat().st_size != int(metadata["bytes"]):
                raise ValueError(f"restored file has an unexpected size: {name}")
            if sha256_file(target) != metadata["sha256"]:
                raise ValueError(f"restored file SHA-256 mismatch: {name}")
            total_bytes += target.stat().st_size
    return {
        "files_verified": len(expected), "bytes_restored": total_bytes,
        "code_commit": manifest.get("code_commit"),
        "experiment_schedule_sha256": manifest.get("experiment_schedule_sha256"),
        "model_checkpoint_sha256": manifest.get("model_checkpoint_sha256"),
        "status": "passed",
    }
