"""Hashed, empty-directory-restorable recovery packages for G V4.1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any
import zipfile

from .v4_1_config import load_v4_1_config

MANIFEST_NAME = "v4_1_snapshot_manifest.json"
DRIVE_RECEIPT_PATH = "research/g_v4_1/results/recovery/drive_verification.json"


def _is_recovery_payload_path(name: str) -> bool:
    """Exclude transient files and Drive bookkeeping that is not needed to resume."""
    ignored_fragments = {"__pycache__", ".pytest_cache", ".pyc"}
    return (name != DRIVE_RECEIPT_PATH
            and not any(part in name for part in ignored_fragments)
            and not name.endswith((".tmp", ".zip")))


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(raw: str) -> str:
    path = PurePosixPath(str(raw))
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe snapshot path: {raw!r}")
    return path.as_posix()


def create_snapshot_archive(
    root: str | Path,
    archive_path: str | Path,
    relative_paths: list[str],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a deterministic file list and SHA-256 manifest atomically."""
    root_path = Path(root).resolve()
    target = Path(archive_path).resolve()
    names = [_safe_relative(raw) for raw in relative_paths]
    if len(names) != len(set(names)) or MANIFEST_NAME in names:
        raise ValueError("snapshot paths must be unique and cannot replace the manifest")
    entries = []
    for name in sorted(names):
        source = root_path / name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"snapshot source is missing or not a regular file: {name}")
        data = source.read_bytes()
        entries.append({"path": name, "size_bytes": len(data), "sha256": _sha_bytes(data)})
    manifest = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4-1",
        "metadata": dict(metadata),
        "files_count": len(entries),
        "total_file_bytes": sum(int(row["size_bytes"]) for row in entries),
        "files": entries,
    }
    payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for row in entries:
                archive.write(root_path / row["path"], arcname=row["path"])
            archive.writestr(MANIFEST_NAME, payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def restore_snapshot_archive(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify every archived byte and materialize only into an empty directory."""
    source = Path(archive_path)
    target = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise ValueError("snapshot restore destination must be empty")
    staging = Path(tempfile.mkdtemp(prefix=".v4-1-restore-", dir=target))
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = archive.namelist()
            if len(members) != len(set(members)) or MANIFEST_NAME not in members:
                raise ValueError("snapshot archive has duplicate entries or no manifest")
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            if manifest.get("format_version") != 1 or manifest.get("experiment_id") != "wuziq-alphazero-g-v4-1":
                raise ValueError("unsupported V4.1 snapshot manifest")
            entries = manifest.get("files")
            if not isinstance(entries, list) or int(manifest.get("files_count", -1)) != len(entries):
                raise ValueError("snapshot manifest file list is incomplete")
            expected_names = {_safe_relative(row["path"]) for row in entries}
            if set(members) != expected_names | {MANIFEST_NAME}:
                raise ValueError("snapshot archive entries differ from its manifest")
            for row in entries:
                name = _safe_relative(row["path"])
                data = archive.read(name)
                if len(data) != int(row["size_bytes"]) or _sha_bytes(data) != row["sha256"]:
                    raise ValueError(f"snapshot file SHA-256 mismatch: {name}")
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
        for row in entries:
            name = _safe_relative(row["path"])
            final = target / name
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, final)
        shutil.rmtree(staging)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def create_v4_1_snapshot(root: str | Path, archive_path: str | Path) -> dict[str, Any]:
    """Package all changed V4.1 inputs/results and the source code needed to resume."""
    root_path = Path(root).resolve()
    config = load_v4_1_config(root_path)
    diff = subprocess.run(
        ["git", "diff", "--name-only", config["base_commit"]], cwd=root_path,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root_path,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    paths = sorted({name for name in [*diff, *untracked]
                    if name and _is_recovery_payload_path(name)})
    if not paths or not any(name.startswith("research/g_v4_1/") for name in paths):
        raise ValueError("snapshot has no V4.1 experiment state")
    progress_path = root_path / "research/g_v4_1/results/progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"phase": "PREREGISTERED"}
    metadata = {
        "branch": config["branch"],
        "base_commit": config["base_commit"],
        "current_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root_path,
                                          check=True, capture_output=True, text=True).stdout.strip(),
        "config_sha256": config["config_sha256"],
        "opening_schedule_sha256": config["opening_schedule_sha256"],
        "progress": progress,
    }
    return create_snapshot_archive(root_path, archive_path, paths, metadata=metadata)
