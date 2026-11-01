"""Hash-verified, path-safe recovery archives for long G V1 runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile
from typing import Any

_DEFAULT_INCLUDE = (
    "src/research/alphazero_g",
    "scripts/preflight_g_v1.py",
    "scripts/register_g_v1_budget.py",
    "scripts/run_g_v1.py",
    "scripts/evaluate_g_v1.py",
    "scripts/report_g_v1.py",
    "scripts/create_g_v1_snapshot.py",
    "scripts/restore_g_v1_snapshot.py",
    "tests/test_alphazero_g_puct.py",
    "tests/test_alphazero_g_selfplay.py",
    "tests/test_alphazero_g_training.py",
    "tests/test_alphazero_g_experiment.py",
    "tests/test_alphazero_g_snapshot.py",
    "research/g_v1",
    "research/f_v2/checkpoints/best_f2.pt",
    "research/f_v2/checkpoints/f1_reference.pt",
)


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("snapshot contains an unsafe empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "" in path.parts or "\\" in value:
        raise ValueError(f"snapshot contains an unsafe path: {value}")
    normalized = path.as_posix()
    if normalized in {".", "manifest.json"}:
        raise ValueError(f"snapshot contains a reserved path: {value}")
    return normalized


def _source_files(root: Path, includes: list[str]) -> list[tuple[str, Path]]:
    result: dict[str, Path] = {}
    for requested in includes:
        relative = _validate_relative_path(requested)
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(f"required snapshot input does not exist: {relative}")
        if source.is_symlink():
            raise ValueError(f"snapshot input may not be a symlink: {relative}")
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"snapshot input contains a symlink: {candidate}")
            if not candidate.is_file():
                continue
            archive_path = _validate_relative_path(candidate.relative_to(root).as_posix())
            result[archive_path] = candidate
    if not result:
        raise ValueError("snapshot has no files")
    return sorted(result.items())


def create_snapshot(
    root: str | Path,
    output: str | Path,
    commit: str,
    active_seconds: float,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Create an atomic ZIP containing an allowlisted set of recovery files."""
    if not isinstance(commit, str) or not commit:
        raise ValueError("commit must be nonempty")
    if not math.isfinite(float(active_seconds)) or active_seconds < 0:
        raise ValueError("active_seconds must be finite and nonnegative")
    root_path = Path(root).resolve()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(manifest or {})
    includes = list(metadata.pop("include", _DEFAULT_INCLUDE))
    files = _source_files(root_path, includes)
    file_records = []
    for relative, source in files:
        data = source.read_bytes()
        file_records.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mode": stat.S_IMODE(source.stat().st_mode),
        })
    archive_manifest = {
        "format_version": 1,
        "commit": commit,
        "active_seconds": float(active_seconds),
        "metadata": metadata,
        "files": file_records,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", json.dumps(archive_manifest, sort_keys=True, separators=(",", ":")))
            for relative, source in files:
                archive.write(source, relative)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output_path)
        parent_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def restore_snapshot(archive_path: str | Path, empty_directory: str | Path) -> dict[str, Any]:
    """Verify every member before atomically populating an empty destination."""
    archive_path = Path(archive_path)
    destination = Path(empty_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("snapshot restore destination must be empty")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("snapshot archive contains duplicate paths")
        if names.count("manifest.json") != 1:
            raise ValueError("snapshot archive must contain one manifest.json")
        for info in infos:
            if info.is_dir():
                raise ValueError("snapshot archive may contain files only")
            name = info.filename
            if name != "manifest.json":
                _validate_relative_path(name)
            file_type = stat.S_IFMT(info.external_attr >> 16)
            if file_type == stat.S_IFLNK:
                raise ValueError(f"snapshot archive contains a symlink: {name}")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("snapshot manifest is invalid JSON") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise ValueError("unsupported snapshot format")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("snapshot manifest has no file records")
        expected_paths: set[str] = set()
        verified: list[tuple[str, bytes, int]] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("snapshot file record must be an object")
            relative = _validate_relative_path(record.get("path"))
            if relative in expected_paths:
                raise ValueError("snapshot manifest contains duplicate paths")
            expected_paths.add(relative)
            try:
                data = archive.read(relative)
            except KeyError as exc:
                raise ValueError(f"snapshot archive is missing {relative}") from exc
            if len(data) != record.get("size"):
                raise ValueError(f"snapshot size mismatch for {relative}")
            if hashlib.sha256(data).hexdigest() != record.get("sha256"):
                raise ValueError(f"snapshot SHA-256 mismatch for {relative}")
            mode = record.get("mode", 0o600)
            if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o777:
                raise ValueError(f"snapshot mode is invalid for {relative}")
            verified.append((relative, data, mode))
        actual_paths = set(names) - {"manifest.json"}
        if actual_paths != expected_paths:
            raise ValueError("snapshot file list does not match its manifest")

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent))
    try:
        for relative, data, mode in verified:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            target.chmod(mode)
        if any(destination.iterdir()):
            raise ValueError("snapshot restore destination changed during restore")
        destination.rmdir()
        staging.replace(destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def next_snapshot_threshold(active_seconds: float) -> int:
    """Return the next cumulative snapshot threshold for 1,2,4,6,10,16,26h…"""
    if not math.isfinite(float(active_seconds)) or active_seconds < 0:
        raise ValueError("active_seconds must be finite and nonnegative")
    thresholds = [3600, 10800, 25200, 46800, 82800, 140400]
    next_value = thresholds[-1]
    while next_value <= active_seconds:
        next_value += 26 * 3600
    for threshold in thresholds:
        if threshold > active_seconds:
            return threshold
    return next_value
