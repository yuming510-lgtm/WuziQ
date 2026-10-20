"""Minimal SHA-256-checked V4B recovery archives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile

from .checkpoint import load_game_records

_ALLOWED_FILES = {
    "config.json",
    "progress.json",
    "9x9_games.jsonl",
    "reuse_audit.json",
    "v3_reused_games.jsonl",
    "code_version.txt",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_snapshot(
    results_dir: str | Path,
    destination: str | Path,
    *,
    code_version: str,
) -> dict[str, Any]:
    """Archive only config, resume progress, completed games, reuse proof, and code ID."""

    root = Path(results_dir)
    required = (root / "config.json", root / "progress.json", root / "reuse_audit.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("V4B snapshot requires config, progress, and reuse audit")
    records_path = root / "9x9_games.jsonl"
    records = load_game_records(records_path) if records_path.is_file() else []
    identifiers = [str(row["game_id"]) for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Cannot snapshot duplicate V4B game IDs")
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if int(progress.get("completed_games", -1)) != len(records):
        raise ValueError("V4B progress count differs from durable game records")
    if set(progress.get("completed_ids", ())) != set(identifiers):
        raise ValueError("V4B progress IDs differ from durable game records")
    audit = json.loads((root / "reuse_audit.json").read_text(encoding="utf-8"))
    if audit.get("verified"):
        source = root / "v3_reused_games.jsonl"
        if not source.is_file() or _sha256(source.read_bytes()) != audit.get("source_games_sha256"):
            raise ValueError("V4B reused V3 source bytes do not match the reuse audit")
    sources = ["config.json", "progress.json", "reuse_audit.json"]
    if records_path.is_file():
        sources.append("9x9_games.jsonl")
    if audit.get("verified"):
        sources.append("v3_reused_games.jsonl")
    contents = {name: (root / name).read_bytes() for name in sources}
    contents["code_version.txt"] = (str(code_version).strip() + "\n").encode("utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": json.loads(contents["config.json"])["experiment_id"],
        "completed_games": len(records),
        "completed_game_ids_sha256": _sha256("\n".join(sorted(identifiers)).encode("utf-8")),
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
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {**manifest, "archive_sha256": _sha256(output.read_bytes())}


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify every archive digest, restore atomically, and reconcile IDs/progress."""

    with zipfile.ZipFile(snapshot_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("V4B snapshot contains duplicate paths")
        if "manifest.json" not in names or not {"config.json", "progress.json", "reuse_audit.json", "code_version.txt"}.issubset(names):
            raise ValueError("V4B snapshot is missing required recovery files")
        for name in names:
            if name != "manifest.json" and (name not in _ALLOWED_FILES or PurePosixPath(name).name != name):
                raise ValueError(f"V4B snapshot contains unsupported path: {name}")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported V4B snapshot schema")
        contents = {name: archive.read(name) for name in names if name != "manifest.json"}
    if set(contents) != set(manifest.get("files", {})):
        raise ValueError("V4B snapshot manifest file list differs from archive")
    for name, data in contents.items():
        if _sha256(data) != manifest["files"][name]:
            raise ValueError(f"V4B snapshot checksum mismatch: {name}")

    output = Path(destination)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("V4B restore destination must be empty")
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
    records_path = output / "9x9_games.jsonl"
    records = load_game_records(records_path) if records_path.exists() else []
    game_ids = sorted(str(record["game_id"]) for record in records)
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("Restored V4B results contain duplicate game IDs")
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    if int(progress.get("completed_games", -1)) != len(records) or set(progress.get("completed_ids", ())) != set(game_ids):
        raise ValueError("Restored V4B progress differs from restored result records")
    audit = json.loads((output / "reuse_audit.json").read_text(encoding="utf-8"))
    if audit.get("verified"):
        source_path = output / "v3_reused_games.jsonl"
        if not source_path.is_file() or _sha256(source_path.read_bytes()) != audit.get("source_games_sha256"):
            raise ValueError("Restored V3 reuse source does not match its audit digest")
    expected_id_hash = _sha256("\n".join(game_ids).encode("utf-8"))
    if expected_id_hash != manifest.get("completed_game_ids_sha256"):
        raise ValueError("Restored game IDs do not match the snapshot manifest")
    return {
        "archive_sha256": _sha256(Path(snapshot_path).read_bytes()),
        "experiment_id": manifest.get("experiment_id"),
        "completed_games": len(game_ids),
        "game_ids": game_ids,
        "restored_files": sorted(contents),
        "code_version": (output / "code_version.txt").read_text(encoding="utf-8").strip(),
    }
