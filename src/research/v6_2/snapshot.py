"""Minimal, checksummed V6.2 recovery snapshots and empty-directory restore."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import zipfile

from ..checkpoint import artifact_lock, load_game_records
from ..v6.config import canonical, sha256
from .config import ROOT, load_v6_2_config
from .schedule import build_schedule
from .validation import validate_v6_2_records


SNAPSHOT_FILES = ("config.json", "schedule.json", "progress.json", "9x9_games.jsonl")
CODE_FILES = (
    "src/research/v6_2/__init__.py",
    "src/research/v6_2/config.py",
    "src/research/v6_2/schedule.py",
    "src/research/v6_2/runner.py",
    "src/research/v6_2/validation.py",
    "src/research/v6_2/snapshot.py",
)
_MANIFEST = "manifest.json"
_IDENTITY = "source_identity.json"


def _source_identity(progress: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot identify the V6.2 recovery source commit") from exc
    commit = str(progress.get("implementation_commit", ""))
    if not commit or commit != head:
        raise ValueError("V6.2 snapshot progress does not match the checked-out source commit")
    if config.get("branch") != branch:
        raise ValueError("V6.2 snapshot is not being created on its registered branch")
    return {
        "branch": branch,
        "commit": head,
        "baseline_commit": config["baseline_commit"],
        "experiment_id": config["experiment_id"],
        "code_sha256": {relative: sha256((ROOT / relative).read_bytes()) for relative in CODE_FILES},
        "engine_sha256": config["engine_sha256"],
    }


def _write_archive(contents: dict[str, bytes], destination: Path, identity: dict[str, Any]) -> dict[str, Any]:
    contents[_IDENTITY] = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    manifest = {
        "schema_version": 1,
        "experiment_id": identity["experiment_id"],
        "implementation_commit": identity["commit"],
        "files": {name: sha256(data) for name, data in sorted(contents.items())},
    }
    contents[_MANIFEST] = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(name, data)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
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
    raw = destination.read_bytes()
    return {
        "path": str(destination),
        "sha256": sha256(raw),
        "size_bytes": len(raw),
        "experiment_id": identity["experiment_id"],
        "implementation_commit": identity["commit"],
        "game_count": len(json.loads(contents["progress.json"]).get("completed_ids", [])),
    }


def create_snapshot(output_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    """Capture only the frozen settings, six-game schedule, progress, and completed games."""
    root = Path(output_dir)
    output = Path(destination)
    with artifact_lock(root / ".v6_2_checkpoint.lock", shared=True):
        contents: dict[str, bytes] = {}
        for name in SNAPSHOT_FILES:
            source = root / name
            if not source.is_file():
                if name == "9x9_games.jsonl":
                    contents[name] = b""
                    continue
                raise FileNotFoundError(f"V6.2 snapshot is missing {name}")
            contents[name] = source.read_bytes()

    config = load_v6_2_config(json.loads(contents["config.json"]))
    schedule = json.loads(contents["schedule.json"])
    progress = json.loads(contents["progress.json"])
    if canonical(schedule) != canonical(build_schedule(config)):
        raise ValueError("V6.2 snapshot schedule differs from the preregistered schedule")
    if progress.get("config_sha256") != sha256(canonical(config)):
        raise ValueError("V6.2 snapshot progress config hash mismatch")
    if progress.get("schedule_sha256") != sha256(canonical(schedule)):
        raise ValueError("V6.2 snapshot progress schedule hash mismatch")
    records_path = root / "9x9_games.jsonl"
    records = load_game_records(records_path) if records_path.is_file() else []
    validation = validate_v6_2_records(records, schedule, require_complete=False)
    ids = {str(row["game_id"]) for row in records}
    if not set(map(str, progress.get("completed_ids", ()))) <= ids:
        raise ValueError("V6.2 snapshot progress claims a game absent from JSONL")
    if len(records) != int(progress.get("completed_games", -1)):
        raise ValueError("V6.2 snapshot progress count differs from completed game records")
    if validation["games"] != len(records):
        raise ValueError("V6.2 snapshot contains invalid game records")
    identity = _source_identity(progress, config)
    return _write_archive(contents, output, identity)


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Validate a V6.2 archive fully, then restore it into an empty directory."""
    archive_path = Path(snapshot_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        expected = {*SNAPSHOT_FILES, _IDENTITY, _MANIFEST}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("V6.2 snapshot has missing, duplicate, or unsupported files")
        contents = {name: archive.read(name) for name in (*SNAPSHOT_FILES, _IDENTITY)}
        manifest = json.loads(archive.read(_MANIFEST))
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(contents):
        raise ValueError("V6.2 snapshot manifest is malformed")
    for name, data in contents.items():
        if sha256(data) != manifest["files"][name]:
            raise ValueError(f"V6.2 snapshot checksum mismatch for {name}")

    identity = json.loads(contents[_IDENTITY])
    config = load_v6_2_config(json.loads(contents["config.json"]))
    schedule = json.loads(contents["schedule.json"])
    progress = json.loads(contents["progress.json"])
    if (identity.get("experiment_id") != config["experiment_id"]
            or identity.get("experiment_id") != progress.get("experiment_id")
            or identity.get("commit") != manifest.get("implementation_commit")
            or progress.get("implementation_commit") != identity.get("commit")
            or config.get("baseline_commit") != identity.get("baseline_commit")):
        raise ValueError("V6.2 snapshot experiment or source identity mismatch")
    if progress.get("config_sha256") != sha256(canonical(config)):
        raise ValueError("V6.2 snapshot config identity mismatch")
    if progress.get("schedule_sha256") != sha256(canonical(schedule)):
        raise ValueError("V6.2 snapshot schedule identity mismatch")
    if canonical(schedule) != canonical(build_schedule(config)):
        raise ValueError("V6.2 snapshot does not contain the deterministic schedule")
    for relative, expected_hash in identity.get("code_sha256", {}).items():
        code_path = ROOT / relative
        if not code_path.is_file() or sha256(code_path.read_bytes()) != expected_hash:
            raise ValueError(f"V6.2 recovery code mismatch: {relative}")
    if identity.get("engine_sha256") != config["engine_sha256"]:
        raise ValueError("V6.2 snapshot decision-engine hashes differ from the frozen config")

    staging_records = contents["9x9_games.jsonl"]
    staging = Path(tempfile.mkdtemp(prefix="v6_2_restore_validate_"))
    try:
        (staging / "9x9_games.jsonl").write_bytes(staging_records)
        records = load_game_records(staging / "9x9_games.jsonl")
        validated = validate_v6_2_records(records, schedule, require_complete=False)
        record_ids = {str(row["game_id"]) for row in records}
        progress_ids = set(map(str, progress.get("completed_ids", ())))
        if not progress_ids <= record_ids:
            raise ValueError("V6.2 snapshot progress claims missing completed games")
        if len(records) != int(progress.get("completed_games", -1)):
            raise ValueError("V6.2 snapshot game count disagrees with progress")
        if validated["games"] != len(records):
            raise ValueError("V6.2 snapshot replay count mismatch")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=False, exist_ok=True)
    if not target.is_dir() or any(target.iterdir()):
        raise ValueError("V6.2 snapshot restoration requires an empty destination directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore.", dir=target.parent))
    try:
        for name, data in contents.items():
            (stage / name).write_bytes(data)
        target.rmdir()
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "restored_files": [*SNAPSHOT_FILES, _IDENTITY],
        "game_count": len(records),
        "progress_game_count": len(progress_ids),
        "implementation_commit": identity["commit"],
        "empty_restore": True,
    }
