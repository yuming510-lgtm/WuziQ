"""Checksummed minimal recovery snapshot and validated empty-directory restore."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any
import zipfile

from ..checkpoint import artifact_lock, load_game_records
from ..v6.config import canonical, sha256
from .config import ROOT, load_v6_3_config
from .schedule import build_schedule
from .validation import validate_v6_3_records


SNAPSHOT_FILES = ("config.json", "schedule.json", "progress.json", "9x9_games.jsonl")
CODE_FILES = (
    "configs/v6_3/championship.json",
    "src/research/v6_3/__init__.py",
    "src/research/v6_3/config.py",
    "src/research/v6_3/schedule.py",
    "src/research/v6_3/validation.py",
    "src/research/v6_3/runner.py",
    "src/research/v6_3/snapshot.py",
    "src/research/v6_3/analysis.py",
    "scripts/preflight_v6_3.py",
    "scripts/run_v6_3_championship.py",
    "scripts/validate_v6_3_results.py",
    "scripts/build_v6_3_report.py",
    "scripts/create_v6_3_snapshot.py",
    "scripts/restore_v6_3_snapshot.py",
)
IDENTITY_FILE = "source_identity.json"
MANIFEST_FILE = "manifest.json"


def _branch_head() -> tuple[str, str]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    return branch, commit


def _identity(progress: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    branch, head = _branch_head()
    if branch != config["branch"]:
        raise ValueError("V6.3 snapshots must be produced on the registered branch")
    if str(progress.get("implementation_commit")) != head:
        raise ValueError("V6.3 snapshot progress does not match the checked-out implementation commit")
    code_hashes = {}
    for relative in CODE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V6.3 recovery source is missing {relative}")
        code_hashes[relative] = sha256(path.read_bytes())
    return {
        "branch": branch,
        "implementation_commit": head,
        "source_v6_2_commit": config["source_v6_2_commit"],
        "experiment_id": config["experiment_id"],
        "engine_sha256": config["engine_sha256"],
        "code_sha256": code_hashes,
    }


def _archive(contents: dict[str, bytes], destination: Path, identity: dict[str, Any]) -> dict[str, Any]:
    contents[IDENTITY_FILE] = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    for relative in CODE_FILES:
        contents[f"source/{relative}"] = (ROOT / relative).read_bytes()
    manifest = {
        "schema_version": 1,
        "experiment_id": identity["experiment_id"],
        "implementation_commit": identity["implementation_commit"],
        "files": {name: sha256(data) for name, data in sorted(contents.items())},
    }
    contents[MANIFEST_FILE] = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(name, data)
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    raw = destination.read_bytes()
    return {
        "path": str(destination),
        "sha256": sha256(raw),
        "size_bytes": len(raw),
        "experiment_id": identity["experiment_id"],
        "implementation_commit": identity["implementation_commit"],
        "game_count": json.loads(contents["progress.json"]).get("completed_games", 0),
    }


def create_snapshot(output_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    root, output = Path(output_dir), Path(destination)
    with artifact_lock(root / ".v6_3_checkpoint.lock", shared=True):
        contents: dict[str, bytes] = {}
        for name in SNAPSHOT_FILES:
            source = root / name
            if not source.is_file():
                if name == "9x9_games.jsonl":
                    contents[name] = b""
                    continue
                raise FileNotFoundError(f"V6.3 snapshot is missing {name}")
            contents[name] = source.read_bytes()
    config = load_v6_3_config(json.loads(contents["config.json"]))
    schedule = json.loads(contents["schedule.json"])
    progress = json.loads(contents["progress.json"])
    if canonical(schedule) != canonical(build_schedule(config)):
        raise ValueError("V6.3 snapshot schedule differs from preregistration")
    if progress.get("config_sha256") != sha256(canonical(config)):
        raise ValueError("V6.3 snapshot config hash mismatch")
    if progress.get("schedule_sha256") != sha256(canonical(schedule)):
        raise ValueError("V6.3 snapshot schedule hash mismatch")
    staging_dir = Path(tempfile.mkdtemp(prefix="v6_3_snapshot_validate_"))
    try:
        (staging_dir / "9x9_games.jsonl").write_bytes(contents["9x9_games.jsonl"])
        records = load_game_records(staging_dir / "9x9_games.jsonl")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    checked = validate_v6_3_records(records, schedule, require_complete=False)
    record_ids = {str(record["game_id"]) for record in records}
    if not set(map(str, progress.get("completed_ids", ()))) <= record_ids:
        raise ValueError("V6.3 snapshot progress claims a missing game")
    if len(records) != int(progress.get("completed_games", -1)) or checked["games"] != len(records):
        raise ValueError("V6.3 snapshot progress count differs from replayed results")
    identity = _identity(progress, config)
    return _archive(contents, output, identity)


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    snapshot = Path(snapshot_path)
    with zipfile.ZipFile(snapshot, "r") as archive:
        names = archive.namelist()
        expected = {
            *SNAPSHOT_FILES, IDENTITY_FILE, MANIFEST_FILE,
            *(f"source/{relative}" for relative in CODE_FILES),
        }
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("V6.3 snapshot has missing, duplicate, or unsupported files")
        contents = {name: archive.read(name) for name in names if name != MANIFEST_FILE}
        manifest = json.loads(archive.read(MANIFEST_FILE))
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(contents):
        raise ValueError("V6.3 snapshot manifest is malformed")
    for name, data in contents.items():
        if sha256(data) != manifest["files"][name]:
            raise ValueError(f"V6.3 snapshot checksum mismatch for {name}")
    identity = json.loads(contents[IDENTITY_FILE])
    config = load_v6_3_config(json.loads(contents["config.json"]))
    schedule = json.loads(contents["schedule.json"])
    progress = json.loads(contents["progress.json"])
    if (identity.get("experiment_id") != config["experiment_id"]
            or progress.get("experiment_id") != config["experiment_id"]
            or identity.get("implementation_commit") != manifest.get("implementation_commit")
            or identity.get("implementation_commit") != progress.get("implementation_commit")
            or identity.get("source_v6_2_commit") != config["source_v6_2_commit"]
            or identity.get("engine_sha256") != config["engine_sha256"]):
        raise ValueError("V6.3 snapshot source identity mismatch")
    if (progress.get("config_sha256") != sha256(canonical(config))
            or progress.get("schedule_sha256") != sha256(canonical(schedule))
            or canonical(schedule) != canonical(build_schedule(config))):
        raise ValueError("V6.3 snapshot frozen config or schedule mismatch")
    if set(identity.get("code_sha256", {})) != set(CODE_FILES):
        raise ValueError("V6.3 snapshot code manifest is incomplete")
    for relative in CODE_FILES:
        source_name = f"source/{relative}"
        if (sha256(contents[source_name]) != identity["code_sha256"][relative]
                or contents[source_name] != (ROOT / relative).read_bytes()):
            raise ValueError(f"V6.3 recovery code mismatch: {relative}")
    staging = Path(tempfile.mkdtemp(prefix="v6_3_restore_validate_"))
    try:
        (staging / "9x9_games.jsonl").write_bytes(contents["9x9_games.jsonl"])
        records = load_game_records(staging / "9x9_games.jsonl")
        checked = validate_v6_3_records(records, schedule, require_complete=False)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ids = {str(row["game_id"]) for row in records}
    if (len(records) != int(progress.get("completed_games", -1))
            or not set(map(str, progress.get("completed_ids", ()))) <= ids
            or checked["games"] != len(records)):
        raise ValueError("V6.3 snapshot progress and recovered games disagree")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir() or any(target.iterdir()):
        raise ValueError("V6.3 restoration requires an empty destination directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore.", dir=target.parent))
    try:
        for name in SNAPSHOT_FILES:
            (stage / name).write_bytes(contents[name])
        source_root = stage / "source"
        for relative in CODE_FILES:
            restored = source_root / relative
            restored.parent.mkdir(parents=True, exist_ok=True)
            restored.write_bytes(contents[f"source/{relative}"])
        (stage / IDENTITY_FILE).write_bytes(contents[IDENTITY_FILE])
        target.rmdir()
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "empty_restore": True,
        "game_count": len(records),
        "completed_ids": sorted(ids),
        "implementation_commit": identity["implementation_commit"],
        "restored_files": [*SNAPSHOT_FILES, IDENTITY_FILE, *(f"source/{p}" for p in CODE_FILES)],
        "tiebreak_status": checked["tiebreak_status"],
    }

