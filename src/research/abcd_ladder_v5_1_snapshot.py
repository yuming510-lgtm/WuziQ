"""Minimal, hash-verified V5.1 recovery snapshots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from .abcd_ladder_v5 import _project_root
from .abcd_ladder_v5_1 import engine_config, load_config
from .abcd_ladder_v5_1_reuse import derive_state
from .abcd_ladder_v5_1_runner import _validate_v51_record
from .checkpoint import artifact_lock, load_game_records

_FILES = ("config.json", "schedule.json", "reuse_manifest.json", "progress.json", "9x9_games.jsonl")
_ROOT = _project_root()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_payload(files: dict[str, bytes]) -> tuple[list[str], dict[str, Any]]:
    if set(files) != set(_FILES):
        raise ValueError("V5.1 snapshot does not contain exactly the required recovery files")
    config = load_config(json.loads(files["config.json"].decode("utf-8")))
    schedule = json.loads(files["schedule.json"].decode("utf-8"))
    reuse = json.loads(files["reuse_manifest.json"].decode("utf-8"))
    progress = json.loads(files["progress.json"].decode("utf-8"))
    if len(schedule) != 168 or sum(len(group.get("games", [])) for group in schedule) != 336:
        raise ValueError("V5.1 snapshot schedule is not the frozen 168-group grid")
    config_hash = _sha(_canonical(config))
    if (progress.get("experiment_id") != config["experiment_id"]
            or progress.get("config_sha256") != config_hash
            or progress.get("schedule_sha256") != _sha(_canonical(schedule))
            or progress.get("reuse_manifest_sha256") != _sha(_canonical(reuse))):
        raise ValueError("V5.1 snapshot progress metadata does not match frozen inputs")
    if progress.get("engine_sha256") != config.get("engine_sha256"):
        raise ValueError("V5.1 snapshot decision-engine pins do not match its config")
    records_path = Path("9x9_games.jsonl")
    # Reuse the JSONL parser's duplicate and corrupt-tail checks without touching
    # the source artifact by writing the bytes to a temporary file.
    with tempfile.TemporaryDirectory(prefix="wuziq-v51-snapshot-check-") as temporary:
        temporary_records = Path(temporary) / records_path.name
        temporary_records.write_bytes(files[records_path.name])
        records = load_game_records(temporary_records)
    if len(records) != len({str(record["game_id"]) for record in records}):
        raise ValueError("V5.1 snapshot contains duplicate game IDs")
    schedule_games = {str(game["game_id"]): game for group in schedule for game in group["games"]}
    if len(schedule_games) != 336:
        raise ValueError("V5.1 snapshot schedule contains duplicate game IDs")
    engine = engine_config(config)
    for record in records:
        spec = schedule_games.get(str(record["game_id"]))
        if spec is None:
            raise ValueError(f"V5.1 snapshot contains unscheduled game {record['game_id']}")
        _validate_v51_record(record, spec, engine)
    derived = derive_state(records, config, schedule)
    ids = sorted(str(record["game_id"]) for record in records)
    if ids != sorted(str(game_id) for game_id in progress.get("completed_ids", [])):
        raise ValueError("V5.1 snapshot progress game IDs differ from durable game records")
    expected_progress = {
        "completed_games": len(ids),
        "completed_groups": derived["completed_groups"],
        "pending_group_id": derived["pending_group_id"],
        "complete": derived["complete"],
        "routes": derived["routes"],
        "groups": derived["groups"],
    }
    for key, expected in expected_progress.items():
        if progress.get(key) != expected:
            raise ValueError(f"V5.1 snapshot progress field {key} is stale")
    return ids, progress


def create_snapshot(results_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write a small zip containing only the inputs and completed game records."""
    root = Path(results_dir)
    source_files = {name: root / name for name in _FILES}
    with artifact_lock(root / ".v5_1_checkpoint.lock", shared=True):
        if any(not path.is_file() for path in source_files.values()):
            raise FileNotFoundError("V5.1 snapshot requires config, schedule, reuse manifest, progress and game JSONL")
        contents = {name: path.read_bytes() for name, path in source_files.items()}
    ids, _ = _validate_payload(contents)
    manifest = {"schema_version": 1, "files": {name: _sha(data) for name, data in sorted(contents.items())}}
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
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
    return {"sha256": _sha(output.read_bytes()), "files": manifest["files"], "completed_games": len(ids)}


def restore_snapshot(snapshot_path: str | Path, destination: str | Path) -> list[str]:
    """Verify and restore into an empty directory; no existing file is overwritten."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("V5.1 snapshot restore destination must be empty")
    with zipfile.ZipFile(snapshot_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {*_FILES, "manifest.json"}:
            raise ValueError("V5.1 snapshot contains an unsupported or duplicate path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported V5.1 snapshot schema")
        contents = {name: archive.read(name) for name in _FILES}
    if manifest.get("files") != {name: _sha(data) for name, data in sorted(contents.items())}:
        raise ValueError("V5.1 snapshot checksum mismatch")
    ids, _ = _validate_payload(contents)
    temporary = Path(tempfile.mkdtemp(prefix="wuziq-v51-restore-", dir=output.parent))
    try:
        for name, data in contents.items():
            (temporary / name).write_bytes(data)
        for name in _FILES:
            os.replace(temporary / name, output / name)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return ids
