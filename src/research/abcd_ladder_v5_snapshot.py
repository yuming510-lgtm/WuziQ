"""Minimal JSON snapshots and verified restoration for V5 recovery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from .abcd_ladder_v5 import load_config
from .abcd_ladder_v5_analysis import validate_records
from .checkpoint import load_game_records, write_json_atomic

_RECOVERY_FILES = ("config.json", "9x9_games.jsonl", "progress.json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_snapshot(results_dir: str | Path, output_path: str | Path, implementation_commit: str) -> dict[str, Any]:
    """Package only config, code identity, progress and completed game records."""
    root = Path(results_dir)
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in _RECOVERY_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"V5 snapshot is missing required recovery file {path}")
        content = path.read_bytes()
        try:
            files[name] = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"V5 recovery file {name} is not UTF-8 text") from exc
        hashes[name] = _sha256(content)
    config = json.loads(files["config.json"])
    load_config(config)
    progress = json.loads(files["progress.json"])
    if progress.get("experiment_id") != config["experiment_id"]:
        raise ValueError("V5 progress does not match snapshot config")
    if progress.get("implementation_commit") != implementation_commit:
        raise ValueError("V5 progress code version does not match snapshot code version")
    records = [json.loads(line) for line in files["9x9_games.jsonl"].splitlines() if line.strip()]
    ids = [str(row.get("game_id", "")) for row in records]
    if len(ids) != len(set(ids)) or not set(progress.get("completed_ids", ())).issubset(ids):
        raise ValueError("V5 snapshot has duplicate games or progress ahead of durable records")
    validate_records(records, config, require_complete=False)
    snapshot = {
        "format": "wuziq-v5-recovery-json-v1",
        "experiment_id": config["experiment_id"],
        "created_at_utc": _utc_now(),
        "implementation_commit": implementation_commit,
        "config_sha256": hashes["config.json"],
        "file_sha256": hashes,
        "files": files,
        "game_ids": ids,
        "completed_games": len(ids),
    }
    write_json_atomic(Path(output_path), snapshot)
    return {
        "snapshot_path": str(Path(output_path).resolve()),
        "snapshot_sha256": _sha256(Path(output_path).read_bytes()),
        "completed_games": len(ids),
        "implementation_commit": implementation_commit,
        "file_sha256": hashes,
    }


def restore_snapshot(snapshot_source: str | Path, empty_directory: str | Path) -> dict[str, Any]:
    """Verify a downloaded snapshot and restore it atomically into an empty directory."""
    source = Path(snapshot_source)
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    if snapshot.get("format") != "wuziq-v5-recovery-json-v1":
        raise ValueError("Unknown V5 snapshot format")
    if set(snapshot.get("files", {})) != set(_RECOVERY_FILES):
        raise ValueError("V5 snapshot lacks required recovery files")
    for name in _RECOVERY_FILES:
        content = str(snapshot["files"][name]).encode("utf-8")
        if _sha256(content) != snapshot.get("file_sha256", {}).get(name):
            raise ValueError(f"V5 snapshot checksum mismatch for {name}")
    if snapshot.get("file_sha256", {}).get("config.json") != snapshot.get("config_sha256"):
        raise ValueError("V5 config checksum manifest mismatch")
    target = Path(empty_directory)
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"V5 restore target must be empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v5-restore-", dir=target.parent))
    try:
        for name in _RECOVERY_FILES:
            (staging / name).write_text(snapshot["files"][name], encoding="utf-8")
        config = load_config(staging / "config.json")
        records = load_game_records(staging / "9x9_games.jsonl")
        valid = validate_records(records, config, require_complete=False)
        progress = json.loads((staging / "progress.json").read_text(encoding="utf-8"))
        if progress.get("implementation_commit") != snapshot.get("implementation_commit"):
            raise ValueError("V5 restored progress uses a different code version")
        if not set(progress.get("completed_ids", ())).issubset({str(row["game_id"]) for row in records}):
            raise ValueError("V5 restored progress is ahead of restored games")
        for path in staging.iterdir():
            shutil.move(str(path), str(target / path.name))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "restored_to": str(target.resolve()),
        "experiment_id": config["experiment_id"],
        "implementation_commit": snapshot["implementation_commit"],
        "games": len(records),
        "valid": valid["valid"],
        "complete": valid["complete"],
        "file_sha256": snapshot["file_sha256"],
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--results-dir", default="research/v5/results")
    create.add_argument("--output", default="research/v5/v5_recovery_snapshot.json")
    create.add_argument("--implementation-commit", required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--empty-directory", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "create":
        result = create_snapshot(args.results_dir, args.output, args.implementation_commit)
    else:
        result = restore_snapshot(args.snapshot, args.empty_directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
