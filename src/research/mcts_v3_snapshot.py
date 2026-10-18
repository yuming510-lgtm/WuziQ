"""V3-specific validation around the shared durable recovery archive format."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .checkpoint import create_snapshot, load_game_records, restore_snapshot
from .mcts_v2 import build_schedule

_EXPERIMENT_PREFIX = "wuziq-mcts-high-compute-v3"


def _read_run_state(results_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    config_path = results_dir / "config.json"
    progress_path = results_dir / "progress.json"
    if not config_path.is_file() or not progress_path.is_file():
        raise FileNotFoundError("V3 recovery requires config.json and progress.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    experiment_id = str(config.get("experiment_id", ""))
    if not experiment_id.startswith(_EXPERIMENT_PREFIX):
        raise ValueError("snapshot config is not a V3 high-compute experiment")
    if str(progress.get("experiment_id", "")) != experiment_id:
        raise ValueError("progress experiment_id differs from the V3 config")
    code_commit = str(config.get("code_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("V3 snapshot requires a full 40-character code_commit")

    schedule = build_schedule(config)
    scheduled_ids = {str(spec["game_id"]) for spec in schedule}
    game_paths = sorted(results_dir.glob("*_games.jsonl"))
    records = [record for path in game_paths for record in load_game_records(path)]
    identifiers = [str(record["game_id"]) for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("V3 recovery records contain duplicate game IDs")
    unexpected = set(identifiers) - scheduled_ids
    if unexpected:
        raise ValueError(f"V3 recovery contains unscheduled game IDs: {sorted(unexpected)[:3]}")
    if int(progress.get("scheduled_games", -1)) != len(schedule):
        raise ValueError("V3 progress scheduled_games differs from the registered schedule")
    if int(progress.get("games_completed", -1)) != len(identifiers):
        raise ValueError("V3 progress games_completed differs from the durable game records")
    if bool(progress.get("complete", False)) != (len(identifiers) == len(schedule)):
        raise ValueError("V3 progress complete flag differs from durable game records")
    return config, progress, game_paths


def create_v3_snapshot(results_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create a minimal, validated archive and return the archive SHA-256."""

    root = Path(results_dir)
    config, progress, game_paths = _read_run_state(root)
    manifest = create_snapshot(
        root / "config.json",
        root / "progress.json",
        game_paths,
        destination,
    )
    archive_hash = hashlib.sha256(Path(destination).read_bytes()).hexdigest()
    return {
        "archive_sha256": archive_hash,
        "code_commit": config["code_commit"],
        "experiment_id": config["experiment_id"],
        "completed_games": int(progress["games_completed"]),
        "scheduled_games": int(progress["scheduled_games"]),
        "completed_game_ids": sorted(
            str(record["game_id"])
            for path in game_paths
            for record in load_game_records(path)
        ),
        "archived_files": sorted(["config.json", "progress.json", *[p.name for p in game_paths]]),
        "manifest": manifest,
    }


def restore_v3_snapshot(
    snapshot_path: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Verify checksums, restore files, and validate IDs against the V3 schedule."""

    snapshot = Path(snapshot_path)
    restored_root = Path(destination)
    identifiers = restore_snapshot(snapshot, restored_root)
    config, progress, _ = _read_run_state(restored_root)
    if identifiers != sorted(
        str(record["game_id"])
        for path in sorted(restored_root.glob("*_games.jsonl"))
        for record in load_game_records(path)
    ):
        raise ValueError("restored game IDs differ from V3 JSONL records")
    return {
        "archive_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "code_commit": config["code_commit"],
        "experiment_id": config["experiment_id"],
        "completed_games": int(progress["games_completed"]),
        "scheduled_games": int(progress["scheduled_games"]),
        "completed_game_ids": identifiers,
    }
