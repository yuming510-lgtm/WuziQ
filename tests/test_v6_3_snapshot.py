from __future__ import annotations

import json
import subprocess
from pathlib import Path
import zipfile

import pytest

from src.research.checkpoint import write_json_atomic
from src.research.v6.config import canonical, sha256
from src.research.v6_3.config import load_v6_3_config
from src.research.v6_3.schedule import build_schedule
from src.research.v6_3.snapshot import create_snapshot, restore_snapshot


def _prepare_results(root: Path) -> None:
    config = load_v6_3_config()
    schedule = build_schedule(config)
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    progress = {
        "schema_version": 1, "experiment_id": config["experiment_id"],
        "implementation_commit": commit, "config_sha256": sha256(canonical(config)),
        "schedule_sha256": sha256(canonical(schedule)), "completed_ids": [],
        "completed_games": 0, "complete": False, "tiebreak_status": "pending",
        "first_started_at_utc": "2026-10-28T00:00:00Z", "updated_at_utc": "2026-10-28T00:00:00Z",
    }
    (root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    (root / "9x9_games.jsonl").write_bytes(b"")


def test_v6_3_snapshot_restores_verified_contents_to_empty_directory(tmp_path: Path) -> None:
    output = tmp_path / "results"
    _prepare_results(output)
    archive = tmp_path / "v6_3_snapshot.zip"
    created = create_snapshot(output, archive)
    target = tmp_path / "empty_restore"
    target.mkdir()

    restored = restore_snapshot(archive, target)

    assert created["size_bytes"] > 0
    assert restored["empty_restore"] is True
    assert restored["game_count"] == 0
    for name in ("config.json", "schedule.json", "progress.json", "9x9_games.jsonl"):
        assert (target / name).read_bytes() == (output / name).read_bytes()


def test_snapshot_rejects_tampering_and_nonempty_restore_target(tmp_path: Path) -> None:
    output = tmp_path / "results"
    _prepare_results(output)
    archive = tmp_path / "v6_3_snapshot.zip"
    create_snapshot(output, archive)
    with zipfile.ZipFile(archive, "r") as original:
        entries = {name: original.read(name) for name in original.namelist()}
    entries["9x9_games.jsonl"] = b"tampered\n"
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for name, data in entries.items():
            output_zip.writestr(name, data)
    with pytest.raises(ValueError, match="checksum"):
        restore_snapshot(tampered, tmp_path / "tampered_restore")

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        restore_snapshot(archive, nonempty)
    assert (nonempty / "sentinel").read_text(encoding="utf-8") == "keep"

