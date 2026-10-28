from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from src.research.v6.config import canonical, sha256
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.schedule import build_schedule
from src.research.v6_2.snapshot import create_snapshot, restore_snapshot


def _prepare_output(root: Path) -> None:
    config = load_v6_2_config()
    schedule = build_schedule(config)
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.strip()
    progress = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "implementation_commit": commit,
        "config_sha256": sha256(canonical(config)),
        "schedule_sha256": sha256(canonical(schedule)),
        "completed_ids": [],
        "completed_games": 0,
        "complete": False,
        "first_started_at_utc": "2026-10-28T00:00:00Z",
        "updated_at_utc": "2026-10-28T00:00:00Z",
    }
    (root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    (root / "9x9_games.jsonl").write_bytes(b"")


def test_snapshot_restores_to_empty_directory_with_matching_hashes(tmp_path: Path) -> None:
    output = tmp_path / "results"
    _prepare_output(output)
    archive = tmp_path / "v6_2_snapshot.zip"
    created = create_snapshot(output, archive)
    target = tmp_path / "empty_restore"
    target.mkdir()

    restored = restore_snapshot(archive, target)

    assert created["size_bytes"] > 0
    assert restored["empty_restore"] is True
    assert restored["game_count"] == 0
    assert restored["implementation_commit"]
    assert (target / "config.json").read_bytes() == (output / "config.json").read_bytes()


def test_snapshot_restore_rejects_tampered_content_and_nonempty_target(tmp_path: Path) -> None:
    output = tmp_path / "results"
    _prepare_output(output)
    archive = tmp_path / "v6_2_snapshot.zip"
    create_snapshot(output, archive)
    with zipfile.ZipFile(archive, "r") as original:
        entries = {name: original.read(name) for name in original.namelist()}
    entries["9x9_games.jsonl"] = b"tampered\n"
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for name, data in entries.items():
            output_zip.writestr(name, data)

    with pytest.raises(ValueError, match="checksum"):
        restore_snapshot(tampered, tmp_path / "restore_tampered")

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        restore_snapshot(archive, nonempty)
    assert (nonempty / "sentinel").read_text(encoding="utf-8") == "keep"
