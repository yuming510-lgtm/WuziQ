"""Minimal recovery archive creation, integrity, and restore tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module():
    assert importlib.util.find_spec("src.research.abc_search_v4_snapshot") is not None
    from src.research import abc_search_v4_snapshot

    return abc_search_v4_snapshot


def test_snapshot_round_trip_restores_config_progress_runs_and_code_version(
    tmp_path: Path,
) -> None:
    snapshot = _module()
    config = tmp_path / "config.json"
    progress = tmp_path / "progress.json"
    runs = tmp_path / "search_runs.jsonl"
    config.write_text('{"experiment_id":"v4"}\n', encoding="utf-8")
    progress.write_text('{"completed_runs":1}\n', encoding="utf-8")
    runs.write_text('{"run_id":"opening__A__00050ms"}\n', encoding="utf-8")
    archive = tmp_path / "checkpoint.zip"

    manifest = snapshot.create_snapshot(
        config, progress, runs, "3d6135a-code", archive
    )
    restored = snapshot.restore_snapshot(archive, tmp_path / "restored")

    assert set(manifest["files"]) == {
        "config.json",
        "progress.json",
        "search_runs.jsonl",
        "code_version.txt",
    }
    assert restored["run_ids"] == ["opening__A__00050ms"]
    assert (tmp_path / "restored" / "config.json").read_bytes() == config.read_bytes()
    assert (tmp_path / "restored" / "code_version.txt").read_text().strip() == "3d6135a-code"


def test_snapshot_restore_rejects_checksum_mismatch(tmp_path: Path) -> None:
    snapshot = _module()
    config = tmp_path / "config.json"
    progress = tmp_path / "progress.json"
    runs = tmp_path / "search_runs.jsonl"
    config.write_text("{}\n", encoding="utf-8")
    progress.write_text("{}\n", encoding="utf-8")
    runs.write_text("", encoding="utf-8")
    archive = tmp_path / "checkpoint.zip"
    snapshot.create_snapshot(config, progress, runs, "v4-commit", archive)

    altered = tmp_path / "altered.zip"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(altered, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "progress.json":
                payload += b"bad"
            target.writestr(name, payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        snapshot.restore_snapshot(altered, tmp_path / "invalid-restore")
