from __future__ import annotations

import json
import zipfile

import pytest

from src.research.alphazero_g.v4_1_snapshot import (
    _is_recovery_payload_path,
    create_snapshot_archive,
    restore_snapshot_archive,
)


def test_v4_1_snapshot_round_trips_to_an_empty_directory(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "research/g_v4_1").mkdir(parents=True)
    (source / "src/research/alphazero_g").mkdir(parents=True)
    (source / "research/g_v4_1/progress.json").write_text('{"games": 3}\n')
    (source / "src/research/alphazero_g/v4_1_runner.py").write_text("runner\n")
    archive = tmp_path / "snapshot.zip"
    destination = tmp_path / "empty-restore"
    destination.mkdir()

    manifest = create_snapshot_archive(
        source,
        archive,
        ["research/g_v4_1/progress.json", "src/research/alphazero_g/v4_1_runner.py"],
        metadata={"experiment_id": "wuziq-alphazero-g-v4-1"},
    )
    restored = restore_snapshot_archive(archive, destination)

    assert manifest["files_count"] == 2
    assert restored["files_count"] == 2
    assert (destination / "research/g_v4_1/progress.json").read_text() == '{"games": 3}\n'
    assert (destination / "src/research/alphazero_g/v4_1_runner.py").read_text() == "runner\n"


def test_v4_1_snapshot_rejects_nonempty_restore_destination(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("x")
    archive = tmp_path / "snapshot.zip"
    create_snapshot_archive(source, archive, ["a.txt"], metadata={})
    destination = tmp_path / "not-empty"
    destination.mkdir()
    (destination / "other.txt").write_text("keep")

    with pytest.raises(ValueError, match="empty"):
        restore_snapshot_archive(archive, destination)


def test_v4_1_snapshot_detects_modified_archive_content(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("original")
    archive = tmp_path / "snapshot.zip"
    create_snapshot_archive(source, archive, ["a.txt"], metadata={})
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "r") as src, zipfile.ZipFile(tampered, "w") as dst:
        for member in src.infolist():
            payload = src.read(member.filename)
            if member.filename == "a.txt":
                payload = b"changed"
            dst.writestr(member, payload)
    destination = tmp_path / "tamper-restore"
    destination.mkdir()

    with pytest.raises(ValueError, match="SHA-256"):
        restore_snapshot_archive(tampered, destination)


def test_v4_1_recovery_payload_excludes_drive_receipt_and_transient_files() -> None:
    assert not _is_recovery_payload_path("research/g_v4_1/results/recovery/drive_verification.json")
    assert not _is_recovery_payload_path("research/g_v4_1/results/state.tmp")
    assert not _is_recovery_payload_path("research/g_v4_1/recovery.zip")
    assert _is_recovery_payload_path("research/g_v4_1/results/progress.json")
