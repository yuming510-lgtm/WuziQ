from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest


def _make_snapshot(path: Path, payload: bytes = b"recovery-state") -> str:
    name = "research/f_v2/checkpoint.bin"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": 1, "code_commit": "test-commit",
        "experiment_schedule_sha256": "schedule", "model_checkpoint_sha256": "model",
        "files": {name: {"sha256": digest, "bytes": len(payload)}},
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("recovery_manifest.json", json.dumps(manifest))
        archive.writestr(name, payload)
    return digest


def test_snapshot_restores_to_empty_directory_and_checks_sha256(tmp_path) -> None:
    from src.research.imitation_f.v2_snapshot import restore_snapshot

    archive = tmp_path / "snapshot.zip"
    digest = _make_snapshot(archive)
    restored = tmp_path / "empty"
    result = restore_snapshot(archive, restored)

    file = restored / "research/f_v2/checkpoint.bin"
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    assert result["status"] == "passed"
    assert result["files_verified"] == 1
    assert result["code_commit"] == "test-commit"


def test_snapshot_restore_rejects_nonempty_destination_and_tampered_archive(tmp_path) -> None:
    from src.research.imitation_f.v2_snapshot import restore_snapshot

    archive = tmp_path / "snapshot.zip"
    _make_snapshot(archive)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep").write_text("keep")
    with pytest.raises(ValueError, match="destination must be empty"):
        restore_snapshot(archive, nonempty)

    bad_archive = tmp_path / "bad.zip"
    _make_snapshot(archive)
    # Keep the original manifest and replace the payload once, so the failure
    # exercises the SHA-256 check instead of duplicate-path rejection.
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(bad_archive, "w") as target:
        for name in source.namelist():
            target.writestr(
                name,
                b"corruptd-state" if name == "research/f_v2/checkpoint.bin" else source.read(name),
            )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        restore_snapshot(bad_archive, tmp_path / "restore-bad")
