from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from src.research.alphazero_g.snapshot import create_snapshot, next_snapshot_threshold, restore_snapshot


def test_snapshot_round_trips_into_empty_directory_with_hashes(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "research/g_v1/results").mkdir(parents=True)
    (root / "research/g_v1/results/progress.json").write_text('{"step":12}\n')
    (root / "src/research/alphazero_g").mkdir(parents=True)
    (root / "src/research/alphazero_g/runner.py").write_text("resume = True\n")
    archive = tmp_path / "snapshot.zip"
    digest = create_snapshot(
        root, archive, commit="deadbeef", active_seconds=3601,
        manifest={"include": ["research/g_v1", "src/research/alphazero_g"]},
    )
    target = tmp_path / "restored"
    target.mkdir()

    metadata = restore_snapshot(archive, target)

    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    assert metadata["commit"] == "deadbeef"
    assert metadata["active_seconds"] == 3601
    assert (target / "research/g_v1/results/progress.json").read_text() == '{"step":12}\n'
    assert (target / "src/research/alphazero_g/runner.py").read_text() == "resume = True\n"


def test_snapshot_rejects_hash_tampering_and_nonempty_restore_target(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "research/g_v1").mkdir(parents=True)
    (root / "research/g_v1/state.txt").write_text("saved")
    archive = tmp_path / "snapshot.zip"
    create_snapshot(root, archive, "abc", 1, {"include": ["research/g_v1"]})
    target = tmp_path / "empty"
    target.mkdir()
    (target / "keep").write_text("x")
    with pytest.raises(ValueError, match="empty"):
        restore_snapshot(archive, target)

    entries = []
    with zipfile.ZipFile(archive) as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    rewritten = tmp_path / "tampered.zip"
    with zipfile.ZipFile(rewritten, "w") as output:
        for name, data in entries:
            if name == "research/g_v1/state.txt":
                data = b"alter"
            output.writestr(name, data)
    clean = tmp_path / "clean"
    clean.mkdir()
    with pytest.raises(ValueError, match="SHA-256"):
        restore_snapshot(rewritten, clean)


def test_snapshot_rejects_traversal_and_duplicate_archive_paths(tmp_path) -> None:
    def write_archive(path, names):
        manifest = {
            "format_version": 1,
            "commit": "abc",
            "active_seconds": 1,
            "metadata": {},
            "files": [
                {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in names
            ],
        }
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, data in names:
                archive.writestr(name, data)

    traversal = tmp_path / "traversal.zip"
    write_archive(traversal, [("../escape", b"x")])
    target = tmp_path / "target1"
    target.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        restore_snapshot(traversal, target)

    duplicate = tmp_path / "duplicate.zip"
    write_archive(duplicate, [("safe.txt", b"x"), ("safe.txt", b"y")])
    target2 = tmp_path / "target2"
    target2.mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        restore_snapshot(duplicate, target2)


def test_cumulative_snapshot_thresholds_follow_requested_intervals() -> None:
    expected = [3600, 10800, 25200, 46800, 82800, 140400, 234000]
    for current, following in zip(expected, expected[1:], strict=False):
        assert next_snapshot_threshold(current - 1) == current
        assert next_snapshot_threshold(current) == following
    assert next_snapshot_threshold(140400) == 234000
