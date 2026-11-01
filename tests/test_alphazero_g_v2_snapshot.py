from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.research.alphazero_g.v2_snapshot import (
    create_v2_snapshot,
    next_v2_snapshot_threshold,
    restore_v2_snapshot,
)


def _prepare_root(tmp_path):
    for relative in (
        "research/g_v2/config/experiment.json",
        "research/g_v2/config/openings.json",
        "research/f_v2/checkpoints/best_f2.pt",
    ):
        source = Path(relative)
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return tmp_path


def test_snapshot_restores_registered_state_to_empty_directory_with_matching_hashes(tmp_path):
    root = _prepare_root(tmp_path / "root")
    archive = tmp_path / "g_v2.zip"
    created = create_v2_snapshot(root, archive)
    target = tmp_path / "restored"
    restored = restore_v2_snapshot(archive, target)

    assert created["sha256"] == restored["archive_sha256"]
    assert restored["empty_restore"] is True
    assert restored["commit"] == created["commit"]
    assert (target / "research/g_v2/config/experiment.json").read_bytes() == (
        root / "research/g_v2/config/experiment.json"
    ).read_bytes()
    assert (target / "research/f_v2/checkpoints/best_f2.pt").is_file()


def test_snapshot_rejects_tampered_member_and_nonempty_restore_target(tmp_path):
    root = _prepare_root(tmp_path / "root")
    archive = tmp_path / "g_v2.zip"
    create_v2_snapshot(root, archive)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(tampered, "w") as output:
        for name in source.namelist():
            value = source.read(name)
            if name == "research/g_v2/config/experiment.json":
                value += b" "
            output.writestr(name, value)
    with pytest.raises(ValueError, match="SHA-256"):
        restore_v2_snapshot(tampered, tmp_path / "bad-restore")

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="empty destination"):
        restore_v2_snapshot(archive, nonempty)


def test_snapshot_rejects_unsupported_archive_paths(tmp_path):
    root = _prepare_root(tmp_path / "root")
    archive = tmp_path / "g_v2.zip"
    create_v2_snapshot(root, archive)
    bad = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(bad, "w") as output:
        for name in source.namelist():
            output.writestr(name, source.read(name))
        output.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="unsupported or unsafe"):
        restore_v2_snapshot(bad, tmp_path / "traversal-restore")


def test_snapshot_thresholds_use_cumulative_effective_runtime():
    assert [next_v2_snapshot_threshold(value) for value in (0, 3599, 3600, 10800, 25200)] == [
        3600, 3600, 10800, 25200, 46800,
    ]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        next_v2_snapshot_threshold(-1)
