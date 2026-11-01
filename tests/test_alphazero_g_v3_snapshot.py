from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from src.research.alphazero_g.v3_snapshot import (
    create_v3_snapshot,
    next_v3_snapshot_threshold,
    restore_v3_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_required_tree(target: Path) -> Path:
    required = (
        "research/g_v3/config/experiment.json",
        "research/g_v3/config/openings.json",
        "research/g_v3/results/preflight/preflight.json",
        "research/g_v3/results/bn_probe/probe.json",
        "research/g_v3/results/bn_probe/selfplay.jsonl",
        "research/g_v3/results/bn_probe/evaluation.jsonl",
        "research/g_v3/results/recovery/drive_status.json",
        "research/f_v2/checkpoints/best_f2.pt",
    )
    for relative in required:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return target


def test_snapshot_hashes_and_restores_to_empty_directory(tmp_path):
    root = _copy_required_tree(tmp_path / "source")
    archive = tmp_path / "g_v3_recovery.zip"
    created = create_v3_snapshot(root, archive)
    destination = tmp_path / "empty_restore"
    restored = restore_v3_snapshot(archive, destination)

    assert created["sha256"] == restored["archive_sha256"]
    assert restored["empty_restore"] is True
    assert restored["branch"] == "research/alphazero-g-v3"
    assert (destination / "research/g_v3/config/experiment.json").read_bytes() == (
        root / "research/g_v3/config/experiment.json"
    ).read_bytes()
    assert (destination / "source/src/research/alphazero_g/v3_runner.py").is_file()
    assert (destination / "research/f_v2/checkpoints/best_f2.pt").is_file()


def test_snapshot_rejects_tampering_and_nonempty_restore_target(tmp_path):
    root = _copy_required_tree(tmp_path / "source")
    archive = tmp_path / "g_v3_recovery.zip"
    create_v3_snapshot(root, archive)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(tampered, "w") as output:
        for name in source.namelist():
            contents = source.read(name)
            if name == "research/g_v3/config/experiment.json":
                contents += b" "
            output.writestr(name, contents)
    with pytest.raises(ValueError, match="SHA-256"):
        restore_v3_snapshot(tampered, tmp_path / "bad")
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep").write_text("x")
    with pytest.raises(ValueError, match="empty destination"):
        restore_v3_snapshot(archive, nonempty)


def test_snapshot_thresholds_accumulate_across_restarts():
    assert [next_v3_snapshot_threshold(value) for value in (0, 3599, 3600, 10800, 25200)] == [
        3600, 3600, 10800, 25200, 46800,
    ]
    assert next_v3_snapshot_threshold(140400) == 234000
    with pytest.raises(ValueError, match="finite and nonnegative"):
        next_v3_snapshot_threshold(-1)
