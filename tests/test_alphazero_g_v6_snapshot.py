from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _snapshot_module():
    name = "scripts.create_g_v6_snapshot"
    if importlib.util.find_spec(name) is None:
        pytest.fail("V6 recovery snapshot writer is not implemented")
    from scripts import create_g_v6_snapshot

    return create_g_v6_snapshot


def test_v6_snapshot_restores_exact_members_only_into_empty_directory(tmp_path: Path) -> None:
    writer = _snapshot_module()
    root = tmp_path / "repo"
    (root / "research/g_v6/config").mkdir(parents=True)
    (root / "research/g_v6/config/registration.json").write_text('{"id":"v6"}', encoding="utf-8")
    (root / "research/g_v6/checkpoints").mkdir(parents=True)
    (root / "research/g_v6/checkpoints/training_state.pt").write_bytes(b"checkpoint-payload")
    archive = tmp_path / "v6.zip"

    manifest = writer.create_snapshot(root, archive, code_commit="test-commit")

    assert manifest["experiment_id"] == "wuziq-alphazero-g-v6-9x9-400-equalized-training"
    assert manifest["code_commit"] == "test-commit"
    assert manifest["file_count"] == 2

    try:
        from scripts.restore_g_v6_snapshot import restore_snapshot
    except ImportError as exc:
        pytest.fail(f"V6 recovery snapshot restore is not implemented: {exc}")
    target = tmp_path / "empty"
    target.mkdir()
    restored = restore_snapshot(archive, target)
    assert restored["restore_succeeded"] is True
    assert restored["files_verified"] == 2
    assert (target / "research/g_v6/config/registration.json").read_text(encoding="utf-8") == '{"id":"v6"}'
    assert (target / "research/g_v6/checkpoints/training_state.pt").read_bytes() == b"checkpoint-payload"


def test_v6_snapshot_refuses_nonempty_restore_target(tmp_path: Path) -> None:
    writer = _snapshot_module()
    root = tmp_path / "repo"
    (root / "research/g_v6").mkdir(parents=True)
    (root / "research/g_v6/a").write_text("x", encoding="utf-8")
    archive = tmp_path / "v6.zip"
    writer.create_snapshot(root, archive, code_commit="test-commit")
    try:
        from scripts.restore_g_v6_snapshot import restore_snapshot
    except ImportError as exc:
        pytest.fail(f"V6 recovery snapshot restore is not implemented: {exc}")
    target = tmp_path / "nonempty"
    target.mkdir()
    (target / "existing").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        restore_snapshot(archive, target)
