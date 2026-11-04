from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_g_v6_1_snapshot import create_snapshot
from scripts.restore_g_v6_1_snapshot import restore_snapshot


def test_snapshot_restores_only_v61_artifacts_and_validates_hashes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "research/g_v6_1/config").mkdir(parents=True)
    (root / "research/g_v6_1/config/registration.json").write_text('{"id":"v61"}')
    (root / "research/g_v6/config").mkdir(parents=True)
    (root / "research/g_v6/config/keep.json").write_text('{"id":"v6"}')
    # The snapshot function records git metadata; use the live repo only for this fixture.
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@local",
                    "commit", "--allow-empty", "-qm", "fixture"], cwd=root, check=True)
    archive = tmp_path / "recovery.zip"
    result = create_snapshot(root, archive)
    assert result["experiment_id"] == "wuziq-alphazero-g-v6-1-checkpoint-tournament"
    target = tmp_path / "empty"
    target.mkdir()
    restored = restore_snapshot(archive, target)
    assert restored["restore_succeeded"] is True
    assert (target / "research/g_v6_1/config/registration.json").is_file()
    assert not (target / "research/g_v6/config/keep.json").exists()
    with pytest.raises(ValueError, match="empty directory"):
        restore_snapshot(archive, target)
