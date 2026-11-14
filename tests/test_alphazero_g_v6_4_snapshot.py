from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.research.alphazero_g.v6_4_snapshot import create_snapshot_archive, restore_snapshot_archive


def test_snapshot_can_be_downloaded_and_restored_into_an_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "research/g_v6_4/config").mkdir(parents=True)
    (root / "research/g_v6_4/results/matches").mkdir(parents=True)
    (root / "research/g_v6_4/config/opening_schedule.json").write_text('{"seed":640614}\n')
    (root / "research/g_v6_4/results/matches/game.json").write_text('{"winner":"B"}\n')
    archive = tmp_path / "snapshot.zip"

    metadata = create_snapshot_archive(root, archive)
    destination = tmp_path / "empty-restore"
    restored = restore_snapshot_archive(archive, destination)

    assert metadata["sha256"] == restored["sha256"]
    assert restored["empty_directory_restore"] is True
    assert (destination / "research/g_v6_4/config/opening_schedule.json").is_file()
    assert json.loads((destination / "research/g_v6_4/results/matches/game.json").read_text()) == {"winner": "B"}


def test_snapshot_restore_rejects_nonempty_destination(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "research/g_v6_4/config").mkdir(parents=True)
    (root / "research/g_v6_4/config/opening_schedule.json").write_text('{}')
    archive = tmp_path / "snapshot.zip"
    create_snapshot_archive(root, archive)
    destination = tmp_path / "nonempty"
    destination.mkdir()
    (destination / "existing").write_text("keep")

    try:
        restore_snapshot_archive(archive, destination)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("restore must not overwrite a nonempty destination")


def test_snapshot_archive_contains_manifest_and_only_recovery_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "research/g_v6_4/config").mkdir(parents=True)
    (root / "research/g_v6_4/results").mkdir(parents=True)
    (root / "research/g_v6_4/config/opening_schedule.json").write_text('{}')
    (root / "research/g_v6_4/unrelated.log").write_text("ignore")
    archive = tmp_path / "snapshot.zip"

    create_snapshot_archive(root, archive)

    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
    assert "V6_4_RECOVERY_MANIFEST.json" in names
    assert "research/g_v6_4/config/opening_schedule.json" in names
    assert "research/g_v6_4/unrelated.log" not in names
