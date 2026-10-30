from __future__ import annotations

import importlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def _api():
    name = "src.research.imitation_f.snapshot"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail("F V1 recovery snapshot API is not implemented")
    return importlib.import_module(name)


def test_snapshot_restores_exact_files_into_an_empty_directory(tmp_path: Path) -> None:
    api = _api()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"seed":731701}\n')
    (source / "games.jsonl").write_text('{"game_id":"g0"}\n')
    archive = tmp_path / "snapshot.zip"
    created = api.create_snapshot(source, archive, paths=["config.json", "games.jsonl"])
    restore = tmp_path / "empty-restore"
    restored = api.restore_snapshot(archive, restore)
    assert created["sha256"] == restored["sha256"]
    assert (restore / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert (restore / "games.jsonl").read_bytes() == (source / "games.jsonl").read_bytes()


def test_snapshot_rejects_tampering_and_nonempty_restore_destination(tmp_path: Path) -> None:
    api = _api()
    source = tmp_path / "source"
    source.mkdir()
    (source / "progress.json").write_text('{"done":1}\n')
    archive = tmp_path / "snapshot.zip"
    api.create_snapshot(source, archive, paths=["progress.json"])
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep").write_text("x")
    with pytest.raises(ValueError, match="empty"):
        api.restore_snapshot(archive, nonempty)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "r") as original, zipfile.ZipFile(tampered, "w") as replacement:
        for name in original.namelist():
            content = original.read(name)
            if name == "progress.json":
                content = b'{"done":2}\n'
            replacement.writestr(name, content)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="SHA|hash|integrity"):
        api.restore_snapshot(tampered, empty)
