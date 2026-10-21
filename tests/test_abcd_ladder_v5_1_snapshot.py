from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from src.research.abcd_ladder_v5_1_snapshot import create_snapshot, restore_snapshot
from src.research.abcd_ladder_v5_1_runner import run_experiment
from src.research.abcd_ladder_v5_1 import load_config
from tests.test_abcd_ladder_v5_1_runner import _complete_legal_record


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "abcd_ladder_v5_1.json"


def _one_game_partial_results(path: Path) -> dict:
    config = load_config(CONFIG)
    runner = lambda spec, _engine, _commit: _complete_legal_record(spec, config)
    return run_experiment(config, path, reuse_v5=False, max_games=1,
                          implementation_commit="snapshot-test", game_runner=runner)


def test_snapshot_round_trip_verifies_hashes_and_restores_into_empty_directory(tmp_path):
    results = tmp_path / "results"
    _one_game_partial_results(results)
    archive = tmp_path / "snapshot.zip"
    manifest = create_snapshot(results, archive)
    restored = tmp_path / "empty-restore"
    restored.mkdir()

    ids = restore_snapshot(archive, restored)

    assert len(ids) == 1
    assert manifest["files"]
    assert {path.name for path in restored.iterdir()} == {
        "config.json", "schedule.json", "reuse_manifest.json", "progress.json", "9x9_games.jsonl",
    }
    assert json.loads((restored / "progress.json").read_text())["pending_group_id"]


def test_snapshot_rejects_tampered_archive_hash(tmp_path):
    results = tmp_path / "results"
    _one_game_partial_results(results)
    archive = tmp_path / "snapshot.zip"
    create_snapshot(results, archive)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "progress.json":
                data += b" "
            target.writestr(name, data)

    with pytest.raises(ValueError, match="checksum"):
        restore_snapshot(tampered, tmp_path / "restore")


def test_snapshot_restore_requires_an_empty_directory(tmp_path):
    results = tmp_path / "results"
    _one_game_partial_results(results)
    archive = tmp_path / "snapshot.zip"
    create_snapshot(results, archive)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("do not overwrite")

    with pytest.raises(ValueError, match="empty"):
        restore_snapshot(archive, nonempty)


def test_restored_partial_pair_resumes_exactly_once(tmp_path):
    results = tmp_path / "results"
    config = load_config(CONFIG)
    runner = lambda spec, _engine, _commit: _complete_legal_record(spec, config)
    first = run_experiment(config, results, reuse_v5=False, max_games=1,
                           implementation_commit="snapshot-test", game_runner=runner)
    archive = tmp_path / "snapshot.zip"
    create_snapshot(results, archive)
    restored = tmp_path / "recovered"
    restored.mkdir()
    restore_snapshot(archive, restored)

    resumed = run_experiment(config, restored, reuse_v5=False, max_games=1,
                              implementation_commit="snapshot-test", game_runner=runner)

    assert first["completed_games"] == 1
    assert resumed["completed_games"] == 2
    assert resumed["completed_groups"] == 1
    ids = [json.loads(line)["game_id"] for line in (restored / "9x9_games.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 2
