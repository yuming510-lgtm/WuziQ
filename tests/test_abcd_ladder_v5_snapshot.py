import json
from pathlib import Path

import pytest

from src.research.abcd_ladder_v5 import load_config
from src.research.abcd_ladder_v5_snapshot import create_snapshot, restore_snapshot


CONFIG = Path(__file__).parents[1] / "configs" / "abcd_ladder_v5.json"


def _initial_results(path: Path) -> None:
    config = load_config(CONFIG)
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "9x9_games.jsonl").write_text("", encoding="utf-8")
    (path / "progress.json").write_text(json.dumps({
        "experiment_id": config["experiment_id"],
        "implementation_commit": "test-commit",
        "completed_ids": [],
        "completed_games": 0,
    }), encoding="utf-8")


def test_initial_snapshot_restores_into_empty_directory_and_hashes_files(tmp_path):
    results = tmp_path / "results"
    _initial_results(results)
    snapshot = tmp_path / "snapshot.json"
    created = create_snapshot(results, snapshot, "test-commit")
    restored = tmp_path / "empty-restore"

    result = restore_snapshot(snapshot, restored)

    assert created["completed_games"] == 0
    assert result["valid"] is True
    assert result["games"] == 0
    for filename in ("config.json", "9x9_games.jsonl", "progress.json"):
        assert (restored / filename).read_bytes() == (results / filename).read_bytes()


def test_snapshot_rejects_tampering_before_restore(tmp_path):
    results = tmp_path / "results"
    _initial_results(results)
    snapshot = tmp_path / "snapshot.json"
    create_snapshot(results, snapshot, "test-commit")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["files"]["progress.json"] += " "
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_snapshot(snapshot, tmp_path / "empty-restore")
