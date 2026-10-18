"""V3 recovery archives contain only verified restart-critical state."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _small_v3_config() -> dict:
    return {
        "experiment_id": "wuziq-mcts-high-compute-v3-snapshot-test",
        "code_commit": "a" * 40,
        "base_seed": 20261018,
        "opening_seeds": [101, 102, 103],
        "opening_plies": 4,
        "search": {"max_depth": 1, "time_limit_seconds": 0.05, "mcts_simulations": 1000000},
        "phases": [
            {
                "name": "d3-10s-vs-a",
                "track": "high-compute",
                "board_size": 9,
                "games_per_matchup": 6,
                "matchups": [["D3", "A"]],
                "budget_seconds_by_model": {"D3": 10.0, "A": 0.05},
            }
        ],
    }


def _module():
    return importlib.import_module("src.research.mcts_v3_snapshot")


def test_v3_snapshot_round_trip_restores_only_registered_run_state(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "results"
    root.mkdir()
    config = _small_v3_config()
    schedule = importlib.import_module("src.research.mcts_v2").build_schedule(config)
    identifier = schedule[0]["game_id"]
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "progress.json").write_text(
        json.dumps({"experiment_id": config["experiment_id"], "scheduled_games": 6, "games_completed": 1}),
        encoding="utf-8",
    )
    (root / "9x9_d3-10s-vs-a_games.jsonl").write_text(
        json.dumps({"game_id": identifier, "outcome": "draw"}) + "\n",
        encoding="utf-8",
    )
    (root / "temporary.log").write_text("not needed for recovery", encoding="utf-8")
    archive = tmp_path / "recovery.zip"

    manifest = module.create_v3_snapshot(root, archive)
    restored = tmp_path / "restored"
    recovered = module.restore_v3_snapshot(archive, restored)

    assert recovered["completed_game_ids"] == [identifier]
    assert recovered["code_commit"] == "a" * 40
    assert recovered["archive_sha256"] == manifest["archive_sha256"]
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {
            "config.json",
            "progress.json",
            "9x9_d3-10s-vs-a_games.jsonl",
            "manifest.json",
        }
    assert not (restored / "temporary.log").exists()


def test_v3_restore_rejects_archive_with_changed_game_bytes(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "results"
    root.mkdir()
    config = _small_v3_config()
    schedule = importlib.import_module("src.research.mcts_v2").build_schedule(config)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "progress.json").write_text(
        json.dumps({"experiment_id": config["experiment_id"], "scheduled_games": 6, "games_completed": 1}),
        encoding="utf-8",
    )
    (root / "9x9_d3-10s-vs-a_games.jsonl").write_text(
        json.dumps({"game_id": schedule[0]["game_id"]}) + "\n", encoding="utf-8"
    )
    archive = tmp_path / "recovery.zip"
    module.create_v3_snapshot(root, archive)

    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(changed, "w") as output:
        for name in source.namelist():
            data = source.read(name)
            if name.endswith("_games.jsonl"):
                data = data.replace(b"game_id", b"GAME_ID")
            output.writestr(name, data)

    with pytest.raises(ValueError, match="checksum"):
        module.restore_v3_snapshot(changed, tmp_path / "bad-restore")


def test_v3_snapshot_requires_a_recorded_code_commit(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "results"
    root.mkdir()
    config = _small_v3_config()
    del config["code_commit"]
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "progress.json").write_text(
        json.dumps({"experiment_id": config["experiment_id"], "scheduled_games": 6, "games_completed": 0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="code_commit"):
        module.create_v3_snapshot(root, tmp_path / "recovery.zip")
