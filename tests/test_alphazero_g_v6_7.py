from src.research.alphazero_g.v6_7_training import (
    MILESTONES,
    TARGET_GAMES,
    batch_target_is_stale,
    exclusive_training_lock,
    next_batch_target,
    snapshot_due,
)
from src.research.alphazero_g.v6_7_snapshot import register_snapshot, restore_snapshot
import json
import zipfile
import pytest
from src.research.alphazero_g import v6_7_training as training


def test_v67_training_milestones_and_batch_boundaries():
    assert MILESTONES == (10_000, 12_000, 16_000, 20_000)
    assert TARGET_GAMES == 20_000
    assert next_batch_target(8_000) == 8_250
    assert next_batch_target(9_900) == 10_000
    assert next_batch_target(19_900) == 20_000
    assert next_batch_target(9_512, last_snapshot_games=9_500) == 9_750


def test_v67_stale_batch_target_requests_a_snapshot_instead_of_running_noop():
    assert batch_target_is_stale(current_games=11_250, target_games=11_250)
    assert not batch_target_is_stale(current_games=11_249, target_games=11_250)


def test_v67_batch_with_stale_progress_requests_recovery_without_training(tmp_path, monkeypatch):
    artifact = tmp_path / "research/g_v6_7"
    (artifact / "config").mkdir(parents=True)
    (artifact / "results").mkdir(parents=True)
    (artifact / "config/reference_training_config.json").write_text("{}", encoding="utf-8")
    (artifact / "results/baseline.json").write_text(
        json.dumps({"cumulative_training_cpu_seconds": 100.0}), encoding="utf-8")
    progress_path = artifact / "results/progress.json"
    progress_path.write_text(json.dumps({
        "games_completed": 11_250,
        "last_snapshot_games": 11_000,
        "last_snapshot_total_cpu_seconds": 0.0,
        "snapshot_due": False,
    }), encoding="utf-8")
    validation = {
        "game_count": 11_250,
        "environment_states": 196_573,
        "optimizer_updates": 11_249,
        "replay_rows": 10_000,
        "sha256": "verified-state",
        "cumulative_training_cpu_seconds": 6_000.0,
    }
    monkeypatch.setattr(training, "artifact_path", lambda root: artifact)
    monkeypatch.setattr(training, "progress_path", lambda root: progress_path)
    monkeypatch.setattr(training, "state_path", lambda root: artifact / "checkpoint.pt")
    monkeypatch.setattr(training, "validate_training_state", lambda *args, **kwargs: validation)
    monkeypatch.setattr(training, "_load_accounting", lambda root: {
        "evaluation_cpu_seconds": 0.0,
        "audit_cpu_seconds": 0.0,
        "snapshot_packaging_cpu_seconds": 0.0,
    })
    monkeypatch.setattr(training, "train_reference_arm",
                        lambda *args, **kwargs: pytest.fail("no-op resume must not enter training"))

    result = training.run_training_batch(tmp_path)
    saved_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert result["stopped"] == "stale_batch_target"
    assert result["snapshot_due"] is True
    assert saved_progress["snapshot_due"] is True
    assert saved_progress["games_completed"] == 11_250


def test_v67_batch_lock_rejects_a_second_writer(tmp_path):
    lock_path = tmp_path / "training.lock"
    with exclusive_training_lock(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with exclusive_training_lock(lock_path):
                pass


def test_v67_unregistrable_snapshot_does_not_pollute_drive_status(tmp_path):
    results = tmp_path / "research/g_v6_7/results"
    results.mkdir(parents=True)
    status_path = results / "drive_backup_status.json"
    original = {"experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
                "folder_id": "test-folder", "snapshots": []}
    status_path.write_text(json.dumps(original), encoding="utf-8")
    (results / "progress.json").write_text(json.dumps({"games_completed": 11_250}), encoding="utf-8")
    row = {"snapshot_id": "bad-row", "games_completed": 11_250,
           "folder_id": "test-folder", "sha256": "abc", "downloaded_sha256": "abc",
           "empty_directory_restore": True}
    with pytest.raises(ValueError, match="total_new_cpu_seconds"):
        register_snapshot(tmp_path, row)
    assert json.loads(status_path.read_text(encoding="utf-8")) == original


def test_v67_snapshot_is_due_on_first_of_250_games_or_three_cpu_hours():
    assert not snapshot_due(8_249, 8_000, 1_000, 0)
    assert snapshot_due(8_250, 8_000, 1_000, 0)
    assert snapshot_due(8_100, 8_000, 10_801, 0)


def test_v67_restore_rejects_path_traversal_before_extracting(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("V6_7_RECOVERY_MANIFEST.json", json.dumps({
            "experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
            "file_hashes": {},
        }))
        handle.writestr("../escape.txt", "no")
    destination = tmp_path / "empty"
    try:
        restore_snapshot(archive, destination)
    except ValueError as exc:
        assert "unsafe archive" in str(exc)
    else:
        raise AssertionError("unsafe member path was accepted")
    assert not (tmp_path / "escape.txt").exists()
