from src.research.alphazero_g.v6_7_training import (
    MILESTONES,
    TARGET_GAMES,
    next_batch_target,
    snapshot_due,
)
from src.research.alphazero_g.v6_7_snapshot import restore_snapshot
import json
import zipfile


def test_v67_training_milestones_and_batch_boundaries():
    assert MILESTONES == (10_000, 12_000, 16_000, 20_000)
    assert TARGET_GAMES == 20_000
    assert next_batch_target(8_000) == 8_250
    assert next_batch_target(9_900) == 10_000
    assert next_batch_target(19_900) == 20_000
    assert next_batch_target(9_512, last_snapshot_games=9_500) == 9_750


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
