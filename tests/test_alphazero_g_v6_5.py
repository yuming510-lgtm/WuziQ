from __future__ import annotations

import json

import pytest

from src.research.alphazero_g import v6_5_audit, v6_5_training, v6_5_snapshot


def _winning_record():
    moves = []
    for col in range(5):
        moves.append({"player": "B", "row": 0, "col": col, "opening": False, "configured_playouts": 400})
        if col < 4:
            moves.append({"player": "W", "row": 1, "col": col, "opening": False, "configured_playouts": 400})
    return {
        "source": "self_play", "game_id": "reference400_empty-game-005000",
        "board_size": 9, "n_in_row": 5, "simulations_configured_per_move": 400,
        "opening": [], "moves": moves, "move_count": len(moves),
        "training_state_count": len(moves), "winner_color": "B",
        "outcome": "B_win", "termination": "five_in_row",
    }


def test_v65_starts_at_frozen_g5000_and_targets_g8000_without_changing_search():
    assert (v6_5_training.BASE_GAMES, v6_5_training.BASE_STATES, v6_5_training.BASE_UPDATES) == (5000, 87999, 4999)
    assert v6_5_training.TARGET_GAMES == 8000
    assert v6_5_training.SELFPLAY_SIMULATIONS == 400
    assert v6_5_training.MILESTONES == (6000, 7000, 8000)
    assert v6_5_training.SOURCE_STATE_SHA256 == "0fb9886068a014d000227ae7b275ff9043322a678d7f259e9730d44ac95398d2"


def test_independent_replay_accepts_a_legal_five_in_row_terminal():
    result = v6_5_audit.audit_game_record(_winning_record())
    assert result["winner_color"] == "B"
    assert result["move_count"] == 9
    assert result["root_visits_values"] == [-1]


def test_independent_replay_rejects_moves_after_win_and_wrong_result():
    record = _winning_record()
    record["moves"].append({"player": "W", "row": 2, "col": 0, "opening": False, "configured_playouts": 400})
    record["move_count"] += 1
    record["training_state_count"] += 1
    with pytest.raises(ValueError, match="after a five-in-row win"):
        v6_5_audit.audit_game_record(record)
    record = _winning_record()
    record["winner_color"] = "W"
    with pytest.raises(ValueError, match="winner/outcome"):
        v6_5_audit.audit_game_record(record)


def test_drive_snapshot_registration_requires_download_and_empty_restore(tmp_path):
    artifact = tmp_path / "research/g_v6_5"
    (artifact / "results").mkdir(parents=True)
    (artifact / "results/drive_backup_status.json").write_text(json.dumps({
        "experiment_id": "wuziq-alphazero-g-v6-5-selfplay-scale",
        "folder_id": "drive-folder", "verified_snapshots": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unverified"):
        v6_5_snapshot.register_verified_snapshot(tmp_path, {
            "name": "snapshot.zip", "sha256": "a", "downloaded_sha256": "b",
            "empty_directory_restore": False, "games_completed": 5000,
            "training_state_sha256": "state",
        })
    status = v6_5_snapshot.register_verified_snapshot(tmp_path, {
        "name": "snapshot.zip", "sha256": "a", "downloaded_sha256": "a",
        "empty_directory_restore": True, "games_completed": 5000,
        "training_state_sha256": "state",
    })
    assert status["latest_games"] == 5000
    assert len(status["verified_snapshots"]) == 1


def test_snapshot_restore_requires_empty_destination(tmp_path):
    archive = tmp_path / "missing.zip"
    archive.write_bytes(b"not-a-zip")
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "existing").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        v6_5_snapshot.restore_snapshot(archive, destination)
