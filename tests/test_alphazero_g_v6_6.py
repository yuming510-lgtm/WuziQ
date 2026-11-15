from __future__ import annotations

import json

import pytest

from src.research.alphazero_g.v6_6_evaluation import (
    BASELINE_OPPONENTS,
    build_match_plan,
    finalize_v6_5_progress,
)


def _openings(count: int = 40):
    rows = []
    for index in range(count):
        rows.append({
            "opening_group_id": f"v6-6-test-{index:03d}",
            "seed": 660606 + index,
            "moves": [
                {"player": "B", "row": 4, "col": 4},
                {"player": "W", "row": 3, "col": 4},
                {"player": "B", "row": 5, "col": 5},
                {"player": "W", "row": 3, "col": 3},
                {"player": "B", "row": 4, "col": 5},
                {"player": "W", "row": 5, "col": 3},
            ],
        })
    return rows


def test_v66_schedule_has_exact_registered_counts_and_color_pairs():
    plan = build_match_plan(_openings())

    assert len(plan) == 504
    assert len({row["game_id"] for row in plan}) == 504
    assert {row["candidate_color"] for row in plan} == {"B", "W"}
    for opponent in BASELINE_OPPONENTS[:-1]:
        for candidate in ("G5000", "G8000"):
            matches = [row for row in plan if row["candidate"] == candidate and row["opponent"] == opponent]
            assert len(matches) == 40
            assert len({row["opening_group_id"] for row in matches}) == 20
    for candidate in ("G5000", "G8000"):
        matches = [row for row in plan if row["candidate"] == candidate and row["opponent"] == "C@5s"]
        assert len(matches) == 12
        assert len({row["opening_group_id"] for row in matches}) == 6
    direct = [row for row in plan if row["series"] == "G8000_vs_G5000"]
    assert len(direct) == 80
    assert len({row["opening_group_id"] for row in direct}) == 40


def test_v66_uses_identical_opening_color_seed_for_g5000_and_g8000_against_same_opponent():
    plan = build_match_plan(_openings())
    left = {
        (row["opponent"], row["opening_group_id"], row["candidate_color"]): row["seed"]
        for row in plan if row["candidate"] == "G5000" and row["opponent"] in BASELINE_OPPONENTS
    }
    right = {
        (row["opponent"], row["opening_group_id"], row["candidate_color"]): row["seed"]
        for row in plan if row["candidate"] == "G8000" and row["opponent"] in BASELINE_OPPONENTS
    }
    assert left == right


def test_v66_marks_v65_training_complete_without_changing_checkpoint_identity(tmp_path):
    progress = tmp_path / "training_progress.json"
    progress.write_text(json.dumps({
        "completed": False,
        "phase": "self_play_training",
        "games_completed": 8000,
        "environment_states": 139884,
        "optimizer_updates": 7999,
        "checkpoint_sha256": "frozen-state",
    }), encoding="utf-8")

    result = finalize_v6_5_progress(progress, expected_games=8000)

    updated = json.loads(progress.read_text(encoding="utf-8"))
    assert result["completed"] is True
    assert updated["completed"] is True
    assert updated["phase"] == "training_complete"
    assert updated["checkpoint_sha256"] == "frozen-state"


def test_v66_refuses_to_mark_an_incomplete_v65_run_complete(tmp_path):
    progress = tmp_path / "training_progress.json"
    progress.write_text(json.dumps({"completed": False, "games_completed": 7999}), encoding="utf-8")

    with pytest.raises(ValueError, match="not at G8000"):
        finalize_v6_5_progress(progress, expected_games=8000)

