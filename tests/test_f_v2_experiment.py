from __future__ import annotations

import json
from pathlib import Path

import pytest


def _api():
    from src.research.imitation_f.v2_experiment import analyze_f1_a_losses, build_experiment_plan
    return analyze_f1_a_losses, build_experiment_plan


def test_experiment_plan_builds_disjoint_legal_train_dev_and_20_group_blind_schedules() -> None:
    _, build_experiment_plan = _api()
    plan = build_experiment_plan(
        seed=732101,
        train_count=3,
        development_count=2,
        blind_count=20,
        excluded_groups=("opening-already-used-a", "opening-already-used-b"),
    )

    groups = {
        name: [row["opening_group_id"] for row in plan[name]]
        for name in ("training_openings", "development_openings", "blind_openings")
    }
    assert [len(groups[name]) for name in groups] == [3, 2, 20]
    assert len(set(groups["training_openings"] + groups["development_openings"] + groups["blind_openings"])) == 25
    assert not set().union(*map(set, groups.values())) & {"opening-already-used-a", "opening-already-used-b"}
    assert plan["schedule_sha256"]
    for name in groups:
        for spec in plan[name]:
            assert len(spec["opening"]) == 4
            assert [(move["player"], move["row"], move["col"]) for move in spec["opening"]][0] == ("B", 4, 4)


def test_f1_a_loss_analysis_finds_the_missed_win_and_two_missed_blocks() -> None:
    analyze_f1_a_losses, _ = _api()
    records = [
        json.loads(line)
        for line in Path("research/f_v1/results/matches.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    losses = [row for row in records if row["pair"] == ["F", "A"] and row["winner_model"] == "A"]

    result = analyze_f1_a_losses(losses)

    assert result["games"] == 2
    assert result["missed_immediate_wins"] == 1
    assert result["missed_single_point_blocks"] == 2
    assert {item["game_id"] for item in result["examples"]} == {row["game_id"] for row in losses}


def test_experiment_plan_rejects_nonpositive_schedule_counts() -> None:
    _, build_experiment_plan = _api()
    with pytest.raises(ValueError, match="counts must be positive"):
        build_experiment_plan(732101, 0, 8, 20, ())


def test_preflight_estimate_uses_observed_game_annotation_and_update_rates() -> None:
    from src.research.imitation_f.v2_experiment import estimate_run_seconds

    result = estimate_run_seconds(
        train_games=32,
        mean_game_seconds=0.1,
        mean_f_turns_per_game=10,
        mean_annotation_seconds=0.1,
        train_rows=12000,
        seconds_per_update=0.02,
        expected_epochs=45,
        maximum_epochs=60,
        maximum_training_runs=2,
        evaluation_games=192,
        mean_evaluation_game_seconds=0.1,
        overhead_seconds=20,
    )

    assert result["data_seconds"] == pytest.approx(35.2)
    assert result["central_seconds"] == pytest.approx(243.6)
    assert result["conservative_seconds"] > result["central_seconds"]


def test_network_loss_analysis_finds_missed_exact_single_block() -> None:
    from src.research.imitation_f.v2_experiment import analyze_network_losses

    moves = []
    sequence = [
        ("B", "A", 0, 0), ("W", "F2", 8, 8),
        ("B", "A", 0, 1), ("W", "F2", 8, 7),
        ("B", "A", 0, 2), ("W", "F2", 7, 8),
        ("B", "A", 0, 3), ("W", "F2", 7, 7),
        ("B", "A", 0, 4),
    ]
    for ply, (player, model, row, col) in enumerate(sequence):
        moves.append({"ply": ply, "player": player, "model": model, "row": row, "col": col,
                      "opening": False, "elapsed_seconds": 0.0})
    record = {
        "game_id": "f-v2-test-lost-block", "group_id": "test-opening", "board_size": 9,
        "pair": ["F2", "A"], "challenger_model": "F2", "black_model": "A", "white_model": "F2",
        "winner_model": "A", "winner_color": "B", "outcome": "b_win", "termination": "five_in_row",
        "moves": moves, "move_count": len(moves),
    }

    result = analyze_network_losses([record], "F2")
    assert result["games"] == 1
    assert result["missed_immediate_wins"] == 0
    assert result["missed_single_point_blocks"] == 1
    assert result["examples"][0]["mistakes"][0]["required_block"] == [0, 4]
