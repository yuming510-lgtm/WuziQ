from __future__ import annotations

import time

import src.research.alphazero_g.v5_experiment as experiment
from src.research.alphazero_g.v5_experiment import six_by_six_gate_status


def _stage(node: int, score: float, lower: float) -> dict:
    return {
        "node": node,
        "match": {
            "score_rate": score,
            "paired_opening_bootstrap_ci95": [lower, 1.0],
        },
    }


def test_early_5k_and_10k_scores_do_not_satisfy_two_late_node_gate() -> None:
    settings = {
        "maximum_environment_states": 20_000,
        "continue_to_9x9_gate": {
            "candidate_min_score_rate_vs_frozen_random_initial": 0.6,
            "paired_ci95_lower_above": 0.5,
            "required_later_nodes": 2,
        }
    }
    passed, gate_nodes = six_by_six_gate_status(
        [_stage(5_000, 0.95, 0.85), _stage(10_000, 0.95, 0.85)],
        settings,
    )
    assert not passed
    assert gate_nodes == [10_000]


def test_two_registered_late_nodes_satisfy_gate_when_both_clear_thresholds() -> None:
    settings = {
        "maximum_environment_states": 20_000,
        "continue_to_9x9_gate": {
            "candidate_min_score_rate_vs_frozen_random_initial": 0.6,
            "paired_ci95_lower_above": 0.5,
            "required_later_nodes": 2,
        }
    }
    passed, gate_nodes = six_by_six_gate_status(
        [_stage(5_000, 0.95, 0.85), _stage(10_000, 0.95, 0.85),
         _stage(15_000, 0.90, 0.75), _stage(20_000, 0.90, 0.75)],
        settings,
    )
    assert passed
    assert gate_nodes == [15_000, 20_000]


def test_late_node_failure_blocks_gate_even_with_strong_early_stages() -> None:
    settings = {
        "maximum_environment_states": 20_000,
        "continue_to_9x9_gate": {
            "candidate_min_score_rate_vs_frozen_random_initial": 0.6,
            "paired_ci95_lower_above": 0.5,
            "required_later_nodes": 2,
        }
    }
    passed, gate_nodes = six_by_six_gate_status(
        [_stage(5_000, 0.95, 0.85), _stage(10_000, 0.95, 0.85),
         _stage(15_000, 0.90, 0.75), _stage(20_000, 0.55, 0.40)],
        settings,
    )
    assert not passed
    assert gate_nodes == [15_000, 20_000]


def test_completed_six_by_six_run_serializes_gate_node_ids(tmp_path, monkeypatch) -> None:
    arm_dir = tmp_path / "research/g_v5/checkpoints/reference6x6-400-empty"
    arm_dir.mkdir(parents=True)
    for node in (0, 2_000, 5_000, 10_000, 15_000, 20_000):
        (arm_dir / f"node_{node}.pt").write_bytes(b"fixture")

    train_calls = iter((10_000, 20_000))

    def fake_train(*args, **kwargs):
        states = next(train_calls)
        return {"completed": True, "environment_states": states, "game_count": 10,
                "update_count": 10, "cumulative_cpu_seconds": 1.0}

    def fake_match(*, phase, pairs, **kwargs):
        games = 2 * int(pairs)
        return {"phase": phase, "games": games, "wins": games - 1, "draws": 0,
                "losses": 1, "score_rate": 0.95,
                "paired_opening_bootstrap_ci95": [0.9, 1.0]}

    monkeypatch.setattr(experiment, "train_reference_arm", fake_train)
    monkeypatch.setattr(experiment, "_load_reference_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "_append_match_set", fake_match)
    monkeypatch.setattr(experiment, "_record_tactics",
                        lambda *args, **kwargs: {"network_accuracy": 0.4, "puct_accuracy": 0.7})

    settings = {
        "network_seed": 1, "training_seed": 2, "simulations_per_move": 400,
        "milestones": [2_000, 5_000, 10_000, 15_000, 20_000],
        "development_pairs_at_intermediate_nodes": 8,
        "development_pairs_at_gate_nodes": 20,
        "blind_pairs": 20,
        "maximum_environment_states": 20_000,
        "continue_to_9x9_gate": {
            "candidate_min_score_rate_vs_frozen_random_initial": 0.6,
            "paired_ci95_lower_above": 0.5,
            "required_later_nodes": 2,
        },
    }
    schedule = {"six_by_six": {"development": [{}] * 20, "blind": [{}] * 20}}
    result = experiment._run_6x6(
        tmp_path, {"six_by_six": settings, "config_sha256": "test"},
        schedule, [], time.process_time() + 100,
    )

    assert result["gate_nodes"] == [15_000, 20_000]
    assert result["continue_to_9x9_gate_passed"] is True
    assert result["complete"] is True
