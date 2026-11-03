from __future__ import annotations

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
