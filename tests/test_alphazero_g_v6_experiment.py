from __future__ import annotations

import importlib.util

import pytest


def _experiment_module():
    name = "src.research.alphazero_g.v6_experiment"
    if importlib.util.find_spec(name) is None:
        pytest.fail("V6 stage orchestration and dev-only selection are not implemented")
    from src.research.alphazero_g import v6_experiment

    return v6_experiment


def _row(milestone: int, score: float, *, complete: bool = True) -> dict:
    opponents = ["ref-v6-initial", "G0-V3-510901-5K", "G0-V3-510901-10K",
                 "G0-V3-510901-20K", "A", "B-50ms"]
    comparisons = {}
    for budget in (64, 400):
        comparisons[str(budget)] = {
            name: {
                "complete": complete,
                "games": 16,
                "score_rate": score if name == "G0-V3-510901-10K" else 0.5,
                "paired_opening_bootstrap_ci95": [max(0.0, score - 0.2), min(1.0, score + 0.2)],
            }
            for name in opponents
        }
    return {
        "milestone": milestone,
        "actual_states": milestone + 8,
        "development_seed": 696103,
        "checkpoint_path": f"research/g_v6/checkpoints/node_{milestone}.pt",
        "checkpoint_sha256": f"sha-{milestone}",
        "comparisons": comparisons,
    }


def test_checkpoint_selection_uses_only_complete_puct64_development_and_tie_goes_earlier() -> None:
    module = _experiment_module()
    rows = [_row(2_000, 0.5), _row(5_000, 0.75), _row(10_000, 0.75), _row(15_000, 0.25), _row(20_000, 0.25)]

    selected = module.select_development_candidate(rows)

    assert selected["milestone"] == 5_000
    assert selected["score_rate"] == 0.75
    assert selected["selection_budget"] == 64
    assert selected["opponent"] == "G0-V3-510901-10K"


def test_checkpoint_selection_rejects_incomplete_stage_results() -> None:
    module = _experiment_module()
    rows = [_row(node, 0.5) for node in (2_000, 5_000, 10_000, 15_000, 20_000)]
    rows[3]["comparisons"]["400"]["A"]["complete"] = False

    with pytest.raises(ValueError, match="incomplete"):
        module.select_development_candidate(rows)


def test_v5_milestone_checkpoint_counters_are_read_from_metric_schema() -> None:
    module = _experiment_module()
    checkpoint = {"environment_states": 2_003, "games_completed": 63}
    training_metric = {"game_count": 63, "update_count": 62}

    assert module._milestone_counts(checkpoint, training_metric) == (63, 62)

