from __future__ import annotations

import json

from src.ai import SearchConfig
from src.research.alphazero_g.v6_8_tournament import (
    MODELS, OPPONENTS, _d4_opening_key, _summarize, prepare_experiment, verify_models,
)


def test_six_frozen_checkpoints_load_with_expected_metadata():
    manifest = verify_models()
    assert [row["games"] for row in manifest["models"]] == list(MODELS)
    assert len({row["sha256"] for row in manifest["models"]}) == len(MODELS)
    assert all(row["simulations_per_move"] == 400 for row in manifest["models"])
    assert all(row["board_size"] == 9 and row["n_in_row"] == 5 for row in manifest["models"])
    assert len({row["config_sha256"] for row in manifest["models"]}) == 1


def test_opening_pool_is_frozen_unique_legal_and_historically_disjoint():
    prepared = prepare_experiment()
    schedule = prepared["schedule"]
    assert schedule["historical_d4_overlap"] == 0
    assert len(schedule["opening_groups"]) == 20
    keys = [_d4_opening_key(row["moves"]) for row in schedule["opening_groups"]]
    assert len(set(keys)) == 20
    assert schedule["pairs_per_condition"] == 20
    assert schedule["games_per_condition"] == 40


def test_fixed_schedule_expands_to_960_unique_model_opponent_opening_color_slots():
    assert len(MODELS) * len(OPPONENTS) * 20 * 2 == 960
    assert SearchConfig().max_depth == 4
    assert SearchConfig().candidate_radius == 2
    assert SearchConfig(time_limit_seconds=0.05).time_limit_seconds == 0.05


def test_paired_summary_uses_wins_and_draws_and_color_split():
    rows = [
        {"points": 1.0, "candidate_color": "B", "opening_group_id": "o1", "move_count": 10,
         "candidate_cpu_seconds": 2.0, "opponent_cpu_seconds": .1,
         "moves": [{"actor": "G5000", "cpu_seconds": 2.0}]},
        {"points": .5, "candidate_color": "W", "opening_group_id": "o1", "move_count": 12,
         "candidate_cpu_seconds": 2.5, "opponent_cpu_seconds": .2,
         "moves": [{"actor": "G5000", "cpu_seconds": 2.5}]},
    ]
    summary = _summarize(rows, seed=7)
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 1, 0)
    assert summary["score_rate"] == .75
    assert summary["black_score_rate"] == 1.0
    assert summary["white_score_rate"] == .5
