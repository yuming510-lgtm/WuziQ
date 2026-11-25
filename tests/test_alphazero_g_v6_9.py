from __future__ import annotations

import json
from pathlib import Path

from src.research.alphazero_g import v6_9_training as training


def test_g20000_full_state_is_a_valid_continuation_point():
    result = training.validate_training_state(
        training.ROOT / "research/g_v6_7/checkpoints/reference400_empty/training_state.pt",
        root=training.ROOT,
        expected_games=20_000,
    )
    assert result["sha256"] == training.SOURCE_STATE_SHA256
    assert result["environment_states"] == 348_216
    assert result["optimizer_updates"] == 19_999
    assert result["adam_restored"] and result["replay_restored"]
    assert result["python_rng_restored"] and result["numpy_rng_restored"] and result["torch_rng_restored"]


def test_frozen_reference_config_remains_v6_7_identical():
    original = training.ROOT / "research/g_v6_7/config/reference_training_config.json"
    copied = training.artifact_path() / "config/reference_training_config.json"
    assert json.loads(original.read_text()) == json.loads(copied.read_text())


def test_milestones_cover_all_30k_continuation():
    assert training.MILESTONES == (25_000, 30_000, 35_000, 40_000, 45_000, 50_000)
    assert training.MILESTONES[0] > training.BASE_GAMES
    assert training.MILESTONES[-1] == training.TARGET_GAMES
