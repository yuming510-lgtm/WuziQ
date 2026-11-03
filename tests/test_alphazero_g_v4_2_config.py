from __future__ import annotations

import pytest


def _registered_config():
    from src.research.alphazero_g.v4_2_config import config_sha256

    config = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4-2",
        "branch": "research/alphazero-g-v4-2",
        "base_commit": "4dd0c449c7a4afe1a4f68b3e2a234cdd70e186df",
        "compute": {"cpu_hour_cap": 5.0, "device": "cpu"},
        "training": {"states_per_arm": 10_000, "maximum_total_new_states": 30_000},
        "historical_candidates": [],
        "forbidden_training_inputs": ["A", "B", "C", "E", "F", "F1", "F2", "teacher_labels", "handcrafted_tactical_reward"],
        "arms": {
            "baseline": {"search_variant": "baseline", "opponent_pool": "current"},
            "search_target": {"search_variant": "selected_diagnostic", "opponent_pool": "current"},
            "history_pool": {"search_variant": "baseline", "opponent_pool": "historical_g0"},
        },
    }
    config["config_sha256"] = config_sha256(config)
    return config


def test_v4_2_config_accepts_equal_single_factor_arms() -> None:
    from src.research.alphazero_g.v4_2_config import validate_v4_2_config

    validate_v4_2_config(_registered_config(), require_branch=False)


def test_v4_2_config_rejects_training_budget_above_cap() -> None:
    from src.research.alphazero_g.v4_2_config import config_sha256, validate_v4_2_config

    config = _registered_config()
    config["training"]["maximum_total_new_states"] = 40_001
    config["config_sha256"] = config_sha256(config)
    with pytest.raises(ValueError, match="40,000"):
        validate_v4_2_config(config, require_branch=False)


def test_v4_2_config_rejects_an_arm_that_changes_two_factors() -> None:
    from src.research.alphazero_g.v4_2_config import config_sha256, validate_v4_2_config

    config = _registered_config()
    config["arms"]["search_target"]["opponent_pool"] = "historical_g0"
    config["config_sha256"] = config_sha256(config)
    with pytest.raises(ValueError, match="one registered factor"):
        validate_v4_2_config(config, require_branch=False)


def test_v4_2_config_rejects_modified_digest() -> None:
    from src.research.alphazero_g.v4_2_config import validate_v4_2_config

    config = _registered_config()
    config["training"]["states_per_arm"] = 5_000
    with pytest.raises(ValueError, match="SHA-256"):
        validate_v4_2_config(config, require_branch=False)


def test_v4_2_config_rejects_wrong_branch_identity() -> None:
    from src.research.alphazero_g.v4_2_config import config_sha256, validate_v4_2_config

    config = _registered_config()
    config["branch"] = "research/alphazero-g-v4-1"
    config["config_sha256"] = config_sha256(config)
    with pytest.raises(ValueError, match="branch/base identity"):
        validate_v4_2_config(config, require_branch=False)


def test_v4_2_config_rejects_non_g0_history_opponent() -> None:
    from src.research.alphazero_g.v4_2_config import config_sha256, validate_v4_2_config

    config = _registered_config()
    config["historical_candidates"] = [{"candidate_id": "teacher", "training_family": "F2"}]
    config["training"]["history_opponents"] = ["teacher"]
    config["config_sha256"] = config_sha256(config)
    with pytest.raises(ValueError, match="G0 checkpoints only"):
        validate_v4_2_config(config, require_branch=False)
