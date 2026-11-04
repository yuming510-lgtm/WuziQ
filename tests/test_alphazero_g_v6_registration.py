from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _registration_module():
    name = "src.research.alphazero_g.v6_registration"
    if importlib.util.find_spec(name) is None:
        pytest.fail("V6 registration and checkpoint-resume verifier are not implemented")
    from src.research.alphazero_g import v6_registration

    return v6_registration


def test_v6_opening_schedule_is_fixed_legal_paired_and_disjoint_from_v5() -> None:
    module = _registration_module()
    schedule = module.build_opening_schedule()
    module.validate_opening_schedule(schedule)

    assert schedule["development_seed"] != schedule["blind_seed"]
    assert len(schedule["development"]) == 8
    assert len(schedule["blind"]) == 20
    assert all(len(opening["moves"]) == 6 for opening in schedule["development"])
    assert all(len(opening["moves"]) == 6 for opening in schedule["blind"])
    assert schedule == module.build_opening_schedule()


def test_v6_resume_validator_restores_full_v5_checkpoint_without_mutation() -> None:
    module = _registration_module()
    root = Path(__file__).resolve().parents[1]
    result = module.validate_v5_resume_checkpoint(root)

    assert result["valid"] is True
    assert result["arm_id"] == "reference400_empty"
    assert result["board_size"] == 9
    assert result["simulations_per_move"] == 400
    assert result["environment_states"] == 1514
    assert result["replay_rows"] == 10000
    assert result["optimizer_slots"] > 0
    assert result["python_rng_restored"] is True
    assert result["numpy_rng_restored"] is True
    assert result["torch_rng_restored"] is True
    assert result["model_matches_node_1500"] is True


def test_v6_registration_records_cumulative_state_milestones_and_v5_lineage() -> None:
    module = _registration_module()
    root = Path(__file__).resolve().parents[1]
    registration = module.make_registration(root)

    assert registration["board_size"] == 9
    assert registration["n_in_row"] == 5
    assert registration["simulations_per_move"] == 400
    assert registration["target_cumulative_states"] == 20000
    assert registration["milestones"] == [2000, 5000, 10000, 15000, 20000]
    assert registration["resume_source"].endswith("reference400_empty/training_state.pt")
    assert registration["v5_config_sha256"]
