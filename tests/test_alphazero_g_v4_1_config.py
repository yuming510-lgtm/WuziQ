from __future__ import annotations

import json

from src.research.alphazero_g.v4_1_config import load_v4_1_config


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_registered_v4_1_has_four_isolated_single_factor_arms() -> None:
    config = load_v4_1_config(ROOT)

    assert set(config["arms"]) == {"control_v4", "state_uniform", "winner_balanced", "low_lr"}
    assert config["training"]["maximum_total_new_states"] == 40_000
    assert len(config["evaluation"]["blind_group_ids"]) == 20
    assert config["initial_checkpoint_sha256"] == "ed6e2e6b5a25f355db5b31a9c58ba22158af2b345f86b3bcd3add9c3a49da7f9"


def test_registered_opening_schedule_is_disjoint_and_hash_checked() -> None:
    config = load_v4_1_config(ROOT)
    schedule = json.loads((ROOT / config["opening_schedule_path"]).read_text())

    training_ids = {g["opening_group_id"] for g in schedule["train_openings_by_seed"]["741101"]}
    development_ids = {g["opening_group_id"] for g in schedule["development_openings"]}
    blind_ids = {g["opening_group_id"] for g in schedule["blind_openings"]}
    assert not training_ids & development_ids
    assert not training_ids & blind_ids
    assert not development_ids & blind_ids
