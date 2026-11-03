from __future__ import annotations

from pathlib import Path


def test_historical_candidate_registry_covers_key_v3_v4_and_v4_1_g0_models() -> None:
    from src.research.alphazero_g.v4_2_registration import historical_candidate_registry

    root = Path(__file__).resolve().parents[1]
    candidates = historical_candidate_registry(root)
    ids = {row["candidate_id"] for row in candidates}
    assert {
        "G0-V3-510901-5K", "G0-V3-510901-10K", "G0-V3-510901-20K",
        "G0-V4-540402-30K", "G0-V4-CONT-100K", "G0-V4_1-winner_balanced-10K",
    } <= ids
    assert len(candidates) == 12
    assert all(Path(root, row["checkpoint_path"]).is_file() for row in candidates)
    assert all(row["training_family"] == "G0" for row in candidates)
    assert all(row["checkpoint_sha256"] for row in candidates)


def test_registered_history_opponents_are_frozen_g0_only_and_do_not_include_A_or_B() -> None:
    from src.research.alphazero_g.v4_2_registration import create_v4_2_registration

    root = Path(__file__).resolve().parents[1]
    config, _ = create_v4_2_registration(root, states_per_arm=10_000)
    ids = set(config["training"]["history_opponents"])
    assert ids == {
        "G0-V3-510901-5K", "G0-V3-510901-10K", "G0-V3-510901-20K",
        "G0-V4-540402-30K", "G0-V4_1-control_v4-10K",
    }
    assert all(row["candidate_id"] in ids or row["training_family"] == "G0"
               for row in config["historical_candidates"])
