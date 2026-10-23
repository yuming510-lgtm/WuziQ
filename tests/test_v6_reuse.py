from __future__ import annotations

from copy import deepcopy

import pytest

from src.research.v6.config import canonical
from src.research.v6.config import load_config
from src.research.v6.reuse import prepare_v5_1_reuse, reconstruct_frozen_schedule
from src.research.v6.schedule import build_schedule
from src.research.v6.validation import validate_game_record


def test_reuse_imports_only_full_replay_valid_v5_1_groups() -> None:
    config = load_config()
    schedule, _routes = build_schedule(config)
    frozen_schedule, manifest, records = prepare_v5_1_reuse(schedule, config)
    specs = {
        game["game_id"]: game
        for group in frozen_schedule for game in group["games"]
    }

    assert manifest["source_record_count"] == 147
    assert manifest["source_full_groups"] == 73
    assert manifest["v6_reused_groups"] == 73
    assert manifest["v6_reused_games"] == 146
    assert len(manifest["source_partial_games_excluded"]) == 1
    assert len(records) == 146
    assert len({record["game_id"] for record in records}) == 146
    assert all(record["source_type"] == "reused_v5_1" for record in records)
    assert all(record["game_seed"] == specs[record["game_id"]]["game_seed"] for record in records)
    assert all(validate_game_record(record, specs[record["game_id"]], verify_proofs=False) for record in records)


def test_outcome_never_changes_v5_1_reuse_eligibility() -> None:
    config = load_config()
    schedule, _routes = build_schedule(config)
    frozen, manifest, records = prepare_v5_1_reuse(schedule, config)
    first_group = next(group for group in frozen if group.get("reuse_source"))
    ids = {game["game_id"] for game in first_group["games"]}
    first_records = [record for record in records if record["game_id"] in ids]
    assert len(first_records) == 2
    winners_before = [record["winner_model"] for record in first_records]

    # Eligibility remains the same when the source outcome is only read later.
    _, same_manifest, same_records = prepare_v5_1_reuse(schedule, config)
    assert same_manifest["groups"] == manifest["groups"]
    assert [record["winner_model"] for record in same_records if record["game_id"] in ids] == winners_before


def test_reconstruction_accepts_exact_reuse_decoration_and_rejects_tampering() -> None:
    config = load_config()
    schedule, _routes = build_schedule(config)
    frozen, manifest, _records = prepare_v5_1_reuse(schedule, config)

    rebuilt, rebuilt_manifest = reconstruct_frozen_schedule(config, frozen, manifest)
    assert canonical(rebuilt) == canonical(frozen)
    assert canonical(rebuilt_manifest) == canonical(manifest)

    tampered = deepcopy(frozen)
    reused = next(group for group in tampered if group.get("reuse_source"))
    reused["reuse_source"]["source_commit"] = "incorrect"
    with pytest.raises(ValueError, match="deterministic source and reuse manifest"):
        reconstruct_frozen_schedule(config, tampered, manifest)
