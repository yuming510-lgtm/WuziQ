from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.research.alphazero_g.v5_training import train_reference_arm


def _tiny_config() -> dict:
    return {
        "experiment_id": "wuziq-alphazero-g-v5-reference-and-closed-loop",
        "config_sha256": "v5-fixture-config",
        "training": {
            "buffer_size": 128,
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.002,
            "kl_target": 0.02,
        },
    }


def test_training_artifact_namespace_keeps_v5_default_path_and_writes_v6_separately(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    try:
        result = train_reference_arm(
            tmp_path, arm="namespace_smoke", board_size=6, n_in_row=4,
            simulations=2, target_states=1, milestones=[1], seed=87,
            config=_tiny_config(), artifact_dir="research/g_v6",
        )
    except TypeError as exc:
        pytest.fail(f"training runner lacks a V6-only artifact namespace: {exc}")

    assert result["completed"] is True
    assert (tmp_path / "research/g_v6/checkpoints/namespace_smoke/training_state.pt").is_file()
    assert len(list((tmp_path / "research/g_v6/raw/selfplay/namespace_smoke").glob("*.json"))) == 1
    assert not (tmp_path / "research/g_v5/checkpoints/namespace_smoke/training_state.pt").exists()


def test_v6_continuation_copies_and_resumes_all_v5_training_state_without_mutation(tmp_path: Path) -> None:
    try:
        from src.research.alphazero_g.v6_training import prepare_v6_continuation, train_v6
    except ImportError as exc:
        pytest.fail(f"V6 continuation wrapper is not implemented: {exc}")

    source_root = Path(__file__).resolve().parents[1]
    source_path = source_root / "research/g_v5/checkpoints/reference400_empty/training_state.pt"
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    prepared = prepare_v6_continuation(tmp_path, source_root=source_root)
    assert prepared["resume_validated"] is True

    result = train_v6(tmp_path, source_root=source_root, target_states=1514)
    assert result["completed"] is True
    assert result["environment_states"] == 1514
    assert result["game_count"] == 47
    assert result["update_count"] == 46
    target = tmp_path / "research/g_v6/checkpoints/reference400_empty/training_state.pt"
    assert target.is_file()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
    assert result["simulations_per_move"] == 400

