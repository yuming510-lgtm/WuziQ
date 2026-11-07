from __future__ import annotations

import numpy as np
import json
import pytest

from src.research.alphazero_g import v5_training
from src.research.alphazero_g.v6_2_registration import create_opening_schedule
from src.research.alphazero_g.v6_2_training import registration_value, train_to_games


def test_reference_training_can_stop_on_exact_cumulative_game_count(tmp_path, monkeypatch):
    def one_game(*, game_id, seed, **kwargs):
        state = np.zeros((1, 4, 9, 9), dtype=np.float32)
        policy = np.zeros((1, 81), dtype=np.float32)
        policy[0, 40] = 1.0
        return (
            {
                "game_id": game_id,
                "seed": seed,
                "training_state_count": 1,
                "runner_cpu_seconds": 0.01,
                "runner_wall_seconds": 0.01,
            },
            state,
            policy,
            np.asarray([1], dtype=np.float32),
        )

    monkeypatch.setattr(v5_training, "_run_game", one_game)
    result = v5_training.train_reference_arm(
        tmp_path,
        arm="reference400_empty",
        board_size=9,
        n_in_row=5,
        simulations=400,
        target_states=20_001,
        target_games=1,
        progress_experiment_id="wuziq-alphazero-g-v6-2-9x9-selfplay-scale",
        milestones=[],
        seed=20261107,
        config={
            "config_sha256": "a" * 64,
            "experiment_id": "v6-2-test",
            "training": {
                "buffer_size": 10_000,
                "batch_size": 512,
                "epochs": 5,
                "learning_rate": 0.002,
                "kl_target": 0.02,
            },
        },
        artifact_dir="research/g_v6_2_test",
    )

    assert result["game_count"] == 1
    assert result["environment_states"] == 1
    assert result["completed"] is True
    metric = (tmp_path / "research/g_v6_2_test/results/training/reference400_empty.jsonl").read_text()
    assert '"policy_loss"' in metric
    assert '"value_loss"' in metric
    progress = json.loads((tmp_path / "research/g_v6_2_test/results/training_progress.json").read_text())
    assert progress["experiment_id"] == "wuziq-alphazero-g-v6-2-9x9-selfplay-scale"


def test_resume_game_ids_continue_from_cumulative_v6_counter(tmp_path, monkeypatch):
    game_ids = []

    def one_game(*, game_id, seed, **kwargs):
        game_ids.append(game_id)
        state = np.zeros((1, 4, 9, 9), dtype=np.float32)
        policy = np.zeros((1, 81), dtype=np.float32)
        policy[0, 40] = 1.0
        return (
            {"game_id": game_id, "seed": seed, "training_state_count": 1,
             "runner_cpu_seconds": 0.01, "runner_wall_seconds": 0.01},
            state,
            policy,
            np.asarray([1], dtype=np.float32),
        )

    monkeypatch.setattr(v5_training, "_run_game", one_game)
    common = {
        "root": tmp_path,
        "arm": "reference400_empty",
        "board_size": 9,
        "n_in_row": 5,
        "simulations": 400,
        "target_states": 1_000_000,
        "milestones": [],
        "seed": 20261107,
        "config": {
            "config_sha256": "b" * 64,
            "experiment_id": "v6-2-resume-test",
            "training": {
                "buffer_size": 10_000,
                "batch_size": 512,
                "epochs": 5,
                "learning_rate": 0.002,
                "kl_target": 0.02,
            },
        },
        "artifact_dir": "research/g_v6_2_test",
        "progress_experiment_id": "wuziq-alphazero-g-v6-2-9x9-selfplay-scale",
    }
    first = v5_training.train_reference_arm(**common, target_games=1)
    checkpoint_path = tmp_path / "research/g_v6_2_test/checkpoints/reference400_empty/training_state.pt"
    checkpoint = v5_training.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["game_count"] = 1_012
    checkpoint["environment_states"] = 20_001
    checkpoint["update_count"] = 1_011
    v5_training.torch.save(checkpoint, checkpoint_path)

    resumed = v5_training.train_reference_arm(**common, target_games=1_013)

    assert first["completed"] is True
    assert resumed["game_count"] == 1_013
    assert game_ids == ["reference400_empty-game-000000", "reference400_empty-game-001012"]


def test_registered_development_and_blind_openings_are_new_and_legal():
    schedule = create_opening_schedule()
    assert len(schedule["development"]) == 80
    assert len(schedule["blind"]) == 40
    assert schedule["validation"]["historical_d4_overlap"] == 0
    assert schedule["validation"]["all_openings_legal_and_nonterminal"] is True


def test_training_beyond_2000_requires_the_registered_gate(tmp_path):
    config = tmp_path / "research/g_v6_2/config/registration.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"cpu_hard_cap_total_new_hours": 30}), encoding="utf-8")
    gate = tmp_path / "research/g_v6_2/results/gate_2000.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({"passed": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="advancement gate"):
        train_to_games(tmp_path, target_games=2_500, session_cpu_seconds=60)


def test_cpu_budget_is_read_from_registered_compute_section(tmp_path):
    config = tmp_path / "research/g_v6_2/config/registration.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"compute": {"cpu_hard_cap_total_new_hours": 30}}), encoding="utf-8")

    assert registration_value(tmp_path, "cpu_hard_cap_total_new_hours") == 30
