from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from src.research.alphazero_g.puct import PUCTConfig
from src.research.alphazero_g.replay import ReplayBuffer, load_game_log
from src.research.alphazero_g.selfplay import play_selfplay_game
from src.research.alphazero_g.training import (
    F2_SHA256,
    GTrainer,
    initialize_arm,
    load_g_checkpoint,
    load_training_state,
    save_g_checkpoint,
    save_training_state,
)
from src.research.alphazero_g.experiment import build_g_schedule
from src.research.alphazero_g.runner import run_training_arm


class FixedPolicyValue(torch.nn.Module):
    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((planes.shape[0], 81), device=planes.device), torch.zeros(
            (planes.shape[0],), device=planes.device
        )


def _replay() -> ReplayBuffer:
    opening = [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 3, "col": 5},
        {"player": "W", "row": 5, "col": 3},
    ]
    game = play_selfplay_game(
        FixedPolicyValue(), opening, arm_id="G0", game_id="training-test-game",
        opening_group_id="training-only", simulations=1, seed=5201,
    )
    replay = ReplayBuffer(256, seed=5202)
    replay.add_complete_game(game)
    return replay


def _same_weights(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    return all(torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict())


def test_g0_initialization_is_seeded_and_f2_initialization_is_exact() -> None:
    first = initialize_arm("G0", seed=5203)
    second = initialize_arm("G0", seed=5203)
    assert _same_weights(first, second)

    f2 = initialize_arm("G-F2", seed=999, f2_checkpoint="research/f_v2/checkpoints/best_f2.pt")
    assert f2.checkpoint_metadata["checkpoint_sha256"] == F2_SHA256
    assert hashlib.sha256(open("research/f_v2/checkpoints/best_f2.pt", "rb").read()).hexdigest() == F2_SHA256
    frozen = torch.load("research/f_v2/checkpoints/best_f2.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(f2.state_dict()[key], frozen["state_dict"][key]) for key in f2.state_dict())
    assert not _same_weights(first, f2)


def test_one_update_uses_replay_visit_and_terminal_targets() -> None:
    trainer = GTrainer(initialize_arm("G0", seed=5204), seed=5205, learning_rate=1e-3)
    metrics = trainer.train_updates(_replay(), updates=2, batch_size=4)
    assert trainer.update_count == 2
    assert metrics["updates"] == 2
    assert metrics["policy_loss"] > 0
    assert metrics["value_loss"] >= 0
    assert metrics["total_loss"] == pytest.approx(metrics["policy_loss"] + metrics["value_loss"])


def test_training_checkpoint_restores_optimizer_replay_and_random_stream_exactly(tmp_path) -> None:
    replay = _replay()
    trainer = GTrainer(initialize_arm("G0", seed=5206), seed=5207, learning_rate=1e-3)
    trainer.train_updates(replay, updates=1, batch_size=4)
    checkpoint = tmp_path / "resume.pt"
    save_training_state(checkpoint, trainer, replay, progress={"environment_steps": 8})

    expected_metrics = trainer.train_updates(replay, updates=1, batch_size=4)
    expected_weights = {key: value.clone() for key, value in trainer.model.state_dict().items()}

    resumed_replay = ReplayBuffer(256, seed=0)
    resumed = GTrainer(initialize_arm("G0", seed=5206), seed=999, learning_rate=1e-3)
    progress = load_training_state(checkpoint, resumed, resumed_replay)
    actual_metrics = resumed.train_updates(resumed_replay, updates=1, batch_size=4)

    assert progress == {"environment_steps": 8}
    assert resumed.update_count == trainer.update_count
    for key in expected_metrics:
        if key not in {"cpu_seconds", "wall_seconds"}:
            assert actual_metrics[key] == expected_metrics[key]
    assert all(torch.equal(resumed.model.state_dict()[key], expected_weights[key]) for key in expected_weights)


def test_g_checkpoint_is_f_compatible_and_loadable(tmp_path) -> None:
    model = initialize_arm("G0", seed=5208)
    path = tmp_path / "g-final.pt"
    digest = save_g_checkpoint(path, model, metadata={"arm_id": "G0", "step": 12})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    loaded = load_g_checkpoint(path)
    assert _same_weights(model, loaded)
    assert loaded.checkpoint_metadata["arm_id"] == "G0"


def _tiny_runner_root(path):
    import subprocess

    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "G test"], cwd=path, check=True)
    (path / "README.md").write_text("test root\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "test root"], cwd=path, check=True)
    return path


def _tiny_config():
    schedule = build_g_schedule(5209, 1, 1, 1)
    return {
        "seed": 5209,
        "arms": {
            "G0": {"environment_steps": 20, "updates": 2, "simulations_per_move": 1},
            "G-F2": {"environment_steps": 20, "updates": 2, "simulations_per_move": 1},
        },
        "stage_environment_steps": [20],
        "stage_updates": [2],
        "opening_schedule": schedule,
        "initialization": {"G0": {"type": "random", "seed": 5210}},
        "replay_capacity": 256,
        "batch_size": 4,
        "torch_threads": 1,
        "device": "cpu",
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0, "temperature_moves": 12, "late_temperature": 0.1},
    }


def test_runner_game_checkpoint_resume_is_idempotent_and_exact(tmp_path) -> None:
    interrupted_root = _tiny_runner_root(tmp_path / "interrupted")
    uninterrupted_root = _tiny_runner_root(tmp_path / "uninterrupted")
    config = _tiny_config()
    first = run_training_arm(interrupted_root, config, "G0", stop_after_games=1)
    assert first["completed"] is False
    second = run_training_arm(interrupted_root, config, "G0")
    baseline = run_training_arm(uninterrupted_root, config, "G0")
    assert second["completed"] is True
    assert baseline["completed"] is True
    assert second["budgeted_environment_steps"] == 20
    assert baseline["budgeted_environment_steps"] == 20
    assert second["optimizer_updates"] == baseline["optimizer_updates"] == 2
    resumed_model = load_g_checkpoint(interrupted_root / "research/g_v1/checkpoints/G0/final.pt")
    baseline_model = load_g_checkpoint(uninterrupted_root / "research/g_v1/checkpoints/G0/final.pt")
    assert _same_weights(resumed_model, baseline_model)
    resumed_games, _ = load_game_log(interrupted_root / "research/g_v1/results/selfplay/G0.jsonl")
    baseline_games, _ = load_game_log(uninterrupted_root / "research/g_v1/results/selfplay/G0.jsonl")
    assert [game["game_id"] for game in resumed_games] == [game["game_id"] for game in baseline_games]
