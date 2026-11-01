from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from src.research.alphazero_g.puct import PUCTConfig
from src.research.alphazero_g.replay import ReplayBuffer
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
