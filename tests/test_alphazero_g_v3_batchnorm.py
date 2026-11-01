import copy

import pytest
import torch

from src.research.alphazero_g.replay import ReplayBuffer
from src.research.alphazero_g.selfplay import play_selfplay_game
from src.research.alphazero_g.training import GTrainer, initialize_arm
from src.research.alphazero_g.v3_bn import (
    capture_bn_buffers,
    restore_bn_buffers,
    set_bn_training_mode,
)


class FixedPolicyValue(torch.nn.Module):
    def forward(self, planes):
        return torch.zeros((planes.shape[0], 81)), torch.zeros((planes.shape[0],))


def _replay():
    opening = [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 3, "col": 5},
        {"player": "W", "row": 5, "col": 3},
    ]
    game = play_selfplay_game(
        FixedPolicyValue(), opening, arm_id="G0", game_id="bn-game",
        opening_group_id="test-opening", simulations=1, seed=991,
    )
    replay = ReplayBuffer(256, seed=992)
    replay.add_complete_game(game)
    return replay


def _non_buffer_state(model):
    return {key: value.clone() for key, value in model.state_dict().items()
            if not any(key.endswith(suffix) for suffix in ("running_mean", "running_var", "num_batches_tracked"))}


def test_bn_buffers_restore_exactly_without_changing_parameters():
    model = initialize_arm("G0", seed=993)
    initial = capture_bn_buffers(model)
    initial_parameters = _non_buffer_state(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.add_(2.0)
                module.running_var.mul_(1.5)
                module.num_batches_tracked.add_(4)
    restore_bn_buffers(model, initial)

    assert all(torch.equal(capture_bn_buffers(model)[name][field], initial[name][field])
               for name in initial for field in initial[name])
    assert all(torch.equal(model.state_dict()[name], tensor) for name, tensor in initial_parameters.items())


def test_bn_freeze_keeps_affine_parameters_trainable_without_running_stat_drift():
    torch.manual_seed(994)
    model = initialize_arm("G0", seed=994)
    initial_buffers = capture_bn_buffers(model)
    initial_affine = model.input_bn.weight.detach().clone()
    trainer = GTrainer(model, seed=995, learning_rate=1e-4, expected_updates=2, bn_mode="freeze")
    trainer.train_updates(_replay(), updates=2, batch_size=4)

    after = capture_bn_buffers(model)
    assert all(torch.equal(after[name][field], initial_buffers[name][field])
               for name in initial_buffers for field in initial_buffers[name])
    assert not torch.equal(model.input_bn.weight.detach(), initial_affine)
    assert model.input_bn.weight.grad is not None


def test_bn_update_preserves_legacy_running_stat_behavior_and_modes_validate():
    model = initialize_arm("G0", seed=996)
    initial = capture_bn_buffers(model)
    trainer = GTrainer(model, seed=997, learning_rate=1e-4, expected_updates=1, bn_mode="update")
    trainer.train_updates(_replay(), updates=1, batch_size=4)
    after = capture_bn_buffers(model)
    assert any(not torch.equal(after[name][field], initial[name][field])
               for name in initial for field in initial[name])

    with pytest.raises(ValueError, match="bn mode|bn_mode"):
        set_bn_training_mode(model, "restore")


def test_bn_restore_rejects_layer_or_buffer_shape_mismatch():
    model = initialize_arm("G0", seed=998)
    buffers = copy.deepcopy(capture_bn_buffers(model))
    buffers.pop(next(iter(buffers)))
    with pytest.raises(ValueError, match="layer|buffer"):
        restore_bn_buffers(model, buffers)
