from __future__ import annotations

import importlib
import importlib.util
import random

import numpy as np
import pytest
import torch


def _api():
    name = "src.research.imitation_f.training"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail("F V1 resumable training API is not implemented")
    return importlib.import_module(name)


def test_training_checkpoint_restores_optimizer_scheduler_and_all_rngs(tmp_path) -> None:
    api = _api()
    from src.research.imitation_f.model import PolicyValueNet

    torch.manual_seed(731601)
    random.seed(731601)
    np.random.seed(731601)
    model = PolicyValueNet(channels=8, blocks=1, seed=731601)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    x = torch.randn(2, 3, 9, 9)
    policy, value = model(x)
    loss = policy.square().mean() + value.square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    state = {
        "epoch": 1,
        "sampler_cursor": 7,
        "update_step": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    path = tmp_path / "resume.pt"
    digest = api.save_training_state(path, state)
    expected_x = torch.randn(2, 3, 9, 9)
    expected_torch = torch.rand(4)
    expected_numpy = np.random.random(4)
    expected_python = [random.random() for _ in range(4)]
    next_policy, next_value = model(expected_x)
    next_loss = next_policy.square().mean() + next_value.square().mean()
    optimizer.zero_grad(set_to_none=True)
    next_loss.backward()
    optimizer.step()
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    restored_model = PolicyValueNet(channels=8, blocks=1, seed=1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=9, gamma=0.9)
    loaded = api.load_training_state(path, restored_model, restored_optimizer, restored_scheduler)
    assert digest
    assert loaded["epoch"] == 1
    assert loaded["sampler_cursor"] == 7
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    actual_x = torch.randn(2, 3, 9, 9)
    assert torch.equal(actual_x, expected_x)
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.allclose(np.random.random(4), expected_numpy)
    assert [random.random() for _ in range(4)] == expected_python
    actual_policy, actual_value = restored_model(actual_x)
    actual_loss = actual_policy.square().mean() + actual_value.square().mean()
    restored_optimizer.zero_grad(set_to_none=True)
    actual_loss.backward()
    restored_optimizer.step()
    assert all(torch.equal(left, right) for left, right in zip(restored_model.parameters(), expected_parameters))


def test_train_runs_a_small_policy_value_dataset_and_writes_best_checkpoint(tmp_path) -> None:
    api = _api()
    from src.research.imitation_f.model import PolicyValueNet

    def row(index: int) -> dict:
        board = ["." for _ in range(81)]
        board[40] = "1"
        target = [0.0] * 81
        target[(index + 1) % 81] = 1.0
        return {
            "state_key": f"p{index}",
            "canonical_board": "".join(board),
            "policy_target": target,
            "value_target": 1.0 if index % 2 else -1.0,
            "value_count": 1,
            "opening_group_ids": [f"g{index}"],
        }

    dataset = {"train": [row(index) for index in range(8)], "validation": [row(index + 8) for index in range(4)]}
    result = api.train(
        {
            "seed": 731602, "epochs": 2, "batch_size": 4, "channels": 8, "blocks": 1,
            "learning_rate": 0.002, "checkpoint_every_updates": 1, "patience": 2,
        },
        dataset, tmp_path, resume=False,
    )
    assert result["updates"] == 4
    assert result["elapsed_seconds"] > 0
    assert result["best_checkpoint"].is_file()
    assert result["history"]
    loaded = PolicyValueNet()
    assert result["best_checkpoint"].stat().st_size > 0
    state = torch.load(result["state_checkpoint"], map_location="cpu", weights_only=False)
    assert state["elapsed_training_seconds"] == pytest.approx(result["elapsed_seconds"])


def test_resume_rejects_changed_dataset_instead_of_silently_mixing_runs(tmp_path) -> None:
    api = _api()

    def row(action: int) -> dict:
        board = ["."] * 81
        board[40] = "1"
        target = [0.0] * 81
        target[action] = 1.0
        return {
            "state_key": f"dataset-{action}",
            "canonical_board": "".join(board),
            "policy_target": target,
            "value_target": 0.0,
            "value_count": 1,
            "opening_group_ids": [f"g{action}"],
        }

    config = {
        "seed": 731603, "epochs": 1, "batch_size": 2, "channels": 8, "blocks": 1,
        "checkpoint_every_updates": 1,
    }
    api.train(config, {"train": [row(0), row(1)], "validation": [row(2)]}, tmp_path, resume=False)
    changed = {"train": [row(0), row(3)], "validation": [row(2)]}
    with pytest.raises(ValueError, match="dataset fingerprint"):
        api.train(config, changed, tmp_path, resume=True)
