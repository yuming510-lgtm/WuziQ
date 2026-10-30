from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from src.gomoku import GomokuBoard


def _api():
    name = "src.research.imitation_f.model"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail("F V1 model API is not implemented")
    return importlib.import_module(name)


def test_policy_value_shapes_ranges_and_deterministic_initialization() -> None:
    api = _api()
    first = api.PolicyValueNet(seed=731501)
    second = api.PolicyValueNet(seed=731501)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    policy, value = first(torch.zeros(2, 3, 9, 9))
    assert policy.shape == (2, 81)
    assert value.shape == (2,)
    assert torch.isfinite(policy).all()
    assert torch.all(value >= -1) and torch.all(value <= 1)


def test_encoding_and_prediction_mask_every_occupied_action() -> None:
    api = _api()
    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    board.place_stone("W", (3, 3))
    encoded = api.encode_board(board, "B")
    assert encoded.shape == (3, 9, 9)
    assert encoded[0, 4, 4] == 1
    assert encoded[1, 3, 3] == 1
    assert encoded[2, 4, 4] == 0
    model = api.PolicyValueNet(seed=731502)
    probabilities, value = api.predict_policy_value(model, board, "B")
    assert probabilities.shape == (81,)
    assert probabilities[4 * 9 + 4].item() == 0
    assert probabilities[3 * 9 + 3].item() == 0
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert -1 <= value <= 1


def test_full_board_prediction_rejects_no_legal_move() -> None:
    api = _api()
    board = GomokuBoard(9)
    for row in range(9):
        for col in range(9):
            board.place_stone("B" if (row + col) % 2 == 0 else "W", (row, col))
    with pytest.raises(ValueError, match="legal"):
        api.predict_policy_value(api.PolicyValueNet(), board, "B")


def test_checkpoint_round_trip_preserves_model_predictions(tmp_path) -> None:
    api = _api()
    model = api.PolicyValueNet(seed=731503)
    path = tmp_path / "f_checkpoint.pt"
    api.save_f_checkpoint(path, model, metadata={"dataset_version": "test-v1"})
    loaded = api.load_f_checkpoint(path, device="cpu")
    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    before = api.predict_policy_value(model, board, "W")
    after = api.predict_policy_value(loaded, board, "W")
    assert torch.allclose(before[0], after[0])
    assert before[1] == pytest.approx(after[1])
