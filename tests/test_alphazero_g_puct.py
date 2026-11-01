from __future__ import annotations

import numpy as np
import pytest
import torch

from src.gomoku import GomokuBoard


class FixedPolicyValue(torch.nn.Module):
    def __init__(self, preferred_action: int | None = None) -> None:
        super().__init__()
        self.preferred_action = preferred_action
        self.evaluations = 0

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.evaluations += 1
        logits = torch.zeros((planes.shape[0], 81), dtype=planes.dtype, device=planes.device)
        if self.preferred_action is not None:
            logits[:, self.preferred_action] = 20.0
        values = torch.zeros((planes.shape[0],), dtype=planes.dtype, device=planes.device)
        return logits, values


def test_backup_flips_value_view_at_each_parent() -> None:
    from src.research.alphazero_g.puct import PUCTNode, backup_path

    root = PUCTNode(prior=1.0, to_move="B")
    child = PUCTNode(prior=1.0, to_move="W")
    leaf = PUCTNode(prior=1.0, to_move="B")

    backup_path([root, child, leaf], leaf_value=1.0)

    assert (root.visit_count, child.visit_count, leaf.visit_count) == (1, 1, 1)
    assert root.value_sum == 1.0
    assert child.value_sum == -1.0
    assert leaf.value_sum == 1.0


def test_terminal_result_is_expressed_from_side_to_move_view() -> None:
    from src.research.alphazero_g.puct import terminal_value

    assert terminal_value(winner="B", to_move="W") == -1.0
    assert terminal_value(winner="W", to_move="W") == 1.0
    assert terminal_value(winner=None, to_move="B") == 0.0


def test_puct_masks_occupied_actions_and_restores_input_board() -> None:
    from src.research.alphazero_g.puct import PUCTSearch

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    board.place_stone("W", (4, 5))
    original = board.to_dict()
    result = PUCTSearch(FixedPolicyValue()).search(
        board, "B", 16, add_root_noise=False, temperature=1.0, seed=77
    )

    assert board.to_dict() == original
    assert board.get(result.move) is None
    assert result.simulations == 16
    assert result.root_visits.sum() == 16
    assert np.isclose(result.visit_policy.sum(), 1.0)
    assert result.visit_policy[4 * 9 + 4] == 0.0
    assert result.visit_policy[4 * 9 + 5] == 0.0
    assert np.all(result.root_priors[[4 * 9 + 4, 4 * 9 + 5]] == 0.0)
    assert np.all(np.isfinite(result.root_priors))


def test_root_dirichlet_noise_is_seeded_and_temperature_policy_is_valid() -> None:
    from src.research.alphazero_g.puct import PUCTSearch

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    plain = PUCTSearch(FixedPolicyValue()).search(
        board, "W", 12, add_root_noise=False, temperature=1.0, seed=19
    )
    noisy_a = PUCTSearch(FixedPolicyValue()).search(
        board, "W", 12, add_root_noise=True, temperature=1.0, seed=19
    )
    noisy_b = PUCTSearch(FixedPolicyValue()).search(
        board, "W", 12, add_root_noise=True, temperature=1.0, seed=19
    )

    assert np.isclose(noisy_a.root_priors.sum(), 1.0)
    assert np.all(noisy_a.root_priors >= 0.0)
    assert not np.allclose(plain.root_priors, noisy_a.root_priors)
    assert np.array_equal(noisy_a.root_priors, noisy_b.root_priors)
    assert np.array_equal(noisy_a.root_visits, noisy_b.root_visits)
    assert np.isclose(noisy_a.visit_policy.sum(), 1.0)
    assert np.isclose(noisy_a.temperature_zero_policy.sum(), 1.0)
    assert np.count_nonzero(noisy_a.temperature_zero_policy) == 1


def test_winning_terminal_child_is_backed_up_without_network_evaluation() -> None:
    from src.research.alphazero_g.puct import PUCTSearch

    board = GomokuBoard(9)
    for col in range(4):
        board.place_stone("B", (4, col))
        board.place_stone("W", (0, col))
    model = FixedPolicyValue(preferred_action=4 * 9 + 4)

    result = PUCTSearch(model).search(
        board, "B", 1, add_root_noise=False, temperature=0.0, seed=11
    )

    assert result.move == (4, 4)
    assert result.root_value == 1.0
    assert result.network_evaluations == 1


def test_search_rejects_terminal_root_and_zero_simulations() -> None:
    from src.research.alphazero_g.puct import PUCTSearch

    board = GomokuBoard(9)
    for col in range(5):
        board.place_stone("B", (4, col))
    search = PUCTSearch(FixedPolicyValue())
    with pytest.raises(ValueError, match="terminal"):
        search.search(board, "W", 1)
    with pytest.raises(ValueError, match="simulations"):
        search.search(GomokuBoard(9), "B", 0)
