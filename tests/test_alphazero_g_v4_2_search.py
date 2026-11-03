from __future__ import annotations

import numpy as np
import pytest
import torch

from src.gomoku import GomokuBoard


class _FixedModel(torch.nn.Module):
    def __init__(self, preferred_action: int = 0, value: float = 0.0) -> None:
        super().__init__()
        self.preferred_action = preferred_action
        self.value = value

    def forward(self, planes: torch.Tensor):
        logits = torch.zeros((planes.shape[0], 81), dtype=planes.dtype, device=planes.device)
        logits[:, self.preferred_action] = 12.0
        values = torch.full((planes.shape[0],), self.value, dtype=planes.dtype, device=planes.device)
        return logits, values


def test_neutral_search_controls_preserve_legacy_puct_results() -> None:
    from src.research.alphazero_g.puct import PUCTConfig, PUCTSearch

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    default = PUCTSearch(_FixedModel()).search(board, "W", 20, seed=91)
    neutral = PUCTSearch(_FixedModel(), PUCTConfig(
        c_puct=1.5, prior_smoothing_fraction=0.0, leaf_value_scale=1.0
    )).search(board, "W", 20, seed=91)

    assert default.move == neutral.move
    assert np.array_equal(default.root_visits, neutral.root_visits)
    assert np.array_equal(default.root_priors, neutral.root_priors)
    assert default.root_value == neutral.root_value


def test_prior_smoothing_mixes_uniform_mass_only_into_legal_actions() -> None:
    from src.research.alphazero_g.puct import PUCTConfig, PUCTSearch

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    result = PUCTSearch(_FixedModel(preferred_action=0), PUCTConfig(
        prior_smoothing_fraction=0.1
    )).search(board, "W", 4, seed=17)

    legal_count = 80
    assert np.isclose(result.root_priors.sum(), 1.0)
    assert result.root_priors[4 * 9 + 4] == 0.0
    assert np.all(result.root_priors[np.arange(81) != 4 * 9 + 4] >= 0.1 / legal_count - 1e-12)
    assert result.root_priors[0] < PUCTSearch(_FixedModel(preferred_action=0)).search(
        board, "W", 4, seed=17
    ).root_priors[0]


def test_leaf_value_scale_changes_network_evaluation_but_not_terminal_result() -> None:
    from src.research.alphazero_g.puct import PUCTConfig, PUCTSearch

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    unscaled = PUCTSearch(_FixedModel(value=0.75)).search(board, "W", 1, seed=5)
    zeroed = PUCTSearch(_FixedModel(value=0.75), PUCTConfig(
        leaf_value_scale=0.0
    )).search(board, "W", 1, seed=5)
    assert unscaled.root_value == pytest.approx(-0.75)
    assert zeroed.root_value == pytest.approx(0.0)

    winning = GomokuBoard(9)
    for col in range(4):
        winning.place_stone("B", (4, col))
        winning.place_stone("W", (0, col))
    terminal = PUCTSearch(_FixedModel(preferred_action=4 * 9 + 4), PUCTConfig(
        leaf_value_scale=0.0
    )).search(winning, "B", 1, seed=5)
    assert terminal.move == (4, 4)
    assert terminal.root_value == 1.0


def test_frozen_match_engine_can_use_distinct_attacker_and_opponent_search_configs() -> None:
    from src.research.imitation_f.model import PolicyValueNet
    from src.research.alphazero_g.evaluation import play_evaluation_match
    from src.research.alphazero_g.puct import PUCTConfig

    model = PolicyValueNet(board_size=9, channels=32, blocks=2, seed=17)
    attacker = PUCTConfig(c_puct=5.0)
    defender = PUCTConfig(leaf_value_scale=0.0)
    result = play_evaluation_match(
        model, game_id="g-v4-2-distinct-search-configs", opening_group_id="test-open",
        arm_id="frozen-model", mode="puct", opponent="MODEL", opponent_mode="puct",
        opponent_model=model, opening=[], agent_color="B", seed=2, simulations=1,
        opponent_simulations=1, config=attacker, opponent_config=defender,
    )

    assert result["search_config"]["c_puct"] == 5.0
    assert result["opponent_search_config"]["leaf_value_scale"] == 0.0


@pytest.mark.parametrize("field,value", [("prior_smoothing_fraction", -0.1),
                                          ("prior_smoothing_fraction", 1.1),
                                          ("leaf_value_scale", -0.1),
                                          ("leaf_value_scale", 1.1)])
def test_search_calibration_rejects_out_of_range_values(field: str, value: float) -> None:
    from src.research.alphazero_g.puct import PUCTConfig

    with pytest.raises(ValueError):
        PUCTConfig(**{field: value})
