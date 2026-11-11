from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.ai import SearchConfig
from src.research.alphazero_g import v6_3_evaluation as evaluation
from src.research.alphazero_g.v6_3_training import (
    ARTIFACT, BASE_GAMES, BASE_STATES, BASE_UPDATES, ROOT, build_opening_schedule,
    prepare_experiment, validate_training_state,
)


def test_frozen_g2500_state_restores_all_training_components():
    prepared = prepare_experiment(ROOT)
    validation = validate_training_state(
        ROOT / ARTIFACT / "checkpoints/reference400_empty/training_state.pt",
        expected_games=BASE_GAMES,
    )
    assert validation["environment_states"] == BASE_STATES
    assert validation["optimizer_updates"] == BASE_UPDATES
    assert all(validation[key] for key in (
        "model_restored", "adam_restored", "replay_restored", "python_rng_restored",
        "numpy_rng_restored", "torch_rng_restored",
    ))
    assert prepared["openings"]["historical_d4_overlap"] == 0


def test_opening_schedule_has_40_legal_fresh_paired_positions():
    schedule = build_opening_schedule(ROOT, seed=630603, groups=40)
    assert len(schedule["openings"]) == 40
    assert schedule["validation"] == {
        "all_openings_legal_and_nonterminal": True,
        "unique_d4_openings": 40,
        "historical_d4_overlap": 0,
        "same_openings_for_all_models_and_opponents": True,
    }


@pytest.mark.parametrize(("identity", "budget"), [("A", 1e9), ("B", 0.05), ("C", 0.05)])
def test_original_a_minimax_b_and_alpha_beta_c_keep_their_engine_identity(identity, budget):
    from src.gomoku import GomokuBoard

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    fake_result = SimpleNamespace(
        move=(4, 5), simulations=0, completed_depth=2, nodes=17,
        cutoffs=3, elapsed_seconds=0.05, timed_out=True,
    )
    with patch.object(evaluation, "choose_move", return_value=fake_result) as choose:
        move, metrics = evaluation._agent_choice({"kind": identity},
                                                  evaluation.make_board(9, 5, 0),
                                                  "W", 17)
    assert move == (4, 5)
    assert choose.call_args.args[2] == identity
    config = choose.call_args.args[3]
    assert isinstance(config, SearchConfig)
    assert config.time_limit_seconds == budget
    assert metrics["search_depth"] == 2
    assert metrics["cutoffs"] == 3
    assert metrics["timed_out"] is True


def test_duplicate_challenge_game_requires_same_registered_pair():
    opening = {"opening_group_id": "open-1", "seed": 77}
    game = {
        "game_id": "v63-example", "candidate": "G2500", "opponent": "C",
        "opening_group_id": "open-1", "candidate_color": "B",
        "simulations_per_candidate_turn": 400, "moves": [{"row": 0, "col": 0}],
        "move_count": 1, "termination": "five_in_row",
    }
    evaluation._validate_existing(game, game_id="v63-example", candidate="G2500",
                                  opponent="C", opening=opening, color="B")
    with pytest.raises(ValueError, match="collision"):
        evaluation._validate_existing(game, game_id="v63-example", candidate="G2500",
                                      opponent="B", opening=opening, color="B")
