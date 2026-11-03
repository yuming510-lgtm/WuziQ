from __future__ import annotations

import numpy as np
import torch

from src.research.alphazero_g.v4_2_training import (
    V4SampleReplay,
    play_history_pool_game,
    prefix_selfplay_game,
)
from src.research.alphazero_g.puct import PUCTConfig
from src.research.imitation_f.model import PolicyValueNet
from src.research.alphazero_g.training import GTrainer, load_training_state, save_training_state


class _WinNow(torch.nn.Module):
    def forward(self, planes):
        logits = torch.zeros((planes.shape[0], 81), device=planes.device)
        logits[:, 40] = 20.0
        return logits, torch.zeros((planes.shape[0],), device=planes.device)


def _winning_opening():
    return [
        {"player": "B", "row": 4, "col": 0}, {"player": "W", "row": 0, "col": 0},
        {"player": "B", "row": 4, "col": 1}, {"player": "W", "row": 0, "col": 2},
        {"player": "B", "row": 4, "col": 2}, {"player": "W", "row": 0, "col": 4},
        {"player": "B", "row": 4, "col": 3}, {"player": "W", "row": 0, "col": 6},
    ]


def test_history_pool_game_logs_only_current_g0_learner_targets_and_checks_replay():
    game = play_history_pool_game(
        _WinNow(), _WinNow(), game_id="pool-game-1", opening_group_id="opening-1",
        arm_id="history_pool", opening=_winning_opening(), learner_color="B",
        simulations=1, seed=9, config=PUCTConfig(),
    )

    assert game["winner_color"] == "B"
    assert game["stats"]["simulations"] == 1
    assert len(game["experience_rows"]) == 1
    assert game["experience_rows"][0]["value_target"] == 1.0
    assert game["experience_rows"][0]["player"] == "B"


def test_v4_replay_deduplicates_games_and_restores_rng_state():
    row = {"source": "self_play", "game_id": "g1", "policy_target": np.ones(81).tolist()}
    replay = V4SampleReplay(20, seed=3)
    game = {"game_id": "g1", "winner_color": "B", "arm_id": "baseline"}
    assert replay.add_game(game, [row]) is True
    assert replay.add_game(game, [row]) is False
    saved = replay.state_dict()
    expected = replay.sample(3)
    restored = V4SampleReplay(20, seed=77)
    restored.load_state_dict(saved)
    assert restored.sample(3) == expected
    assert restored.position_count == 1


def test_state_budget_prefix_keeps_complete_game_metadata_and_exact_row_count():
    rows = [{"sample_id": f"g:ply:{index}"} for index in range(3)]
    game = {"game_id": "g", "moves": [{} for _ in range(11)], "opening": [{} for _ in range(8)],
            "experience_rows": rows, "completed": True}
    prefix = prefix_selfplay_game(game, 2)
    assert len(prefix["experience_rows"]) == 2
    assert prefix["replay_prefix"] is True
    assert prefix["budgeted_environment_steps"] == 2
    assert prefix["completion_overrun_steps"] == 1
    assert len(game["experience_rows"]) == 3


def test_v4_sample_replay_trains_and_restores_optimizer_with_exact_resume_state(tmp_path):
    policy = np.zeros(81, dtype=np.float32)
    policy[4 * 9 + 4] = 1.0
    row = {
        "source": "self_play", "arm_id": "baseline", "game_id": "train-game",
        "sample_id": "train-game:ply:0", "opening_group_id": "open-0", "ply": 0,
        "player": "B", "canonical_board": "." * 81, "policy_target": policy.tolist(),
        "value_target": 0.0, "game_complete": True,
    }
    replay = V4SampleReplay(100, seed=88)
    replay.add_game({"game_id": "train-game", "winner_color": None, "arm_id": "baseline"}, [row])
    trainer = GTrainer(PolicyValueNet(board_size=9, channels=32, blocks=2, seed=18),
                       seed=99, expected_updates=1, learning_rate=1e-4, bn_mode="freeze")
    result = trainer.train_updates(replay, updates=1, batch_size=2)
    assert result["updates"] == 1
    assert np.isfinite(result["total_loss"])

    state_path = tmp_path / "training.pt"
    saved_rng = trainer.rng.bit_generator.state
    save_training_state(state_path, trainer, replay, progress={"training_states": 1})
    restored = GTrainer(PolicyValueNet(board_size=9, channels=32, blocks=2, seed=22),
                        seed=1, expected_updates=1, learning_rate=1e-4, bn_mode="freeze")
    restored_replay = V4SampleReplay(100, seed=1)
    progress = load_training_state(state_path, restored, restored_replay)
    assert progress["training_states"] == 1
    assert restored.update_count == 1
    assert restored_replay.position_count == 1
    assert restored.rng.bit_generator.state == saved_rng
