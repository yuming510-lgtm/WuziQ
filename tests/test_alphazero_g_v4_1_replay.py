from __future__ import annotations

import numpy as np
import torch

from src.gomoku import GomokuBoard
from src.research.alphazero_g.replay import ReplayBuffer
from src.research.alphazero_g.training import GTrainer, initialize_arm


def _valid_game(game_id: str, winner: str, filler_pairs: int = 0) -> dict:
    moves: list[tuple[str, int, int]] = []
    for index in range(filler_pairs):
        moves.extend([("B", 8, index * 2), ("W", 7, index * 2 + 1)])
    if winner == "B":
        for col in range(4):
            moves.extend([("B", 4, col), ("W", 0, col)])
        moves.append(("B", 4, 4))
    else:
        for col in range(4):
            moves.extend([("B", 0, col), ("W", 4, col)])
        moves.extend([("B", 1, 8), ("W", 4, 4)])
    board = GomokuBoard(9)
    rows = []
    recorded_moves = []
    for ply, (player, row, col) in enumerate(moves):
        cells = "".join(
            "1" if cell == player else "2" if cell is not None else "."
            for line in board.grid for cell in line
        )
        legal = np.fromiter((cell == "." for cell in cells), dtype=np.float32)
        policy = legal / legal.sum()
        rows.append({
            "source": "self_play", "arm_id": "G0-test", "game_id": game_id,
            "sample_id": f"{game_id}:ply:{ply}", "opening_group_id": "test-open",
            "ply": ply, "player": player, "canonical_board": cells,
            "policy_target": policy.tolist(),
            "value_target": 1.0 if player == winner else -1.0,
            "game_complete": True,
        })
        board.place_stone(player, (row, col))
        recorded_moves.append({"player": player, "row": row, "col": col, "opening": False, "simulations": 1})
    assert board.check_victory((moves[-1][1], moves[-1][2]), winner)
    return {
        "source": "self_play", "arm_id": "G0-test", "game_id": game_id,
        "opening_group_id": "test-open", "opening": [], "seed": 1,
        "simulations_per_move": 1, "completed": True, "termination": "five_in_row",
        "winner_color": winner, "outcome": f"{winner}_win", "move_count": len(moves),
        "moves": recorded_moves, "experience_rows": rows,
        "stats": {"simulations": len(moves)},
    }


def _replay(seed: int = 91) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=128, seed=seed)
    replay.add_complete_game(_valid_game("b-short", "B"))
    replay.add_complete_game(_valid_game("w-short", "W"))
    return replay


def test_state_sampling_is_uniform_over_rows_and_default_remains_game_uniform() -> None:
    replay = ReplayBuffer(capacity=128, seed=11)
    replay.add_complete_game(_valid_game("b-short", "B"))
    replay.add_complete_game(_valid_game("b-long", "B", filler_pairs=4))
    game_batch = replay.sample(2000)
    state_batch = replay.sample(2000, mode="uniform_state")

    game_short_rate = np.mean([row["game_id"] == "b-short" for row in game_batch])
    state_short_rate = np.mean([row["game_id"] == "b-short" for row in state_batch])
    assert 0.43 <= game_short_rate <= 0.57
    assert 0.30 <= state_short_rate <= 0.39


def test_winner_balanced_sampling_equalizes_non_draw_winner_games() -> None:
    replay = ReplayBuffer(capacity=512, seed=12)
    for index in range(4):
        replay.add_complete_game(_valid_game(f"b-{index}", "B"))
    replay.add_complete_game(_valid_game("w-0", "W"))

    batch = replay.sample(5000, mode="winner_balanced")
    black_winner_rate = np.mean([row["game_id"].startswith("b-") for row in batch])

    assert 0.46 <= black_winner_rate <= 0.54


def test_replay_sampling_modes_are_seed_reproducible() -> None:
    first = _replay(seed=13).sample(20, mode="winner_balanced")
    second = _replay(seed=13).sample(20, mode="winner_balanced")

    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]


def test_gtrainer_accepts_and_uses_explicit_replay_sampling_mode(monkeypatch) -> None:
    replay = _replay(seed=14)
    observed: list[str] = []
    original = replay.sample

    def sample(batch_size: int, *, mode: str = "uniform_game"):
        observed.append(mode)
        return original(batch_size, mode=mode)

    monkeypatch.setattr(replay, "sample", sample)
    trainer = GTrainer(initialize_arm("G0", seed=15), seed=16, learning_rate=1e-4)

    trainer.train_updates(replay, updates=1, batch_size=2, sampling_mode="winner_balanced")

    assert observed == ["winner_balanced"]
