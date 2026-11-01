from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.research.alphazero_g.puct import PUCTConfig
from src.research.alphazero_g.selfplay import play_selfplay_game
from src.research.alphazero_g.replay import (
    ReplayBuffer,
    append_complete_game,
    augment_sample,
    budgeted_prefix,
    load_game_log,
)


class FixedPolicyValue(torch.nn.Module):
    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros((planes.shape[0], 81), dtype=planes.dtype, device=planes.device)
        values = torch.zeros((planes.shape[0],), dtype=planes.dtype, device=planes.device)
        return logits, values


def _opening() -> list[dict]:
    return [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 3, "col": 5},
        {"player": "W", "row": 5, "col": 3},
    ]


def test_selfplay_records_a_complete_legal_game_and_terminal_value_targets() -> None:
    model = FixedPolicyValue()
    game = play_selfplay_game(
        model,
        _opening(),
        arm_id="G0",
        game_id="g0-test-0001",
        opening_group_id="train-open-01",
        simulations=2,
        seed=5101,
        config=PUCTConfig(),
        temperature_moves=12,
    )

    assert game["completed"] is True
    assert game["source"] == "self_play"
    assert game["termination"] in {"five_in_row", "full_board"}
    assert game["move_count"] == len(game["moves"])
    assert len(game["experience_rows"]) == game["move_count"] - len(_opening())
    assert all(row["source"] == "self_play" and row["game_complete"] for row in game["experience_rows"])
    for row in game["experience_rows"]:
        assert row["value_target"] == (0.0 if game["winner_color"] is None else
                                        (1.0 if row["player"] == game["winner_color"] else -1.0))
        policy = np.asarray(row["policy_target"])
        assert np.isclose(policy.sum(), 1.0)
        occupied = np.fromiter((cell != "." for cell in row["canonical_board"]), bool, count=81)
        assert np.all(policy[occupied] == 0.0)
    assert all(move["simulations"] == 2 for move in game["moves"] if not move["opening"])


def test_selfplay_game_id_and_moves_are_reproducible_for_same_seed() -> None:
    kwargs = {
        "opening": _opening(), "arm_id": "G-F2", "game_id": "gf2-test-0042",
        "opening_group_id": "train-open-02", "simulations": 2, "seed": 5102,
        "config": PUCTConfig(), "temperature_moves": 12,
    }
    first = play_selfplay_game(FixedPolicyValue(), **kwargs)
    second = play_selfplay_game(FixedPolicyValue(), **kwargs)

    assert [(row["row"], row["col"]) for row in first["moves"]] == [
        (row["row"], row["col"]) for row in second["moves"]
    ]
    assert [row["policy_target"] for row in first["experience_rows"]] == [
        row["policy_target"] for row in second["experience_rows"]
    ]


def test_replay_accepts_only_complete_selfplay_rows_and_deduplicates_games() -> None:
    game = play_selfplay_game(
        FixedPolicyValue(), _opening(), arm_id="G0", game_id="g0-replay-01",
        opening_group_id="train-open-03", simulations=1, seed=5103,
    )
    replay = ReplayBuffer(capacity=512, seed=5104)

    assert replay.add_complete_game(game) is True
    assert replay.add_complete_game(game) is False
    batch = replay.sample(min(4, len(game["experience_rows"])))
    assert batch
    assert all(row["source"] == "self_play" and row["arm_id"] == "G0" for row in batch)
    assert replay.position_count == len(game["experience_rows"])

    corrupted = copy.deepcopy(game)
    corrupted["experience_rows"][0]["source"] = "f_supervised"
    with pytest.raises(ValueError, match="self-play"):
        ReplayBuffer(capacity=512, seed=5105).add_complete_game(corrupted)


def test_replay_applies_same_d4_transform_to_board_and_visit_policy() -> None:
    board = ["."] * 81
    board[1 * 9 + 2] = "1"
    board[2 * 9 + 5] = "2"
    policy = [0.0] * 81
    policy[0 * 9 + 6] = 1.0
    sample = {"canonical_board": "".join(board), "policy_target": policy, "source": "self_play"}

    augmented = augment_sample(sample, rotations=1, reflect=True)

    assert augmented["canonical_board"] != sample["canonical_board"]
    assert np.isclose(sum(augmented["policy_target"]), 1.0)
    assert augmented["policy_target"].index(1.0) == 2 * 9 + 8
    assert augmented["canonical_board"].count("1") == 1
    assert augmented["canonical_board"].count("2") == 1


def test_equal_step_budget_keeps_exact_prefix_and_reports_terminal_tail() -> None:
    rows = [{"sample_id": f"game:ply:{index}"} for index in range(7)]

    accepted, completion_overrun = budgeted_prefix(rows, remaining_steps=5)

    assert len(accepted) == 5
    assert completion_overrun == 2
    assert budgeted_prefix(rows, remaining_steps=0) == ([], 7)


def test_selfplay_rejects_illegal_or_terminal_opening() -> None:
    with pytest.raises(ValueError, match="opening"):
        play_selfplay_game(
            FixedPolicyValue(), [{"player": "B", "row": 9, "col": 0}],
            arm_id="G0", game_id="bad-opening", opening_group_id="bad",
            simulations=1, seed=5106,
        )


def test_game_log_repairs_partial_tail_and_appends_unique_complete_games(tmp_path) -> None:
    game = play_selfplay_game(
        FixedPolicyValue(), _opening(), arm_id="G0", game_id="g0-log-01",
        opening_group_id="train-open-04", simulations=1, seed=5107,
    )
    path = tmp_path / "games.jsonl"
    seen: set[str] = set()

    assert append_complete_game(path, game, seen) is True
    assert append_complete_game(path, game, seen) is False
    with path.open("ab") as handle:
        handle.write(b'{"game_id":"interrupted')

    records, repaired_ids = load_game_log(path)

    assert len(records) == 1
    assert repaired_ids == {"g0-log-01"}
    assert path.read_bytes().endswith(b"\n")

