from __future__ import annotations

from src.research.alphazero_g.v4_1_runner import budgeted_game_prefix, deterministic_game_seed


def test_v4_1_budgeted_prefix_preserves_full_game_and_accepts_exact_rows() -> None:
    game = {
        "moves": [{"ply": index} for index in range(7)],
        "opening": [{"ply": 0}],
        "experience_rows": [{"sample_id": str(index)} for index in range(6)],
        "game_id": "g-1",
    }

    accepted, replay_game = budgeted_game_prefix(game, remaining_states=4)

    assert [row["sample_id"] for row in accepted] == ["0", "1", "2", "3"]
    assert replay_game["budgeted_environment_steps"] == 4
    assert replay_game["completion_overrun_steps"] == 2
    assert len(replay_game["experience_rows"]) == 4
    assert len(game["experience_rows"]) == 6


def test_v4_1_game_seeds_are_deterministic_and_unique_by_game_index() -> None:
    first = deterministic_game_seed(7, 123456)
    second = deterministic_game_seed(7, 123456)

    assert first == second
    assert 0 <= first < 2**32
    assert first != deterministic_game_seed(8, 123456)
