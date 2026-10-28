from __future__ import annotations

from pathlib import Path

from src.gomoku import GomokuBoard
from src.research.v6.config import load_config as load_v6_config
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.schedule import build_schedule, derive_seed


ROOT = Path(__file__).resolve().parents[1]


def test_v6_2_config_pins_frozen_engines_and_fresh_common_opening() -> None:
    config = load_v6_2_config()
    v6 = load_v6_config()

    assert config["baseline_commit"] == "0c7e0e5881b8e0bcd2bdd8c07c247912ad86783d"
    assert config["board_size"] == 9
    assert config["time_limit_seconds"] == 200.0
    assert config["models"] == ["C", "D", "E5"]
    assert config["engine_sha256"] == v6["engine_sha256"]
    opening = config["opening"]
    assert [(move["player"], move["row"], move["col"]) for move in opening] == [
        ("B", 4, 4), ("W", 3, 5), ("B", 5, 3), ("W", 2, 6)
    ]
    board = GomokuBoard(9)
    for move in opening:
        coord = (move["row"], move["col"])
        board.place_stone(move["player"], coord)
        assert not board.check_victory(coord, move["player"])
    assert tuple((row, col) for _, row, col in v6["opening_catalog"]["731201"]) == tuple(
        (move["row"], move["col"]) for move in opening
    ) if "731201" in v6["opening_catalog"] else True
    assert config["score_rule"] == {"win": 1.0, "draw": 0.5, "loss": 0.0, "tie": "joint_champion"}


def test_v6_2_schedule_has_six_unique_paired_games_and_is_deterministic() -> None:
    config = load_v6_2_config()

    schedule = build_schedule(config)
    again = build_schedule(config)

    assert schedule == again
    assert len(schedule) == 6
    assert len({row["game_id"] for row in schedule}) == 6
    assert len({row["game_seed"] for row in schedule}) == 6
    assert {tuple(row["pair"]) for row in schedule} == {
        ("C", "D"), ("C", "E5"), ("D", "E5")
    }
    for pair in {tuple(row["pair"]) for row in schedule}:
        rows = [row for row in schedule if tuple(row["pair"]) == pair]
        assert len(rows) == 2
        assert rows[0]["opening"] == rows[1]["opening"] == config["opening"]
        assert rows[0]["opening_seed"] == rows[1]["opening_seed"] == int(config["opening_id"])
        assert rows[0]["color_swap"] is False and rows[1]["color_swap"] is True
        assert rows[0]["black_model"] == rows[1]["white_model"]
        assert rows[0]["white_model"] == rows[1]["black_model"]
        assert rows[0]["budget_seconds_by_model"] == rows[1]["budget_seconds_by_model"] == {
            pair[0]: 200.0,
            pair[1]: 200.0,
        }
    assert all(row["source_type"] == "new" for row in schedule)


def test_game_seed_derivation_is_stable_and_disambiguated() -> None:
    assert derive_seed(20261028, "game", "same-id") == derive_seed(20261028, "game", "same-id")
    assert derive_seed(20261028, "game", "same-id") != derive_seed(20261028, "game", "other-id")
