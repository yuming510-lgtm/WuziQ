from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.gomoku import GomokuBoard
def _api():
    name = "src.research.imitation_f.generation"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail("F V1 generation API is not implemented")
    return importlib.import_module(name)


def test_preregistered_schedule_is_color_paired_and_openings_are_legal() -> None:
    api = _api()
    config = api.build_generation_config(seed=731401, openings_per_pair=2)
    assert len(config["games"]) == 24
    assert len({game["game_id"] for game in config["games"]}) == 24
    for pair in config["pairings"]:
        for opening_seed in config["opening_seeds_by_pair"]["-".join(pair)]:
            games = [g for g in config["games"] if tuple(g["pair"]) == tuple(pair) and g["opening_seed"] == opening_seed]
            assert len(games) == 2
            assert games[0]["opening"] == games[1]["opening"]
            assert games[0]["black_model"] == games[1]["white_model"]
            assert games[0]["white_model"] == games[1]["black_model"]
            board = GomokuBoard(9)
            for index, move in enumerate(games[0]["opening"]):
                assert move["player"] == ("B" if index % 2 == 0 else "W")
                board.place_stone(move["player"], (move["row"], move["col"]))
                assert not board.check_victory((move["row"], move["col"]), move["player"])


def test_evaluation_openings_are_reproducible_and_exclude_training_groups() -> None:
    api = _api()
    config = api.build_generation_config(seed=731401, openings_per_pair=1)
    used = {game["opening_group_id"] for game in config["games"]}
    first = api.build_evaluation_openings(731901, 3, tuple(used))
    second = api.build_evaluation_openings(731901, 3, tuple(used))
    assert first == second
    assert not used.intersection(row["opening_group_id"] for row in first)
    assert len({row["opening_group_id"] for row in first}) == 3


def test_annotation_uses_frozen_search_and_returns_legal_labeled_action() -> None:
    api = _api()
    board = GomokuBoard(9)
    for col in range(4):
        board.place_stone("B", (4, col))
        board.place_stone("W", (0, col))
    annotation = api.annotate_position(board, "B", "C", 0.05, 731402)
    assert annotation["teacher"] == "C"
    assert 0 <= annotation["action_index"] < 81
    row, col = divmod(annotation["action_index"], 9)
    assert board.get((row, col)) is None
    assert annotation["nodes"] >= 0


def test_generation_is_resumable_and_deduplicates_completed_game_ids(tmp_path: Path) -> None:
    api = _api()
    spec = {
        "game_id": "unit-a-b",
        "pair": ["A", "B"],
        "black_model": "A",
        "white_model": "B",
        "board_size": 9,
        "opening_seed": 731403,
        "opening": [
            {"player": "B", "row": 4, "col": 4},
            {"player": "W", "row": 4, "col": 5},
            {"player": "B", "row": 5, "col": 3},
            {"player": "W", "row": 3, "col": 6},
        ],
        "game_seed": 731403,
    }
    config = {"seed": 731403, "budget_seconds": 0.0, "games": [spec]}
    frozen = {"splits": {"validation": [], "blind_test": []}, "opening_group_assignment": {}}
    first = api.run_generation(config, tmp_path, frozen)
    second = api.run_generation(config, tmp_path, frozen)
    assert first["completed_games"] == second["completed_games"] == 1
    records = [json.loads(line) for line in (tmp_path / "generated_games.jsonl").read_text().splitlines()]
    assert len(records) == 1
    api.validate_game_record(records[0])
    assert (tmp_path / "generated_annotations.jsonl").exists()
