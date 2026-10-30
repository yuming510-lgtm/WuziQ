from __future__ import annotations

import importlib
import importlib.util

import pytest

from src.gomoku import GomokuBoard


def _api():
    name = "src.research.imitation_f.evaluation"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail("F V1 evaluation API is not implemented")
    return importlib.import_module(name)


def test_direct_win_and_unique_block_are_detected_by_exact_rules() -> None:
    api = _api()
    winning = GomokuBoard(9)
    for col in range(4):
        winning.place_stone("B", (4, col))
        winning.place_stone("W", (0, col))
    assert api.find_immediate_wins(winning, "B") == [(4, 4)]

    blocking = GomokuBoard(9)
    for col in range(4):
        blocking.place_stone("W", (4, col))
    for col in (0, 2, 4, 6):
        blocking.place_stone("B", (0, col))
    assert api.find_immediate_wins(blocking, "W") == [(4, 4)]
    row = {"canonical_board": "." * 81, "to_move": "B"}
    board_text = ["."] * 81
    for col in range(4):
        board_text[4 * 9 + col] = "2"
    for col in (0, 2, 4, 6):
        board_text[0 * 9 + col] = "1"
    row["canonical_board"] = "".join(board_text)
    tactics = api.extract_direct_tactics([row])
    assert len(tactics) == 1
    assert tactics[0]["kind"] == "block"
    assert tactics[0]["correct_actions"] == [4 * 9 + 4]


def test_eval_reports_top1_value_and_legal_rate_for_a_small_blind_set() -> None:
    api = _api()
    from src.research.imitation_f.model import PolicyValueNet

    model = PolicyValueNet(channels=8, blocks=1, seed=731801)
    board_text = ["."] * 81
    board_text[40] = "1"
    target = [0.0] * 81
    target[0] = 1.0
    rows = [{
        "state_key": "p0", "canonical_board": "".join(board_text),
        "policy_target": target, "value_target": 1.0, "value_count": 1,
    }]
    metrics = api.evaluate_policy(model, rows, device="cpu")
    assert 0 <= metrics["top1_accuracy"] <= 1
    assert 0 <= metrics["target_probability_mass"] <= 1
    assert metrics["legal_rate"] == 1
    assert metrics["value_mae"] >= 0
    assert metrics["inference_latency_ms"]["median"] > 0


def test_tactical_evaluation_counts_only_direct_win_and_force_block_positions() -> None:
    api = _api()
    from src.research.imitation_f.model import PolicyValueNet

    win_board = ["."] * 81
    for col in range(4):
        win_board[4 * 9 + col] = "1"
    block_board = ["."] * 81
    for col in range(4):
        block_board[4 * 9 + col] = "2"
    cases = api.extract_direct_tactics([
        {"canonical_board": "".join(win_board), "to_move": "B"},
        {"canonical_board": "".join(block_board), "to_move": "B"},
    ])
    report = api.evaluate_tactics(PolicyValueNet(channels=8, blocks=1, seed=731802), cases, device="cpu")
    assert report["positions"] == 2
    assert report["by_kind"]["win"]["positions"] == 1
    assert report["by_kind"]["block"]["positions"] == 1
    assert report["correct"] <= 2


def test_color_paired_match_uses_full_legal_games_and_unique_ids() -> None:
    api = _api()
    from src.research.imitation_f.model import PolicyValueNet

    opening = [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 4, "col": 5},
        {"player": "B", "row": 5, "col": 3},
        {"player": "W", "row": 3, "col": 6},
    ]
    model = PolicyValueNet(channels=8, blocks=1, seed=731803)
    black = api.play_match(model, "A", opening, "B", 731901)
    white = api.play_match(model, "A", opening, "W", 731901)
    assert black["game_id"] != white["game_id"]
    assert black["opening"] == white["opening"]
    assert {black["black_model"], white["white_model"]} == {"F"}
    assert all(move["nodes"] == 0 for move in black["moves"] if move["model"] == "F")
    assert black["move_count"] == len(black["moves"])
    assert white["move_count"] == len(white["moves"])
