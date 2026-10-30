from __future__ import annotations

import importlib
import importlib.util
import pytest

from src.gomoku import GomokuBoard


def _dataset_api():
    module_name = "src.research.imitation_f.dataset"
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail("F V1 dataset API is not implemented")
    return importlib.import_module(module_name)


def _game(game_id: str, black_model: str, white_model: str, black_finish: bool) -> dict:
    prefix = [
        ("B", 4, 0), ("W", 0, 0),
        ("B", 4, 1), ("W", 0, 1),
        ("B", 4, 2), ("W", 0, 2),
        ("B", 4, 3), ("W", 0, 3),
    ]
    if black_finish:
        tail = [("B", 4, 4)]
        winner_color, winner_model = "B", black_model
    else:
        tail = [("B", 5, 5), ("W", 0, 4)]
        winner_color, winner_model = "W", white_model
    moves = []
    for ply, (player, row, col) in enumerate(prefix + tail):
        opening = ply < 2
        model = None if opening else (black_model if player == "B" else white_model)
        moves.append({
            "ply": ply, "player": player, "model": model,
            "row": row, "col": col, "opening": opening,
        })
    return {
        "game_id": game_id,
        "board_size": 9,
        "black_model": black_model,
        "white_model": white_model,
        "opening": [
            {"player": player, "row": row, "col": col}
            for player, row, col in prefix[:2]
        ],
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "b_win" if winner_color == "B" else "w_win",
        "termination": "five_in_row",
    }


def test_game_replay_accepts_a_complete_five_in_row_record() -> None:
    api = _dataset_api()
    api.validate_game_record(_game("c-win", "C", "E5", True))


def test_game_replay_rejects_a_move_after_the_board_is_terminal() -> None:
    api = _dataset_api()
    record = _game("c-win-extra", "C", "E5", True)
    record["moves"].append({
        "ply": 10, "player": "W", "model": "E5",
        "row": 8, "col": 8, "opening": False,
    })
    record["move_count"] += 1
    with pytest.raises(ValueError, match="terminal|move_count|record"):
        api.validate_game_record(record)


def test_game_replay_rejects_wrong_winner_and_out_of_turn_records() -> None:
    api = _dataset_api()
    wrong_winner = _game("wrong-winner", "C", "E5", True)
    wrong_winner["winner_color"] = "W"
    with pytest.raises(ValueError, match="winner|outcome"):
        api.validate_game_record(wrong_winner)

    out_of_turn = _game("out-of-turn", "C", "E5", True)
    out_of_turn["moves"][3]["player"] = "B"
    with pytest.raises(ValueError, match="out-of-turn"):
        api.validate_game_record(out_of_turn)


def test_d4_canonicalization_transforms_the_label_with_the_board() -> None:
    api = _dataset_api()
    board = GomokuBoard(9)
    board.place_stone("B", (0, 1))
    board.place_stone("W", (3, 4))
    state_a, action_a = api.canonicalize_position(board, "B", (2, 6))

    rotated = GomokuBoard(9)
    rotated.place_stone("B", (1, 8))
    rotated.place_stone("W", (4, 5))
    state_b, action_b = api.canonicalize_position(rotated, "B", (6, 6))

    assert state_a == state_b
    assert action_a == action_b


def test_color_swap_is_canonical_for_the_current_player_perspective() -> None:
    api = _dataset_api()
    board = GomokuBoard(9)
    board.place_stone("B", (1, 2))
    board.place_stone("W", (6, 7))
    state, action = api.canonicalize_position(board, "B", (4, 3))
    swapped = GomokuBoard(9)
    swapped.place_stone("W", (1, 2))
    swapped.place_stone("B", (6, 7))
    swapped_state, swapped_action = api.canonicalize_position(swapped, "W", (4, 3))
    assert state == swapped_state
    assert action == swapped_action


def test_same_state_teacher_conflict_becomes_a_normalized_soft_target() -> None:
    api = _dataset_api()
    c_game = _game("c-label", "C", "E5", True)
    e5_game = _game("e5-label", "E5", "C", False)

    rows, _ = api.records_to_positions([c_game, e5_game])
    state = GomokuBoard(9)
    for player, row, col in [
        ("B", 4, 0), ("W", 0, 0), ("B", 4, 1), ("W", 0, 1),
        ("B", 4, 2), ("W", 0, 2), ("B", 4, 3), ("W", 0, 3),
    ]:
        state.place_stone(player, (row, col))
    state_key, _ = api.canonicalize_position(state, "B")
    target = next(row for row in rows if row["state_key"] == state_key)

    nonzero = [value for value in target["policy_target"] if value > 0]
    assert len(nonzero) == 2
    assert sorted(nonzero) == [0.5, 0.5]
    assert target["value_target"] == 0.0


def test_exact_duplicate_game_is_counted_once() -> None:
    api = _dataset_api()
    game = _game("same-a", "C", "E5", True)
    duplicate = dict(game, game_id="same-b")
    rows, stats = api.records_to_positions([game, duplicate])
    assert stats["duplicate_games"] == 1
    assert stats["unique_games"] == 1
    assert sum(row["policy_count"] for row in rows) == 7


def test_records_with_same_symmetric_position_are_merged_before_split() -> None:
    api = _dataset_api()
    game = _game("base", "C", "E5", True)
    mirrored = _game("mirror", "C", "E5", True)
    # Reflect all moves and the opening across the vertical centerline.
    for move in mirrored["moves"]:
        move["col"] = 8 - move["col"]
    for move in mirrored["opening"]:
        move["col"] = 8 - move["col"]
    rows, _ = api.records_to_positions([game, mirrored])
    opening_board = GomokuBoard(9)
    for move in game["opening"]:
        opening_board.place_stone(move["player"], (move["row"], move["col"]))
    state, _ = api.canonicalize_position(opening_board, "B")
    assert sum(row["state_key"] == state for row in rows) == 1
    row = next(row for row in rows if row["state_key"] == state)
    assert row["policy_count"] == 2
    assert len(row["opening_group_ids"]) == 1


def test_opening_components_and_symmetric_states_never_cross_splits() -> None:
    api = _dataset_api()
    rows = [
        {"state_key": "shared", "opening_group_ids": ["g0", "g1"], "policy_target": [1.0]},
        {"state_key": "s1", "opening_group_ids": ["g0"], "policy_target": [1.0]},
        {"state_key": "s2", "opening_group_ids": ["g1"], "policy_target": [1.0]},
        {"state_key": "s3", "opening_group_ids": ["g2"], "policy_target": [1.0]},
        {"state_key": "s4", "opening_group_ids": ["g3"], "policy_target": [1.0]},
        {"state_key": "s5", "opening_group_ids": ["g4"], "policy_target": [1.0]},
    ]

    split = api.split_positions(rows, seed=731301, validation_min=1, test_min=1)
    assert all(split[name] for name in ("train", "validation", "blind_test"))
    keys = [{row["state_key"] for row in split[name]} for name in ("train", "validation", "blind_test")]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    group_splits = {
        group: name
        for name in ("train", "validation", "blind_test")
        for row in split[name]
        for group in row["opening_group_ids"]
    }
    assert group_splits["g0"] == group_splits["g1"]
    assert len(group_splits) == sum(len({g for row in split[name] for g in row["opening_group_ids"]})
                                    for name in ("train", "validation", "blind_test"))


def test_generated_annotations_only_extend_train_and_merge_policy_value_counts() -> None:
    api = _dataset_api()
    train_key = "P:" + "." * 40 + "1" + "." * 40
    heldout_key = "P:" + "." * 39 + "2" + "." * 41
    train_policy = [0.0] * 81
    train_policy[3] = 1.0
    rows = [{
        "state_key": train_key, "canonical_board": "." * 40 + "1" + "." * 40,
        "to_move": "B", "policy_target": train_policy,
        "policy_action_counts": {"3": 1}, "policy_count": 1,
        "value_target": 1.0, "value_count": 1,
        "opening_group_ids": ["train-group"], "source_game_ids": [],
        "teacher_counts": {}, "teacher_elapsed_seconds": [],
    }, {
        "state_key": heldout_key, "canonical_board": "." * 39 + "2" + "." * 41,
        "to_move": "B", "policy_target": [1.0] + [0.0] * 80,
        "policy_action_counts": {"0": 1}, "policy_count": 1,
        "value_target": 0.0, "value_count": 1,
        "opening_group_ids": ["blind-group"], "source_game_ids": [],
        "teacher_counts": {}, "teacher_elapsed_seconds": [],
    }]
    annotations = [
        {"state_key": train_key, "opening_group_id": "generated-train", "to_move": "B",
         "action_index": 4, "value_target": -1.0, "value_count": 1, "teacher": "C"},
        {"state_key": heldout_key, "opening_group_id": "generated-other", "to_move": "B",
         "action_index": 1, "value_target": 1.0, "value_count": 1, "teacher": "E5"},
        {"state_key": "P:" + "." * 80 + "1", "opening_group_id": "generated-new", "to_move": "B",
         "action_index": 8, "value_target": 0.0, "value_count": 1, "teacher": "C"},
    ]
    import json
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        annotation_path = Path(directory) / "annotations.jsonl"
        annotation_path.write_text("".join(json.dumps(row) + "\n" for row in annotations))
        assignment = {"train-group": "train", "blind-group": "blind_test"}
        merged, accepted = api._merge_annotations(rows, annotation_path, assignment)
    updated = next(row for row in merged if row["state_key"] == train_key)
    assert accepted == 2
    assert assignment["generated-train"] == "train"
    assert updated["policy_count"] == 2
    assert updated["policy_target"][3] == pytest.approx(0.5)
    assert updated["policy_target"][4] == pytest.approx(0.5)
    assert updated["value_target"] == pytest.approx(0.0)
    assert not any(row["state_key"] == heldout_key and row["policy_target"][1] for row in merged)
