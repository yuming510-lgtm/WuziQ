from __future__ import annotations

import json
import importlib
import importlib.util

import pytest

from src.gomoku import GomokuBoard
from src.research.imitation_f.dataset import canonicalize_position


def _api():
    name = "src.research.imitation_f.v2_data"
    if importlib.util.find_spec(name) is None:
        pytest.fail("F V2 train-only tactical/DAgger data API is not implemented")
    return importlib.import_module(name)


def _position(board: str, *, to_move: str = "B", action: int = 0, key: str | None = None) -> dict:
    target = [0.0] * 81
    target[action] = 1.0
    return {
        "state_key": key or f"P:{board}",
        "canonical_board": board,
        "to_move": to_move,
        "policy_target": target,
        "value_target": 0.0,
        "value_count": 1,
        "opening_group_ids": ["training-group"],
    }


def _board_with(stones: dict[tuple[int, int], str]) -> str:
    cells = ["."] * 81
    for (row, col), player in stones.items():
        cells[row * 9 + col] = player
    return "".join(cells)


def test_exact_tactical_labels_find_immediate_win_and_single_block_but_not_dual_threat() -> None:
    api = _api()
    winning = _position(_board_with({(4, col): "1" for col in range(4)}), action=40)
    forced_block = _position(_board_with({(4, col): "2" for col in range(4)}), action=0)
    dual_threat = _position(_board_with({**{(4, col): "2" for col in range(4)}, **{(0, col): "2" for col in range(4)}}))

    labels = api.exact_tactical_targets([winning, forced_block, dual_threat])

    by_key = {row["state_key"]: row for row in labels}
    assert set(by_key) == {winning["state_key"], forced_block["state_key"]}
    assert by_key[winning["state_key"]]["tactical_kind"] == "win"
    assert by_key[winning["state_key"]]["policy_target"][40] == 1.0
    assert by_key[forced_block["state_key"]]["tactical_kind"] == "block"
    assert by_key[forced_block["state_key"]]["policy_target"][40] == 1.0
    assert dual_threat["policy_target"][0] == 1.0


def test_weighted_dataset_keeps_all_ordinary_rows_and_balances_attack_defense() -> None:
    api = _api()
    normal_rows = []
    seen = set()
    rng = __import__("random").Random(732402)
    while len(normal_rows) < 100:
        board = GomokuBoard(9)
        coords = rng.sample([(row, col) for row in range(9) for col in range(9)], 8)
        legal = True
        for index, coord in enumerate(coords):
            player = "B" if index % 2 == 0 else "W"
            board.place_stone(player, coord)
            if board.check_victory(coord, player):
                legal = False
                break
        if not legal:
            continue
        key, _ = canonicalize_position(board, "B")
        if key in seen:
            continue
        seen.add(key)
        action = next(index for index, cell in enumerate(key.split(":", 1)[1]) if cell == ".")
        normal_rows.append(_position(key.split(":", 1)[1], action=action, key=key))
    win = _position(_board_with({(4, col): "1" for col in range(4)}), action=0, key="train-win")
    block = _position(_board_with({(4, col): "2" for col in range(4)}), action=0, key="train-block")

    rows, manifest = api.build_weighted_training_rows(
        [*normal_rows, win, block], [], set(), set(), tactical_share=0.3, seed=732401,
    )

    assert {row["state_key"] for row in rows if row["tactical_kind"] == "ordinary"} == {row["state_key"] for row in normal_rows}
    assert manifest["tactical_win_rows"] == manifest["tactical_block_rows"]
    assert 0.29 <= manifest["actual_tactical_share"] <= 0.32
    assert all(row["policy_target"][40] == 1.0 for row in rows if row["tactical_kind"] in {"win", "block"})


def test_weighted_dataset_rejects_any_dagger_state_from_validation_or_blind() -> None:
    api = _api()
    row = _position("." * 81, key="held-out-state")
    with pytest.raises(ValueError, match="held-out canonical state"):
        api.build_weighted_training_rows([], [row], {"held-out-state"}, set(), tactical_share=0.3, seed=1)


def test_duplicate_dagger_generator_rows_are_aggregated_without_losing_count() -> None:
    api = _api()
    row = _position("." * 81, key="generator-state", action=40)
    rows, manifest = api.build_weighted_training_rows(
        [], (item for item in (row, row)), set(), set(), tactical_share=0.3, seed=2,
    )
    assert manifest["dagger_unique_states"] == 1
    assert manifest["dagger_duplicate_rows_aggregated"] == 1
    assert len(rows) == 1


def test_dagger_collection_records_heldout_collisions_and_resume_is_idempotent(tmp_path) -> None:
    api = _api()
    from pathlib import Path

    from src.research.imitation_f.v2_data import collect_dagger_data

    opening = {
        "opening_seed": 732501,
        "opening_group_id": "opening-test-v2-dagger",
        "opening": [
            {"player": "B", "row": 4, "col": 4},
            {"player": "W", "row": 3, "col": 4},
            {"player": "B", "row": 4, "col": 3},
            {"player": "W", "row": 5, "col": 4},
        ],
    }
    checkpoint = Path("research/f_v1/checkpoints/best.pt")
    source_dir = tmp_path / "source"
    result = collect_dagger_data(checkpoint, [opening], source_dir, 0.0, 732502)
    annotations = [json.loads(line) for line in (source_dir / "dagger_annotations.jsonl").read_text().splitlines()]
    assert result["completed_games"] == 2
    assert annotations

    heldout_key = annotations[0]["state_key"]
    heldout_dir = tmp_path / "heldout"
    heldout_result = collect_dagger_data(checkpoint, [opening], heldout_dir, 0.0, 732502, {heldout_key})
    progress = json.loads((heldout_dir / "dagger_progress.json").read_text())
    assert heldout_result["annotation_rows"] == len(annotations) - 1
    assert progress["heldout_state_keys_skipped"] == 1
    collect_dagger_data(checkpoint, [opening], heldout_dir, 0.0, 732502, {heldout_key})
    repeated_progress = json.loads((heldout_dir / "dagger_progress.json").read_text())
    assert repeated_progress["heldout_state_keys_skipped"] == 1
    saved = (source_dir / "dagger_annotations.jsonl").read_bytes()
    resumed = collect_dagger_data(checkpoint, [opening], source_dir, 0.0, 732502)
    assert resumed["annotation_rows"] == len(annotations)
    assert (source_dir / "dagger_annotations.jsonl").read_bytes() == saved


def test_network_action_audit_accepts_only_masked_policy_argmax() -> None:
    api = _api()
    from src.research.imitation_f.model import PolicyValueNet
    from src.research.imitation_f.evaluation import play_match

    model = PolicyValueNet(channels=8, blocks=1, seed=732503)
    opening = [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 4, "col": 3},
        {"player": "W", "row": 5, "col": 4},
    ]
    record = play_match(model, "A", opening, "B", 732504, opponent_budget_seconds=0.0)

    result = api.verify_network_actions(model, [record])
    assert result["network_actions"] == sum(move.get("model") == "F" for move in record["moves"])
    assert result["mismatches"] == 0

    altered = json.loads(json.dumps(record))
    first_f_move = next(move for move in altered["moves"] if move.get("model") == "F")
    original = (first_f_move["row"], first_f_move["col"])
    replacement = next(
        (row, col) for row in range(9) for col in range(9)
        if (row, col) != original and not any(
            item["row"] == row and item["col"] == col
            for item in altered["moves"][: first_f_move["ply"]]
        )
    )
    first_f_move["row"], first_f_move["col"] = replacement
    with pytest.raises(ValueError, match="masked-policy argmax"):
        api.verify_network_actions(model, [altered])


def test_dagger_exact_tactical_actions_are_in_canonical_orientation() -> None:
    api = _api()
    from src.gomoku import GomokuBoard
    from src.research.imitation_f.dataset import canonicalize_position
    from src.research.imitation_f.evaluation import find_immediate_wins

    sequence = [
        ("B", "F1", 0, 1), ("W", "A", 7, 7),
        ("B", "F1", 0, 2), ("W", "A", 6, 7),
        ("B", "F1", 0, 3), ("W", "A", 7, 6),
        ("B", "F1", 0, 4), ("W", "A", 6, 6),
        ("B", "F1", 4, 4),
    ]
    record = {
        "game_id": "canonical-tactical-label", "group_id": "training-group",
        "winner_color": "B", "moves": [
            {"ply": ply, "player": player, "model": model, "row": row, "col": col,
             "opening": False}
            for ply, (player, model, row, col) in enumerate(sequence)
        ],
    }

    annotations, counts, _ = api._game_annotations(record, 0.0, 7, set(), set())
    label = next(row for row in annotations if row["source_ply"] == 8)
    board = GomokuBoard(9)
    for player, _, row, col in sequence[:-1]:
        board.place_stone(player, (row, col))
    actual_wins = find_immediate_wins(board, "B")
    expected = {canonicalize_position(board, "B", coord)[1] for coord in actual_wins}
    observed = {index for index, probability in enumerate(label["policy_target"]) if probability > 0}
    actual = {row * 9 + col for row, col in actual_wins}
    assert label["tactical_kind"] == "win"
    assert expected != actual
    assert observed == expected
    assert counts.get("exact-rule") == 1
