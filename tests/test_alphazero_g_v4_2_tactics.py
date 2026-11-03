from __future__ import annotations


def test_tactical_diagnostics_use_many_legal_nonsymmetric_mother_positions() -> None:
    from src.research.alphazero_g.v4_2_diagnostics import (
        build_v4_2_tactical_set,
        puzzle_symmetry_key,
        validate_v4_2_puzzle,
    )

    puzzles = build_v4_2_tactical_set(seed=742_202, per_category=4)
    assert len(puzzles) == 16
    assert {row["category"] for row in puzzles} == {
        "direct_win", "unique_block", "false_threat", "edge_win"
    }
    keys = [puzzle_symmetry_key(puzzle) for puzzle in puzzles]
    assert len(keys) == len(set(keys))
    for puzzle in puzzles:
        validate_v4_2_puzzle(puzzle)
        assert puzzle["optimal_actions"]
        assert puzzle["label_source"] == "exact_legal_five_in_a_row_rules"


def test_tactical_labels_reject_forged_move_sets() -> None:
    import copy
    import pytest

    from src.research.alphazero_g.v4_2_diagnostics import (
        build_v4_2_tactical_set,
        validate_v4_2_puzzle,
    )

    puzzle = copy.deepcopy(build_v4_2_tactical_set(seed=742_203, per_category=1)[0])
    puzzle["optimal_actions"] = [(puzzle["optimal_actions"][0] + 1) % 81]
    with pytest.raises(ValueError, match="exact rule labels"):
        validate_v4_2_puzzle(puzzle)
