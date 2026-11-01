import copy
import json

import pytest

from src.research.alphazero_g.v2_schedule import build_v2_schedule, validate_v2_schedule


def _opening(coords):
    return [
        {"player": "B" if index % 2 == 0 else "W", "row": row, "col": col}
        for index, (row, col) in enumerate(coords)
    ]


def _rotated(opening):
    return [
        {"player": item["player"], "row": item["col"], "col": 8 - item["row"]}
        for item in opening
    ]


def test_v2_schedule_is_deterministic_and_has_registered_counts():
    first = build_v2_schedule(510701, 128, 8, 20)
    second = build_v2_schedule(510701, 128, 8, 20)

    assert first == second
    assert len(first["train_openings"]) == 128
    assert len(first["development_openings"]) == 8
    assert len(first["blind_openings"]) == 20
    validate_v2_schedule(first)


def test_v2_schedule_excludes_rotated_historical_opening():
    history = _opening([(1, 2), (2, 3), (4, 5), (7, 1)])
    schedule = build_v2_schedule(510701, 0, 1, 1, excluded_openings=[history])
    historical_cells = {(m["player"], m["row"], m["col"]) for m in history}
    rotated_cells = {(m["player"], m["row"], m["col"]) for m in _rotated(history)}

    for section in ("development_openings", "blind_openings"):
        for group in schedule[section]:
            cells = {(m["player"], m["row"], m["col"]) for m in group["opening"]}
            assert cells != historical_cells
            assert cells != rotated_cells


def test_v2_schedule_rejects_checksum_tampering():
    schedule = build_v2_schedule(510701, 2, 1, 1)
    corrupted = copy.deepcopy(schedule)
    corrupted["blind_openings"][0]["opening"][0]["row"] = 8

    with pytest.raises(ValueError, match="checksum"):
        validate_v2_schedule(corrupted)
