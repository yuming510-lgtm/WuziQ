import copy
import hashlib
import json

import pytest

from src.research.alphazero_g.v3_schedule import build_v3_schedule, validate_v3_schedule


def _d4_key(opening):
    board = [["." for _ in range(9)] for _ in range(9)]
    for move in opening:
        board[int(move["row"])][int(move["col"])] = str(move["player"])
    forms = []
    for reflected in (False, True):
        grid = [row[:] for row in board]
        if reflected:
            grid = [list(reversed(row)) for row in grid]
        for _ in range(4):
            forms.append("".join(cell for row in grid for cell in row))
            grid = [list(row) for row in zip(*grid[::-1])]
    return min(forms)


def _opening_digest(opening):
    return hashlib.sha256(_d4_key(opening).encode()).hexdigest()


def _resign(schedule):
    unsigned = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    schedule["schedule_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prior_opening():
    return [
        {"player": "B", "row": 0, "col": 0},
        {"player": "W", "row": 4, "col": 4},
        {"player": "B", "row": 0, "col": 1},
        {"player": "W", "row": 4, "col": 5},
    ]


def test_schedule_is_reproducible_paired_and_has_exact_training_mix():
    kwargs = dict(
        seeds=[510601, 510602],
        train_groups_per_seed=8,
        development_groups=4,
        blind_groups=4,
        excluded_openings=[_prior_opening()],
    )
    first = build_v3_schedule(**kwargs)
    second = build_v3_schedule(**kwargs)

    assert first == second
    assert first["seeds"] == [510601, 510602]
    assert len(first["development_openings"]) == 4
    assert len(first["blind_openings"]) == 4
    assert first["development_openings"] != first["blind_openings"]

    random_train = []
    all_group_ids = []
    for seed in ("510601", "510602"):
        groups = first["train_openings_by_seed"][seed]
        assert len(groups) == 8
        assert [group["kind"] for group in groups].count("empty") == 4
        assert [group["kind"] for group in groups].count("random4") == 4
        assert all(group["opening"] == [] for group in groups if group["kind"] == "empty")
        assert all(len(group["opening"]) == 4 for group in groups if group["kind"] == "random4")
        random_train.extend(group["opening"] for group in groups if group["kind"] == "random4")
        all_group_ids.extend(group["opening_group_id"] for group in groups)
    held_out = first["development_openings"] + first["blind_openings"]
    assert all(group["kind"] == "random4" and len(group["opening"]) == 4 for group in held_out)
    all_group_ids.extend(group["opening_group_id"] for group in held_out)
    assert len(all_group_ids) == len(set(all_group_ids))

    train_keys = {_d4_key(opening) for opening in random_train}
    dev_keys = {_d4_key(group["opening"]) for group in first["development_openings"]}
    blind_keys = {_d4_key(group["opening"]) for group in first["blind_openings"]}
    assert len(train_keys) == len(random_train)
    assert len(dev_keys) == 4
    assert len(blind_keys) == 4
    assert train_keys.isdisjoint(dev_keys)
    assert train_keys.isdisjoint(blind_keys)
    assert dev_keys.isdisjoint(blind_keys)
    assert _opening_digest(_prior_opening()) in first["excluded_opening_digests"]
    validate_v3_schedule(first)


def test_schedule_rejects_symmetric_dev_blind_leakage_after_valid_rehash():
    schedule = build_v3_schedule(
        seeds=[3], train_groups_per_seed=4, development_groups=4, blind_groups=4,
        excluded_openings=[],
    )
    corrupted = copy.deepcopy(schedule)
    corrupted["blind_openings"][0]["opening"] = copy.deepcopy(
        corrupted["development_openings"][0]["opening"]
    )
    _resign(corrupted)

    with pytest.raises(ValueError, match="overlap|duplicate|symmetr"):
        validate_v3_schedule(corrupted)


def test_schedule_rejects_malformed_or_terminal_opening_even_with_valid_hash():
    schedule = build_v3_schedule(
        seeds=[4], train_groups_per_seed=4, development_groups=4, blind_groups=4,
        excluded_openings=[],
    )
    corrupted = copy.deepcopy(schedule)
    corrupted["development_openings"][0]["opening"][1]["player"] = "B"
    _resign(corrupted)

    with pytest.raises(ValueError, match="alternate|opening|legal"):
        validate_v3_schedule(corrupted)


def test_schedule_rejects_modified_payload_without_updated_digest():
    schedule = build_v3_schedule(
        seeds=[5], train_groups_per_seed=4, development_groups=4, blind_groups=4,
        excluded_openings=[],
    )
    schedule["blind_openings"][0]["opening_seed"] += 1

    with pytest.raises(ValueError, match="digest|checksum|hash"):
        validate_v3_schedule(schedule)
