"""Deterministic, symmetry-aware opening schedules for G V3."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from ...gomoku import GomokuBoard


def _opening_key(opening: list[dict[str, Any]]) -> str:
    cells = [["." for _ in range(9)] for _ in range(9)]
    for move in opening:
        cells[int(move["row"])][int(move["col"])] = str(move["player"])
    variants: list[str] = []
    for reflect in (False, True):
        for turns in range(4):
            board = [row[:] for row in cells]
            if reflect:
                board = [list(reversed(row)) for row in board]
            for _ in range(turns):
                board = [list(row) for row in zip(*board[::-1])]
            variants.append("".join(cell for row in board for cell in row))
    return min(variants)


def _opening_digest(opening: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_opening_key(opening).encode("ascii")).hexdigest()


def _validate_opening(opening: Any, *, allow_empty: bool = False) -> None:
    if not isinstance(opening, list):
        raise ValueError("opening must be a list")
    if not opening and allow_empty:
        return
    if len(opening) != 4:
        raise ValueError("random opening must contain exactly four plies")
    board = GomokuBoard(9)
    expected = "B"
    for move in opening:
        if not isinstance(move, dict) or move.get("player") != expected:
            raise ValueError("opening players must alternate from B")
        try:
            coordinate = (int(move["row"]), int(move["col"]))
            board.place_stone(expected, coordinate)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("opening contains an illegal move") from exc
        if board.check_victory(coordinate, expected) or board.is_full():
            raise ValueError("opening must be nonterminal")
        expected = "W" if expected == "B" else "B"


def _make_random_opening(rng: np.random.Generator) -> list[dict[str, Any]]:
    board = GomokuBoard(9)
    result: list[dict[str, Any]] = []
    player = "B"
    for _ in range(4):
        legal = [(row, col) for row in range(9) for col in range(9) if board.get((row, col)) is None]
        coordinate = legal[int(rng.integers(0, len(legal)))]
        board.place_stone(player, coordinate)
        result.append({"player": player, "row": coordinate[0], "col": coordinate[1]})
        player = "W" if player == "B" else "B"
    return result


def _group(group_id: str, kind: str, opening: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    return {
        "opening_group_id": group_id,
        "kind": kind,
        "opening_seed": int(rng.integers(0, 2**63 - 1)),
        "opening": opening,
        "symmetry_digest": _opening_digest(opening),
    }


def _seed_value(seeds: list[int]) -> int:
    payload = json.dumps(seeds, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_v3_schedule(
    seeds: list[int],
    *,
    train_groups_per_seed: int,
    development_groups: int,
    blind_groups: int,
    excluded_openings: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build paired G0/G-F2 training starts and held-out random-open schedules."""
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or int(seed) != seed for seed in seeds
    ):
        raise ValueError("seeds must be a nonempty list of integers")
    if len(set(map(int, seeds))) != len(seeds):
        raise ValueError("seeds must be unique")
    if isinstance(train_groups_per_seed, bool) or train_groups_per_seed < 2 or train_groups_per_seed % 2:
        raise ValueError("training group count must be a positive even integer")
    if any(isinstance(value, bool) or int(value) != value or value < 1
           for value in (development_groups, blind_groups)):
        raise ValueError("development and blind group counts must be positive integers")
    if not isinstance(excluded_openings, list):
        raise ValueError("excluded_openings must be a list")

    excluded: set[str] = set()
    for opening in excluded_openings:
        if opening:
            _validate_opening(opening)
            excluded.add(_opening_digest(opening))

    rng = np.random.default_rng(_seed_value(list(map(int, seeds))))
    used_random_keys = set(excluded)
    seen_ids: set[str] = set()
    train: dict[str, list[dict[str, Any]]] = {}
    for seed in map(int, seeds):
        groups: list[dict[str, Any]] = []
        kinds = ["empty"] * (train_groups_per_seed // 2) + ["random4"] * (train_groups_per_seed // 2)
        rng.shuffle(kinds)
        for index, kind in enumerate(kinds):
            group_id = f"g-v3-train-s{seed}-{index:04d}"
            if group_id in seen_ids:
                raise ValueError("duplicate opening group identifier")
            seen_ids.add(group_id)
            if kind == "empty":
                opening: list[dict[str, Any]] = []
            else:
                for _attempt in range(100_000):
                    opening = _make_random_opening(rng)
                    digest = _opening_digest(opening)
                    if digest not in used_random_keys:
                        used_random_keys.add(digest)
                        break
                else:
                    raise RuntimeError("could not generate a unique random training opening")
            groups.append(_group(group_id, kind, opening, rng))
        train[str(seed)] = groups

    def heldout(kind: str, count: int) -> list[dict[str, Any]]:
        rows = []
        for index in range(count):
            for _attempt in range(100_000):
                opening = _make_random_opening(rng)
                digest = _opening_digest(opening)
                if digest not in used_random_keys:
                    used_random_keys.add(digest)
                    break
            else:
                raise RuntimeError(f"could not generate unique {kind} openings")
            group_id = f"g-v3-{kind}-{index:04d}"
            if group_id in seen_ids:
                raise ValueError("duplicate opening group identifier")
            seen_ids.add(group_id)
            rows.append(_group(group_id, "random4", opening, rng))
        return rows

    payload: dict[str, Any] = {
        "format_version": 1,
        "seeds": list(map(int, seeds)),
        "train_groups_per_seed": int(train_groups_per_seed),
        "development_groups": int(development_groups),
        "blind_groups": int(blind_groups),
        "excluded_opening_digests": sorted(excluded),
        "train_openings_by_seed": train,
        "development_openings": heldout("development", int(development_groups)),
        "blind_openings": heldout("blind", int(blind_groups)),
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    validate_v3_schedule(payload)
    return payload


def validate_v3_schedule(schedule: dict[str, Any]) -> None:
    """Validate schedule digest, opening legality, mix, and nonempty D4 isolation."""
    if not isinstance(schedule, dict) or schedule.get("format_version") != 1:
        raise ValueError("unsupported G V3 schedule format")
    unsigned = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if schedule.get("schedule_sha256") != expected_sha:
        raise ValueError("schedule checksum does not match its contents")
    seeds = schedule.get("seeds")
    count = schedule.get("train_groups_per_seed")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("schedule seed list is invalid")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError("schedule training mix size is invalid")
    train = schedule.get("train_openings_by_seed")
    if not isinstance(train, dict) or set(train) != {str(seed) for seed in seeds}:
        raise ValueError("schedule training seeds do not match its registration")

    excluded = set(map(str, schedule.get("excluded_opening_digests", [])))
    used_digests = set(excluded)
    group_ids: set[str] = set()

    def add_group(group: Any, *, allow_empty: bool, unique_random: bool) -> None:
        if not isinstance(group, dict):
            raise ValueError("opening group must be an object")
        group_id = str(group.get("opening_group_id", ""))
        if not group_id or group_id in group_ids:
            raise ValueError("opening group identifier is empty or duplicated")
        group_ids.add(group_id)
        kind = group.get("kind")
        opening = group.get("opening")
        if kind == "empty":
            if not allow_empty or opening != []:
                raise ValueError("empty start is only allowed in training groups")
            digest = _opening_digest([])
        elif kind == "random4":
            _validate_opening(opening)
            digest = _opening_digest(opening)
            if unique_random and digest in used_digests:
                raise ValueError("random opening overlaps or duplicates a D4-symmetric opening")
            if unique_random:
                used_digests.add(digest)
        else:
            raise ValueError("opening kind must be empty or random4")
        if group.get("symmetry_digest") != digest:
            raise ValueError("opening symmetry digest is invalid")
        opening_seed = group.get("opening_seed")
        if isinstance(opening_seed, bool) or not isinstance(opening_seed, int):
            raise ValueError("opening seed must be an integer")

    for seed in seeds:
        groups = train[str(seed)]
        if not isinstance(groups, list) or len(groups) != count:
            raise ValueError("training opening count differs from registration")
        if sum(group.get("kind") == "empty" for group in groups) != count // 2:
            raise ValueError("training starts must be exactly 50% empty and 50% random4")
        for group in groups:
            add_group(group, allow_empty=True, unique_random=True)
    for section in ("development_openings", "blind_openings"):
        groups = schedule.get(section)
        expected_count = schedule.get("development_groups" if section.startswith("development") else "blind_groups")
        if not isinstance(groups, list) or len(groups) != expected_count:
            raise ValueError(f"{section} count differs from registration")
        for group in groups:
            if group.get("kind") != "random4":
                raise ValueError("development and blind schedules use random4 openings only")
            add_group(group, allow_empty=False, unique_random=True)
