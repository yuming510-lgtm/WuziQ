"""Deterministic, symmetry-disjoint opening schedules for the G V2 experiment."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np

from ...gomoku import GomokuBoard


def _board_key(opening: list[dict[str, Any]]) -> str:
    cells = [["." for _ in range(9)] for _ in range(9)]
    for move in opening:
        cells[int(move["row"])][int(move["col"])] = str(move["player"])
    transformed: list[str] = []
    for reflect in (False, True):
        for turns in range(4):
            board = [row[:] for row in cells]
            if reflect:
                board = [list(reversed(row)) for row in board]
            for _ in range(turns):
                board = [list(row) for row in zip(*board[::-1])]
            transformed.append("".join(value for row in board for value in row))
    return min(transformed)


def _opening_digest(opening: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_board_key(opening).encode()).hexdigest()


def _validate_opening(opening: list[dict[str, Any]]) -> None:
    if not isinstance(opening, list) or len(opening) != 4:
        raise ValueError("V2 opening must contain exactly four plies")
    board = GomokuBoard(9)
    expected = "B"
    for move in opening:
        if not isinstance(move, dict) or move.get("player") != expected:
            raise ValueError("V2 opening colors must alternate from black")
        try:
            coord = (int(move["row"]), int(move["col"]))
            board.place_stone(expected, coord)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("V2 opening contains an illegal move") from exc
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("V2 opening must be nonterminal")
        expected = "W" if expected == "B" else "B"


def build_v2_schedule(
    seed: int,
    train_groups: int = 128,
    development_groups: int = 8,
    blind_groups: int = 20,
    excluded_openings: Iterable[list[dict[str, Any]]] = (),
    excluded_group_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Build nonoverlapping 4-ply schedules with dihedral opening deduplication."""
    counts = (train_groups, development_groups, blind_groups)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
        raise ValueError("opening group counts must be nonnegative integers")
    if development_groups < 1 or blind_groups < 1:
        raise ValueError("development and blind schedules must be nonempty")
    historical_openings = [list(opening) for opening in excluded_openings]
    for opening in historical_openings:
        _validate_opening(opening)
    excluded_ids = sorted(set(str(value) for value in excluded_group_ids))
    excluded_digests = {_opening_digest(opening) for opening in historical_openings}
    rng = np.random.default_rng(int(seed))
    sections: dict[str, list[dict[str, Any]]] = {"train": [], "development": [], "blind": []}
    limits = dict(zip(sections, counts, strict=True))
    seen_digests = set(excluded_digests)
    seen_ids = set(excluded_ids)
    target = sum(counts)
    attempts = 0
    while sum(map(len, sections.values())) < target:
        attempts += 1
        if attempts > max(10_000, target * 200):
            raise RuntimeError("could not build a symmetry-disjoint V2 opening schedule")
        board = GomokuBoard(9)
        opening: list[dict[str, Any]] = []
        player = "B"
        for _ in range(4):
            row, col = divmod(int(rng.integers(0, 81)), 9)
            while board.get((row, col)) is not None:
                row, col = divmod(int(rng.integers(0, 81)), 9)
            board.place_stone(player, (row, col))
            opening.append({"player": player, "row": row, "col": col})
            player = "W" if player == "B" else "B"
        digest = _opening_digest(opening)
        if digest in seen_digests:
            continue
        section = next(name for name in sections if len(sections[name]) < limits[name])
        index = len(sections[section])
        group_id = f"g-v2-{section}-{index:04d}-{digest[:12]}"
        if group_id in seen_ids:
            continue
        seen_digests.add(digest)
        seen_ids.add(group_id)
        sections[section].append({
            "opening_group_id": group_id,
            "opening_seed": int(rng.integers(0, 2**63 - 1)),
            "opening": opening,
            "symmetry_digest": digest,
        })
    payload: dict[str, Any] = {
        "format_version": 1,
        "seed": int(seed),
        "excluded_group_ids": excluded_ids,
        "excluded_opening_digests": sorted(excluded_digests),
        "train_openings": sections["train"],
        "development_openings": sections["development"],
        "blind_openings": sections["blind"],
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    validate_v2_schedule(payload, excluded_openings=historical_openings)
    return payload


def validate_v2_schedule(
    schedule: dict[str, Any], *, excluded_openings: Iterable[list[dict[str, Any]]] = ()
) -> None:
    if schedule.get("format_version") != 1:
        raise ValueError("unsupported V2 schedule format")
    canonical = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    expected = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if schedule.get("schedule_sha256") != expected:
        raise ValueError("V2 schedule checksum does not match its contents")
    excluded = {_opening_digest(list(opening)) for opening in excluded_openings}
    excluded.update(str(value) for value in schedule.get("excluded_opening_digests", []))
    seen_digests = set(excluded)
    seen_ids = set(map(str, schedule.get("excluded_group_ids", [])))
    for key in ("train_openings", "development_openings", "blind_openings"):
        groups = schedule.get(key)
        if not isinstance(groups, list):
            raise ValueError("V2 schedule section must be a list")
        for group in groups:
            group_id = str(group.get("opening_group_id", ""))
            if not group_id or group_id in seen_ids:
                raise ValueError("V2 schedule contains an empty or duplicate opening group ID")
            seen_ids.add(group_id)
            opening = group.get("opening")
            _validate_opening(opening)
            digest = _opening_digest(opening)
            if digest in seen_digests:
                raise ValueError("V2 schedule contains an overlapping or excluded symmetric opening")
            if group.get("symmetry_digest") != digest:
                raise ValueError("V2 schedule opening digest is invalid")
            seen_digests.add(digest)
