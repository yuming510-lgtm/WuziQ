"""Pre-registered, deterministic opening schedules with history exclusions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np

from ...gomoku import GomokuBoard


def _opening_digest(opening: list[dict[str, Any]]) -> str:
    normalized = [{"player": move["player"], "row": int(move["row"]), "col": int(move["col"])} for move in opening]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _random_opening(rng: np.random.Generator) -> list[dict[str, Any]]:
    board = GomokuBoard(9)
    opening: list[dict[str, Any]] = []
    player = "B"
    for _ in range(4):
        legal = [(row, col) for row in range(9) for col in range(9) if board.get((row, col)) is None]
        row, col = legal[int(rng.integers(0, len(legal)))]
        board.place_stone(player, (row, col))
        if board.check_victory((row, col), player):
            raise RuntimeError("generated four-ply opening unexpectedly terminated")
        opening.append({"player": player, "row": row, "col": col})
        player = "W" if player == "B" else "B"
    return opening


def build_g_schedule(
    seed: int,
    train_groups: int,
    development_groups: int,
    blind_groups: int,
    excluded_group_ids: Iterable[str] = (),
    excluded_openings: Iterable[list[dict[str, Any]]] = (),
) -> dict[str, Any]:
    """Create distinct legal four-ply train/dev/blind openings and hash them."""
    counts = (train_groups, development_groups, blind_groups)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
        raise ValueError("opening group counts must be nonnegative integers")
    if development_groups < 1 or blind_groups < 1:
        raise ValueError("development and blind schedules each need at least one group")
    excluded_openings = tuple(list(opening) for opening in excluded_openings)
    excluded_ids = set(str(value) for value in excluded_group_ids)
    excluded_digests = {_opening_digest(list(opening)) for opening in excluded_openings}
    rng = np.random.default_rng(int(seed))
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "development": [], "blind": []}
    seen_digests = set(excluded_digests)
    total = sum(counts)
    attempts = 0
    while sum(len(items) for items in groups.values()) < total:
        attempts += 1
        if attempts > max(10000, total * 100):
            raise RuntimeError("could not create a collision-free G opening schedule")
        opening = _random_opening(rng)
        digest = _opening_digest(opening)
        if digest in seen_digests:
            continue
        seed_value = int(rng.integers(0, 2**63 - 1))
        group_id = "g-open-" + hashlib.sha256(f"{int(seed)}:{seed_value}:{digest}".encode()).hexdigest()[:20]
        if group_id in excluded_ids:
            continue
        seen_digests.add(digest)
        group = {"opening_group_id": group_id, "opening_seed": seed_value, "opening": opening}
        for section, limit in zip(groups, counts, strict=True):
            if len(groups[section]) < limit:
                groups[section].append(group)
                break
    payload = {
        "format_version": 1,
        "seed": int(seed),
        "excluded_group_ids": sorted(excluded_ids),
        "train_openings": groups["train"],
        "development_openings": groups["development"],
        "blind_openings": groups["blind"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["schedule_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    validate_g_schedule(payload, excluded_openings=excluded_openings)
    return payload


def validate_g_schedule(
    schedule: dict[str, Any], *, excluded_openings: Iterable[list[dict[str, Any]]] = ()
) -> None:
    seen_ids: set[str] = set()
    seen_digests = {_opening_digest(list(opening)) for opening in excluded_openings}
    for key in ("train_openings", "development_openings", "blind_openings"):
        for group in schedule.get(key, []):
            group_id = group.get("opening_group_id")
            if not isinstance(group_id, str) or not group_id or group_id in seen_ids:
                raise ValueError("opening group ids must be unique and nonempty")
            seen_ids.add(group_id)
            board = GomokuBoard(9)
            expected = "B"
            opening = group.get("opening")
            if not isinstance(opening, list) or len(opening) != 4:
                raise ValueError("each G opening must contain exactly four plies")
            for move in opening:
                if move.get("player") != expected:
                    raise ValueError("G opening colors must alternate from black")
                coord = (int(move["row"]), int(move["col"]))
                board.place_stone(expected, coord)
                if board.check_victory(coord, expected):
                    raise ValueError("G opening is terminal")
                expected = "W" if expected == "B" else "B"
            digest = _opening_digest(opening)
            if digest in seen_digests:
                raise ValueError("G opening overlaps another or an excluded opening")
            seen_digests.add(digest)
    if set(schedule.get("excluded_group_ids", [])) & seen_ids:
        raise ValueError("G opening id collides with excluded historical groups")
    stored_hash = schedule.get("schedule_sha256")
    if stored_hash is not None:
        payload = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if stored_hash != expected:
            raise ValueError("G schedule hash does not match its contents")
