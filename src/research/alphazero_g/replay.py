"""Validated self-play replay and crash-safe complete-game logs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _validate_game(game: dict[str, Any]) -> None:
    if game.get("completed") is not True or game.get("source") != "self_play":
        raise ValueError("only completed self-play games may enter replay")
    game_id, arm_id = game.get("game_id"), game.get("arm_id")
    if not isinstance(game_id, str) or not game_id or not isinstance(arm_id, str) or not arm_id:
        raise ValueError("complete self-play game needs game_id and arm_id")
    if game.get("termination") not in {"five_in_row", "full_board"}:
        raise ValueError("self-play game has no valid terminal result")
    winner = game.get("winner_color")
    if winner not in {"B", "W", None}:
        raise ValueError("winner_color must be B, W, or None")
    expected_outcome = "draw" if winner is None else "B_win" if winner == "B" else "W_win"
    if game.get("outcome") != expected_outcome:
        raise ValueError("outcome does not match winner_color")
    rows = game.get("experience_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("complete self-play game has no training rows")
    move_count = game.get("move_count")
    if isinstance(move_count, bool) or move_count != len(game.get("moves", [])):
        raise ValueError("move_count does not match complete move list")
    for row in rows:
        if row.get("source") != "self_play":
            raise ValueError("replay rows must come from self-play, not self-play supervision")
        if row.get("game_complete") is not True or row.get("game_id") != game_id or row.get("arm_id") != arm_id:
            raise ValueError("self-play row does not belong to this completed game")
        if row.get("player") not in {"B", "W"}:
            raise ValueError("self-play row has invalid side-to-move")
        canonical = row.get("canonical_board")
        if not isinstance(canonical, str) or len(canonical) != 81 or set(canonical) - {"1", "2", "."}:
            raise ValueError("self-play row has invalid canonical board")
        policy = np.asarray(row.get("policy_target"), dtype=np.float64)
        if policy.shape != (81,) or not np.isfinite(policy).all() or np.any(policy < 0):
            raise ValueError("policy target must be a finite nonnegative 81-cell vector")
        if not math.isclose(float(policy.sum()), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("policy target must sum to one")
        occupied = np.fromiter((cell != "." for cell in canonical), dtype=bool, count=81)
        if np.any(policy[occupied] != 0):
            raise ValueError("policy target assigns mass to an illegal occupied cell")
        expected_value = 0.0 if winner is None else 1.0 if row["player"] == winner else -1.0
        value = row.get("value_target")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) != expected_value:
            raise ValueError("value target does not match the final game result")


def _game_digest(game: dict[str, Any]) -> str:
    payload = json.dumps(game, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReplayBuffer:
    """Bounded, game-aware replay sampled uniformly by game then position."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(int(seed))
        self._games: list[dict[str, Any]] = []
        self._digests: dict[str, str] = {}
        self._position_count = 0

    @property
    def position_count(self) -> int:
        return self._position_count

    @property
    def game_count(self) -> int:
        return len(self._games)

    def add_complete_game(self, game: dict[str, Any]) -> bool:
        _validate_game(game)
        game_id = str(game["game_id"])
        digest = _game_digest(game)
        if game_id in self._digests:
            if self._digests[game_id] != digest:
                raise ValueError("conflicting self-play game with duplicate game_id")
            return False
        clean = copy.deepcopy(game)
        self._games.append(clean)
        self._digests[game_id] = digest
        self._position_count += len(clean["experience_rows"])
        while self._position_count > self.capacity and self._games:
            removed = self._games.pop(0)
            removed_id = str(removed["game_id"])
            self._position_count -= len(removed["experience_rows"])
            del self._digests[removed_id]
        return True

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self._games:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self._rng.integers(0, len(self._games), size=int(batch_size))
        result: list[dict[str, Any]] = []
        for game_index in indices:
            rows = self._games[int(game_index)]["experience_rows"]
            row_index = int(self._rng.integers(0, len(rows)))
            result.append(copy.deepcopy(rows[row_index]))
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "capacity": self.capacity,
            "games": copy.deepcopy(self._games),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1 or int(state.get("capacity", 0)) != self.capacity:
            raise ValueError("incompatible replay checkpoint")
        rebuilt = ReplayBuffer(self.capacity, seed=0)
        for game in state.get("games", []):
            rebuilt.add_complete_game(game)
        self._games = rebuilt._games
        self._digests = rebuilt._digests
        self._position_count = rebuilt._position_count
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])


def augment_sample(sample: dict[str, Any], rotations: int = 0, reflect: bool = False) -> dict[str, Any]:
    """Apply one D4 board symmetry consistently to canonical state and policy."""
    if isinstance(rotations, bool) or int(rotations) != rotations or rotations not in range(4):
        raise ValueError("rotations must be an integer from 0 through 3")
    board = sample.get("canonical_board")
    policy = np.asarray(sample.get("policy_target"), dtype=np.float64)
    if not isinstance(board, str) or len(board) != 81 or set(board) - {"1", "2", "."}:
        raise ValueError("canonical_board must contain 81 valid cells")
    if policy.shape != (81,) or not np.isfinite(policy).all():
        raise ValueError("policy_target must be a finite 81-cell vector")
    board_array = np.asarray(list(board)).reshape(9, 9)
    policy_array = policy.reshape(9, 9)
    board_array = np.rot90(board_array, int(rotations))
    policy_array = np.rot90(policy_array, int(rotations))
    if reflect:
        board_array = np.fliplr(board_array)
        policy_array = np.fliplr(policy_array)
    result = copy.deepcopy(sample)
    result["canonical_board"] = "".join(board_array.reshape(-1).tolist())
    result["policy_target"] = policy_array.reshape(-1).tolist()
    return result


def budgeted_prefix(rows: Sequence[dict[str, Any]], remaining_steps: int) -> tuple[list[dict[str, Any]], int]:
    """Select exactly the remaining matched environment-step budget.

    A self-play game is completed before persistence so terminal labels are real;
    rows beyond the shared step quota are counted as completion overrun and are
    excluded from replay/training by the caller.
    """
    if isinstance(remaining_steps, bool) or int(remaining_steps) != remaining_steps or remaining_steps < 0:
        raise ValueError("remaining_steps must be a nonnegative integer")
    accepted_count = min(len(rows), int(remaining_steps))
    return list(rows[:accepted_count]), len(rows) - accepted_count


def append_complete_game(path: str | Path, game: dict[str, Any], seen_ids: set[str]) -> bool:
    """Append one fsynced, complete JSON record; duplicate IDs are idempotent."""
    _validate_game(game)
    game_id = str(game["game_id"])
    if game_id in seen_ids:
        return False
    payload = (json.dumps(game, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("failed to append complete self-play game")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    seen_ids.add(game_id)
    return True


def load_game_log(path: str | Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Read complete games and truncate an interrupted, unterminated tail."""
    target = Path(path)
    if not target.exists():
        return [], set()
    raw = target.read_bytes()
    last_newline = raw.rfind(b"\n")
    complete_bytes = raw[:last_newline + 1] if last_newline >= 0 else b""
    if len(complete_bytes) != len(raw):
        with target.open("r+b") as handle:
            handle.truncate(len(complete_bytes))
            handle.flush()
            os.fsync(handle.fileno())
    games: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(complete_bytes.splitlines(), start=1):
        try:
            game = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid self-play JSONL record at line {line_number}") from exc
        _validate_game(game)
        game_id = str(game["game_id"])
        if game_id in ids:
            raise ValueError(f"duplicate game_id in self-play log: {game_id}")
        games.append(game)
        ids.add(game_id)
    return games, ids
