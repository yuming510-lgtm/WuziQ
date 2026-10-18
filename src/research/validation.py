"""Replay tournament records and verify schedule, pairing, and outcome integrity."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable

from ..gomoku import GomokuBoard, Player
from .runner import build_schedule, paired_opening


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def validate_tournament(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    """Verify all IDs and paired quotas, then replay every saved move legally."""

    games = list(records)
    by_id: dict[str, dict[str, Any]] = {}
    for record in games:
        identifier = str(record.get("game_id", ""))
        if not identifier:
            raise ValueError("A result record is missing game_id")
        if identifier in by_id:
            raise ValueError(f"Duplicate game_id: {identifier}")
        by_id[identifier] = record

    schedule = build_schedule(config)
    scheduled = {item["game_id"]: item for item in schedule}
    unexpected = set(by_id) - set(scheduled)
    if unexpected:
        raise ValueError(f"Unexpected game IDs: {sorted(unexpected)[:3]}")
    missing = set(scheduled) - set(by_id)
    if require_complete and missing:
        raise ValueError(f"Tournament is incomplete: {len(missing)} scheduled games are missing")

    pairs: dict[tuple[int, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    blocks: dict[tuple[int, tuple[str, str], int], list[dict[str, Any]]] = defaultdict(list)
    opening_layouts: dict[int, set[str]] = defaultdict(set)
    for record in games:
        identifier = str(record["game_id"])
        spec = scheduled[identifier]
        for key in (
            "board_size",
            "matchup",
            "opening_index",
            "color_swap",
            "black_model",
            "white_model",
            "opening_seed",
            "game_seed",
        ):
            actual = tuple(record[key]) if key == "matchup" else record.get(key)
            expected = tuple(spec[key]) if key == "matchup" else spec[key]
            if actual != expected:
                raise ValueError(f"Game {identifier} has mismatched scheduled field {key}")
        size = int(record["board_size"])
        matchup = tuple(record["matchup"])
        pairs[(size, matchup)].append(record)
        blocks[(size, matchup, int(record["opening_index"]))].append(record)
        opening_layouts[size].add(
            json.dumps(record["opening"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    for (size, matchup), pair_games in pairs.items():
        expected_count = next(
            int(phase["games_per_matchup"])
            for phase in config["phases"]
            if int(phase["board_size"]) == size
        )
        if require_complete and len(pair_games) != expected_count:
            raise ValueError(f"{size}x{size} {matchup} has {len(pair_games)} games; expected {expected_count}")
        colors = Counter(record["black_model"] for record in pair_games)
        for model in matchup:
            expected_model_count = len(pair_games) // 2
            if colors[model] != expected_model_count:
                raise ValueError(f"{size}x{size} {matchup} has unbalanced black/white games")

    for (size, matchup, opening_index), block_games in blocks.items():
        if len(block_games) != 2:
            raise ValueError(f"Opening block {size}x{size} {matchup} #{opening_index} is incomplete")
        first, second = block_games
        if first["opening"] != second["opening"]:
            raise ValueError(f"Opening block {size}x{size} {matchup} #{opening_index} differs")
        if first["black_model"] != second["white_model"] or first["white_model"] != second["black_model"]:
            raise ValueError(f"Opening block {size}x{size} {matchup} #{opening_index} lacks color swap")
        expected_opening = paired_opening(size, int(first["opening_seed"]))
        normalized = [{"player": player, "row": coord[0], "col": coord[1]} for player, coord in expected_opening]
        if first["opening"] != normalized:
            raise ValueError(f"Opening block {size}x{size} {matchup} #{opening_index} is not reproducible")

    illegal_moves = 0
    timeout_moves = 0
    over_budget_moves = 0
    max_move_seconds = 0.0
    termination_counts: Counter[str] = Counter()
    placed_moves = 0
    time_limit = float(config["search"]["time_limit_seconds"])
    for record in games:
        identifier = str(record["game_id"])
        size = int(record["board_size"])
        board = GomokuBoard(size)
        expected_player: Player = "B"
        derived_winner_color: Player | None = None
        derived_winner_model: str | None = None
        derived_termination = "full_board"
        moves = record.get("moves")
        if not isinstance(moves, list) or not moves:
            raise ValueError(f"Game {identifier} has no move list")

        terminal = False
        for index, move in enumerate(moves):
            if move.get("player") != expected_player:
                raise ValueError(f"Game {identifier} has out-of-turn move at ply {index}")
            timeout_moves += int(bool(move.get("timed_out", False)))
            elapsed = float(move.get("elapsed_seconds", 0.0))
            max_move_seconds = max(max_move_seconds, elapsed)
            if elapsed > time_limit:
                over_budget_moves += 1

            if move.get("illegal"):
                if index != len(moves) - 1:
                    raise ValueError(f"Game {identifier} continues after an illegal move")
                illegal_moves += 1
                derived_winner_color = _other(expected_player)
                derived_winner_model = (
                    record["black_model"]
                    if derived_winner_color == "B"
                    else record["white_model"]
                )
                derived_termination = "illegal_move_forfeit"
                terminal = True
                break

            try:
                row, col = int(move["row"]), int(move["col"])
                if type(move["row"]) is not int or type(move["col"]) is not int:
                    raise ValueError("coordinates must be integers")
                board.place_stone(expected_player, (row, col))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Game {identifier} contains an illegal move at ply {index}") from exc

            is_opening_ply = index < 2
            if bool(move.get("opening")) != is_opening_ply:
                raise ValueError(f"Game {identifier} has an incorrect opening marker at ply {index}")
            expected_model = None if is_opening_ply else (
                record["black_model"] if expected_player == "B" else record["white_model"]
            )
            if move.get("model") != expected_model:
                raise ValueError(f"Game {identifier} has incorrect model ownership at ply {index}")
            if is_opening_ply:
                opening_move = record["opening"][index]
                if (
                    opening_move.get("player") != expected_player
                    or opening_move.get("row") != row
                    or opening_move.get("col") != col
                ):
                    raise ValueError(f"Game {identifier} has an opening log that differs from its moves")
            placed_moves += 1

            if board.check_victory((row, col), expected_player):
                derived_winner_color = expected_player
                derived_winner_model = (
                    record["black_model"] if expected_player == "B" else record["white_model"]
                )
                derived_termination = "five_in_row"
                terminal = True
            elif board.is_full():
                derived_termination = "full_board"
                terminal = True

            if terminal and index != len(moves) - 1:
                raise ValueError(f"Game {identifier} continues after termination")
            expected_player = _other(expected_player)

        if record.get("move_count") != len(moves):
            raise ValueError(f"Game {identifier} has inconsistent move_count")
        if not terminal:
            raise ValueError(f"Game {identifier} ended before a terminal board state")
        if record.get("winner_color") != derived_winner_color:
            raise ValueError(f"Game {identifier} has an incorrect winner_color")
        if record.get("winner_model") != derived_winner_model:
            raise ValueError(f"Game {identifier} has an incorrect winner_model")
        if record.get("termination") != derived_termination:
            raise ValueError(f"Game {identifier} has an incorrect termination")
        expected_outcome = "draw" if derived_winner_color is None else f"{derived_winner_color.lower()}_win"
        if record.get("outcome") != expected_outcome:
            raise ValueError(f"Game {identifier} has an incorrect outcome")
        termination_counts[derived_termination] += 1

    return {
        "game_count": len(games),
        "expected_game_count": len(schedule),
        "complete": len(games) == len(schedule),
        "unique_game_ids": len(by_id),
        "board_counts": dict(sorted(Counter(int(record["board_size"]) for record in games).items())),
        "matchup_counts": {
            f"{size}x{size}-{'-'.join(pair)}": len(pair_games)
            for (size, pair), pair_games in sorted(pairs.items())
        },
        "paired_opening_blocks": len(blocks),
        "distinct_opening_layouts": {
            str(size): len(layouts) for size, layouts in sorted(opening_layouts.items())
        },
        "placed_moves": placed_moves,
        "illegal_moves": illegal_moves,
        "timeout_moves": timeout_moves,
        "over_budget_moves": over_budget_moves,
        "max_move_seconds": max_move_seconds,
        "terminations": dict(sorted(termination_counts.items())),
    }
