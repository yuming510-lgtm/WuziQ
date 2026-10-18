"""Reproducibility, quota, and legal-replay validation for V2 games."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable

from ..gomoku import GomokuBoard
from .mcts_v2 import build_schedule, paired_opening


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-6)


def validate_records(
    records: Iterable[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate all observed games and report missing preregistered IDs."""

    games = list(records)
    identifiers = [str(game["game_id"]) for game in games]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate game_id in V2 result records")

    schedule = build_schedule(config)
    expected_by_id = {str(spec["game_id"]): spec for spec in schedule}
    actual_ids = set(identifiers)
    unexpected = actual_ids - set(expected_by_id)
    if unexpected:
        raise ValueError(f"Unscheduled game IDs: {sorted(unexpected)[:3]}")

    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    openings_by_board: dict[str, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    termination_counts: Counter[str] = Counter()
    illegal_moves = 0
    replayed_placements = 0

    for game in games:
        identifier = str(game["game_id"])
        spec = expected_by_id[identifier]
        for key in (
            "experiment_id",
            "phase_name",
            "track",
            "board_size",
            "pair_id",
            "opening_index",
            "color_swap",
            "black_model",
            "white_model",
            "opening_seed",
            "game_seed",
            "budget_seconds_by_model",
        ):
            if game.get(key) != spec.get(key):
                raise ValueError(f"Game {identifier} has mismatched {key}")
        if tuple(game.get("matchup", ())) != tuple(spec["matchup"]):
            raise ValueError(f"Game {identifier} has mismatched matchup")
        pair_groups[str(game["pair_id"])].append(game)

        size = int(game["board_size"])
        board = GomokuBoard(size)
        moves = list(game.get("moves", ()))
        opening_plies = int(spec["opening_plies"])
        if len(moves) != int(game.get("move_count", -1)):
            raise ValueError(f"Game {identifier} move_count mismatch")
        if len(moves) < opening_plies:
            raise ValueError(f"Game {identifier} ended before its paired opening")

        opening_coordinates: list[tuple[int, int]] = []
        opening_rows: list[dict[str, Any]] = []
        seen_terminal = False
        winning_model: str | None = None
        winning_color: str | None = None
        illegal_forfeit_winner: str | None = None

        for index, move in enumerate(moves):
            if seen_terminal:
                raise ValueError(f"Game {identifier} has moves after termination")
            player = str(move.get("player"))
            if player not in {"B", "W"}:
                raise ValueError(f"Game {identifier} has invalid player at ply {index}")
            model = move.get("model")
            expected_model = spec["black_model"] if player == "B" else spec["white_model"]
            coordinate_values = move.get("row"), move.get("col")

            if index < opening_plies:
                if model is not None or not move.get("opening"):
                    raise ValueError(f"Game {identifier} opening log is malformed at ply {index}")
                if type(coordinate_values[0]) is not int or type(coordinate_values[1]) is not int:
                    raise ValueError(f"Game {identifier} opening coordinate is invalid")
                opening_coordinates.append((coordinate_values[0], coordinate_values[1]))
                opening_rows.append(
                    {
                        "player": player,
                        "row": coordinate_values[0],
                        "col": coordinate_values[1],
                    }
                )
            else:
                if model != expected_model or move.get("opening"):
                    raise ValueError(f"Game {identifier} model/color mismatch at ply {index}")
                elapsed = float(move.get("elapsed_seconds", -1.0))
                if elapsed < 0:
                    raise ValueError(f"Game {identifier} has negative elapsed time")
                for metric in (
                    "nodes",
                    "simulations",
                    "rollout_plies",
                    "tree_depth",
                    "completed_depth",
                    "terminal_simulations",
                    "truncated_simulations",
                ):
                    if int(move.get(metric, -1)) < 0:
                        raise ValueError(f"Game {identifier} has invalid {metric}")

            if move.get("illegal"):
                illegal_moves += 1
                if index != len(moves) - 1:
                    raise ValueError(f"Game {identifier} continues after an illegal move")
                illegal_forfeit_winner = (
                    spec["white_model"] if player == "B" else spec["black_model"]
                )
                seen_terminal = True
                continue

            row, col = coordinate_values
            if (
                type(row) is not int
                or type(col) is not int
                or not board.in_bounds(row, col)
                or board.get((row, col)) is not None
            ):
                raise ValueError(f"Game {identifier} has an illegal coordinate at ply {index}")
            board.place_stone(player, (row, col))
            replayed_placements += 1
            if board.check_victory((row, col), player):
                seen_terminal = True
                winning_model = expected_model
                winning_color = player
                if index != len(moves) - 1:
                    raise ValueError(f"Game {identifier} continues after five in a row")

        if game.get("opening") != opening_rows:
            raise ValueError(f"Game {identifier} opening metadata differs from move log")
        seeded_opening = paired_opening(
            size,
            int(spec["opening_seed"]),
            opening_plies=opening_plies,
            color_swap=bool(spec["color_swap"]),
        )
        expected_opening_rows = [
            {"player": player, "row": coord[0], "col": coord[1]}
            for player, coord in seeded_opening
        ]
        if opening_rows != expected_opening_rows:
            raise ValueError(f"Game {identifier} opening does not match its fixed seed")
        openings_by_board[str(size)].add(tuple(opening_coordinates))

        termination = str(game.get("termination"))
        termination_counts[termination] += 1
        if termination == "five_in_row":
            if winning_model is None or game.get("winner_model") != winning_model:
                raise ValueError(f"Game {identifier} winner does not match replay")
            if game.get("winner_color") != winning_color:
                raise ValueError(f"Game {identifier} winner color does not match replay")
        elif termination == "full_board":
            if len(board.history) != size * size or winning_model is not None:
                raise ValueError(f"Game {identifier} full-board termination is invalid")
            if game.get("winner_model") is not None or game.get("winner_color") is not None:
                raise ValueError(f"Game {identifier} draw has a winner")
        elif termination == "illegal_move_forfeit":
            if illegal_forfeit_winner is None or game.get("winner_model") != illegal_forfeit_winner:
                raise ValueError(f"Game {identifier} illegal-move forfeit winner is invalid")
        else:
            raise ValueError(f"Game {identifier} has unknown termination {termination}")

        if game.get("outcome") != (
            "draw" if game.get("winner_model") is None else f"{str(game['winner_color']).lower()}_win"
        ):
            raise ValueError(f"Game {identifier} outcome does not match winner")

        totals = {
            "timeout_count": sum(bool(move.get("timed_out", False)) for move in moves),
            "search_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
            "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
            "rollout_plies_total": sum(int(move.get("rollout_plies", 0)) for move in moves),
            "terminal_simulations_total": sum(
                int(move.get("terminal_simulations", 0)) for move in moves
            ),
            "truncated_simulations_total": sum(
                int(move.get("truncated_simulations", 0)) for move in moves
            ),
        }
        for key, value in totals.items():
            if int(game.get(key, -1)) != value:
                raise ValueError(f"Game {identifier} {key} does not match move metrics")
        thinking_time = sum(float(move.get("elapsed_seconds", 0.0)) for move in moves)
        if not _close(float(game.get("thinking_seconds_total", -1.0)), thinking_time):
            raise ValueError(f"Game {identifier} thinking time does not match move metrics")
        maximum_depth = max((int(move.get("tree_depth", 0)) for move in moves), default=0)
        if int(game.get("tree_depth_max", -1)) != maximum_depth:
            raise ValueError(f"Game {identifier} tree_depth_max does not match move metrics")

    unpaired_groups = []
    for pair_id, group in pair_groups.items():
        if len(group) != 2 or {bool(game["color_swap"]) for game in group} != {False, True}:
            unpaired_groups.append(pair_id)
            continue
        false_game = next(game for game in group if not game["color_swap"])
        true_game = next(game for game in group if game["color_swap"])
        if [
            (move["row"], move["col"]) for move in false_game["opening"]
        ] != [(move["row"], move["col"]) for move in true_game["opening"]]:
            raise ValueError(f"Paired opening coordinates differ for {pair_id}")
        if false_game["opening"] != true_game["opening"]:
            raise ValueError(f"Paired opening colors differ for {pair_id}")
        if (
            false_game["black_model"] != true_game["white_model"]
            or false_game["white_model"] != true_game["black_model"]
        ):
            raise ValueError(f"Paired models did not exchange colors for {pair_id}")
        if false_game["moves"][len(false_game["opening"])]["model"] == true_game[
            "moves"
        ][len(true_game["opening"])]["model"]:
            raise ValueError(f"Paired fifth-ply decision owner did not change for {pair_id}")

    missing_ids = set(expected_by_id) - actual_ids
    return {
        "game_count": len(games),
        "scheduled_game_count": len(schedule),
        "unique_game_ids": len(actual_ids),
        "missing_game_count": len(missing_ids),
        "complete": not missing_ids,
        "illegal_moves": illegal_moves,
        "replayed_placements": replayed_placements,
        "termination_counts": dict(sorted(termination_counts.items())),
        "paired_opening_groups": len(pair_groups) - len(unpaired_groups),
        "unpaired_opening_group_count": len(unpaired_groups),
        "unpaired_opening_group_ids": sorted(unpaired_groups),
        "unique_opening_positions_by_board": {
            size: len(openings) for size, openings in sorted(openings_by_board.items())
        },
        "missing_game_id_sample": sorted(missing_ids)[:10],
    }
