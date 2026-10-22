"""V6 game replay, proof-certificate, and exact-once integrity checks."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ...gomoku import GomokuBoard, Player
from .engine import V6Agent


_COUNTERS = (
    "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations",
    "rollout_plies", "terminal_simulations", "truncated_simulations",
    "root_candidates", "root_min_visits", "reused_nodes",
)


def validate_game_record(
    record: dict[str, Any],
    spec: dict[str, Any],
    *,
    verify_proofs: bool = True,
) -> dict[str, Any]:
    game_id = str(spec["game_id"])
    if record.get("game_id") != game_id or record.get("group_id") != spec["group_id"]:
        raise ValueError(f"V6 game {game_id} has mismatched identity")
    for key in (
        "experiment_id", "board_size", "pair", "model_versions",
        "budget_seconds_by_model", "opening_index", "opening_seed", "opening",
        "color_swap", "black_model", "white_model", "game_seed", "route_refs",
    ):
        if record.get(key) != spec.get(key):
            raise ValueError(f"V6 game {game_id} disagrees with scheduled field {key}")
    if not record.get("implementation_commit"):
        raise ValueError(f"V6 game {game_id} lacks an implementation commit")
    moves = record.get("moves")
    if not isinstance(moves, list) or len(moves) <= 4 or len(moves) > 81:
        raise ValueError(f"V6 game {game_id} is incomplete or has too many moves")

    board = GomokuBoard(int(spec["board_size"]))
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    timeout_count = nodes_total = simulations_total = cutoffs_total = 0
    thinking_total = 0.0
    max_tree_depth = max_search_depth = proof_count = proof_verified = 0
    for index, move in enumerate(moves):
        if int(move.get("ply", -1)) != index or move.get("player") != current:
            raise ValueError(f"V6 game {game_id} has a ply/color mismatch at {index}")
        opening = index < 4
        if bool(move.get("opening")) != opening:
            raise ValueError(f"V6 game {game_id} has an opening marker mismatch at {index}")
        expected_model = None if opening else (spec["black_model"] if current == "B" else spec["white_model"])
        if move.get("model") != expected_model:
            raise ValueError(f"V6 game {game_id} has an unexpected model at {index}")
        elapsed = float(move.get("elapsed_seconds", -1))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"V6 game {game_id} has an invalid elapsed time at {index}")
        thinking_total += elapsed
        timeout_count += bool(move.get("timed_out"))
        for counter in _COUNTERS:
            value = move.get(counter, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"V6 game {game_id} has an invalid {counter} at {index}")
        if int(move.get("tree_depth", move.get("completed_depth", 0))) != int(move.get("completed_depth", 0)):
            raise ValueError(f"V6 game {game_id} has inconsistent tree depth at {index}")
        nodes_total += int(move.get("nodes", 0))
        simulations_total += int(move.get("simulations", 0))
        cutoffs_total += int(move.get("cutoffs", 0))
        max_tree_depth = max(max_tree_depth, int(move.get("tree_depth", move.get("completed_depth", 0))))
        max_search_depth = max(max_search_depth, int(move.get("completed_depth", 0)))

        if opening:
            expected = spec["opening"][index]
            coord = int(expected["row"]), int(expected["col"])
            if (move.get("row"), move.get("col")) != coord:
                raise ValueError(f"V6 game {game_id} changed its frozen opening")
        elif move.get("illegal"):
            if index != len(moves) - 1:
                raise ValueError(f"V6 game {game_id} continues after an illegal move")
            winner_color = "W" if current == "B" else "B"
            winner_model = spec["black_model"] if winner_color == "B" else spec["white_model"]
            termination = "illegal_move_forfeit"
            break
        else:
            if type(move.get("row")) is not int or type(move.get("col")) is not int:
                raise ValueError(f"V6 game {game_id} has invalid coordinates at {index}")
            coord = int(move["row"]), int(move["col"])
            if not board.in_bounds(*coord) or board.get(coord) is not None:
                raise ValueError(f"V6 game {game_id} contains an illegal move at {index}")

        if not opening and verify_proofs and move.get("proof_status") == "proven":
            model = str(move["model"])
            certificate = move.get("proof_certificate")
            if model not in {"E1", "E2", "E3", "E4"} or not isinstance(certificate, dict):
                raise ValueError(f"V6 game {game_id} has a proof claim without a certificate")
            if tuple(map(int, certificate.get("root", {}).get("move", ()))) != coord:
                raise ValueError(f"V6 game {game_id} proof move differs from the played move")
            if not V6Agent.verify_proof_certificate(board, current, certificate):
                raise ValueError(f"V6 game {game_id} has an invalid tactical proof at {index}")
            proof_count += 1
            proof_verified += 1
        try:
            board.place_stone(current, coord)
        except ValueError as exc:
            raise ValueError(f"V6 game {game_id} repeats a move at {index}") from exc
        if board.check_victory(coord, current):
            if index != len(moves) - 1:
                raise ValueError(f"V6 game {game_id} continues after a five-in-row win")
            winner_color = current
            winner_model = expected_model
            termination = "five_in_row"
            break
        if board.is_full():
            if index != len(moves) - 1:
                raise ValueError(f"V6 game {game_id} continues after board fill")
            termination = "full_board"
            break
        current = "W" if current == "B" else "B"

    outcome = "draw" if winner_color is None else f"{winner_color.lower()}_win"
    if winner_color is None and (termination != "full_board" or not board.is_full()):
        raise ValueError(f"V6 game {game_id} ended without a win or full-board draw")
    expected_totals = {
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": outcome,
        "termination": termination,
        "move_count": len(moves),
        "timeout_count": timeout_count,
        "search_nodes_total": nodes_total,
        "simulations_total": simulations_total,
        "tree_depth_max": max_tree_depth,
        "search_depth_max": max_search_depth,
        "cutoffs_total": cutoffs_total,
    }
    for key, value in expected_totals.items():
        if record.get(key) != value:
            raise ValueError(f"V6 game {game_id} has incorrect {key}")
    if not math.isclose(float(record.get("thinking_seconds_total", -1)), thinking_total, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError(f"V6 game {game_id} has incorrect thinking_seconds_total")
    return {
        "game_id": game_id,
        "group_id": spec["group_id"],
        "winner_model": winner_model,
        "winner_color": winner_color,
        "outcome": outcome,
        "move_count": len(moves),
        "proofs": proof_verified,
    }


def validate_records(
    records: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    *,
    require_complete_groups: bool = False,
    verify_proofs: bool = True,
) -> dict[str, Any]:
    specs = {
        str(game["game_id"]): game
        for group in schedule for game in group["games"]
    }
    seen: set[str] = set()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proof_count = move_count = 0
    for record in records:
        game_id = str(record.get("game_id", ""))
        if game_id in seen:
            raise ValueError(f"Duplicate V6 game ID: {game_id}")
        seen.add(game_id)
        spec = specs.get(game_id)
        if spec is None:
            raise ValueError(f"V6 results contain an unscheduled game: {game_id}")
        result = validate_game_record(record, spec, verify_proofs=verify_proofs)
        proof_count += int(result["proofs"])
        move_count += int(result["move_count"])
        by_group[str(spec["group_id"])].append(record)
    if any(len(rows) > 2 for rows in by_group.values()):
        raise ValueError("A V6 paired group contains more than two games")
    if require_complete_groups and any(len(rows) != 2 for rows in by_group.values()):
        raise ValueError("A played V6 group is missing a color-swapped game")
    return {
        "valid": True,
        "games": len(records),
        "paired_groups": sum(len(rows) == 2 for rows in by_group.values()),
        "partial_groups": sum(len(rows) == 1 for rows in by_group.values()),
        "proofs_verified": proof_count,
        "replayed_moves": move_count,
        "unique_ids": len(seen),
    }
