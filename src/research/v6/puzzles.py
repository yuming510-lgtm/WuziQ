"""Frozen tactical-position loader and proof-aware correctness evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...gomoku import GomokuBoard, Player
from .config import ROOT
from .engine import V6Agent


PUZZLE_PATH = ROOT / "configs/v6/tactical_positions.json"
MODELS = ("E1", "E2", "E3", "E4", "E5")


def load_tactical_positions(path: str | Path | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path is not None else PUZZLE_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("board_size") != 9:
        raise ValueError("Unsupported V6 tactical puzzle set")
    positions = payload.get("positions")
    if not isinstance(positions, list) or len(positions) < 10:
        raise ValueError("V6 tactical suite must contain at least ten positions")
    identifiers: set[str] = set()
    for position in positions:
        position_id = str(position.get("position_id", ""))
        if not position_id or position_id in identifiers:
            raise ValueError("V6 tactical puzzle IDs must be unique and nonempty")
        identifiers.add(position_id)
        board = GomokuBoard(9)
        current: Player = "B"
        for index, raw in enumerate(position.get("moves", [])):
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError(f"Malformed move in V6 puzzle {position_id}")
            player, row, col = str(raw[0]), int(raw[1]), int(raw[2])
            if player != current:
                raise ValueError(f"V6 puzzle {position_id} does not alternate colors at ply {index}")
            coordinate = row, col
            board.place_stone(player, coordinate)
            if board.check_victory(coordinate, player):
                raise ValueError(f"V6 puzzle {position_id} is terminal before the puzzle turn")
            current = "W" if current == "B" else "B"
        if not position.get("moves"):
            raise ValueError(f"V6 puzzle {position_id} has no reproducible position")
        if position.get("to_move") != current:
            raise ValueError(f"V6 puzzle {position_id} has incorrect player to move")
        if position.get("expectation") not in {
            "no_required_proof", "must_win_now", "must_block", "must_prove",
            "must_not_prove", "every_proof_must_verify",
        }:
            raise ValueError(f"V6 puzzle {position_id} has an unknown expectation")
        position["board_size"] = 9
        position["fixture_sha256"] = hashlib.sha256(
            json.dumps(position, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    categories = {str(position["category"]) for position in positions}
    required_categories = {"opening", "attack", "defense", "continuous_attack", "counterattack", "false_threat", "edge_attack", "edge_defense", "multi_threat_attack"}
    if not required_categories.issubset(categories):
        raise ValueError("V6 tactical suite is missing a required coverage category")
    return positions


def board_for_position(position: dict[str, Any]) -> GomokuBoard:
    board = GomokuBoard(int(position["board_size"]))
    for player, row, col in position["moves"]:
        board.place_stone(str(player), (int(row), int(col)))
    return board


def evaluate_tactical_result(
    position: dict[str, Any], model: str, result: Any, board: GomokuBoard,
) -> dict[str, Any]:
    """Score declared oracle expectations; unknown search is not a false proof."""
    expectation = str(position["expectation"])
    proof_status = str(getattr(result, "proof_status", "not_applicable"))
    certificate = getattr(result, "proof_certificate", None)
    proof_claimed = proof_status == "proven"
    certificate_valid = bool(
        proof_claimed
        and certificate
        and V6Agent.verify_proof_certificate(board, str(position["to_move"]), certificate)
    )
    correct: bool | None = None
    criterion = "proof_soundness_only"
    if expectation == "must_win_now":
        correct = tuple(result.move) in {tuple(map(int, item)) for item in position["winning_moves"]}
        criterion = "selected_an_exact_immediate_win"
    elif expectation == "must_block":
        correct = tuple(result.move) in {tuple(map(int, item)) for item in position["blocking_moves"]}
        criterion = "selected_the_only_immediate_block"
    elif expectation == "must_prove" and model in position.get("proof_required_models", []):
        correct = proof_claimed and certificate_valid
        criterion = "required_independently_verified_forced_win"
    elif expectation == "must_not_prove":
        correct = not proof_claimed
        criterion = "rejected_a_counterwin_refutation"
    elif expectation == "every_proof_must_verify":
        correct = not proof_claimed or certificate_valid
        criterion = "every_reported_proof_replays"
    elif expectation == "must_prove":
        criterion = "proof_optional_for_this_model"
    if proof_claimed and not certificate_valid:
        correct = False
        criterion = "proof_certificate_must_verify"
    return {
        "correct": correct,
        "criterion": criterion,
        "proof_claimed": proof_claimed,
        "proof_certificate_valid": certificate_valid,
    }


def run_tactical_suite(
    output_path: str | Path,
    *,
    max_attack_depth: int = 8,
    max_nodes: int = 50_000,
) -> dict[str, Any]:
    """Run all five V6 players on the frozen tactical corpus and write JSON."""
    import random
    import time

    positions = load_tactical_positions()
    rows: list[dict[str, Any]] = []
    for position_index, position in enumerate(positions):
        for model_index, model in enumerate(MODELS):
            board = board_for_position(position)
            before = (tuple(tuple(row) for row in board.grid), tuple(board.history))
            budget = float(position.get("validation_budget_seconds", 0.5))
            seed = int.from_bytes(hashlib.sha256(
                f"v6-tactical:{position['position_id']}:{model}".encode()
            ).digest()[:8], "big")
            started = time.perf_counter()
            result = V6Agent(model, max_attack_depth=max_attack_depth, max_nodes=max_nodes).choose_move(
                board, str(position["to_move"]), time_limit_seconds=budget, seed=seed,
            )
            elapsed = time.perf_counter() - started
            legal = board.in_bounds(*result.move) and board.get(result.move) is None
            evaluation = evaluate_tactical_result(position, model, result, board)
            if not legal:
                evaluation["correct"] = False
                evaluation["criterion"] = "legal_move_required"
            row = {
                "position_id": position["position_id"],
                "category": position["category"],
                "fixture_sha256": position["fixture_sha256"],
                "model": model,
                "to_move": position["to_move"],
                "budget_seconds": budget,
                "move": list(result.move),
                "legal": legal,
                "proof_status": str(getattr(result, "proof_status", "not_applicable")),
                "proof_kind": getattr(result, "proof_kind", None),
                "proof_claimed": evaluation["proof_claimed"],
                "proof_certificate_valid": evaluation["proof_certificate_valid"],
                "correct": evaluation["correct"],
                "criterion": evaluation["criterion"],
                "nodes": int(result.nodes),
                "completed_depth": int(result.completed_depth),
                "tree_depth": int(result.stats.get("max_attack_depth", result.completed_depth)),
                "cutoffs": int(result.cutoffs),
                "simulations": int(result.simulations),
                "elapsed_seconds": float(result.elapsed_seconds),
                "wall_seconds": elapsed,
                "timed_out": bool(result.timed_out),
                "stats": dict(result.stats),
            }
            if (tuple(tuple(row) for row in board.grid), tuple(board.history)) != before:
                row["board_restored"] = False
            else:
                row["board_restored"] = True
            rows.append(row)
    evaluated = [row for row in rows if row["correct"] is not None]
    summary = {
        "schema_version": 1,
        "experiment_id": "wuziq-advanced-search-v6-2026-10-22",
        "positions": len(positions),
        "models": list(MODELS),
        "runs": len(rows),
        "scored_cases": len(evaluated),
        "correct_cases": sum(row["correct"] is True for row in evaluated),
        "incorrect_cases": sum(row["correct"] is False for row in evaluated),
        "proof_claims": sum(row["proof_claimed"] for row in rows),
        "invalid_proofs": sum(row["proof_claimed"] and not row["proof_certificate_valid"] for row in rows),
        "timeouts": sum(row["timed_out"] for row in rows),
        "illegal_moves": sum(not row["legal"] for row in rows),
        "board_restore_failures": sum(not row["board_restored"] for row in rows),
        "rows": rows,
    }
    summary["all_proofs_valid"] = summary["invalid_proofs"] == 0
    summary["all_moves_legal"] = summary["illegal_moves"] == 0
    summary["all_boards_restored"] = summary["board_restore_failures"] == 0
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
