"""Independent 9x9 self-play and evaluation replay audit for V6.7."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from . import v5_evaluation as v5_eval
from . import v6_5_audit, v6_5_training, v6_7_training as training


def _audit_evaluation_game(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("source") != "evaluation" or (record.get("board_size"), record.get("n_in_row")) != (9, 5):
        raise ValueError("evaluation record has the wrong source or board rules")
    moves = record.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("evaluation record has no moves")
    board = np.zeros((9, 9), dtype=np.int8)
    expected = "B"
    terminal = False
    winner = None
    for index, move in enumerate(moves):
        if terminal:
            raise ValueError(f"evaluation has moves after terminal at ply {index}")
        color = move.get("player")
        if color != expected:
            raise ValueError(f"evaluation ply {index} breaks alternating color order")
        row, col = int(move.get("row", -1)), int(move.get("col", -1))
        if not (0 <= row < 9 and 0 <= col < 9) or board[row, col] != 0:
            raise ValueError(f"evaluation ply {index} is illegal or occupied")
        board[row, col] = 1 if color == "B" else 2
        if not move.get("opening", False):
            if move.get("actor") not in {record["candidate"], record["opponent"]}:
                raise ValueError(f"evaluation ply {index} has an unknown actor")
            if move["actor"] == record["opponent"] and str(record["opponent"]).startswith("C@"):
                if (move.get("algorithm") != "Alpha-Beta" or move.get("max_depth") != 69
                        or move.get("candidate_radius") != 2):
                    raise ValueError("C move does not use the frozen V6.4 Alpha-Beta protocol")
                budget = 0.05 if record["opponent"] == "C@50ms" else 0.2
                if float(move.get("configured_time_limit_seconds", -1)) != budget:
                    raise ValueError("C per-move time budget differs from its registered condition")
        terminal = _has_five(board, row, col, 1 if color == "B" else 2)
        if terminal:
            winner = color
        elif not np.any(board == 0):
            terminal = True
        if terminal and index != len(moves) - 1:
            raise ValueError("evaluation record continues after a terminal position")
        expected = "W" if expected == "B" else "B"
    if not terminal:
        raise ValueError("evaluation record ended before a win or full-board draw")
    if winner != record.get("winner_color"):
        raise ValueError("evaluation winner differs from independent replay")
    if int(record.get("move_count", -1)) != len(moves):
        raise ValueError("evaluation move count is inconsistent")
    expected_points = 0.5 if winner is None else 1.0 if winner == record.get("candidate_color") else 0.0
    if float(record.get("points", -1)) != expected_points:
        raise ValueError("evaluation points are inconsistent with the winner")
    return {"game_id": record["game_id"], "winner_color": winner,
            "points": expected_points, "move_count": len(moves)}


def _has_five(board: np.ndarray, row: int, col: int, player: int) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            rr, cc = row + sign * dr, col + sign * dc
            while 0 <= rr < 9 and 0 <= cc < 9 and board[rr, cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        if count >= 5:
            return True
    return False


def audit_training(root: str | Path = training.ROOT, *, require_complete: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = training.artifact_path(root)
    state_validation = training.validate_training_state(root=root)
    games = int(state_validation["game_count"])
    state_count = int(state_validation["environment_states"])
    updates = int(state_validation["optimizer_updates"])
    directory = artifact / "raw/selfplay" / training.ARM
    expected = {f"{training.ARM}-game-{i:06d}" for i in range(training.BASE_GAMES, games)}
    pending_id = f"{training.ARM}-game-{games:06d}"
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(record.get("game_id", ""))
        if game_id in found or path.stem != game_id:
            raise ValueError(f"duplicate or mismatched V6.7 game record: {path.name}")
        index = int(game_id.rsplit("-", 1)[-1])
        if index < training.BASE_GAMES or index > games:
            raise ValueError(f"training game ID outside G8000..G{games}: {game_id}")
        if int(record.get("seed", -1)) != training.SEED + index * 10007:
            raise ValueError(f"self-play seed differs from the frozen sequence: {game_id}")
        found[game_id] = record
    pending = {pending_id} if pending_id in found else set()
    if set(found) != expected | pending:
        raise ValueError(f"training ID range does not reconcile: missing={sorted(expected-set(found))[:5]}")
    if len(list(directory.glob("*.npz"))) != len(found):
        raise ValueError("self-play JSON/NPZ counts differ")
    total_states = 0
    winner_counts = {"B": 0, "W": 0, "draw": 0}
    move_total = 0
    for game_id in sorted(expected):
        record = found[game_id]
        v6_5_audit.audit_game_record(record, directory / f"{game_id}.npz")
        total_states += int(record["training_state_count"])
        move_total += int(record["move_count"])
        winner_counts[record.get("winner_color") or "draw"] += 1
    if pending:
        v6_5_audit.audit_game_record(found[pending_id], directory / f"{pending_id}.npz")
    baseline = json.loads((artifact / "results/baseline.json").read_text(encoding="utf-8"))
    expected_delta = state_count - int(baseline["environment_states"])
    if total_states != expected_delta:
        raise ValueError(f"self-play states do not reconcile: records={total_states}, checkpoint={expected_delta}")
    if updates != games - 1:
        raise ValueError("optimizer updates do not equal cumulative completed games minus one")
    history_path = artifact / "results/training" / f"{training.ARM}.jsonl"
    history_ids = {str(json.loads(line)["game_id"]) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()} if history_path.exists() else set()
    if history_ids != expected:
        raise ValueError("optimizer history IDs do not match the complete raw game records")
    for boundary in (training.BASE_GAMES, *training.MILESTONES):
        if boundary > games:
            continue
        if boundary == training.BASE_GAMES:
            checkpoint = artifact / "checkpoints" / training.ARM / "training_state_games_8000.pt"
            model_path = artifact / "models/G8000.pt"
        else:
            checkpoint = artifact / "checkpoints" / training.ARM / f"training_state_games_{boundary}.pt"
            model_path = artifact / "models" / f"G{boundary}.pt"
        if not checkpoint.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"missing full-state or model milestone G{boundary}")
        v = training.validate_training_state(checkpoint, root=root, expected_games=boundary)
        node = torch.load(model_path, map_location="cpu", weights_only=False)
        if int(node.get("game_count", -1)) != boundary:
            raise ValueError(f"G{boundary} model metadata is inconsistent")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not v6_5_training._state_equal(node["model_state"], state["model_state"]):
            raise ValueError(f"G{boundary} model and full training state weights differ")
        if int(node["environment_states"]) != int(v["environment_states"]):
            raise ValueError(f"G{boundary} model state count differs from its full checkpoint")
    complete = games == training.TARGET_GAMES
    if require_complete and not complete:
        raise ValueError(f"V6.7 is incomplete at G{games}; target G{training.TARGET_GAMES}")
    result = {
        "experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
        "base_games": training.BASE_GAMES, "target_games": training.TARGET_GAMES,
        "games_completed": games, "new_games_replayed": len(expected),
        "unique_game_ids": len(expected), "legal_terminal_games": len(expected),
        "pending_recovery_game_id": pending_id if pending else None,
        "new_training_states": total_states, "checkpoint_training_state_delta": expected_delta,
        "cumulative_training_states": state_count, "optimizer_updates": updates,
        "winner_distribution": winner_counts,
        "mean_selfplay_moves": move_total / len(expected) if expected else None,
        "checkpoint": state_validation, "complete": complete, "valid": True,
    }
    training.atomic_json(artifact / "results/training_audit.json", result)
    return result


def audit_evaluations(root: str | Path = training.ROOT, *, require_complete: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    games_dir = training.artifact_path(root) / "results/evaluations/games"
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(games_dir.glob("*.json")) if games_dir.is_dir() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(record.get("game_id", ""))
        if path.stem != game_id or game_id in records:
            raise ValueError(f"duplicate or mismatched evaluation ID: {path.name}")
        _audit_evaluation_game(record)
        records[game_id] = record
    by_condition: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for record in records.values():
        key = (int(record["candidate_games"]), str(record["opponent"]),
               "blind" if "_blind_" in str(record["game_id"]) else "development")
        by_condition.setdefault(key, []).append(record)
    summaries = []
    for (games, opponent, pool), rows in sorted(by_condition.items()):
        summary = v5_eval.summarize_paired_games(rows, seed=670701 if pool == "development" else 670702)
        summaries.append({"candidate_games": games, "opponent": opponent, "pool": pool, **summary})
    completed_stage_games = int(training.validate_training_state(root=root)["game_count"])
    if require_complete and completed_stage_games < training.TARGET_GAMES:
        raise ValueError("final evaluation audit requested before G20000")
    expected_dev = {(node, opp) for node in training.MILESTONES for opp in training.EVAL_OPPONENTS}
    expected_blind = {(training.TARGET_GAMES, opp) for opp in training.EVAL_OPPONENTS}
    observed = {(g, opp, pool) for g, opp, pool in by_condition}
    missing = sorted([(g, opp, "development") for g, opp in expected_dev if (g, opp, "development") not in observed]
                     + [(g, opp, "blind") for g, opp in expected_blind if (g, opp, "blind") not in observed])
    complete = not missing
    if require_complete and missing:
        raise ValueError(f"evaluation conditions are incomplete: {missing[:5]}")
    result = {"games": len(records), "unique_game_ids": len(records),
              "conditions": summaries, "missing_conditions": missing,
              "complete": complete, "valid": True}
    training.atomic_json(training.artifact_path(root) / "results/evaluation_audit.json", result)
    return result


def audit_experiment(root: str | Path = training.ROOT, *, require_complete: bool = True) -> dict[str, Any]:
    started = time.process_time()
    training_result = audit_training(root, require_complete=require_complete)
    evaluation_result = audit_evaluations(root, require_complete=require_complete)
    cpu = time.process_time() - started
    training.record_aux_cpu(root, "audit_cpu_seconds", cpu)
    result = {"experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
              "training": training_result, "evaluation": evaluation_result,
              "audit_cpu_seconds": cpu, "valid": True,
              "complete": training_result["complete"] and evaluation_result["complete"]}
    training.atomic_json(training.artifact_path(root) / "results/integrity_audit.json", result)
    return result

