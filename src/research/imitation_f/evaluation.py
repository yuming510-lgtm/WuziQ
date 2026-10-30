"""Leakage-aware policy, value, tactical and move-time evaluation for F V1."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...gomoku import GomokuBoard
from ...ai import SearchConfig, choose_move
from .model import PolicyValueNet, predict_policy_value, select_device
from .training import _encode_rows


def find_immediate_wins(board: GomokuBoard, player: str) -> list[tuple[int, int]]:
    """Return every legal move that wins immediately by the exact game rules."""
    if player not in ("B", "W"):
        raise ValueError("player must be B or W")
    wins = []
    for row in range(board.size):
        for col in range(board.size):
            if board.get((row, col)) is None:
                board.place_stone(player, (row, col))
                won = board.check_victory((row, col), player)
                board.undo()
                if won:
                    wins.append((row, col))
    return wins


def _board_from_canonical(board_text: str, to_move: str) -> tuple[GomokuBoard, torch.Tensor]:
    if len(board_text) != 81 or any(cell not in ".12" for cell in board_text):
        raise ValueError("invalid perspective-canonical board")
    if to_move not in ("B", "W"):
        raise ValueError("to_move must be B or W")
    board = GomokuBoard(9)
    own, opponent = to_move, ("W" if to_move == "B" else "B")
    features = torch.zeros((3, 9, 9), dtype=torch.float32)
    for index, cell in enumerate(board_text):
        row, col = divmod(index, 9)
        if cell == "1":
            board.grid[row][col] = own
            features[0, row, col] = 1
        elif cell == "2":
            board.grid[row][col] = opponent
            features[1, row, col] = 1
        else:
            features[2, row, col] = 1
    return board, features


def extract_direct_tactics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select exact direct-win and forced-single-block positions from rows."""
    tactics = []
    for row in rows:
        to_move = str(row.get("to_move", "B"))
        board, _ = _board_from_canonical(str(row["canonical_board"]), to_move)
        own_wins = find_immediate_wins(board, to_move)
        if own_wins:
            kind, correct = "win", own_wins
        else:
            opponent = "W" if to_move == "B" else "B"
            threats = find_immediate_wins(board, opponent)
            if len(threats) != 1:
                continue
            kind, correct = "block", threats
        tactics.append({
            "state_key": row.get("state_key"),
            "canonical_board": row["canonical_board"],
            "to_move": to_move,
            "kind": kind,
            "correct_actions": [r * 9 + c for r, c in correct],
        })
    return tactics


@torch.inference_mode()
def evaluate_policy(
    model: PolicyValueNet,
    rows: list[dict[str, Any]],
    device: str | torch.device = "auto",
    batch_size: int = 128,
    latency_sample_size: int = 100,
) -> dict[str, Any]:
    """Evaluate blind policy/value labels, legal masking, and single-state latency."""
    if not rows:
        raise ValueError("evaluation set cannot be empty")
    x, targets, target_values = _encode_rows(rows)
    selected_device = select_device(str(device))
    model.to(selected_device)
    model.eval()
    policy_losses, top1, target_mass, legal_predictions = [], [], [], []
    value_errors, value_squared_errors = [], []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(selected_device)
        target = targets[start:start + batch_size].to(selected_device)
        value_target = target_values[start:start + batch_size].to(selected_device)
        logits, value = model(xb)
        legal = xb[:, 2].reshape(-1, 81).bool()
        logits = logits.masked_fill(~legal, -1e9)
        logp = F.log_softmax(logits, dim=-1)
        probabilities = logp.exp()
        prediction = probabilities.argmax(-1)
        max_target = target.max(dim=-1, keepdim=True).values
        target_optima = target >= (max_target - 1e-7)
        policy_losses.extend((-(target * logp).sum(-1)).cpu().tolist())
        top1.extend(target_optima.gather(1, prediction[:, None]).squeeze(1).float().cpu().tolist())
        target_mass.extend((target * probabilities).sum(-1).cpu().tolist())
        legal_predictions.extend(legal.gather(1, prediction[:, None]).squeeze(1).float().cpu().tolist())
        for index, row in enumerate(rows[start:start + len(value)]):
            if int(row.get("value_count", 1)) > 0:
                error = float(value[index].clamp(-1, 1).cpu()) - float(value_target[index].cpu())
                value_errors.append(abs(error))
                value_squared_errors.append(error * error)

    sample_count = min(max(1, latency_sample_size), len(x))
    if selected_device.type == "cuda":
        torch.cuda.synchronize()
    # Warm the kernels and allocator before collecting per-position latency.
    for _ in range(5):
        model(x[:1].to(selected_device))
    if selected_device.type == "cuda":
        torch.cuda.synchronize()
    latencies = []
    for index in range(sample_count):
        item = x[index:index + 1].to(selected_device)
        if selected_device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits, _ = model(item)
        logits = logits.masked_fill(~item[:, 2].reshape(-1, 81).bool(), -1e9)
        torch.softmax(logits, dim=-1)
        if selected_device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
    result = {
        "positions": len(rows),
        "policy_loss": float(np.mean(policy_losses)),
        "top1_accuracy": float(np.mean(top1)),
        "target_probability_mass": float(np.mean(target_mass)),
        "legal_rate": float(np.mean(legal_predictions)),
        "value_trained": bool(value_errors),
        "value_positions": len(value_errors),
        "value_mae": float(np.mean(value_errors)) if value_errors else None,
        "value_mse": float(np.mean(value_squared_errors)) if value_squared_errors else None,
        "inference_latency_ms": {
            "sample_positions": sample_count,
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
    }
    return result


@torch.inference_mode()
def predict_rows(
    model: PolicyValueNet,
    rows: list[dict[str, Any]],
    device: str | torch.device = "auto",
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    """Export per-position masked predictions for a frozen evaluation split."""
    x, targets, _ = _encode_rows(rows)
    selected_device = select_device(str(device))
    model.to(selected_device)
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        xb = x[start:start + batch_size].to(selected_device)
        logits, value = model(xb)
        legal = xb[:, 2].reshape(-1, 81).bool()
        probabilities = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)
        predicted = probabilities.argmax(-1)
        for offset, action in enumerate(predicted.cpu().tolist()):
            index = start + offset
            target = targets[index]
            correct_actions = torch.where(target >= target.max() - 1e-7)[0].tolist()
            predictions.append({
                "state_key": rows[index]["state_key"],
                "predicted_action": int(action),
                "predicted_probability": float(probabilities[offset, action].cpu()),
                "correct_actions": [int(item) for item in correct_actions],
                "top1_correct": action in correct_actions,
                "value_prediction": float(value[offset].clamp(-1, 1).cpu()),
            })
    return predictions


@torch.inference_mode()
def evaluate_tactics(
    model: PolicyValueNet,
    cases: list[dict[str, Any]],
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Measure top-1 exact move on immediate win and single-point block puzzles."""
    selected_device = select_device(str(device))
    model.to(selected_device)
    model.eval()
    by_kind: dict[str, dict[str, int]] = {}
    details = []
    for case in cases:
        _, features = _board_from_canonical(str(case["canonical_board"]), str(case.get("to_move", "B")))
        legal = features[2].reshape(1, 81).bool().to(selected_device)
        logits, _ = model(features.unsqueeze(0).to(selected_device))
        probabilities = torch.softmax(logits[0].masked_fill(~legal[0], -1e9), dim=-1)
        action = int(probabilities.argmax().item())
        top3 = set(int(value) for value in probabilities.topk(min(3, int(legal.sum()))).indices.cpu().tolist())
        top5 = set(int(value) for value in probabilities.topk(min(5, int(legal.sum()))).indices.cpu().tolist())
        correct = action in {int(value) for value in case["correct_actions"]}
        correct_actions = {int(value) for value in case["correct_actions"]}
        kind = str(case["kind"])
        item = by_kind.setdefault(kind, {
            "positions": 0, "correct": 0, "top3_correct": 0, "top5_correct": 0,
        })
        item["positions"] += 1
        item["correct"] += int(correct)
        item["top3_correct"] += int(bool(top3 & correct_actions))
        item["top5_correct"] += int(bool(top5 & correct_actions))
        details.append({
            "kind": kind, "predicted_action": action, "correct_actions": case["correct_actions"],
            "correct": correct, "top3_correct": bool(top3 & correct_actions),
            "top5_correct": bool(top5 & correct_actions),
        })
    for item in by_kind.values():
        item["accuracy"] = item["correct"] / item["positions"] if item["positions"] else 0.0
        item["top3_accuracy"] = item["top3_correct"] / item["positions"] if item["positions"] else 0.0
        item["top5_accuracy"] = item["top5_correct"] / item["positions"] if item["positions"] else 0.0
    total = sum(item["positions"] for item in by_kind.values())
    correct = sum(item["correct"] for item in by_kind.values())
    top3_correct = sum(item["top3_correct"] for item in by_kind.values())
    top5_correct = sum(item["top5_correct"] for item in by_kind.values())
    return {
        "positions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / total if total else 0.0,
        "top5_correct": top5_correct,
        "top5_accuracy": top5_correct / total if total else 0.0,
        "by_kind": by_kind,
        "details": details,
    }


def play_match(
    model_f: PolicyValueNet,
    opponent: str,
    opening: list[dict[str, Any]],
    f_color: str,
    seed: int,
    opponent_budget_seconds: float = 0.05,
) -> dict[str, Any]:
    """Play one full F-vs-A/B/C game and preserve every move metric."""
    opponent = str(opponent).upper()
    if opponent not in ("A", "B", "C"):
        raise ValueError("F V1 match opponent must be A, B, or C")
    if f_color not in ("B", "W"):
        raise ValueError("f_color must be B or W")
    board = GomokuBoard(9)
    moves = []
    current = "B"
    for ply, item in enumerate(opening):
        player = str(item["player"])
        coord = int(item["row"]), int(item["col"])
        if player != current or board.get(coord) is not None:
            raise ValueError("opening is illegal or out of turn")
        board.place_stone(player, coord)
        if board.check_victory(coord, player):
            raise ValueError("opening must be nonterminal")
        moves.append({
            "ply": ply, "player": player, "model": None, "row": coord[0], "col": coord[1],
            "opening": True, "elapsed_seconds": 0.0, "nodes": 0,
            "completed_depth": 0, "cutoffs": 0, "simulations": 0, "timed_out": False,
        })
        current = "W" if current == "B" else "B"
    if current != "B" or len(opening) != 4:
        raise ValueError("registered evaluation openings must contain four plies")

    winner_color = winner_model = None
    termination = "full_board"
    while not board.is_full():
        model_name = "F" if current == f_color else opponent
        started = time.perf_counter()
        if model_name == "F":
            probabilities, value = predict_policy_value(model_f, board, current)
            action = int(probabilities.argmax().item())
            move = divmod(action, 9)
            elapsed = time.perf_counter() - started
            stats = {
                "predicted_value": value,
                "policy_probability": float(probabilities[action].item()),
                "inference_latency_seconds": elapsed,
            }
            nodes = depth = cutoffs = simulations = 0
            timed_out = False
        else:
            result = choose_move(
                board, current, opponent,
                SearchConfig(time_limit_seconds=opponent_budget_seconds),
                seed=int(seed) + len(moves) * 7919,
            )
            move = tuple(map(int, result.move))
            elapsed = float(result.elapsed_seconds)
            stats = {}
            nodes = int(result.nodes)
            depth = int(result.completed_depth)
            cutoffs = int(result.cutoffs)
            simulations = int(result.simulations)
            timed_out = bool(result.timed_out or elapsed > opponent_budget_seconds)
        if not board.in_bounds(*move) or board.get(move) is not None:
            raise ValueError(f"{model_name} selected an illegal match move")
        board.place_stone(current, move)
        moves.append({
            "ply": len(moves), "player": current, "model": model_name,
            "row": move[0], "col": move[1], "opening": False,
            "elapsed_seconds": elapsed, "nodes": nodes, "completed_depth": depth,
            "cutoffs": cutoffs, "simulations": simulations,
            "timed_out": timed_out, "stats": stats,
        })
        if board.check_victory(move, current):
            winner_color = current
            winner_model = model_name
            termination = "five_in_row"
            break
        current = "W" if current == "B" else "B"
    outcome = "draw" if winner_color is None else ("b_win" if winner_color == "B" else "w_win")
    record = {
        "game_id": f"f-v1-match-9x9-f-vs-{opponent.lower()}-s{seed}-f{f_color.lower()}",
        "group_id": "evaluation-" + str(seed),
        "experiment_id": "wuziq-imitation-learning-f-v1",
        "stage": "blind-opening-exploratory-match",
        "board_size": 9,
        "pair": ["F", opponent],
        "black_model": "F" if f_color == "B" else opponent,
        "white_model": opponent if f_color == "B" else "F",
        "opening_seed": int(seed),
        "opening": [dict(move) for move in opening],
        "color_swap": f_color == "W",
        "game_seed": int(seed),
        "budget_seconds_by_model": {"F": None, opponent: opponent_budget_seconds},
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": outcome,
        "termination": termination,
        "thinking_seconds_total": sum(float(move["elapsed_seconds"]) for move in moves),
        "search_nodes_total": sum(int(move["nodes"]) for move in moves),
        "simulations_total": sum(int(move["simulations"]) for move in moves),
        "timeout_count": sum(bool(move["timed_out"]) for move in moves),
    }
    from .dataset import validate_game_record
    validate_game_record(record)
    return record
