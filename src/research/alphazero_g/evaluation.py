"""Frozen-opponent evaluation for pure-network and fixed-budget PUCT play."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard, Player
from ..imitation_f.model import PolicyValueNet, encode_board, load_f_checkpoint, select_device
from .puct import PUCTConfig, PUCTSearch

F1_SHA256 = "8d3cbeefb3b0ec75a4ec769b9852327f7411e8836f87a4add80d9cff2516e51f"
F2_SHA256 = "50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257"


def _root_path(relative: str) -> Path:
    return Path(__file__).resolve().parents[3] / relative


def load_frozen_opponent(name: str, device: str = "cpu") -> PolicyValueNet:
    normalized = name.upper()
    checkpoints = {
        "F1": ("research/f_v2/checkpoints/f1_reference.pt", F1_SHA256),
        "F2": ("research/f_v2/checkpoints/best_f2.pt", F2_SHA256),
    }
    if normalized not in checkpoints:
        raise ValueError("only F1 and F2 are frozen neural-network opponents")
    relative, expected = checkpoints[normalized]
    path = _root_path(relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{normalized} checkpoint hash mismatch")
    model = load_f_checkpoint(path, device=device)
    model.checkpoint_metadata = {"opponent": normalized, "checkpoint_sha256": actual}
    return model


def _network_move(
    model: PolicyValueNet, board: GomokuBoard, player: Player, device: torch.device
) -> tuple[tuple[int, int], dict[str, Any]]:
    encoded = encode_board(board, player)
    legal = encoded[2].reshape(-1).bool().to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits, value = model(encoded.unsqueeze(0).to(device))
            masked = logits[0].masked_fill(~legal, float("-inf"))
            probabilities = torch.softmax(masked, dim=-1)
            action = int(torch.argmax(masked).item())
        entropy = -float((probabilities[legal] * torch.log(probabilities[legal].clamp_min(1e-30))).sum().cpu())
        return divmod(action, 9), {"value": float(value[0].clamp(-1, 1).cpu()), "policy_entropy": entropy}
    finally:
        if was_training:
            model.train()


def play_evaluation_match(
    model: PolicyValueNet,
    *,
    game_id: str,
    opening_group_id: str,
    arm_id: str,
    mode: str,
    opponent: str,
    opening: list[dict[str, Any]],
    agent_color: Player,
    seed: int,
    simulations: int = 16,
    config: PUCTConfig | None = None,
    opponent_model: PolicyValueNet | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one full, color-paired evaluation game with timing and search stats."""
    if not game_id or not opening_group_id or not arm_id:
        raise ValueError("game_id, opening_group_id and arm_id must be nonempty")
    if mode not in {"network", "puct"}:
        raise ValueError("mode must be 'network' or 'puct'")
    if agent_color not in {"B", "W"}:
        raise ValueError("agent_color must be B or W")
    opponent = opponent.upper()
    if opponent not in {"A", "B", "F1", "F2"}:
        raise ValueError("opponent must be A, B, F1, or F2")
    board = GomokuBoard(9)
    expected = "B"
    moves: list[dict[str, Any]] = []
    for item in opening:
        if item.get("player") != expected:
            raise ValueError("evaluation opening must alternate from black")
        coord = (int(item["row"]), int(item["col"]))
        try:
            board.place_stone(expected, coord)
        except ValueError as exc:
            raise ValueError("evaluation opening has an illegal move") from exc
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("evaluation opening must be nonterminal")
        moves.append({"player": expected, "row": coord[0], "col": coord[1], "opening": True})
        expected = "W" if expected == "B" else "B"
    to_move: Player = expected

    selected_device = select_device(device)
    model.to(selected_device)
    frozen_model = opponent_model
    opponent_sha256 = None
    if opponent in {"F1", "F2"}:
        frozen_model = frozen_model or load_frozen_opponent(opponent, device=device)
        frozen_model.to(selected_device)
        opponent_sha256 = getattr(frozen_model, "checkpoint_metadata", {}).get("checkpoint_sha256")
    puct = PUCTSearch(model, config=config or PUCTConfig(), device=device) if mode == "puct" else None
    ai_config = SearchConfig(time_limit_seconds=0.05) if opponent == "B" else SearchConfig(time_limit_seconds=1e9)
    rng = np.random.default_rng(int(seed))
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    agent_seconds = opponent_seconds = 0.0
    agent_simulations = agent_nodes = agent_network_evals = agent_max_depth = 0
    opponent_nodes = opponent_depth = opponent_cutoffs = 0
    winner: Player | None = None
    termination = ""
    try:
        while True:
            is_agent = to_move == agent_color
            decision_started = time.perf_counter()
            extra: dict[str, Any]
            if is_agent and mode == "puct":
                assert puct is not None
                result = puct.search(
                    board, to_move, simulations, add_root_noise=False,
                    temperature=0.0, seed=int(rng.integers(0, 2**63 - 1)),
                )
                coord = result.move
                extra = {
                    "simulations": result.simulations,
                    "nodes": result.tree_nodes,
                    "network_evaluations": result.network_evaluations,
                    "search_depth": result.max_depth,
                    "root_value": result.root_value,
                    "search_entropy": result.visit_entropy,
                }
                agent_simulations += result.simulations
                agent_nodes += result.tree_nodes
                agent_network_evals += result.network_evaluations
                agent_max_depth = max(agent_max_depth, result.max_depth)
            elif is_agent:
                coord, extra = _network_move(model, board, to_move, selected_device)
            elif opponent == "A" or opponent == "B":
                response = choose_move(
                    board, to_move, opponent, config=ai_config,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                coord = response.move
                extra = {
                    "nodes": response.nodes,
                    "search_depth": response.completed_depth,
                    "cutoffs": response.cutoffs,
                    "timed_out": response.timed_out,
                    "simulations": response.simulations,
                }
                opponent_nodes += response.nodes
                opponent_depth = max(opponent_depth, response.completed_depth)
                opponent_cutoffs += response.cutoffs
            else:
                if frozen_model is None:
                    raise RuntimeError("frozen model opponent is not loaded")
                coord, extra = _network_move(frozen_model, board, to_move, selected_device)
            elapsed = time.perf_counter() - decision_started
            if is_agent:
                agent_seconds += elapsed
            else:
                opponent_seconds += elapsed
            board.place_stone(to_move, coord)
            moves.append({
                "player": to_move, "row": coord[0], "col": coord[1],
                "opening": False, "actor": "agent" if is_agent else opponent,
                "elapsed_seconds": elapsed, **extra,
            })
            if board.check_victory(coord, to_move):
                winner, termination = to_move, "five_in_row"
                break
            if board.is_full():
                winner, termination = None, "full_board"
                break
            to_move = "W" if to_move == "B" else "B"
        score = 0.5 if winner is None else 1.0 if winner == agent_color else 0.0
        return {
            "format_version": 1,
            "game_id": game_id,
            "opening_group_id": opening_group_id,
            "arm_id": arm_id,
            "mode": mode,
            "opponent": opponent,
            "opening": [dict(item) for item in opening],
            "agent_color": agent_color,
            "seed": int(seed),
            "simulations_per_agent_move": int(simulations) if mode == "puct" else 0,
            "completed": True,
            "termination": termination,
            "winner_color": winner,
            "outcome_for_agent": "draw" if winner is None else "win" if winner == agent_color else "loss",
            "score_for_agent": score,
            "move_count": len(moves),
            "moves": moves,
            "stats": {
                "agent_seconds": agent_seconds,
                "opponent_seconds": opponent_seconds,
                "total_wall_seconds": time.perf_counter() - started,
                "agent_simulations": agent_simulations,
                "agent_nodes": agent_nodes,
                "agent_network_evaluations": agent_network_evals,
                "agent_max_search_depth": agent_max_depth,
                "opponent_nodes": opponent_nodes,
                "opponent_max_search_depth": opponent_depth,
                "opponent_cutoffs": opponent_cutoffs,
            },
            "opponent_checkpoint_sha256": opponent_sha256,
        }
    finally:
        if was_training:
            model.train()


def summarize_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired color-swapped opening groups with a group bootstrap."""
    if not matches:
        raise ValueError("cannot summarize an empty match list")
    scores = np.asarray([float(match["score_for_agent"]) for match in matches])
    wins = sum(match["outcome_for_agent"] == "win" for match in matches)
    draws = sum(match["outcome_for_agent"] == "draw" for match in matches)
    losses = sum(match["outcome_for_agent"] == "loss" for match in matches)
    n = len(scores)
    p = float(scores.mean())
    grouped: dict[str, list[float]] = {}
    for match in matches:
        group_id = match.get("opening_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("paired summaries require opening_group_id for every game")
        grouped.setdefault(group_id, []).append(float(match["score_for_agent"]))
    group_scores = np.asarray([np.mean(scores) for scores in grouped.values()], dtype=np.float64)
    if len(group_scores) >= 2:
        rng = np.random.default_rng(731091)
        sampled = rng.choice(group_scores, size=(10000, len(group_scores)), replace=True).mean(axis=1)
        ci = [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]
    else:
        ci = None
    by_color = {}
    for color in ("B", "W"):
        subset = [match for match in matches if match["agent_color"] == color]
        by_color[color] = {
            "games": len(subset),
            "points": sum(float(match["score_for_agent"]) for match in subset),
            "score_rate": float(np.mean([match["score_for_agent"] for match in subset])) if subset else None,
        }
    return {
        "games": n, "wins": wins, "draws": draws, "losses": losses,
        "points": float(scores.sum()), "score_rate": p,
        "paired_opening_bootstrap_ci95": ci,
        "paired_opening_groups": len(group_scores),
        "by_color": by_color,
    }
