"""Deterministic, terminally labelled AlphaZero-style self-play games."""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

from ...gomoku import GomokuBoard, Player
from ..imitation_f.model import PolicyValueNet
from .puct import PUCTConfig, PUCTSearch, temperature_policy


def _canonical_board(board: GomokuBoard, to_move: Player) -> str:
    opponent = "W" if to_move == "B" else "B"
    return "".join(
        "1" if cell == to_move else "2" if cell == opponent else "."
        for row in board.grid for cell in row
    )


def _opening_board(opening: Sequence[dict[str, Any]]) -> tuple[GomokuBoard, Player]:
    board = GomokuBoard(9)
    expected: Player = "B"
    for move in opening:
        if not isinstance(move, dict) or move.get("player") != expected:
            raise ValueError("opening must contain legal alternating B/W moves")
        try:
            coord = (int(move["row"]), int(move["col"]))
            board.place_stone(expected, coord)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("opening contains an illegal move") from exc
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("opening must be a nonterminal position")
        expected = "W" if expected == "B" else "B"
    return board, expected


def play_selfplay_game(
    model: PolicyValueNet,
    opening: Sequence[dict[str, Any]],
    *,
    arm_id: str,
    game_id: str,
    opening_group_id: str,
    simulations: int,
    seed: int,
    config: PUCTConfig | None = None,
    temperature_moves: int = 12,
    late_temperature: float = 0.1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Play to a real terminal state and return visit targets plus final values.

    Opening plies are recorded but never become training samples. Each subsequent
    state is encoded from its side-to-move perspective and receives a terminal
    win/loss/draw label after the game ends.
    """
    if not arm_id or not game_id or not opening_group_id:
        raise ValueError("arm_id, game_id and opening_group_id must be nonempty")
    if isinstance(simulations, bool) or int(simulations) != simulations or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if isinstance(temperature_moves, bool) or int(temperature_moves) != temperature_moves or temperature_moves < 0:
        raise ValueError("temperature_moves must be a nonnegative integer")
    if not np.isfinite(late_temperature) or late_temperature < 0:
        raise ValueError("late_temperature must be finite and nonnegative")

    board, to_move = _opening_board(opening)
    moves = [{"player": str(m["player"]), "row": int(m["row"]), "col": int(m["col"]),
              "opening": True} for m in opening]
    samples: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    search = PUCTSearch(model, config=config, device=device)
    prior_training = model.training
    model.eval()
    started = time.perf_counter()
    cpu_started = time.process_time()
    simulations_total = network_evaluations = expanded_nodes = tree_nodes = 0
    max_depth = 0
    step_times: list[float] = []
    winner: Player | None = None
    termination = ""
    try:
        while True:
            step_started = time.perf_counter()
            result = search.search(
                board, to_move, simulations, add_root_noise=True,
                temperature=1.0, seed=int(rng.integers(0, 2**63 - 1)),
            )
            temperature = 1.0 if len(moves) - len(opening) < temperature_moves else float(late_temperature)
            action_probabilities = temperature_policy(result.root_visits, temperature)
            action = int(rng.choice(np.arange(81), p=action_probabilities))
            row, col = divmod(action, 9)
            before = len(moves)
            samples.append({
                "source": "self_play",
                "arm_id": arm_id,
                "game_id": game_id,
                "sample_id": f"{game_id}:ply:{before}",
                "opening_group_id": opening_group_id,
                "ply": before,
                "player": to_move,
                "canonical_board": _canonical_board(board, to_move),
                "policy_target": result.visit_policy.tolist(),
                "value_target": None,
                "game_complete": False,
                "simulations": int(result.simulations),
                "root_value": float(result.root_value),
                "policy_entropy": float(result.visit_entropy),
                "search_max_depth": int(result.max_depth),
            })
            board.place_stone(to_move, (row, col))
            moves.append({
                "player": to_move, "row": row, "col": col, "opening": False,
                "simulations": int(result.simulations),
                "network_evaluations": int(result.network_evaluations),
                "tree_nodes": int(result.tree_nodes),
                "search_max_depth": int(result.max_depth),
                "root_value": float(result.root_value),
                "search_seconds": float(result.elapsed_seconds),
                "temperature": temperature,
            })
            step_times.append(time.perf_counter() - step_started)
            simulations_total += result.simulations
            network_evaluations += result.network_evaluations
            expanded_nodes += result.expanded_nodes
            tree_nodes += result.tree_nodes
            max_depth = max(max_depth, result.max_depth)
            if board.check_victory((row, col), to_move):
                winner, termination = to_move, "five_in_row"
                break
            if board.is_full():
                winner, termination = None, "full_board"
                break
            to_move = "W" if to_move == "B" else "B"
        for sample in samples:
            sample["value_target"] = (
                0.0 if winner is None else 1.0 if sample["player"] == winner else -1.0
            )
            sample["game_complete"] = True
        return {
            "format_version": 1,
            "source": "self_play",
            "arm_id": arm_id,
            "game_id": game_id,
            "opening_group_id": opening_group_id,
            "opening": [dict(move) for move in opening],
            "seed": int(seed),
            "simulations_per_move": int(simulations),
            "completed": True,
            "termination": termination,
            "winner_color": winner,
            "outcome": "draw" if winner is None else "B_win" if winner == "B" else "W_win",
            "move_count": len(moves),
            "moves": moves,
            "experience_rows": samples,
            "stats": {
                "simulations": simulations_total,
                "network_evaluations": network_evaluations,
                "expanded_nodes": expanded_nodes,
                "tree_nodes_sum": tree_nodes,
                "max_search_depth": max_depth,
                "mean_step_seconds": float(np.mean(step_times)) if step_times else 0.0,
                "cpu_seconds": time.process_time() - cpu_started,
                "wall_seconds": time.perf_counter() - started,
                "cpu_wall_seconds": time.perf_counter() - started,
            },
        }
    finally:
        if prior_training:
            model.train()
