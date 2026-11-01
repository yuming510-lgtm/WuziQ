"""Frozen-opponent evaluation for pure-network and fixed-budget PUCT play."""
from __future__ import annotations

import hashlib
import json
import os
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
    opponent_mode: str = "network",
    opponent_simulations: int | None = None,
    config: PUCTConfig | None = None,
    opponent_model: PolicyValueNet | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one full, color-paired evaluation game with timing and search stats."""
    if not game_id or not opening_group_id or not arm_id:
        raise ValueError("game_id, opening_group_id and arm_id must be nonempty")
    if mode not in {"network", "puct"}:
        raise ValueError("mode must be 'network' or 'puct'")
    if opponent_mode not in {"network", "puct"}:
        raise ValueError("opponent_mode must be 'network' or 'puct'")
    if opponent_simulations is None:
        opponent_simulations = simulations
    if isinstance(opponent_simulations, bool) or int(opponent_simulations) != opponent_simulations or opponent_simulations < 1:
        if opponent_mode == "puct":
            raise ValueError("opponent_simulations must be a positive integer for PUCT opponents")
    if agent_color not in {"B", "W"}:
        raise ValueError("agent_color must be B or W")
    opponent = opponent.upper()
    if opponent not in {"A", "B", "F1", "F2", "MODEL"}:
        raise ValueError("opponent must be A, B, F1, F2, or MODEL")
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
    if opponent in {"F1", "F2", "MODEL"}:
        if opponent == "MODEL" and frozen_model is None:
            raise ValueError("MODEL opponent requires opponent_model")
        frozen_model = frozen_model or load_frozen_opponent(opponent, device=device)
        frozen_model.to(selected_device)
        opponent_sha256 = getattr(frozen_model, "checkpoint_metadata", {}).get("checkpoint_sha256")
        if opponent == "F2" and opponent_model is not None and opponent_sha256 != F2_SHA256:
            raise ValueError("provided F2 model does not carry the frozen F2 checkpoint hash")
        if opponent_sha256 is None:
            state_hash = hashlib.sha256()
            for name, value in sorted(frozen_model.state_dict().items()):
                state_hash.update(name.encode("utf-8"))
                state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
            opponent_sha256 = state_hash.hexdigest()
    puct = PUCTSearch(model, config=config or PUCTConfig(), device=device) if mode == "puct" else None
    opponent_puct = (
        PUCTSearch(frozen_model, config=config or PUCTConfig(), device=device)
        if opponent in {"F1", "F2", "MODEL"} and opponent_mode == "puct" and frozen_model is not None else None
    )
    ai_config = SearchConfig(time_limit_seconds=0.05) if opponent == "B" else SearchConfig(time_limit_seconds=1e9)
    rng = np.random.default_rng(int(seed))
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    cpu_started = time.process_time()
    agent_seconds = opponent_seconds = 0.0
    agent_simulations = agent_nodes = agent_network_evals = agent_max_depth = 0
    opponent_nodes = opponent_depth = opponent_cutoffs = 0
    opponent_simulation_total = opponent_network_evals = 0
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
            elif opponent in {"F1", "F2", "MODEL"} and opponent_mode == "puct":
                assert opponent_puct is not None
                result = opponent_puct.search(
                    board, to_move, int(opponent_simulations), add_root_noise=False,
                    temperature=0.0, seed=int(rng.integers(0, 2**31 - 1)),
                )
                coord = result.move
                extra = {
                    "simulations": result.simulations,
                    "nodes": result.tree_nodes,
                    "network_evaluations": result.network_evaluations,
                    "search_depth": result.max_depth,
                    "root_value": result.root_value,
                }
                opponent_simulation_total += result.simulations
                opponent_network_evals += result.network_evaluations
                opponent_nodes += result.tree_nodes
                opponent_depth = max(opponent_depth, result.max_depth)
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
            "source": "evaluation",
            "game_id": game_id,
            "opening_group_id": opening_group_id,
            "arm_id": arm_id,
            "mode": mode,
            "opponent": opponent,
            "opponent_mode": opponent_mode if opponent in {"F1", "F2", "MODEL"} else "traditional",
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
                "total_cpu_seconds": time.process_time() - cpu_started,
                "agent_simulations": agent_simulations,
                "agent_nodes": agent_nodes,
                "agent_network_evaluations": agent_network_evals,
                "agent_max_search_depth": agent_max_depth,
                "opponent_nodes": opponent_nodes,
                "opponent_max_search_depth": opponent_depth,
                "opponent_cutoffs": opponent_cutoffs,
                "opponent_simulations": opponent_simulation_total,
                "opponent_network_evaluations": opponent_network_evals,
            },
            "opponent_checkpoint_sha256": opponent_sha256,
        }
    finally:
        if was_training:
            model.train()


def validate_match_record(match: dict[str, Any]) -> None:
    """Replay a complete evaluation record and verify legality and its result."""
    if match.get("completed") is not True or match.get("source") != "evaluation":
        raise ValueError("only completed evaluation matches are valid records")
    if match.get("format_version") != 1 or not match.get("game_id") or not match.get("opening_group_id"):
        raise ValueError("evaluation match is missing identity metadata")
    moves = match.get("moves")
    if not isinstance(moves, list) or not moves or match.get("move_count") != len(moves):
        raise ValueError("evaluation match move count is invalid")
    opening = match.get("opening")
    if not isinstance(opening, list):
        raise ValueError("evaluation match opening is invalid")
    board = GomokuBoard(9)
    expected: Player = "B"
    winner: Player | None = None
    termination = ""
    for index, move in enumerate(moves):
        if move.get("player") != expected:
            raise ValueError("evaluation match colors do not alternate")
        coord = (int(move.get("row", -1)), int(move.get("col", -1)))
        try:
            board.place_stone(expected, coord)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation match contains an illegal move") from exc
        if index < len(opening):
            initial = opening[index]
            if (initial.get("player"), int(initial.get("row", -1)), int(initial.get("col", -1))) != (
                expected, coord[0], coord[1]
            ) or move.get("opening") is not True:
                raise ValueError("evaluation match opening does not match its move log")
        elif move.get("opening") is True:
            raise ValueError("evaluation match marks a later move as an opening move")
        if board.check_victory(coord, expected):
            winner, termination = expected, "five_in_row"
            if index != len(moves) - 1:
                raise ValueError("evaluation match continued after a win")
        elif board.is_full():
            winner, termination = None, "full_board"
            if index != len(moves) - 1:
                raise ValueError("evaluation match continued after a full-board draw")
        expected = "W" if expected == "B" else "B"
    if len(opening) >= len(moves):
        raise ValueError("evaluation record contains no post-opening move")
    if match.get("termination") != termination or match.get("winner_color") != winner:
        raise ValueError("evaluation match terminal metadata does not match replay")
    color = match.get("agent_color")
    if color not in {"B", "W"}:
        raise ValueError("evaluation match has invalid agent_color")
    outcome = "draw" if winner is None else "win" if winner == color else "loss"
    score = 0.5 if winner is None else 1.0 if winner == color else 0.0
    if match.get("outcome_for_agent") != outcome or float(match.get("score_for_agent")) != score:
        raise ValueError("evaluation match score does not match the replayed result")


def append_evaluation_match(path: str | Path, match: dict[str, Any], seen_ids: set[str]) -> bool:
    validate_match_record(match)
    game_id = str(match["game_id"])
    if game_id in seen_ids:
        return False
    payload = (json.dumps(match, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("failed to append complete evaluation match")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    seen_ids.add(game_id)
    return True


def load_evaluation_log(path: str | Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Read legal completed matches and truncate an interrupted JSONL tail."""
    target = Path(path)
    if not target.exists():
        return [], set()
    raw = target.read_bytes()
    last_newline = raw.rfind(b"\n")
    complete = raw[:last_newline + 1] if last_newline >= 0 else b""
    if len(complete) != len(raw):
        with target.open("r+b") as handle:
            handle.truncate(len(complete))
            handle.flush()
            os.fsync(handle.fileno())
    matches: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(complete.splitlines(), start=1):
        try:
            match = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid evaluation JSONL record at line {line_number}") from exc
        validate_match_record(match)
        game_id = str(match["game_id"])
        if game_id in ids:
            raise ValueError(f"duplicate evaluation game id: {game_id}")
        matches.append(match)
        ids.add(game_id)
    return matches, ids


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
