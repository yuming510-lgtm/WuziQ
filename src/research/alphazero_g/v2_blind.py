"""Frozen, paired blind evaluation for the WuziQ G V2 growth experiment."""
from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...gomoku import GomokuBoard, Player
from .evaluation import (
    F2_SHA256,
    _network_move,
    append_evaluation_match,
    load_evaluation_log,
    load_frozen_opponent,
    play_evaluation_match,
    summarize_matches,
    validate_match_record,
)
from .puct import PUCTSearch
from .training import load_g_checkpoint
from .v2_budget import ensure_cpu_room
from .v2_schedule import validate_v2_schedule
from .v2_selection import _game_seed, _load_checkpoint

CONDITIONS = (
    ("F2-network", "F2", "network"),
    ("F2-PUCT64", "F2", "puct"),
    ("G-F2-network", "G", "network"),
    ("G-F2-PUCT64", "G", "puct"),
)
CONDITION_BY_ID = {condition_id: (checkpoint_key, mode) for condition_id, checkpoint_key, mode in CONDITIONS}
DEVICE = "cpu"


def build_v2_blind_plan(
    schedule: dict[str, Any], *, opponents: tuple[str, ...] = ("A", "B")
) -> list[dict[str, Any]]:
    """Freeze every paired model matchup plus the registered A/B comparisons."""
    validate_v2_schedule(schedule)
    if not opponents or not set(opponents).issubset({"A", "B"}):
        raise ValueError("blind opponents must be a nonempty subset of A and B")
    plan: list[dict[str, Any]] = []
    comparisons = [
        (left[0], right[0]) for left, right in combinations(CONDITIONS, 2)
    ] + [(condition[0], opponent) for condition in CONDITIONS for opponent in opponents]
    for left_id, right_id in comparisons:
        left_mode = CONDITION_BY_ID[left_id][1]
        comparison_id = f"{left_id}__vs__{right_id}"
        for group in schedule["blind_openings"]:
            for color in ("B", "W"):
                game_id = f"g-v2-blind-{left_id}-vs-{right_id}-{group['opening_group_id']}-{color}"
                plan.append({
                    "game_id": game_id,
                    "comparison_id": comparison_id,
                    "opening_group_id": group["opening_group_id"],
                    "opening": [dict(move) for move in group["opening"]],
                    "agent_color": color,
                    "left_condition": left_id,
                    "right_condition": right_id,
                    "mode": left_mode,
                })
    ids = [row["game_id"] for row in plan]
    if len(ids) != len(set(ids)):
        raise ValueError("blind plan contains duplicate game IDs")
    return plan


def _checkpoint_file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_models(
    root: Path,
    checkpoints: dict[str, str | Path],
    device: str,
) -> tuple[dict[str, torch.nn.Module], dict[str, str], dict[str, str]]:
    if set(checkpoints) != {"F2", "G"}:
        raise ValueError("blind evaluation requires exactly frozen F2 and selected G checkpoints")
    models: dict[str, torch.nn.Module] = {}
    file_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    for checkpoint_id, key, mode in CONDITIONS:
        source = Path(checkpoints[key])
        if not source.is_absolute():
            source = root / source
        if not source.is_file():
            raise FileNotFoundError(f"missing blind checkpoint {key}: {source}")
        file_sha = _checkpoint_file_sha(source)
        if key == "F2":
            if file_sha != F2_SHA256:
                raise ValueError("blind F2 path is not the frozen registered checkpoint")
            model = load_frozen_opponent("F2", device=device)
            state = model.state_dict()
            weights_digest = hashlib.sha256(b"".join(
                name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
                for name, tensor in sorted(state.items())
            )).hexdigest()
        else:
            model, weights_digest = _load_checkpoint(root, source, "G@final", device)
        model.eval()
        models[checkpoint_id] = model
        file_hashes[key] = file_sha
        model_hashes[checkpoint_id] = weights_digest
    return models, file_hashes, model_hashes


def _play_model_match(
    models: dict[str, torch.nn.Module],
    *,
    game: dict[str, Any],
    left_id: str,
    right_id: str,
    left_sha: str,
    right_sha: str,
    left_weights_sha: str,
    right_weights_sha: str,
    seed: int,
    device: str,
    simulations: int = 64,
) -> dict[str, Any]:
    left_model, right_model = models[left_id], models[right_id]
    left_mode, right_mode = CONDITION_BY_ID[left_id][1], CONDITION_BY_ID[right_id][1]
    board = GomokuBoard(9)
    moves: list[dict[str, Any]] = []
    expected: Player = "B"
    for item in game["opening"]:
        if item["player"] != expected:
            raise ValueError("blind opening must alternate from black")
        coord = (int(item["row"]), int(item["col"]))
        board.place_stone(expected, coord)
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("blind opening must be legal and nonterminal")
        moves.append({"player": expected, "row": coord[0], "col": coord[1], "opening": True})
        expected = "W" if expected == "B" else "B"
    to_move = expected
    left_color: Player = game["agent_color"]
    searchers = {
        left_id: PUCTSearch(left_model, device=device) if left_mode == "puct" else None,
        right_id: PUCTSearch(right_model, device=device) if right_mode == "puct" else None,
    }
    rng = np.random.default_rng(int(seed))
    started, cpu_started = time.perf_counter(), time.process_time()
    seconds = {left_id: 0.0, right_id: 0.0}
    sims = {left_id: 0, right_id: 0}
    nodes = {left_id: 0, right_id: 0}
    depths = {left_id: 0, right_id: 0}
    evaluations = {left_id: 0, right_id: 0}
    winner: Player | None = None
    termination = ""
    device_obj = torch.device(device)
    while True:
        actor_id = left_id if to_move == left_color else right_id
        actor_model = models[actor_id]
        decision_started = time.perf_counter()
        if CONDITION_BY_ID[actor_id][1] == "puct":
            search = searchers[actor_id]
            assert search is not None
            result = search.search(
                board, to_move, simulations, add_root_noise=False, temperature=0.0,
                seed=int(rng.integers(0, 2**63 - 1)),
            )
            coord = result.move
            extra = {
                "simulations": int(result.simulations), "nodes": int(result.tree_nodes),
                "network_evaluations": int(result.network_evaluations),
                "search_depth": int(result.max_depth), "root_value": float(result.root_value),
                "search_entropy": float(result.visit_entropy),
            }
            sims[actor_id] += int(result.simulations)
            nodes[actor_id] += int(result.tree_nodes)
            depths[actor_id] = max(depths[actor_id], int(result.max_depth))
            evaluations[actor_id] += int(result.network_evaluations)
        else:
            coord, extra = _network_move(actor_model, board, to_move, device_obj)
            extra = {"simulations": 0, "nodes": 0, "network_evaluations": 1,
                     "search_depth": 0, **extra}
            evaluations[actor_id] += 1
        elapsed = time.perf_counter() - decision_started
        seconds[actor_id] += elapsed
        moves.append({
            "player": to_move, "row": coord[0], "col": coord[1], "opening": False,
            "actor": actor_id, "elapsed_seconds": elapsed, **extra,
        })
        board.place_stone(to_move, coord)
        if board.check_victory(coord, to_move):
            winner, termination = to_move, "five_in_row"
            break
        if board.is_full():
            termination = "full_board"
            break
        to_move = "W" if to_move == "B" else "B"
    score = 0.5 if winner is None else 1.0 if winner == left_color else 0.0
    return {
        "format_version": 1, "source": "evaluation", "evaluation_phase": "blind",
        "game_id": game["game_id"], "opening_group_id": game["opening_group_id"],
        "comparison_id": game["comparison_id"], "left_condition": left_id,
        "right_condition": right_id, "candidate_id": left_id, "arm_id": left_id,
        "candidate_checkpoint_id": left_id, "candidate_checkpoint_sha256": left_sha,
        "opponent": right_id, "opponent_checkpoint_id": right_id,
        "opponent_checkpoint_sha256": right_sha, "left_weights_sha256": left_weights_sha,
        "right_weights_sha256": right_weights_sha, "mode": left_mode,
        "opponent_mode": right_mode, "simulations_per_agent_move": simulations if left_mode == "puct" else 0,
        "opening": [dict(item) for item in game["opening"]],
        "agent_color": left_color, "seed": int(seed), "completed": True,
        "termination": termination, "winner_color": winner,
        "outcome_for_agent": "draw" if winner is None else "win" if winner == left_color else "loss",
        "score_for_agent": score, "move_count": len(moves), "moves": moves,
        "stats": {
            "agent_seconds": seconds[left_id], "opponent_seconds": seconds[right_id],
            "total_wall_seconds": time.perf_counter() - started,
            "total_cpu_seconds": time.process_time() - cpu_started,
            "agent_simulations": sims[left_id], "opponent_simulations": sims[right_id],
            "agent_nodes": nodes[left_id], "opponent_nodes": nodes[right_id],
            "agent_network_evaluations": evaluations[left_id],
            "opponent_network_evaluations": evaluations[right_id],
            "agent_max_search_depth": depths[left_id],
            "opponent_max_search_depth": depths[right_id],
        },
    }


def run_v2_blind(
    root: str | Path,
    config: dict[str, Any],
    schedule: dict[str, Any],
    checkpoints: dict[str, str | Path],
    *,
    opponents: tuple[str, ...] = ("A", "B"),
    device: str | None = None,
    log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run or resume the fixed blind plan; blind outcomes never enter selection."""
    if config.get("registered") is not True:
        raise ValueError("blind evaluation requires the frozen registered G V2 config")
    validate_v2_schedule(schedule)
    registered_groups = int(config["evaluation"]["blind_groups"])
    if len(schedule["blind_openings"]) != registered_groups:
        raise ValueError("blind schedule count differs from the registered config")
    device = device or str(config.get("device", DEVICE))
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    root_path = Path(root).resolve()
    models, file_hashes, weights_hashes = _condition_models(root_path, checkpoints, device)
    plan = build_v2_blind_plan(schedule, opponents=opponents)
    target = Path(log_path) if log_path else root_path / "research/g_v2/results/evaluation/blind.jsonl"
    previous, seen = load_evaluation_log(target)
    existing = {str(row["game_id"]): row for row in previous}
    results: list[dict[str, Any]] = []
    seed = int(config["seed"])
    cpu_cap_seconds = float(config.get("hard_cpu_hour_cap", 2.0)) * 3600.0
    for game in plan:
        game_id = game["game_id"]
        left_id, right_id = game["left_condition"], game["right_condition"]
        left_key = CONDITION_BY_ID[left_id][0]
        record = existing.get(game_id)
        if record is None:
            ensure_cpu_room(root_path, cpu_cap_seconds)
            if right_id in CONDITION_BY_ID:
                right_key = CONDITION_BY_ID[right_id][0]
                record = _play_model_match(
                    models, game=game, left_id=left_id, right_id=right_id,
                    left_sha=file_hashes[left_key], right_sha=file_hashes[right_key],
                    left_weights_sha=weights_hashes[left_id],
                    right_weights_sha=weights_hashes[right_id],
                    seed=_game_seed(seed, game_id), device=device,
                )
            else:
                record = play_evaluation_match(
                    models[left_id], game_id=game_id,
                    opening_group_id=game["opening_group_id"], arm_id=left_id,
                    mode=game["mode"], opponent=right_id, opening=game["opening"],
                    agent_color=game["agent_color"], seed=_game_seed(seed, game_id),
                    simulations=64, device=device,
                )
                record.update({
                    "evaluation_phase": "blind", "comparison_id": game["comparison_id"],
                    "left_condition": left_id, "right_condition": right_id,
                    "candidate_id": left_id, "candidate_checkpoint_id": left_id,
                    "candidate_checkpoint_sha256": file_hashes[left_key],
                    "left_weights_sha256": weights_hashes[left_id],
                    "opponent_checkpoint_id": right_id,
                })
            append_evaluation_match(target, record, seen)
            existing[game_id] = record
        if (record.get("evaluation_phase") != "blind"
                or record.get("comparison_id") != game["comparison_id"]
                or record.get("candidate_checkpoint_sha256") != file_hashes[left_key]
                or record.get("left_weights_sha256") != weights_hashes[left_id]):
            raise ValueError("existing blind game ID belongs to another comparison or checkpoint")
        if right_id in CONDITION_BY_ID:
            right_key = CONDITION_BY_ID[right_id][0]
            if (record.get("opponent_checkpoint_sha256") != file_hashes[right_key]
                    or record.get("right_weights_sha256") != weights_hashes[right_id]):
                raise ValueError("existing blind game ID has a different opponent checkpoint")
        results.append(record)
    return results


def _invert_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(summary))
    result["wins"], result["losses"] = summary["losses"], summary["wins"]
    result["points"] = summary["games"] - summary["points"]
    result["score_rate"] = 1.0 - summary["score_rate"]
    low, high = summary["paired_opening_bootstrap_ci95"] or (None, None)
    result["paired_opening_bootstrap_ci95"] = None if low is None else [1.0 - high, 1.0 - low]
    for color in ("B", "W"):
        info = result["by_color"][color]
        info["wins"], info["losses"] = info.get("losses", 0), info.get("wins", 0)
        info["points"] = info["games"] - info["points"]
        info["score_rate"] = 1.0 - info["score_rate"]
    return result


def summarize_v2_results(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only complete blind records, paired by opening and color."""
    if not matches:
        raise ValueError("cannot summarize an empty blind result set")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in matches:
        validate_match_record(row)
        if row.get("evaluation_phase") != "blind" or not row.get("comparison_id"):
            raise ValueError("summary accepts complete blind records with comparison IDs only")
        grouped.setdefault(str(row["comparison_id"]), []).append(row)
    comparisons: dict[str, Any] = {}
    for comparison_id, rows in grouped.items():
        by_opening: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_opening.setdefault(str(row["opening_group_id"]), []).append(row)
        if any(len(pair) != 2 or {item["agent_color"] for item in pair} != {"B", "W"}
               for pair in by_opening.values()):
            raise ValueError(f"blind comparison {comparison_id} lacks paired color games")
        summary = summarize_matches(rows)
        stats = [row.get("stats", {}) for row in rows]
        summary["mean_agent_seconds_per_game"] = float(np.mean([float(item.get("agent_seconds", 0.0)) for item in stats]))
        summary["mean_opponent_seconds_per_game"] = float(np.mean([float(item.get("opponent_seconds", 0.0)) for item in stats]))
        summary["total_cpu_seconds"] = float(sum(float(item.get("total_cpu_seconds", 0.0)) for item in stats))
        summary["total_agent_simulations"] = int(sum(int(item.get("agent_simulations", 0)) for item in stats))
        summary["total_opponent_simulations"] = int(sum(int(item.get("opponent_simulations", 0)) for item in stats))
        summary["mean_agent_nodes_per_move"] = float(
            sum(int(item.get("agent_nodes", 0)) for item in stats)
            / max(1, sum(sum(1 for move in row["moves"] if not move.get("opening") and move.get("actor") in (row.get("candidate_id"), "agent")) for row in rows))
        )
        for color in ("B", "W"):
            color_rows = [row for row in rows if row["agent_color"] == color]
            color_wins = sum(row["outcome_for_agent"] == "win" for row in color_rows)
            color_draws = sum(row["outcome_for_agent"] == "draw" for row in color_rows)
            summary["by_color"][color]["wins"] = color_wins
            summary["by_color"][color]["draws"] = color_draws
            summary["by_color"][color]["losses"] = sum(row["outcome_for_agent"] == "loss" for row in color_rows)
        comparisons[comparison_id] = summary
    learning = {}
    f2_g = comparisons.get("F2-PUCT64__vs__G-F2-PUCT64")
    if f2_g is not None:
        g_view = _invert_summary(f2_g)
        learning["F2-PUCT64_to_G-F2-PUCT64"] = {
            "f2_score_rate": f2_g["score_rate"], "g_score_rate": g_view["score_rate"],
            "g_minus_f2_score_rate": g_view["score_rate"] - f2_g["score_rate"],
            "g_view": g_view,
        }
    search = {}
    for model in ("F2", "G-F2"):
        network_id = f"{model}-network"
        puct_id = f"{model}-PUCT64"
        row = comparisons.get(f"{network_id}__vs__{puct_id}")
        if row is not None:
            puct_view = _invert_summary(row)
            search[f"{model}_network_to_PUCT64"] = {
                "network_score_rate": row["score_rate"],
                "puct_score_rate": puct_view["score_rate"],
                "puct_minus_network_score_rate": puct_view["score_rate"] - row["score_rate"],
                "puct_view": puct_view,
            }
    return {"comparisons": comparisons, "learning": learning, "search": search}
