"""Paired G2500/G5000 final challenge against the original A/B/C engines."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard
from .v5_evaluation import (
    _IndependentRuleBoard, _project_board, other, paired_bootstrap_ci,
    summarize_paired_games,
)
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board
from .v6_3_training import ARTIFACT, ARM, MAX_NEW_CPU_SECONDS, ROOT, sha_file


OPPONENTS = ("A", "B", "C")
MODELS = ("G2500", "G5000")
PAIRS = 40
GAMES_PER_CONDITION = 80
TRAINING_CPU_CAP_SECONDS = 45 * 3600
FINAL_AUDIT_RESERVE_SECONDS = 3600


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"v63-{digest}"


def _paired_game_seed(opponent: str, opening_group_id: str, color: str) -> int:
    """Share a fixed RNG seed between G2500/G5000 for the same paired game."""
    digest = hashlib.sha256(
        f"v6.3|{opponent}|{opening_group_id}|{color}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _make_agent(name: str, model: ReferencePolicyValueNet | None) -> dict[str, Any]:
    if name.startswith("G"):
        if model is None:
            raise ValueError("AlphaZero candidate model is required")
        player = instrumented_player(model.policy_value_fn, c_puct=5.0, n_playout=401, is_selfplay=False)
        return {"kind": "G", "model": model, "player": player}
    if name in OPPONENTS:
        return {"kind": name}
    raise ValueError(f"unknown V6.3 agent {name}")


def _agent_choice(agent: dict[str, Any], board, color: str, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
    kind = agent["kind"]
    started_cpu = time.process_time()
    if kind == "G":
        move, _policy = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(move), int(board.width))
        result = dict(agent["player"].last_search_stats)
        result["simulations"] = int(agent["player"].mcts._n_playout - 1)
        result["root_action_visits"] = int(result.get("root_visits", 0))
        result["search_depth"] = int(result.get("tree_depth", 0))
        result["search_nodes"] = int(result.get("tree_nodes", 0))
        return (row, col), {**result, "cpu_seconds": time.process_time() - started_cpu}
    projected = _project_board(board)
    time_limit = 1e9 if kind == "A" else 0.05
    result = choose_move(projected, color, kind,
                         SearchConfig(time_limit_seconds=time_limit), int(seed))
    return result.move, {
        "simulations": int(result.simulations),
        "root_action_visits": 0,
        "search_depth": int(result.completed_depth),
        "search_nodes": int(result.nodes),
        "cutoffs": int(result.cutoffs),
        "cpu_seconds": time.process_time() - started_cpu,
        "elapsed_seconds": float(result.elapsed_seconds),
        "timed_out": bool(result.timed_out),
    }


def play_match(*, game_id: str, candidate_name: str, opponent_name: str,
               opening: dict[str, Any], candidate_color: str, seed: int,
               candidate_model: ReferencePolicyValueNet) -> dict[str, Any]:
    if candidate_color not in {"B", "W"}:
        raise ValueError("candidate color must be black or white")
    candidate = _make_agent(candidate_name, candidate_model)
    opponent = _make_agent(opponent_name, None)
    board = make_board(9, 5, 0)
    independent = _IndependentRuleBoard(9, 5)
    stored_moves: list[dict[str, Any]] = []
    expected_color = "B"
    for item in opening.get("moves", []):
        color = str(item["player"])
        row, col = int(item["row"]), int(item["col"])
        if color != expected_color:
            raise ValueError("opening colors do not alternate from black")
        flat = row * 9 + col
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError("V6.3 opening contains an illegal move")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        if board.game_end()[0] or independent.check_victory((row, col), color):
            raise ValueError("V6.3 opening is terminal")
        stored_moves.append({"player": color, "row": row, "col": col, "opening": True})
        expected_color = other(color)

    np.random.seed(int(seed) & 0xFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    winner: str | None = None
    termination = ""
    candidate_cpu = opponent_cpu = 0.0
    while True:
        color = "B" if board.current_player == 1 else "W"
        is_candidate = color == candidate_color
        agent = candidate if is_candidate else opponent
        actor_name = candidate_name if is_candidate else opponent_name
        move_seed = int(rng.integers(0, 2**31 - 1))
        decision_started = time.perf_counter()
        coord, metrics = _agent_choice(agent, board, color, move_seed)
        elapsed_wall = time.perf_counter() - decision_started
        if is_candidate:
            candidate_cpu += float(metrics.get("cpu_seconds", 0.0))
        else:
            opponent_cpu += float(metrics.get("cpu_seconds", 0.0))
        row, col = coord
        flat = row * 9 + col
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError(f"{actor_name} selected illegal move {(row, col)}")
        board.do_move(flat)
        independent.place_stone(color, coord)
        stored_moves.append({"player": color, "row": row, "col": col,
                             "opening": False, "actor": actor_name,
                             "elapsed_seconds": elapsed_wall, **metrics})
        engine_end, engine_winner = board.game_end()
        project_end = independent.check_victory(coord, color)
        if bool(engine_end) != bool(project_end or independent.is_full()):
            raise AssertionError("reference engine and independent rule oracle disagree")
        if project_end:
            winner, termination = color, "five_in_row"
            if engine_winner != board.current_player ^ 3:
                raise AssertionError("reference engine winner differs from independent replay")
            break
        if independent.is_full():
            winner, termination = None, "full_board"
            break
    points = 0.5 if winner is None else 1.0 if winner == candidate_color else 0.0
    return {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
        "game_id": game_id, "source": "new_paired_challenge",
        "candidate": candidate_name, "opponent": opponent_name,
        "board_size": 9, "n_in_row": 5, "candidate_color": candidate_color,
        "winner_color": winner, "points": points, "termination": termination,
        "opening_group_id": opening["opening_group_id"],
        "opening_seed": int(opening["seed"]), "opening": opening["moves"],
        "seed": int(seed), "candidate_search": "reference PUCT-400",
        "opponent_config": ({"A": {"engine": "rules A", "time_limit_seconds": 1e9},
                             "B": {"engine": "pure Minimax", "time_limit_seconds": 0.05},
                             "C": {"engine": "Alpha-Beta", "time_limit_seconds": 0.05}}[opponent_name]),
        "simulations_per_candidate_turn": 400,
        "candidate_configured_playouts_per_turn": 401,
        "moves": stored_moves, "move_count": len(stored_moves),
        "candidate_cpu_seconds": candidate_cpu, "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }


def _validate_existing(game: dict[str, Any], *, game_id: str, candidate: str,
                       opponent: str, opening: dict[str, Any], color: str) -> None:
    expected = {"game_id": game_id, "candidate": candidate, "opponent": opponent,
                "opening_group_id": opening["opening_group_id"], "candidate_color": color,
                "simulations_per_candidate_turn": 400}
    for key, value in expected.items():
        if game.get(key) != value:
            raise ValueError(f"V6.3 duplicate ID/config collision in stored match: {key}")
    if len(game.get("moves", [])) != int(game.get("move_count", -1)) or not game.get("termination"):
        raise ValueError("stored V6.3 game is incomplete")


def _complete_pair_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["opening_group_id"]), []).append(row)
    paired = []
    for opening_id, pair in sorted(groups.items()):
        colors = [str(row["candidate_color"]) for row in pair]
        if len(colors) != len(set(colors)):
            raise ValueError(f"opening {opening_id} repeats a candidate color")
        if len(pair) == 2 and set(colors) == {"B", "W"}:
            paired.extend(pair)
    return paired, len(paired) // 2


def run_all(root: str | Path = ROOT, *, cpu_soft_limit_seconds: float | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    if len(schedule["openings"]) != PAIRS or schedule["validation"]["historical_d4_overlap"] != 0:
        raise ValueError("V6.3 fixed opening schedule does not satisfy registration")
    milestone_path = artifact / "checkpoints" / ARM / "node_games_5000.pt"
    state_path = artifact / "checkpoints" / ARM / "training_state.pt"
    if not milestone_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("final G5000 training state/model must be complete before the challenge")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state["game_count"]) != 5000:
        raise ValueError("the final challenge may only begin after exact G5000")
    training_base = float(torch.load(artifact / "checkpoints" / ARM / "training_state_games_2500.pt",
                                     map_location="cpu", weights_only=False)["cumulative_cpu_seconds"])
    model_nodes = {}
    for model_name in MODELS:
        node_path = artifact / "checkpoints" / ARM / f"node_games_{2500 if model_name == 'G2500' else 5000}.pt"
        node = torch.load(node_path, map_location="cpu", weights_only=False)
        if int(node["game_count"]) != (2500 if model_name == "G2500" else 5000):
            raise ValueError(f"model checkpoint mismatch for {model_name}")
        if model_name == "G2500" and node_path.stat().st_size == 0:
            raise ValueError("G2500 comparison checkpoint is empty")
        model_nodes[model_name] = node

    total_match_cpu_before = sum(
        float(json.loads(path.read_text(encoding="utf-8")).get("total_cpu_seconds", 0.0))
        for path in (artifact / "results/matches").rglob("*.json") if path.is_file()
        and path.name != "summary.json"
    ) if (artifact / "results/matches").exists() else 0.0
    stop_total = float(cpu_soft_limit_seconds) if cpu_soft_limit_seconds is not None else float(MAX_NEW_CPU_SECONDS - FINAL_AUDIT_RESERVE_SECONDS)
    completed_summaries: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for model_name in MODELS:
        model_node = model_nodes[model_name]
        model = ReferencePolicyValueNet(9, seed=int(model_node.get("base_seed", 590502)))
        model.load_policy_param(model_node["model_state"])
        model.network.eval()
        for opponent in OPPONENTS:
            condition = f"{model_name}_vs_{opponent}"
            directory = artifact / "results/matches" / condition
            directory.mkdir(parents=True, exist_ok=True)
            rows = []
            for opening in schedule["openings"]:
                for color in ("B", "W"):
                    if total_match_cpu_before + (float(state["cumulative_cpu_seconds"]) - training_base) >= stop_total:
                        break
                    game_id = _stable_id("v6.3", model_name, opponent,
                                         str(opening["opening_group_id"]), color)
                    path = directory / f"{game_id}.json"
                    if path.exists():
                        game = json.loads(path.read_text(encoding="utf-8"))
                        _validate_existing(game, game_id=game_id, candidate=model_name,
                                           opponent=opponent, opening=opening, color=color)
                    else:
                        game_seed = _paired_game_seed(opponent, str(opening["opening_group_id"]), color)
                        game = play_match(game_id=game_id, candidate_name=model_name,
                                          opponent_name=opponent, opening=opening,
                                          candidate_color=color, seed=game_seed,
                                          candidate_model=model)
                        _atomic_json(path, game)
                        total_match_cpu_before += float(game["total_cpu_seconds"])
                    rows.append(game)
                if len(rows) and (total_match_cpu_before + (float(state["cumulative_cpu_seconds"]) - training_base) >= stop_total):
                    break
            all_rows[condition] = rows
            paired_rows, completed_pairs = _complete_pair_rows(rows)
            summary = summarize_paired_games(paired_rows, seed=int.from_bytes(hashlib.sha256(condition.encode()).digest()[:4], "big"))
            summary.update({
                "candidate": model_name, "opponent": opponent,
                "search_budget": "candidate PUCT-400; opponent A natural / B,C 50ms",
                "planned_pairs": PAIRS, "planned_games": GAMES_PER_CONDITION,
                "completed_pairs": completed_pairs,
                "raw_completed_games": len(rows),
                "unpaired_completed_games": len(rows) - 2 * completed_pairs,
                "complete": len(rows) == GAMES_PER_CONDITION and completed_pairs == PAIRS,
                "candidate_model_sha256": sha_file(artifact / "checkpoints" / ARM /
                                                     f"node_games_{2500 if model_name == 'G2500' else 5000}.pt"),
                "game_rows": [{key: row.get(key) for key in (
                    "game_id", "opening_group_id", "candidate_color", "winner_color", "points",
                    "termination", "move_count", "candidate_cpu_seconds", "opponent_cpu_seconds",
                    "total_cpu_seconds", "total_wall_seconds")}
                    for row in rows],
            })
            _atomic_json(directory / "summary.json", summary)
            completed_summaries[condition] = summary
            if not summary["complete"]:
                break
        if not all(completed_summaries.get(f"{model_name}_vs_{opponent}", {}).get("complete", False)
                   for opponent in OPPONENTS):
            break
    aggregate = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
        "opening_schedule_sha256": hashlib.sha256(json.dumps(schedule, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "training_new_cpu_seconds": float(state["cumulative_cpu_seconds"]) - training_base,
        "evaluation_cpu_seconds": total_match_cpu_before,
        "planned_games": 480,
        "completed_games": sum(int(summary.get("raw_completed_games", 0)) for summary in completed_summaries.values()),
        "complete": len(completed_summaries) == 6 and all(row.get("complete") for row in completed_summaries.values()),
        "conditions": completed_summaries,
    }
    _atomic_json(artifact / "results/final_challenge.json", aggregate)
    return aggregate

