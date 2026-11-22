"""Paired, color-swapped V6.7 stage evaluation; never changes training state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard
from . import v5_evaluation as v5_eval
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board
from .v5_training import _atomic_json
from .v6_7_training import ARTIFACT, ARM, MILESTONES, ROOT, SIMULATIONS, atomic_json, record_aux_cpu, sha_file

DEV_SEED = 670_701
BLIND_SEED = 670_702
OPENING_PAIRS = 10
OPENING_PLIES = 6
OPPONENTS = ("G8000", "A", "C@50ms", "C@200ms")


def _model_path(root: Path, games: int) -> Path:
    return root / ARTIFACT / "models" / f"G{int(games)}.pt"


def _load_model(root: Path, games: int, cache: dict[int, ReferencePolicyValueNet]) -> ReferencePolicyValueNet:
    games = int(games)
    if games in cache:
        return cache[games]
    path = _model_path(root, games)
    node = torch.load(path, map_location="cpu", weights_only=False)
    if int(node.get("game_count", node.get("games_completed", -1))) != games:
        raise ValueError(f"model file {path.name} has the wrong game count")
    if int(node.get("simulations_per_move", -1)) != SIMULATIONS:
        raise ValueError(f"model file {path.name} is not registered as PUCT-400")
    model = ReferencePolicyValueNet(9, seed=590_501)
    model.load_policy_param(node["model_state"])
    model.network.eval()
    cache[games] = model
    return model


def _write_opening_pools(root: Path) -> dict[str, Any]:
    path = root / ARTIFACT / "config" / "opening_pools.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("development_seed") != DEV_SEED or value.get("blind_seed") != BLIND_SEED:
            raise ValueError("existing opening pools differ from the pre-registered seeds")
        if len(value.get("development", [])) != OPENING_PAIRS or len(value.get("blind", [])) != OPENING_PAIRS:
            raise ValueError("existing opening pools have the wrong size")
        return value
    dev = v5_eval.generate_opening_pool(size=9, n_in_row=5, groups=OPENING_PAIRS,
                                        plies=OPENING_PLIES, seed=DEV_SEED)
    blind = v5_eval.generate_opening_pool(size=9, n_in_row=5, groups=OPENING_PAIRS,
                                          plies=OPENING_PLIES, seed=BLIND_SEED)
    dev_keys = {v5_eval.opening_key(row["moves"], 9) for row in dev}
    blind_keys = {v5_eval.opening_key(row["moves"], 9) for row in blind}
    if len(dev_keys) != OPENING_PAIRS or len(blind_keys) != OPENING_PAIRS or dev_keys & blind_keys:
        raise ValueError("opening pools have a duplicate or development/blind overlap")
    value = {"board_size": 9, "n_in_row": 5, "opening_plies": OPENING_PLIES,
             "pairs_per_condition": OPENING_PAIRS, "development_seed": DEV_SEED,
             "blind_seed": BLIND_SEED, "development": dev, "blind": blind,
             "development_d4_unique": True, "blind_d4_unique": True,
             "pools_disjoint_under_d4": True}
    atomic_json(path, value)
    return value


def create_opening_pools(root: str | Path = ROOT) -> dict[str, Any]:
    return _write_opening_pools(Path(root).resolve())


def _project_reference_board(board) -> GomokuBoard:
    if int(board.width) != 9:
        raise ValueError("V6.7 evaluation is frozen to the 9x9 board")
    converted = GomokuBoard(9)
    for move, player in board.states.items():
        converted.place_stone("B" if int(player) == 1 else "W", divmod(int(move), 9))
    return converted


def _agent_choice(agent: dict[str, Any], board, color: str, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
    kind = agent["kind"]
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    if kind == "reference":
        move, _ = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(move), int(board.width))
        stats = dict(agent["player"].last_search_stats)
        return (row, col), {
            "algorithm": "AlphaZero-reference-PUCT",
            "configured_simulations": SIMULATIONS,
            "root_action_visits": int(stats.get("root_visits", 0)),
            "search_depth": int(stats.get("tree_depth", 0)),
            "search_nodes": int(stats.get("tree_nodes", 0)),
            "network_root_value": stats.get("root_network_value"),
            "cpu_seconds": time.process_time() - cpu_started,
            "elapsed_seconds": time.perf_counter() - wall_started,
            "timed_out": False,
        }
    project = _project_reference_board(board)
    if kind == "A":
        config = SearchConfig(max_depth=69, candidate_radius=2, time_limit_seconds=1e9)
        result = choose_move(project, color, "A", config, int(seed))
    elif kind in {"C@50ms", "C@200ms"}:
        seconds = 0.05 if kind == "C@50ms" else 0.2
        config = SearchConfig(max_depth=69, candidate_radius=2, time_limit_seconds=seconds)
        result = choose_move(project, color, "C", config, int(seed))
    else:
        raise ValueError(f"unsupported opponent kind: {kind}")
    return result.move, {
        "algorithm": "rule-A" if kind == "A" else "Alpha-Beta",
        "configured_time_limit_seconds": config.time_limit_seconds,
        "max_depth": config.max_depth,
        "candidate_radius": config.candidate_radius,
        "search_depth": int(result.completed_depth),
        "search_nodes": int(result.nodes),
        "cutoffs": int(result.cutoffs),
        "cpu_seconds": time.process_time() - cpu_started,
        "elapsed_seconds": time.perf_counter() - wall_started,
        "timed_out": bool(result.timed_out),
    }


def _make_agent(name: str, model: ReferencePolicyValueNet | None = None) -> dict[str, Any]:
    if name in {"A", "C@50ms", "C@200ms"}:
        return {"kind": name}
    if model is None:
        raise ValueError(f"reference model required for {name}")
    player = instrumented_player(model.policy_value_fn, c_puct=5.0,
                                n_playout=SIMULATIONS, is_selfplay=False)
    return {"kind": "reference", "player": player}


def play_match(*, root: Path, game_id: str, candidate_games: int, opponent_name: str,
               opening: dict[str, Any], candidate_color: str, seed: int,
               candidate_sha256: str, opponent_sha256: str) -> dict[str, Any]:
    if candidate_color not in {"B", "W"}:
        raise ValueError("candidate_color must be B or W")
    cache: dict[int, ReferencePolicyValueNet] = getattr(play_match, "_model_cache", {})
    setattr(play_match, "_model_cache", cache)
    candidate_model = _load_model(root, candidate_games, cache)
    opponent_games = BASELINE_GAMES if opponent_name == "G8000" else None
    opponent_model = _load_model(root, opponent_games, cache) if opponent_games else None
    candidate = _make_agent(f"G{candidate_games}", candidate_model)
    opponent = _make_agent(opponent_name, opponent_model)
    board = make_board(9, 5, 0)
    independent = v5_eval._IndependentRuleBoard(9, 5)
    opening_moves = opening.get("moves", [])
    expected = "B"
    stored_moves: list[dict[str, Any]] = []
    for move in opening_moves:
        color = str(move["player"])
        if color != expected:
            raise ValueError("opening colors do not alternate from black")
        row, col = int(move["row"]), int(move["col"])
        flat = row * 9 + col
        if flat not in board.availables:
            raise ValueError("registered opening contains an illegal move")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        if board.game_end()[0] or independent.check_victory((row, col), color):
            raise ValueError("registered opening is terminal")
        stored_moves.append({"player": color, "row": row, "col": col, "opening": True})
        expected = "W" if expected == "B" else "B"

    np.random.seed(int(seed) & 0xFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    candidate_cpu = opponent_cpu = 0.0
    winner: str | None = None
    termination = ""
    ply = len(stored_moves)
    while True:
        color = "B" if int(board.current_player) == 1 else "W"
        is_candidate = color == candidate_color
        agent = candidate if is_candidate else opponent
        move_seed = int(rng.integers(0, 2**31 - 1))
        coord, metrics = _agent_choice(agent, board, color, move_seed)
        row, col = coord
        flat = row * 9 + col
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError(f"{agent['kind']} chose an illegal move {(row, col)}")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        stored_moves.append({"player": color, "row": row, "col": col, "opening": False,
                             "actor": f"G{candidate_games}" if is_candidate else opponent_name,
                             **metrics})
        if is_candidate:
            candidate_cpu += float(metrics["cpu_seconds"])
        else:
            opponent_cpu += float(metrics["cpu_seconds"])
        end, engine_winner = board.game_end()
        independent_win = independent.check_victory((row, col), color)
        if bool(end) != bool(independent_win or independent.is_full()):
            raise AssertionError("reference engine and independent rule oracle disagree")
        ply += 1
        if independent_win:
            winner, termination = color, "five_in_row"
            if engine_winner != (1 if color == "B" else 2):
                raise AssertionError("engine winner differs from independent legal replay")
            break
        if independent.is_full():
            winner, termination = None, "full_board"
            break
    points = 0.5 if winner is None else 1.0 if winner == candidate_color else 0.0
    return {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
        "game_id": game_id, "source": "evaluation", "candidate": f"G{candidate_games}",
        "candidate_games": candidate_games, "opponent": opponent_name,
        "candidate_model_sha256": candidate_sha256, "opponent_model_sha256": opponent_sha256,
        "board_size": 9, "n_in_row": 5, "simulations_per_reference_turn": SIMULATIONS,
        "c_protocol": {"algorithm": "Alpha-Beta", "max_depth": 69,
                       "candidate_radius": 2, "timeout_returns_last_complete_depth": True},
        "candidate_color": candidate_color, "winner_color": winner, "points": points,
        "termination": termination, "opening_group_id": opening["opening_group_id"],
        "opening": opening_moves, "seed": int(seed), "moves": stored_moves,
        "move_count": len(stored_moves), "candidate_cpu_seconds": candidate_cpu,
        "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }


BASELINE_GAMES = 8_000


def _game_path(root: Path, game_id: str) -> Path:
    return root / ARTIFACT / "results/evaluations/games" / f"{game_id}.json"


def run_condition(root: str | Path, *, candidate_games: int, opponent_name: str,
                  pool: str = "development") -> dict[str, Any]:
    root = Path(root).resolve()
    torch.set_num_threads(1)
    if candidate_games not in MILESTONES:
        raise ValueError("only registered V6.7 milestone checkpoints can be evaluated")
    if opponent_name not in OPPONENTS:
        raise ValueError(f"opponent is not registered: {opponent_name}")
    if pool not in {"development", "blind"} or (pool == "blind" and candidate_games != 20_000):
        raise ValueError("blind openings are reserved for the frozen G20000 candidate")
    pools = _write_opening_pools(root)
    openings = pools[pool]
    candidate_path = _model_path(root, candidate_games)
    candidate_sha = sha_file(candidate_path)
    if opponent_name == "G8000":
        opponent_sha = sha_file(_model_path(root, BASELINE_GAMES))
    else:
        opponent_sha = "not_applicable"
    games: list[dict[str, Any]] = []
    color_seed_offset = DEV_SEED if pool == "development" else BLIND_SEED
    for pair_index, opening in enumerate(openings):
        for color_index, candidate_color in enumerate(("B", "W")):
            game_id = f"G{candidate_games}_vs_{opponent_name.replace('@', '').replace('-', '')}_{pool}_{pair_index:02d}_{candidate_color}"
            path = _game_path(root, game_id)
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                if (record.get("candidate_model_sha256") != candidate_sha
                        or record.get("opponent") != opponent_name
                        or record.get("opening_group_id") != opening["opening_group_id"]
                        or record.get("candidate_color") != candidate_color):
                    raise ValueError(f"existing evaluation game conflicts with frozen schedule: {game_id}")
                games.append(record)
                continue
            seed = int(color_seed_offset + pair_index * 2 + color_index)
            record = play_match(root=root, game_id=game_id, candidate_games=candidate_games,
                                opponent_name=opponent_name, opening=opening,
                                candidate_color=candidate_color, seed=seed,
                                candidate_sha256=candidate_sha, opponent_sha256=opponent_sha)
            _atomic_json(path, record)
            record_aux_cpu(root, "evaluation_cpu_seconds", float(record["total_cpu_seconds"]), unique_id=game_id)
            progress = root / ARTIFACT / "results/evaluations/progress.json"
            existing = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {"completed_game_ids": []}
            ids = set(existing.get("completed_game_ids", []))
            ids.add(game_id)
            atomic_json(progress, {"completed_game_ids": sorted(ids), "count": len(ids),
                                  "last_game_id": game_id, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            games.append(record)
    summary = v5_eval.summarize_paired_games(games, seed=color_seed_offset + candidate_games)
    summary.update({"candidate": f"G{candidate_games}", "opponent": opponent_name,
                    "pool": pool, "model_sha256": candidate_sha,
                    "opening_group_ids": [row["opening_group_id"] for row in openings],
                    "condition_complete": len(games) == OPENING_PAIRS * 2,
                    "max_depth": 69 if opponent_name.startswith("C@") else None,
                    "candidate_radius": 2 if opponent_name.startswith("C@") else None})
    summary_path = root / ARTIFACT / "results/evaluations/summaries" / f"G{candidate_games}_{opponent_name.replace('@', '').replace('-', '')}_{pool}.json"
    atomic_json(summary_path, summary)
    return summary


def run_stage(root: str | Path, *, candidate_games: int, pool: str = "development") -> list[dict[str, Any]]:
    if pool == "blind" and candidate_games != 20_000:
        raise ValueError("blind evaluation cannot be used before G20000")
    results = []
    for opponent in OPPONENTS:
        results.append(run_condition(root, candidate_games=candidate_games,
                                     opponent_name=opponent, pool=pool))
    return results

