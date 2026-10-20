"""Resumable, paired 9x9 ABCD compute-ladder tournament (V5)."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from ..ai import SearchConfig, SearchResult, choose_move
from ..gomoku import GomokuBoard, Player
from .checkpoint import append_game, load_game_records, write_json_atomic
from .mcts_d2 import MCTSAgent

_MODELS = ("A", "B", "C", "D")
_MATCHUPS = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
_BUDGET_LABELS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")
_BUDGET_SECONDS = (0.05, 0.2, 1.0, 5.0, 10.0, 20.0, 50.0)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _wall_elapsed_seconds(started_at: str, ended_at: str | None = None) -> float:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat((ended_at or _utc_now()).replace("Z", "+00:00"))
    return max(0.0, (end - start).total_seconds())


def derive_seed(base_seed: int, *parts: object) -> int:
    """Stable unsigned seed independent of interpreter hash randomization."""
    payload = json.dumps([int(base_seed), *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load the frozen schedule and verify that the underlying engines are unchanged."""
    if isinstance(config_source, dict):
        config = deepcopy(config_source)
    else:
        config = json.loads(Path(config_source).read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported V5 config schema")
    if config.get("experiment_id") != "wuziq-abcd-ladder-v5-2026-10-20":
        raise ValueError("Unexpected V5 experiment ID")
    if int(config.get("board_size", 0)) != 9 or tuple(config.get("models", ())) != _MODELS:
        raise ValueError("V5 is frozen to A/B/C/D on a 9x9 board")
    if tuple(tuple(pair) for pair in config.get("matchups", ())) != _MATCHUPS:
        raise ValueError("V5 matchup list or order changed")
    ladder = config.get("budget_ladder", ())
    if tuple(item.get("label") for item in ladder) != _BUDGET_LABELS:
        raise ValueError("V5 must use the seven frozen budget rungs")
    if tuple(float(item.get("seconds", -1)) for item in ladder) != _BUDGET_SECONDS:
        raise ValueError("V5 budget seconds do not match labels")
    if config.get("budgets_seconds_by_model") != {"A": None, "B": 0.05, "C": 0.05, "D": 0.05}:
        raise ValueError("V5 baseline budgets are fixed to natural A and 50ms B/C/D")
    if int(config.get("games_per_group", 0)) != 2 or int(config.get("opening_plies", 0)) != 4:
        raise ValueError("Each V5 group requires two games and four opening plies")
    seeds = tuple(int(seed) for seed in config.get("opening_seeds", ()))
    catalog = config.get("opening_catalog", {})
    if len(seeds) != 12 or set(catalog) != {str(seed) for seed in seeds}:
        raise ValueError("V5 requires the frozen 12-position opening pool")
    for seed in seeds:
        stones = catalog[str(seed)]
        if len(stones) != 4:
            raise ValueError(f"V5 opening {seed} must contain four plies")
        board = GomokuBoard(9)
        for index, stone in enumerate(stones):
            if len(stone) != 3 or str(stone[0]) != ("B" if index % 2 == 0 else "W"):
                raise ValueError(f"V5 opening {seed} must alternate B-W-B-W")
            coord = int(stone[1]), int(stone[2])
            board.place_stone(str(stone[0]), coord)
            if board.check_victory(coord, str(stone[0])):
                raise ValueError(f"V5 opening {seed} is terminal")
    if int(config.get("inference_limits", {}).get("maximum_groups", 0)) != 60:
        raise ValueError("V5 theoretical maximum is fixed at 60 groups")
    if int(config.get("inference_limits", {}).get("maximum_games", 0)) != 120:
        raise ValueError("V5 theoretical maximum is fixed at 120 games")
    if int(config.get("ab_search", {}).get("max_depth", 0)) < 81:
        raise ValueError("V5 B/C search must remain iterative deepening through all empty cells")
    if int(config.get("d_search", {}).get("max_depth", 0)) != 4:
        raise ValueError("V5 D must retain the verified V4C MCTS configuration")
    root = _project_root()
    for relative, expected in config.get("engine_sha256", {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise ValueError(f"Pinned V1-V4C decision engine changed: {relative}")
    return config


def _opening(config: dict[str, Any], seed: int) -> list[tuple[Player, tuple[int, int]]]:
    return [(str(player), (int(row), int(col))) for player, row, col in config["opening_catalog"][str(seed)]]


def _route_id(challenger: str, opponent: str) -> str:
    return f"{challenger}>{opponent}"


def _pair_slug(pair: tuple[str, str]) -> str:
    return "-".join(model.lower() for model in pair)


def _group(
    config: dict[str, Any],
    pair_index: int,
    pair: tuple[str, str],
    *,
    level: int,
    challenger: str | None,
    shared: bool,
    direction_index: int = 0,
) -> dict[str, Any]:
    first, second = pair
    if shared:
        route_ids = [_route_id(first, second), _route_id(second, first)]
        route_slug = "shared-baseline"
        opening_index = (pair_index * 3) % len(config["opening_seeds"])
        budgets = {first: _BUDGET_SECONDS[0], second: _BUDGET_SECONDS[0]}
        challenger_budget_label = _BUDGET_LABELS[0]
        opponent_budget_label = _BUDGET_LABELS[0]
    else:
        if challenger is None:
            raise ValueError("A non-shared group must identify its challenger")
        opponent = second if challenger == first else first
        route_ids = [_route_id(challenger, opponent)]
        route_slug = f"{challenger.lower()}-to-{opponent.lower()}"
        opening_index = (pair_index * 3 + level + direction_index * 5) % len(config["opening_seeds"])
        if challenger == "A":
            budgets = {"A": None, opponent: _BUDGET_SECONDS[level]}
            challenger_budget_label = "natural"
            opponent_budget_label = _BUDGET_LABELS[level]
        else:
            budgets = {challenger: _BUDGET_SECONDS[level], opponent: _BUDGET_SECONDS[0]}
            challenger_budget_label = _BUDGET_LABELS[level]
            opponent_budget_label = _BUDGET_LABELS[0]
    opening_seed = int(config["opening_seeds"][opening_index])
    group_id = f"v5-9x9-{_pair_slug(pair)}-{route_slug}-r{level:02d}-o{opening_seed}"
    opening = [
        {"player": player, "row": coord[0], "col": coord[1]}
        for player, coord in _opening(config, opening_seed)
    ]
    games: list[dict[str, Any]] = []
    for color_swap in (False, True):
        black, white = pair if not color_swap else (pair[1], pair[0])
        game_id = f"{group_id}-s{int(color_swap)}"
        games.append({
            "experiment_id": config["experiment_id"],
            "group_id": group_id,
            "game_id": game_id,
            "board_size": 9,
            "pair": list(pair),
            "matchup": list(pair),
            "route_ids": route_ids,
            "challenge_level": level,
            "challenger_model": challenger,
            "shared_baseline": shared,
            "opening_index": opening_index,
            "opening_seed": opening_seed,
            "color_swap": color_swap,
            "black_model": black,
            "white_model": white,
            "pair_id": group_id,
            "game_seed": derive_seed(int(config["base_seed"]), "v5-game", game_id),
            "opening": opening,
            "budget_seconds_by_model": {model: budgets[model] if model in budgets else None for model in pair},
            "challenger_budget_label": challenger_budget_label,
            "opponent_budget_label": opponent_budget_label,
            "source_type": "new",
        })
    return {
        "experiment_id": config["experiment_id"],
        "group_id": group_id,
        "pair": list(pair),
        "route_ids": route_ids,
        "challenge_level": level,
        "challenger_model": challenger,
        "shared_baseline": shared,
        "opening_index": opening_index,
        "opening_seed": opening_seed,
        "challenger_budget_label": challenger_budget_label,
        "opponent_budget_label": opponent_budget_label,
        "rung_budget_label": _BUDGET_LABELS[level],
        "games": games,
    }


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the maximum 60 groups, sharing only each internal 50ms baseline."""
    frozen = load_config(config)
    schedule: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(_MATCHUPS[:3]):
        challenger, opponent = pair
        for level in range(7):
            schedule.append(_group(
                frozen, pair_index, pair, level=level, challenger=challenger,
                shared=False, direction_index=0,
            ))
    for pair_index, pair in enumerate(_MATCHUPS[3:], start=3):
        schedule.append(_group(frozen, pair_index, pair, level=0, challenger=None, shared=True))
        for level in range(1, 7):
            schedule.append(_group(
                frozen, pair_index, pair, level=level, challenger=pair[0],
                shared=False, direction_index=0,
            ))
            schedule.append(_group(
                frozen, pair_index, pair, level=level, challenger=pair[1],
                shared=False, direction_index=1,
            ))
    ids = [str(group["group_id"]) for group in schedule]
    game_ids = [str(game["game_id"]) for group in schedule for game in group["games"]]
    if len(schedule) != 60 or len(ids) != len(set(ids)) or len(game_ids) != 120 or len(game_ids) != len(set(game_ids)):
        raise ValueError("V5 maximum schedule must contain 60 unique groups and 120 unique games")
    return schedule


def settings_for_model(
    config: dict[str, Any], model: str, occupied: int, budget_seconds: float | None = None
) -> SearchConfig:
    model = model.upper()
    if model not in _MODELS:
        raise ValueError(f"Unknown V5 model: {model}")
    if not 0 <= occupied < 81:
        raise ValueError("occupied must leave at least one empty cell")
    if model == "A":
        settings = dict(config["ab_search"])
        settings["time_limit_seconds"] = math.inf
        settings["max_depth"] = max(1, 81 - occupied)
        return SearchConfig(**settings)
    if budget_seconds is None or budget_seconds <= 0:
        raise ValueError(f"Model {model} requires a positive rung budget")
    if model == "D":
        settings = dict(config["d_search"])
        settings["time_limit_seconds"] = float(budget_seconds)
        return SearchConfig(**settings)
    settings = dict(config["ab_search"])
    settings["time_limit_seconds"] = float(budget_seconds)
    settings["max_depth"] = 81 - occupied
    return SearchConfig(**settings)


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _legal_coordinate(board: GomokuBoard, move: object) -> bool:
    return (
        isinstance(move, (tuple, list)) and len(move) == 2
        and type(move[0]) is int and type(move[1]) is int
        and board.in_bounds(move[0], move[1]) and board.get((move[0], move[1])) is None
    )


def _move_record(result: SearchResult, ply: int, player: Player, model: str, timed_out: bool) -> dict[str, Any]:
    return {
        "ply": ply, "player": player, "model": model,
        "elapsed_seconds": float(result.elapsed_seconds),
        "root_value": int(result.root_value),
        "nodes": int(result.nodes), "completed_depth": int(result.completed_depth),
        "tree_depth": int(result.completed_depth), "cutoffs": int(result.cutoffs),
        "simulations": int(result.simulations), "rollout_plies": int(result.rollout_plies),
        "terminal_simulations": int(result.terminal_simulations),
        "truncated_simulations": int(result.truncated_simulations),
        "root_candidates": int(result.root_candidates), "root_min_visits": int(result.root_min_visits),
        "reused_nodes": int(result.reused_nodes), "timed_out": bool(timed_out), "opening": False,
    }


def _opening_record(player: Player, coord: tuple[int, int], ply: int) -> dict[str, Any]:
    return {
        "ply": ply, "player": player, "model": None, "row": coord[0], "col": coord[1],
        "elapsed_seconds": 0.0, "root_value": 0, "nodes": 0, "completed_depth": 0,
        "tree_depth": 0, "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
        "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
        "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": True,
    }


def run_game(spec: dict[str, Any], config: dict[str, Any], implementation_commit: str = "unknown") -> dict[str, Any]:
    """Run one complete paired game and retain a single MCTS tree for D's moves."""
    frozen = load_config(config)
    board = GomokuBoard(9)
    moves: list[dict[str, Any]] = []
    for ply, (player, coord) in enumerate(_opening(frozen, int(spec["opening_seed"]))):
        board.place_stone(player, coord)
        moves.append(_opening_record(player, coord, ply))

    d_agent = MCTSAgent()
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    for _ in range(81 - len(board.history)):
        model = spec["black_model"] if current == "B" else spec["white_model"]
        budget = spec["budget_seconds_by_model"].get(model)
        settings = settings_for_model(frozen, model, len(board.history), budget)
        seed = derive_seed(int(spec["game_seed"]), "move", len(board.history), model, current)
        result = d_agent.choose_move(board, current, settings, seed) if model == "D" else choose_move(board, current, model, settings, seed)
        elapsed = float(result.elapsed_seconds)
        timed_out = bool(result.timed_out or (budget is not None and elapsed > float(budget)))
        opponent = _other(current)
        opponent_model = spec["black_model"] if opponent == "B" else spec["white_model"]
        move = _move_record(result, len(moves), current, model, timed_out)
        if not _legal_coordinate(board, result.move):
            attempted = list(result.move) if isinstance(result.move, (tuple, list)) else str(result.move)
            move.update({"attempted_move": attempted, "illegal": True})
            moves.append(move)
            winner_color, winner_model, termination = opponent, opponent_model, "illegal_move_forfeit"
            break
        coord = int(result.move[0]), int(result.move[1])
        board.place_stone(current, coord)
        move.update({"row": coord[0], "col": coord[1]})
        moves.append(move)
        if board.check_victory(coord, current):
            winner_color, winner_model, termination = current, model, "five_in_row"
            break
        if board.is_full():
            termination = "full_board"
            break
        current = opponent

    return {
        **spec,
        "pair": list(spec["pair"]),
        "implementation_commit": implementation_commit,
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "draw" if winner_color is None else f"{winner_color.lower()}_win",
        "termination": termination,
        "timeout_count": sum(bool(move.get("timed_out")) for move in moves),
        "search_nodes_total": sum(int(move["nodes"]) for move in moves),
        "simulations_total": sum(int(move["simulations"]) for move in moves),
        "tree_depth_max": max((int(move["tree_depth"]) for move in moves), default=0),
        "thinking_seconds_total": sum(float(move["elapsed_seconds"]) for move in moves),
    }


def _route_specs(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    routes: dict[str, list[dict[str, Any]]] = {}
    for group in build_schedule(config):
        for route_id in group["route_ids"]:
            routes.setdefault(route_id, []).append(group)
    for route_id, groups in routes.items():
        groups.sort(key=lambda group: int(group["challenge_level"]))
        if len(groups) != 7:
            raise ValueError(f"V5 route {route_id} must have seven levels")
    return routes


def _score(game: dict[str, Any], challenger: str) -> float:
    winner = game.get("winner_model")
    return 0.5 if winner is None else 1.0 if winner == challenger else 0.0


def derive_routes(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild advancement decisions from complete paired groups only."""
    frozen = load_config(config)
    schedule = build_schedule(frozen)
    game_by_id = {str(game["game_id"]): game for group in schedule for game in group["games"]}
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in records:
        game_id = str(row.get("game_id", ""))
        if game_id in seen:
            raise ValueError(f"Duplicate V5 game ID: {game_id}")
        seen.add(game_id)
        if game_id not in game_by_id:
            raise ValueError(f"Unscheduled V5 game ID: {game_id}")
        spec = game_by_id[game_id]
        if row.get("group_id") != spec["group_id"]:
            raise ValueError(f"V5 game {game_id} has mismatched group ID")
        rows_by_group.setdefault(str(spec["group_id"]), []).append(row)
    for group_id, rows in rows_by_group.items():
        if len(rows) > 2:
            raise ValueError(f"V5 group {group_id} contains more than two games")

    route_groups = _route_specs(frozen)
    route_state: dict[str, dict[str, Any]] = {}
    active = {route_id: True for route_id in route_groups}
    highest: dict[str, str | None] = {route_id: None for route_id in route_groups}
    failed_at: dict[str, str | None] = {route_id: None for route_id in route_groups}
    next_group_id: str | None = None
    completed_groups = 0
    consumed = 0
    stopped_for_incomplete = False

    for group in schedule:
        group_id = str(group["group_id"])
        rows = rows_by_group.get(group_id, [])
        active_routes = [route_id for route_id in group["route_ids"] if active[route_id]]
        if not active_routes:
            if rows:
                raise ValueError(f"V5 group {group_id} was played after every route using it had failed")
            continue
        if not rows:
            next_group_id = group_id
            stopped_for_incomplete = True
            break
        if len(rows) < 2:
            next_group_id = group_id
            stopped_for_incomplete = True
            break
        by_id = {str(row["game_id"]): row for row in rows}
        expected_ids = {str(game["game_id"]) for game in group["games"]}
        if set(by_id) != expected_ids:
            raise ValueError(f"V5 group {group_id} is missing or has mismatched paired game IDs")
        completed_groups += 1
        consumed += 2
        for route_id in active_routes:
            challenger = route_id.split(">", 1)[0]
            total = sum(_score(by_id[str(game["game_id"])], challenger) for game in group["games"])
            if total >= 1.0:
                highest[route_id] = str(group["rung_budget_label"])
            else:
                active[route_id] = False
                failed_at[route_id] = str(group["challenger_budget_label"])

    for route_id, groups in route_groups.items():
        levels: list[dict[str, Any]] = []
        for group in groups:
            group_id = str(group["group_id"])
            rows = rows_by_group.get(group_id, [])
            label = str(group["challenger_budget_label"])
            if group_id in rows_by_group and len(rows) == 1:
                status = "incomplete_pair"
                total = None
            elif group_id in rows_by_group and len(rows) == 2:
                challenger = route_id.split(">", 1)[0]
                total = sum(_score(row, challenger) for row in rows)
                status = "passed" if total >= 1.0 else "failed"
            elif not active.get(route_id, False):
                status, total = "skipped_after_failure", None
            elif next_group_id == group_id:
                status, total = "pending", None
            else:
                status, total = "not_reached", None
            levels.append({
                "budget": str(group["rung_budget_label"]),
                "opponent_budget": group["opponent_budget_label"],
                "group_id": group_id,
                "status": status,
                "score": total,
                "games": len(rows),
            })
        if failed_at[route_id] is not None:
            overall = "failed"
        elif highest[route_id] == _BUDGET_LABELS[-1] and all(level["status"] == "passed" for level in levels):
            overall = "completed"
        else:
            overall = "active"
        route_state[route_id] = {
            "challenger": route_id.split(">", 1)[0],
            "opponent": route_id.split(">", 1)[1],
            "status": overall,
            "highest_passed_budget": highest[route_id],
            "failed_budget": failed_at[route_id],
            "levels": levels,
        }

    later_game_ids = {
        str(game["game_id"])
        for group in schedule[schedule.index(next(g for g in schedule if g["group_id"] == next_group_id)) + 1:]
        for game in group["games"]
    } if next_group_id and stopped_for_incomplete else set()
    if later_game_ids.intersection(seen):
        raise ValueError("V5 results exist after an unfinished paired group")
    return {
        "routes": route_state,
        "pending_group_id": next_group_id,
        "completed_groups": completed_groups,
        "completed_games": consumed,
        "unfinished_pair": stopped_for_incomplete,
        "complete": next_group_id is None,
    }


def _load_unique_games(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = load_game_records(path)
    ids = [str(row["game_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("V5 JSONL contains duplicate game IDs")
    return rows, set(ids)


def run_experiment(
    config_source: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    max_games: int | None = None,
    max_groups: int | None = None,
    implementation_commit: str = "unknown",
) -> dict[str, Any]:
    """Resume from durable games and never adjudicate a group before both games exist."""
    frozen = load_config(config_source)
    if max_games is not None and max_games < 0:
        raise ValueError("max_games cannot be negative")
    if max_groups is not None and max_groups < 0:
        raise ValueError("max_groups cannot be negative")
    from .abcd_ladder_v5_analysis import validate_game_record, validate_records

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Existing V5 output belongs to a different frozen config")
    if not config_path.exists():
        write_json_atomic(config_path, frozen)
    config_hash = _sha256(_canonical_json(frozen).encode("utf-8"))
    records_path = root / "9x9_games.jsonl"
    if not records_path.exists():
        records_path.touch()
        with records_path.open("rb") as handle:
            import os
            os.fsync(handle.fileno())
    records, completed_ids = _load_unique_games(records_path)
    schedule = build_schedule(frozen)
    validate_records(records, frozen, require_complete=False)
    route_result = derive_routes(records, frozen)

    progress_path = root / "progress.json"
    previous_elapsed = 0.0
    first_started = _utc_now()
    if progress_path.exists():
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        if prior.get("experiment_id") != frozen["experiment_id"] or prior.get("config_sha256") != config_hash:
            raise ValueError("V5 progress metadata does not match frozen config")
        if not set(prior.get("completed_ids", ())).issubset(completed_ids):
            raise ValueError("V5 progress claims games not present in durable JSONL")
        if prior.get("implementation_commit") != implementation_commit:
            raise ValueError("V5 resume code version differs from prior run")
        previous_elapsed = float(prior.get("elapsed_seconds_total", 0.0))
        first_started = str(prior.get("first_started_at_utc", first_started))

    progress: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": frozen["experiment_id"],
        "config_sha256": config_hash,
        "implementation_commit": implementation_commit,
        "maximum_games": 120,
        "completed_games": len(completed_ids),
        "completed_groups": route_result["completed_groups"],
        "completed_ids": sorted(completed_ids),
        "routes": route_result["routes"],
        "all_games_new": True,
        "first_started_at_utc": first_started,
        "last_started_at_utc": _utc_now(),
        "elapsed_seconds_total": previous_elapsed,
        "elapsed_wall_seconds_total": _wall_elapsed_seconds(first_started),
        "updated_at_utc": _utc_now(),
    }
    write_json_atomic(progress_path, progress)
    started = time.perf_counter()
    invocation_games = 0
    invocation_groups = 0
    groups_at_start = int(route_result["completed_groups"])
    by_group = {str(group["group_id"]): group for group in schedule}

    while True:
        if max_groups is not None and invocation_groups >= max_groups:
            break
        route_result = derive_routes(records, frozen)
        group_id = route_result["pending_group_id"]
        if group_id is None:
            break
        group = by_group[group_id]
        group_rows = [row for row in records if row.get("group_id") == group_id]
        existing = {str(row["game_id"]) for row in group_rows}
        for spec in group["games"]:
            identifier = str(spec["game_id"])
            if identifier in existing:
                continue
            if max_games is not None and invocation_games >= max_games:
                break
            record = run_game(spec, frozen, implementation_commit=implementation_commit)
            if str(record.get("game_id")) != identifier or str(record.get("group_id")) != group_id:
                raise ValueError("V5 runner returned a mismatched game or group ID")
            validate_game_record(record, frozen, spec)
            append_game(records_path, record, completed_ids)
            records.append(record)
            existing.add(identifier)
            invocation_games += 1
            route_result = derive_routes(records, frozen)
            invocation_groups = int(route_result["completed_groups"]) - groups_at_start
            elapsed_so_far = time.perf_counter() - started
            progress.update({
                "completed_games": len(completed_ids),
                "completed_groups": route_result["completed_groups"],
                "completed_ids": sorted(completed_ids),
                "last_game_id": identifier,
                "routes": route_result["routes"],
                "pending_group_id": route_result["pending_group_id"],
                "elapsed_seconds_total": previous_elapsed + elapsed_so_far,
                "elapsed_wall_seconds_total": _wall_elapsed_seconds(first_started),
                "updated_at_utc": _utc_now(),
            })
            write_json_atomic(progress_path, progress)
            print(json.dumps({
                "completed_games": len(completed_ids), "completed_groups": route_result["completed_groups"],
                "maximum_games": 120, "last_game_id": identifier,
                "winner": record.get("winner_model"), "elapsed_seconds": record.get("thinking_seconds_total"),
            }, ensure_ascii=False), flush=True)
        if max_games is not None and invocation_games >= max_games:
            break
        if len(existing) < 2:
            break
        if max_groups is not None and invocation_groups >= max_groups:
            break

    invocation_elapsed = time.perf_counter() - started
    route_result = derive_routes(records, frozen)
    completed_at = _utc_now() if route_result["complete"] else None
    progress.update({
        "completed_games": len(completed_ids),
        "completed_groups": route_result["completed_groups"],
        "completed_ids": sorted(completed_ids),
        "routes": route_result["routes"],
        "pending_group_id": route_result["pending_group_id"],
        "remaining_challenge_groups": None if route_result["complete"] else "see pending_group_id and route states",
        "complete": bool(route_result["complete"]),
        "games_completed_this_run": invocation_games,
        "groups_completed_this_run": invocation_groups,
        "elapsed_seconds_this_run": invocation_elapsed,
        "elapsed_seconds_total": previous_elapsed + invocation_elapsed,
        "elapsed_wall_seconds_total": _wall_elapsed_seconds(first_started, completed_at),
        "completed_at_utc": completed_at,
        "updated_at_utc": _utc_now(),
    })
    write_json_atomic(progress_path, progress)
    return progress


def repository_commit(project_root: str | Path | None = None) -> str:
    root = Path(project_root or _project_root())
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def load_records(results_dir: str | Path) -> list[dict[str, Any]]:
    return _load_unique_games(Path(results_dir) / "9x9_games.jsonl")[0]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/abcd_ladder_v5.json")
    parser.add_argument("--output-dir", default="research/v5/results")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--implementation-commit")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    commit = args.implementation_commit or repository_commit()
    progress = run_experiment(
        args.config, args.output_dir, max_games=args.max_games, max_groups=args.max_groups,
        implementation_commit=commit,
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
