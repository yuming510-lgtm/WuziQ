"""Replay validation and descriptive analysis for the V5 compute-ladder tournament."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

from ..gomoku import GomokuBoard, Player
from .abcd_ladder_v5 import build_schedule, derive_routes, load_config
from .checkpoint import load_game_records, write_json_atomic

_Z95 = 1.959963984540054
_MODELS = ("A", "B", "C", "D")
_METRICS = (
    "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations", "rollout_plies",
    "terminal_simulations", "truncated_simulations", "root_candidates", "root_min_visits", "reused_nodes",
)


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _score(winner: str | None, model: str) -> float:
    return 0.5 if winner is None else 1.0 if winner == model else 0.0


def wilson_fractional(values: list[float]) -> list[float] | None:
    """Wilson-style 95% interval over paired group scores; descriptive only."""
    if not values:
        return None
    n = float(len(values))
    p = min(1.0, max(0.0, sum(values) / n))
    z2 = _Z95**2
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def validate_game_record(
    record: dict[str, Any], config: dict[str, Any], expected_spec: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Replay a complete game and verify schedule, legality, result and metrics."""
    frozen = load_config(config)
    schedule = {str(game["game_id"]): game for group in build_schedule(frozen) for game in group["games"]}
    game_id = str(record.get("game_id", ""))
    spec = expected_spec or schedule.get(game_id)
    if spec is None or game_id != str(spec["game_id"]):
        raise ValueError(f"Unknown or mismatched V5 game ID: {game_id}")
    keys = (
        "experiment_id", "group_id", "board_size", "pair", "matchup", "route_ids", "challenge_level",
        "challenger_model", "shared_baseline", "opening_index", "opening_seed", "color_swap", "black_model",
        "white_model", "pair_id", "game_seed", "opening", "budget_seconds_by_model",
        "challenger_budget_label", "opponent_budget_label", "source_type",
    )
    for key in keys:
        if record.get(key) != spec.get(key):
            raise ValueError(f"V5 game {game_id} disagrees with scheduled field {key}")
    if not record.get("implementation_commit"):
        raise ValueError(f"V5 game {game_id} has no implementation commit")
    moves = record.get("moves")
    if not isinstance(moves, list) or len(moves) <= int(frozen["opening_plies"]) or len(moves) > 81:
        raise ValueError(f"V5 game {game_id} is incomplete or has too many plies")

    board = GomokuBoard(9)
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    timeout_count = nodes_total = simulations_total = max_depth = 0
    elapsed_total = 0.0
    for index, move in enumerate(moves):
        if not isinstance(move, dict) or move.get("ply") != index or move.get("player") != current:
            raise ValueError(f"V5 game {game_id} has malformed or out-of-turn ply {index}")
        opening = index < int(frozen["opening_plies"])
        if bool(move.get("opening")) != opening:
            raise ValueError(f"V5 game {game_id} has a wrong opening flag at ply {index}")
        expected_model = None if opening else record["black_model"] if current == "B" else record["white_model"]
        if move.get("model") != expected_model:
            raise ValueError(f"V5 game {game_id} has incorrect model attribution at ply {index}")
        elapsed = float(move.get("elapsed_seconds", -1))
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError(f"V5 game {game_id} has invalid elapsed time at ply {index}")
        if expected_model == "A" and bool(move.get("timed_out")):
            raise ValueError(f"V5 A move was marked timed out despite its natural-time policy")
        elapsed_total += elapsed
        timeout_count += int(bool(move.get("timed_out")))
        for metric in _METRICS:
            if int(move.get(metric, -1)) < 0:
                raise ValueError(f"V5 game {game_id} has invalid {metric} at ply {index}")
        if int(move["tree_depth"]) != int(move["completed_depth"]):
            raise ValueError(f"V5 game {game_id} has inconsistent tree depth at ply {index}")
        nodes_total += int(move["nodes"])
        simulations_total += int(move["simulations"])
        max_depth = max(max_depth, int(move["tree_depth"]))

        if opening:
            open_move = spec["opening"][index]
            coord = int(open_move["row"]), int(open_move["col"])
            if move.get("row") != coord[0] or move.get("col") != coord[1]:
                raise ValueError(f"V5 game {game_id} changed its frozen opening")
        elif move.get("illegal"):
            if index != len(moves) - 1:
                raise ValueError(f"V5 game {game_id} has a non-final illegal move")
            attempted = move.get("attempted_move")
            if isinstance(attempted, (tuple, list)) and len(attempted) == 2 and all(type(value) is int for value in attempted):
                if board.in_bounds(attempted[0], attempted[1]) and board.get((attempted[0], attempted[1])) is None:
                    raise ValueError(f"V5 game {game_id} marked a legal move illegal")
            elif not isinstance(attempted, str):
                raise ValueError(f"V5 game {game_id} has no invalid-move payload")
            winner_color = _other(current)
            winner_model = record["black_model"] if winner_color == "B" else record["white_model"]
            termination = "illegal_move_forfeit"
            break
        else:
            if type(move.get("row")) is not int or type(move.get("col")) is not int:
                raise ValueError(f"V5 game {game_id} has invalid coordinates at ply {index}")
            coord = int(move["row"]), int(move["col"])
        try:
            board.place_stone(current, coord)
        except ValueError as exc:
            raise ValueError(f"V5 game {game_id} repeats or misplaces a move at ply {index}") from exc
        if board.check_victory(coord, current):
            if index != len(moves) - 1:
                raise ValueError(f"V5 game {game_id} continued after victory")
            winner_color = current
            winner_model = expected_model
            termination = "five_in_row"
            break
        if board.is_full():
            if index != len(moves) - 1:
                raise ValueError(f"V5 game {game_id} continued after board fill")
            termination = "full_board"
            break
        current = _other(current)

    if winner_color is None and (termination != "full_board" or not board.is_full()):
        raise ValueError(f"V5 game {game_id} ended without a win or complete draw")
    outcome = "draw" if winner_color is None else f"{winner_color.lower()}_win"
    if record.get("winner_color") != winner_color or record.get("winner_model") != winner_model:
        raise ValueError(f"V5 game {game_id} winner metadata disagrees with replay")
    if record.get("outcome") != outcome or record.get("termination") != termination:
        raise ValueError(f"V5 game {game_id} outcome disagrees with replay")
    expected_totals = {
        "move_count": len(moves), "timeout_count": timeout_count, "search_nodes_total": nodes_total,
        "simulations_total": simulations_total, "tree_depth_max": max_depth,
    }
    for key, value in expected_totals.items():
        if int(record.get(key, -1)) != value:
            raise ValueError(f"V5 game {game_id} has incorrect {key}")
    if not math.isclose(float(record.get("thinking_seconds_total", -1)), elapsed_total, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError(f"V5 game {game_id} has incorrect thinking_seconds_total")
    return {
        "game_id": game_id, "group_id": spec["group_id"], "move_count": len(moves),
        "winner_model": winner_model, "outcome": outcome, "termination": termination,
        "timeouts": timeout_count, "nodes": nodes_total, "simulations": simulations_total,
    }


def validate_records(records: list[dict[str, Any]], config: dict[str, Any], require_complete: bool = False) -> dict[str, Any]:
    """Validate every record and ensure no group or advancement decision is duplicated."""
    frozen = load_config(config)
    schedule = {str(game["game_id"]): game for group in build_schedule(frozen) for game in group["games"]}
    ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        game_id = str(record.get("game_id", ""))
        if game_id in ids:
            raise ValueError(f"Duplicate V5 game ID: {game_id}")
        ids.add(game_id)
        spec = schedule.get(game_id)
        if spec is None:
            raise ValueError(f"Unscheduled V5 game ID: {game_id}")
        validate_game_record(record, frozen, spec)
        grouped[str(spec["group_id"])].append(record)
    if any(len(rows) > 2 for rows in grouped.values()):
        raise ValueError("A V5 challenge group has more than two games")
    route_result = derive_routes(records, frozen)
    if require_complete and not route_result["complete"]:
        raise ValueError(f"V5 challenge schedule is incomplete at {route_result['pending_group_id']}")
    if require_complete and any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("A V5 played group does not contain its complete color-swapped pair")
    return {
        "valid": True,
        "games": len(records),
        "groups": sum(len(rows) == 2 for rows in grouped.values()),
        "partial_groups": sum(len(rows) == 1 for rows in grouped.values()),
        "routes": route_result["routes"],
        "complete": route_result["complete"],
    }


def summarize_records(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    frozen = load_config(config)
    validate_records(records, frozen, require_complete=False)
    schedule = {str(group["group_id"]): group for group in build_schedule(frozen)}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    games_by_id = {str(record["game_id"]): record for record in records}
    for record in records:
        by_group[str(record["group_id"])].append(record)
    route_result = derive_routes(records, frozen)

    model_counts = {model: Counter() for model in _MODELS}
    model_scores: dict[str, list[float]] = {model: [] for model in _MODELS}
    sides = {model: {"black_games": 0, "black_points": 0.0, "white_games": 0, "white_points": 0.0} for model in _MODELS}
    usage: dict[tuple[str, str], dict[str, float]] = {}
    group_summaries: list[dict[str, Any]] = []
    game_summaries: list[dict[str, Any]] = []
    opening_counts: Counter[int] = Counter()

    for game_id, record in sorted(games_by_id.items()):
        opening_counts[int(record["opening_seed"])] += 1
        winner = record.get("winner_model")
        for model in record["pair"]:
            model_counts[model]["games"] += 1
            model_counts[model]["wins"] += int(winner == model)
            model_counts[model]["draws"] += int(winner is None)
            model_counts[model]["losses"] += int(winner is not None and winner != model)
            points = _score(winner, model)
            model_counts[model]["points"] += points
            black = record["black_model"] == model
            sides[model]["black_games" if black else "white_games"] += 1
            sides[model]["black_points" if black else "white_points"] += points
        game_summaries.append({
            "game_id": game_id, "group_id": record["group_id"], "pair": "-".join(record["pair"]),
            "level": int(record["challenge_level"]), "budget_label": record["challenger_budget_label"],
            "opponent_budget_label": record["opponent_budget_label"], "opening_seed": record["opening_seed"],
            "black_model": record["black_model"], "white_model": record["white_model"],
            "winner_model": winner or "draw", "outcome": record["outcome"],
            "move_count": record["move_count"], "thinking_seconds": record["thinking_seconds_total"],
            "timeouts": record["timeout_count"], "nodes": record["search_nodes_total"],
            "simulations": record["simulations_total"], "max_tree_depth": record["tree_depth_max"],
        })
        for move in record["moves"]:
            model = move.get("model")
            if model is None:
                continue
            budget = record["budget_seconds_by_model"].get(model)
            budget_label = "natural" if budget is None else next(
                rung["label"] for rung in frozen["budget_ladder"] if float(rung["seconds"]) == float(budget)
            )
            key = (str(model), budget_label)
            row = usage.setdefault(key, {
                "model": str(model), "budget_label": budget_label, "budget_seconds": float(budget) if budget is not None else 0.0,
                "decisions": 0, "elapsed_seconds": 0.0, "nodes": 0, "cutoffs": 0, "simulations": 0,
                "depth_sum": 0.0, "max_depth": 0, "timeouts": 0, "reused_nodes": 0,
            })
            row["decisions"] += 1
            row["elapsed_seconds"] += float(move["elapsed_seconds"])
            row["nodes"] += int(move["nodes"])
            row["cutoffs"] += int(move["cutoffs"])
            row["simulations"] += int(move["simulations"])
            row["depth_sum"] += int(move["completed_depth"])
            row["max_depth"] = max(row["max_depth"], int(move["completed_depth"]))
            row["timeouts"] += int(bool(move["timed_out"]))
            row["reused_nodes"] += int(move["reused_nodes"])

    for group_id, group_records in sorted(by_group.items()):
        group = schedule[group_id]
        if len(group_records) != 2:
            continue
        challenger_points: dict[str, float] = {}
        for route_id in group["route_ids"]:
            challenger = route_id.split(">", 1)[0]
            challenger_points[route_id] = sum(_score(row.get("winner_model"), challenger) for row in group_records)
        counts = Counter(row.get("winner_model") or "draw" for row in group_records)
        points: dict[str, float] = {}
        for model in group["pair"]:
            points[model] = sum(_score(row.get("winner_model"), model) for row in group_records)
            model_scores[model].append(points[model] / 2.0)
        black_wins = sum(row.get("winner_model") == row["black_model"] for row in group_records)
        white_wins = sum(row.get("winner_model") == row["white_model"] for row in group_records)
        group_summaries.append({
            "group_id": group_id, "pair": "-".join(group["pair"]),
            "route_ids": ";".join(group["route_ids"]), "level": int(group["challenge_level"]),
            "rung_budget": group["rung_budget_label"], "challenger_budget": group["challenger_budget_label"],
            "opponent_budget": group["opponent_budget_label"], "challenger": group["challenger_model"] or "shared",
            "opening_seed": group["opening_seed"], "games": len(group_records),
            "wins_A": sum(row.get("winner_model") == "A" for row in group_records),
            "wins_B": sum(row.get("winner_model") == "B" for row in group_records),
            "wins_C": sum(row.get("winner_model") == "C" for row in group_records),
            "wins_D": sum(row.get("winner_model") == "D" for row in group_records),
            "draws": counts.get("draw", 0), "black_wins": black_wins, "white_wins": white_wins,
            "score_A": points.get("A", 0.0), "score_B": points.get("B", 0.0),
            "score_C": points.get("C", 0.0), "score_D": points.get("D", 0.0),
            "route_scores": json.dumps(challenger_points, ensure_ascii=False, sort_keys=True),
            "passed_routes": ";".join(route_id for route_id, score in challenger_points.items() if score >= 1.0),
        })

    pairwise: list[dict[str, Any]] = []
    for pair in _pair_order():
        pair_groups = [group for group in group_summaries if group["pair"] == "-".join(pair)]
        first, second = pair
        first_points = [float(group[f"score_{first}"]) for group in pair_groups]
        second_points = [float(group[f"score_{second}"]) for group in pair_groups]
        first_score_blocks = [value / 2.0 for value in first_points]
        first_wins = sum(int(group[f"wins_{first}"]) for group in pair_groups)
        second_wins = sum(int(group[f"wins_{second}"]) for group in pair_groups)
        draws = sum(int(group["draws"]) for group in pair_groups)
        games = sum(int(group["games"]) for group in pair_groups)
        pairwise.append({
            "pair": f"{first}-{second}", "groups": len(pair_groups), "games": games,
            "first_wins": first_wins, "draws": draws, "second_wins": second_wins,
            "first_score": sum(first_score_blocks), "second_score": sum(float(group[f"score_{second}"]) for group in pair_groups),
            "first_score_rate": sum(first_points) / games if games else None,
            "second_score_rate": sum(second_points) / games if games else None,
            "first_group_score_ci95": wilson_fractional(first_score_blocks),
            "second_group_score_ci95": wilson_fractional([value / 2.0 for value in second_points]),
        })

    model_summaries: dict[str, Any] = {}
    for model in _MODELS:
        count = model_counts[model]
        games = int(count["games"])
        score_rate = float(count["points"]) / games if games else None
        model_summaries[model] = {
            "games": games, "wins": int(count["wins"]), "draws": int(count["draws"]),
            "losses": int(count["losses"]), "points": float(count["points"]),
            "score_rate": score_rate,
            "score_rate_ci95_paired_groups": wilson_fractional(model_scores[model]),
            **sides[model],
        }
    usage_rows: list[dict[str, Any]] = []
    for key in sorted(usage, key=lambda item: (_MODELS.index(item[0]), item[1])):
        row = usage[key]
        decisions = int(row["decisions"])
        seconds = float(row["elapsed_seconds"])
        usage_rows.append({
            **row,
            "mean_seconds_per_decision": seconds / decisions if decisions else 0.0,
            "mean_nodes_per_decision": int(row["nodes"]) / decisions if decisions else 0.0,
            "nodes_per_second": int(row["nodes"]) / seconds if seconds else 0.0,
            "simulations_per_second": int(row["simulations"]) / seconds if seconds else 0.0,
            "mean_completed_depth": float(row["depth_sum"]) / decisions if decisions else 0.0,
            "cutoff_per_node": int(row["cutoffs"]) / int(row["nodes"]) if row["nodes"] else 0.0,
            "timeout_rate": int(row["timeouts"]) / decisions if decisions else 0.0,
        })

    route_rows: list[dict[str, Any]] = []
    for route_id, route in route_result["routes"].items():
        for level in route["levels"]:
            route_rows.append({
                "route_id": route_id, "challenger": route["challenger"], "opponent": route["opponent"],
                "route_status": route["status"], "highest_passed_budget": route["highest_passed_budget"] or "none",
                "failed_budget": route["failed_budget"] or "none", **level,
            })
    deadline_hit_moves = 0
    nominal_budget_overruns = 0
    over_10ms_budget_overruns = 0
    actual_thinking_seconds = 0.0
    total_nodes = total_simulations = 0
    termination_counts: Counter[str] = Counter()
    for record in records:
        actual_thinking_seconds += float(record["thinking_seconds_total"])
        total_nodes += int(record["search_nodes_total"])
        total_simulations += int(record["simulations_total"])
        termination_counts[str(record["termination"])] += 1
        for move in record["moves"]:
            model = move.get("model")
            if model is None:
                continue
            deadline_hit_moves += int(bool(move.get("timed_out")))
            budget = record["budget_seconds_by_model"].get(model)
            if budget is not None:
                overrun = float(move["elapsed_seconds"]) - float(budget)
                nominal_budget_overruns += int(overrun > 0.0)
                over_10ms_budget_overruns += int(overrun > 0.01)
    return {
        "experiment_id": frozen["experiment_id"],
        "board_size": 9,
        "maximum_groups": 60,
        "maximum_games": 120,
        "actual_games": len(records),
        "actual_groups": len(group_summaries),
        "partial_groups": sum(len(rows) == 1 for rows in by_group.values()),
        "route_state": route_result,
        "models": model_summaries,
        "actual_metrics": {
            "ai_thinking_seconds": actual_thinking_seconds,
            "deadline_hit_moves": deadline_hit_moves,
            "nominal_budget_overrun_moves": nominal_budget_overruns,
            "budget_overrun_over_10ms_moves": over_10ms_budget_overruns,
            "search_nodes_total": total_nodes,
            "mcts_simulations_total": total_simulations,
            "terminations": dict(sorted(termination_counts.items())),
            "unique_game_ids": len(games_by_id),
        },
        "pairwise": pairwise,
        "groups": group_summaries,
        "games": game_summaries,
        "model_usage": usage_rows,
        "routes": route_rows,
        "opening_usage": {str(seed): opening_counts.get(seed, 0) for seed in frozen["opening_seeds"]},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pair_order() -> tuple[tuple[str, str], ...]:
    return (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))


def write_analysis(results_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results = Path(results_dir)
    target = Path(output_dir)
    config = load_config(results / "config.json")
    records = load_game_records(results / "9x9_games.jsonl")
    summary = summarize_records(records, config)
    target.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target / "summary.json", summary)
    _write_csv(target / "group_summary.csv", summary["groups"])
    _write_csv(target / "game_summary.csv", summary["games"])
    _write_csv(target / "pairwise_summary.csv", summary["pairwise"])
    _write_csv(target / "route_summary.csv", summary["routes"])
    _write_csv(target / "model_usage.csv", summary["model_usage"])
    return summary
