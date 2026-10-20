"""Replay validation and descriptive analysis for the V4C B/C/D match."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from ..gomoku import GomokuBoard, Player
from .bcd_high_compute_v4c import build_schedule, load_config

_Z95 = 1.959963984540054
_METRICS = (
    "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations", "rollout_plies",
    "terminal_simulations", "truncated_simulations", "root_candidates", "root_min_visits", "reused_nodes",
)


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _score(record: dict[str, Any], model: str) -> float:
    winner = record.get("winner_model")
    return 0.5 if winner is None else 1.0 if winner == model else 0.0


def wilson_fractional(values: list[float]) -> list[float] | None:
    """Wilson-style 95% interval over paired-opening block scores, for description only."""
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
    """Replay a complete game and verify its schedule, legality, winner, and telemetry totals."""
    frozen = load_config(config)
    schedule = {str(spec["game_id"]): spec for spec in build_schedule(frozen)}
    game_id = str(record.get("game_id", ""))
    spec = expected_spec or schedule.get(game_id)
    if spec is None or game_id != str(spec["game_id"]):
        raise ValueError(f"Unknown or mismatched V4C game ID: {game_id}")
    if record.get("experiment_id") != frozen["experiment_id"]:
        raise ValueError(f"V4C experiment ID mismatch for {game_id}")
    for key in ("board_size", "opening_index", "opening_seed", "color_swap", "black_model", "white_model", "pair_id", "game_seed"):
        if record.get(key) != spec.get(key):
            raise ValueError(f"V4C game {game_id} disagrees with scheduled field {key}")
    if tuple(record.get("matchup", ())) != tuple(spec["matchup"]):
        raise ValueError(f"V4C game {game_id} has a mismatched matchup")
    if record.get("opening") != spec["opening"]:
        raise ValueError(f"V4C game {game_id} has a mismatched opening")
    if record.get("source_type") != "new":
        raise ValueError(f"V4C game {game_id} is not a newly played game")
    if record.get("budget_seconds_by_model") != spec["budget_seconds_by_model"]:
        raise ValueError(f"V4C game {game_id} has incorrect per-model budgets")
    if not record.get("implementation_commit"):
        raise ValueError(f"V4C game {game_id} has no implementation commit")

    moves = record.get("moves")
    if not isinstance(moves, list) or len(moves) <= 4:
        raise ValueError(f"V4C game {game_id} is incomplete")
    board = GomokuBoard(9)
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    timeout_count = nodes_total = simulations_total = rollout_plies_total = 0
    terminal_sims_total = truncated_sims_total = max_tree_depth = 0
    elapsed_total = 0.0

    for index, move in enumerate(moves):
        if not isinstance(move, dict) or move.get("ply") != index or move.get("player") != current:
            raise ValueError(f"V4C game {game_id} has malformed or out-of-turn ply {index}")
        opening = index < 4
        if bool(move.get("opening")) != opening:
            raise ValueError(f"V4C game {game_id} has a wrong opening flag at ply {index}")
        expected_model = None if opening else record["black_model"] if current == "B" else record["white_model"]
        if move.get("model") != expected_model:
            raise ValueError(f"V4C game {game_id} has wrong model attribution at ply {index}")
        elapsed = float(move.get("elapsed_seconds", -1.0))
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError(f"V4C game {game_id} has invalid elapsed time at ply {index}")
        elapsed_total += elapsed
        timeout_count += int(bool(move.get("timed_out")))
        for metric in _METRICS:
            if int(move.get(metric, -1)) < 0:
                raise ValueError(f"V4C game {game_id} has invalid {metric} at ply {index}")
        if int(move["tree_depth"]) != int(move["completed_depth"]):
            raise ValueError(f"V4C game {game_id} tree depth mismatch at ply {index}")
        nodes_total += int(move["nodes"])
        simulations_total += int(move["simulations"])
        rollout_plies_total += int(move["rollout_plies"])
        terminal_sims_total += int(move["terminal_simulations"])
        truncated_sims_total += int(move["truncated_simulations"])
        max_tree_depth = max(max_tree_depth, int(move["tree_depth"]))

        if opening:
            opening_move = spec["opening"][index]
            coord = int(opening_move["row"]), int(opening_move["col"])
            if move.get("row") != coord[0] or move.get("col") != coord[1]:
                raise ValueError(f"V4C game {game_id} changed its registered opening")
        elif move.get("illegal"):
            if index != len(moves) - 1:
                raise ValueError(f"V4C game {game_id} has a non-final illegal move")
            attempted = move.get("attempted_move")
            if isinstance(attempted, (tuple, list)) and len(attempted) == 2 and all(type(item) is int for item in attempted):
                if board.in_bounds(attempted[0], attempted[1]) and board.get((attempted[0], attempted[1])) is None:
                    raise ValueError(f"V4C game {game_id} marks a legal attempted move illegal")
            else:
                if not isinstance(attempted, str):
                    raise ValueError(f"V4C game {game_id} missing illegal attempted move")
            winner_color = _other(current)
            winner_model = record["black_model"] if winner_color == "B" else record["white_model"]
            termination = "illegal_move_forfeit"
            break
        else:
            if type(move.get("row")) is not int or type(move.get("col")) is not int:
                raise ValueError(f"V4C game {game_id} has invalid coordinates at ply {index}")
            coord = int(move["row"]), int(move["col"])

        try:
            board.place_stone(current, coord)
        except ValueError as exc:
            raise ValueError(f"V4C game {game_id} repeats or misplaces a move at ply {index}") from exc
        if board.check_victory(coord, current):
            if index != len(moves) - 1:
                raise ValueError(f"V4C game {game_id} continued after victory")
            winner_color = current
            winner_model = expected_model
            termination = "five_in_row"
            break
        if board.is_full():
            if index != len(moves) - 1:
                raise ValueError(f"V4C game {game_id} continued after board fill")
            termination = "full_board"
            break
        current = _other(current)

    if winner_color is None and termination != "full_board":
        raise ValueError(f"V4C game {game_id} ended without a result")
    if winner_color is None and len(board.history) != 81:
        raise ValueError(f"V4C game {game_id} ended before victory or a full-board draw")
    expected_outcome = "draw" if winner_color is None else f"{winner_color.lower()}_win"
    if record.get("winner_color") != winner_color or record.get("winner_model") != winner_model:
        raise ValueError(f"V4C game {game_id} winner metadata disagrees with replay")
    if record.get("outcome") != expected_outcome or record.get("termination") != termination:
        raise ValueError(f"V4C game {game_id} outcome/termination disagrees with replay")
    if int(record.get("move_count", -1)) != len(moves):
        raise ValueError(f"V4C game {game_id} move count mismatch")
    if int(record.get("timeout_count", -1)) != timeout_count:
        raise ValueError(f"V4C game {game_id} timeout count mismatch")
    expected_totals = {
        "search_nodes_total": nodes_total, "simulations_total": simulations_total,
        "rollout_plies_total": rollout_plies_total, "terminal_simulations_total": terminal_sims_total,
        "truncated_simulations_total": truncated_sims_total, "tree_depth_max": max_tree_depth,
    }
    for key, value in expected_totals.items():
        if int(record.get(key, -1)) != value:
            raise ValueError(f"V4C game {game_id} aggregate mismatch for {key}")
    if not math.isclose(float(record.get("thinking_seconds_total", -1)), elapsed_total, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f"V4C game {game_id} thinking-time total mismatch")
    return {
        "game_id": game_id, "winner_model": winner_model, "outcome": expected_outcome,
        "termination": termination, "move_count": len(moves),
        "thinking_seconds_total": elapsed_total, "timed_out_moves": timeout_count,
    }


def validate_records(records: Iterable[dict[str, Any]], config: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    frozen = load_config(config)
    rows = list(records)
    schedule = build_schedule(frozen)
    expected = {str(spec["game_id"]): spec for spec in schedule}
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("game_id", ""))
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate V4C game ID: {identifier}")
        if identifier not in expected:
            raise ValueError(f"Unscheduled V4C game ID: {identifier}")
        validate_game_record(row, frozen, expected[identifier])
        seen[identifier] = row
    missing = set(expected) - set(seen)
    if require_complete and missing:
        raise ValueError(f"V4C experiment is incomplete: {len(missing)} games missing")
    for pair in _MATCHUP_ORDER(frozen):
        games = [row for row in rows if tuple(row["matchup"]) == pair]
        if require_complete and len(games) != 6:
            raise ValueError(f"V4C matchup {pair} has {len(games)} games, expected 6")
        if require_complete and Counter(row["black_model"] for row in games) != Counter({pair[0]: 3, pair[1]: 3}):
            raise ValueError(f"V4C matchup {pair} is not balanced by color")
        for seed in frozen["opening_seeds"]:
            block = [row for row in games if int(row["opening_seed"]) == int(seed)]
            if len(block) not in ((0, 2) if require_complete else (0, 1, 2)):
                raise ValueError(f"V4C opening block {pair}/{seed} is incomplete")
            if len(block) == 2:
                first, second = block
                if first["opening"] != second["opening"] or first["black_model"] != second["white_model"] or first["white_model"] != second["black_model"]:
                    raise ValueError(f"V4C opening block {pair}/{seed} lacks color-swapped pairing")
    return {
        "complete": not missing, "scheduled_games": len(schedule), "validated_games": len(rows),
        "missing_games": len(missing), "new_games": sum(row.get("source_type") == "new" for row in rows),
        "reused_games": sum(row.get("source_type") != "new" for row in rows),
        "illegal_forfeits": sum(row.get("termination") == "illegal_move_forfeit" for row in rows),
        "timeouts": sum(int(row.get("timeout_count", 0)) for row in rows),
    }


def _matchup_summary(records: list[dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    rows = [record for record in records if tuple(record["matchup"]) == (first, second)]
    if not rows:
        return {}
    wins = {model: sum(record.get("winner_model") == model for record in rows) for model in (first, second)}
    draws = sum(record.get("winner_model") is None for record in rows)
    scores = {model: [_score(record, model) for record in rows] for model in (first, second)}
    paired = {model: [] for model in (first, second)}
    colors = {model: {} for model in (first, second)}
    for model in (first, second):
        for seed in (731091, 731092, 731093):
            block = [record for record in rows if int(record["opening_seed"]) == seed]
            if len(block) == 2:
                paired[model].append(sum(_score(record, model) for record in block) / 2)
        for color in ("B", "W"):
            color_rows = [record for record in rows if (record["black_model"] if color == "B" else record["white_model"]) == model]
            colors[model]["black" if color == "B" else "white"] = {
                "games": len(color_rows),
                "wins": sum(record.get("winner_model") == model for record in color_rows),
                "draws": sum(record.get("winner_model") is None for record in color_rows),
                "score_rate": sum(_score(record, model) for record in color_rows) / len(color_rows) if color_rows else None,
            }
    return {
        "matchup": [first, second], "games": len(rows), "wins": wins, "draws": draws,
        "losses": {first: wins[second], second: wins[first]},
        "score_rate": {model: sum(scores[model]) / len(rows) for model in (first, second)},
        "pure_win_rate": {model: wins[model] / len(rows) for model in (first, second)},
        "paired_block_scores": paired,
        "paired_wilson_95_ci": {model: wilson_fractional(paired[model]) for model in (first, second)},
        "color": colors,
        "game_ids": [record["game_id"] for record in rows],
    }


def _model_usage(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model in ("B", "C", "D"):
        moves = [move for game in records for move in game["moves"] if move.get("model") == model]
        total_seconds = sum(float(move["elapsed_seconds"]) for move in moves)
        nodes = sum(int(move["nodes"]) for move in moves)
        cutoffs = sum(int(move["cutoffs"]) for move in moves)
        simulations = sum(int(move["simulations"]) for move in moves)
        output[model] = {
            "decisions": len(moves), "total_seconds": total_seconds,
            "mean_seconds_per_move": total_seconds / len(moves) if moves else None,
            "max_seconds_per_move": max((float(move["elapsed_seconds"]) for move in moves), default=None),
            "nodes_total": nodes, "mean_nodes_per_move": nodes / len(moves) if moves else None,
            "nodes_per_second": nodes / total_seconds if total_seconds else None,
            "mean_completed_depth": sum(int(move["completed_depth"]) for move in moves) / len(moves) if moves else None,
            "max_completed_depth": max((int(move["completed_depth"]) for move in moves), default=None),
            "cutoffs_total": cutoffs,
            "cutoff_to_node_ratio": cutoffs / nodes if nodes else 0.0,
            "simulations_total": simulations,
            "mean_simulations_per_move": simulations / len(moves) if moves else None,
            "simulations_per_second": simulations / total_seconds if total_seconds else None,
            "timeouts": sum(bool(move.get("timed_out")) for move in moves),
            "reused_nodes_total": sum(int(move["reused_nodes"]) for move in moves),
            "node_metric_semantics": "B/C search nodes" if model in {"B", "C"} else "MCTS tree operations plus rollout plies; simulations are the primary work count",
        }
    output["B_C_comparison"] = {
        "B_nodes_total": output["B"]["nodes_total"],
        "C_nodes_total": output["C"]["nodes_total"],
        "C_cutoffs_total": output["C"]["cutoffs_total"],
        "C_node_reduction_fraction": (output["B"]["nodes_total"] - output["C"]["nodes_total"]) / output["B"]["nodes_total"] if output["B"]["nodes_total"] else None,
        "note": "B and C faced different positions/compute budgets in the tournament; the node reduction is descriptive, not a paired search-efficiency test.",
    }
    return output


def summarize_records(records: Iterable[dict[str, Any]], config: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    rows = list(records)
    validation = validate_records(rows, config, require_complete=require_complete)
    matchups = {"-".join(pair): _matchup_summary(rows, *pair) for pair in _MATCHUP_ORDER(config)}
    models: dict[str, dict[str, Any]] = {}
    for model in ("B", "C", "D"):
        model_scores = [_score(record, model) for record in rows if model in record["matchup"]]
        block_scores: list[float] = []
        for pair in _MATCHUP_ORDER(config):
            if model not in pair:
                continue
            for seed in (731091, 731092, 731093):
                block = [record for record in rows if tuple(record["matchup"]) == pair and int(record["opening_seed"]) == seed]
                if len(block) == 2:
                    block_scores.append(sum(_score(record, model) for record in block) / 2)
        wins = sum(record.get("winner_model") == model for record in rows if model in record["matchup"])
        draws = sum(record.get("winner_model") is None for record in rows if model in record["matchup"])
        games = sum(model in record["matchup"] for record in rows)
        model_colors: dict[str, Any] = {}
        for color, key in (("B", "black"), ("W", "white")):
            subset = [record for record in rows if record["black_model" if color == "B" else "white_model"] == model]
            model_colors[key] = {
                "games": len(subset), "wins": sum(record.get("winner_model") == model for record in subset),
                "draws": sum(record.get("winner_model") is None for record in subset),
                "score_rate": sum(_score(record, model) for record in subset) / len(subset) if subset else None,
            }
        models[model] = {
            "games": games, "wins": wins, "draws": draws, "losses": games - wins - draws,
            "pure_win_rate": wins / games if games else None,
            "score_rate": sum(model_scores) / games if games else None,
            "paired_block_scores": block_scores,
            "paired_wilson_95_ci": wilson_fractional(block_scores),
            "color": model_colors,
        }
    ranking = sorted(models, key=lambda model: (-float(models[model]["score_rate"] or 0), -int(models[model]["wins"]), model))
    return {
        "experiment_id": load_config(config)["experiment_id"],
        "validation": validation,
        "game_count": len(rows),
        "matchups": matchups,
        "models": models,
        "model_usage": _model_usage(rows),
        "exploratory_ranking": ranking,
        "ranking_note": "18局、每组6局、每模型参与12局；区间以每个配对开局块为单位，仅作描述，且D获得明显更多算力。",
        "interval_method": "配对开局块 Wilson-style 描述性 95% 区间；每个对阵 n=3、总排名 n=6，不是经小样本校准的显著性检验。",
        "compute_fairness": "非对等算力挑战：B 1s/步、C 200ms/步、D 50s/步。",
    }


def write_analysis(results_dir: str | Path, config: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(results_dir)
    rows = load_game_records(root / "9x9_games.jsonl")
    summary = summarize_records(rows, config, require_complete=require_complete)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "pairwise_summary.csv").write_text("", encoding="utf-8")
    pair_path = root / "pairwise_summary.csv"
    with pair_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["matchup", "games", "first_wins", "draws", "second_wins", "first_score_rate", "second_score_rate", "first_ci95", "second_ci95"])
        for key, item in summary["matchups"].items():
            first, second = item["matchup"]
            writer.writerow([key, item["games"], item["wins"][first], item["draws"], item["wins"][second], item["score_rate"][first], item["score_rate"][second], item["paired_wilson_95_ci"][first], item["paired_wilson_95_ci"][second]])
    usage_path = root / "model_usage.csv"
    with usage_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *[key for key in summary["model_usage"]["B"]]])
        for model in ("B", "C", "D"):
            writer.writerow([model, *[summary["model_usage"][model][key] for key in summary["model_usage"]["B"]]])
    model_path = root / "model_summary.csv"
    with model_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "games", "wins", "draws", "losses", "pure_win_rate", "score_rate", "paired_ci95", "black_score_rate", "white_score_rate"])
        for model in ("B", "C", "D"):
            item = summary["models"][model]
            writer.writerow([model, item["games"], item["wins"], item["draws"], item["losses"], item["pure_win_rate"], item["score_rate"], item["paired_wilson_95_ci"], item["color"]["black"]["score_rate"], item["color"]["white"]["score_rate"]])
    return summary


_MATCHUP_ORDER = lambda config: [tuple(sorted(pair)) for pair in config["matchups"]]
