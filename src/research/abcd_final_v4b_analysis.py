"""Replay validation and descriptive match analysis for V4B."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

from ..gomoku import GomokuBoard, Player
from .abcd_final_v4b import build_schedule, load_config

_Z95 = 1.959963984540054


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _score(game: dict[str, Any], model: str) -> float:
    if game.get("winner_model") is None:
        return 0.5
    return 1.0 if game.get("winner_model") == model else 0.0


def _wilson_fractional(values: list[float]) -> list[float] | None:
    """Descriptive Wilson interval treating each paired-opening score as one unit."""

    if not values:
        return None
    n = float(len(values))
    p = max(0.0, min(1.0, sum(values) / n))
    z2 = _Z95 * _Z95
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = _Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def validate_game_record(
    record: dict[str, Any],
    config: dict[str, Any],
    expected_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check one game against its frozen schedule and replay all moves to termination."""

    frozen = load_config(config)
    schedule = build_schedule(frozen)
    by_id = {str(spec["game_id"]): spec for spec in schedule}
    game_id = str(record.get("game_id", ""))
    spec = expected_spec or by_id.get(game_id)
    if spec is None or game_id != str(spec["game_id"]):
        raise ValueError(f"Game has an unknown or mismatched ID: {game_id}")
    if record.get("experiment_id") != frozen["experiment_id"]:
        raise ValueError(f"Game {game_id} has a mismatched experiment_id")
    for key in ("board_size", "opening_index", "opening_seed", "color_swap", "black_model", "white_model", "pair_id"):
        if record.get(key) != spec.get(key):
            raise ValueError(f"Game {game_id} disagrees with schedule field {key}")
    if tuple(record.get("matchup", ())) != tuple(spec["matchup"]):
        raise ValueError(f"Game {game_id} has a mismatched matchup")
    if record.get("opening") != spec["opening"]:
        raise ValueError(f"Game {game_id} does not use the frozen opening")
    source_type = record.get("source_type")
    if source_type not in {"new", "reused_v3"}:
        raise ValueError(f"Game {game_id} has an unknown source_type")
    if source_type == "new":
        if int(record.get("game_seed", -1)) != int(spec["game_seed"]):
            raise ValueError(f"Game {game_id} has a mismatched deterministic game seed")
    elif tuple(spec["matchup"]) != ("A", "D") or not record.get("source_game_id") or not record.get("source_game_record_sha256"):
        raise ValueError(f"Game {game_id} has invalid V3 reuse provenance")
    elif record.get("source_experiment_id") != frozen["reuse_v3"]["source_experiment_id"]:
        raise ValueError(f"Game {game_id} has a mismatched source_experiment_id")

    moves = record.get("moves")
    if not isinstance(moves, list) or len(moves) < 4:
        raise ValueError(f"Game {game_id} has no complete move sequence")
    board = GomokuBoard(9)
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    timed_out_count = 0
    nodes_total = simulations_total = rollout_plies_total = 0
    terminal_simulations_total = truncated_simulations_total = 0
    max_tree_depth = 0
    thinking_seconds = 0.0
    for index, move in enumerate(moves):
        if not isinstance(move, dict) or move.get("ply") != index:
            raise ValueError(f"Game {game_id} has a malformed move record at index {index}")
        if move.get("player") != current:
            raise ValueError(f"Game {game_id} has an out-of-turn move at ply {index}")
        is_opening = index < 4
        if bool(move.get("opening")) != is_opening:
            raise ValueError(f"Game {game_id} has an invalid opening marker at ply {index}")
        expected_model = None if is_opening else (record["black_model"] if current == "B" else record["white_model"])
        if move.get("model") != expected_model:
            raise ValueError(f"Game {game_id} has an incorrect model at ply {index}")
        elapsed = float(move.get("elapsed_seconds", -1.0))
        if elapsed < 0:
            raise ValueError(f"Game {game_id} has a negative move time")
        thinking_seconds += elapsed
        timed_out_count += int(bool(move.get("timed_out", False)))
        for metric in (
            "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations", "rollout_plies",
            "terminal_simulations", "truncated_simulations", "root_candidates", "root_min_visits", "reused_nodes",
        ):
            if int(move.get(metric, 0)) < 0:
                raise ValueError(f"Game {game_id} has a negative {metric} value")
        nodes_total += int(move.get("nodes", 0))
        simulations_total += int(move.get("simulations", 0))
        rollout_plies_total += int(move.get("rollout_plies", 0))
        terminal_simulations_total += int(move.get("terminal_simulations", 0))
        truncated_simulations_total += int(move.get("truncated_simulations", 0))
        max_tree_depth = max(max_tree_depth, int(move.get("tree_depth", 0)))
        if move.get("illegal"):
            if is_opening or index != len(moves) - 1:
                raise ValueError(f"Game {game_id} has a nonterminal illegal move")
            attempted = move.get("attempted_move")
            if attempted is None:
                raise ValueError(f"Game {game_id} illegal forfeit is missing attempted_move")
            if (
                isinstance(attempted, (tuple, list))
                and len(attempted) == 2
                and type(attempted[0]) is int
                and type(attempted[1]) is int
                and board.in_bounds(attempted[0], attempted[1])
                and board.get((attempted[0], attempted[1])) is None
            ):
                raise ValueError(f"Game {game_id} attempted_move is legal but marked illegal")
            winner_color = _other(current)
            winner_model = record["black_model"] if winner_color == "B" else record["white_model"]
            termination = "illegal_move_forfeit"
            break
        if not all(type(move.get(coordinate)) is int for coordinate in ("row", "col")):
            raise ValueError(f"Game {game_id} has invalid coordinates at ply {index}")
        coord = int(move["row"]), int(move["col"])
        try:
            board.place_stone(current, coord)
        except ValueError as exc:
            raise ValueError(f"Game {game_id} repeats or misplaces a stone at ply {index}") from exc
        if board.check_victory(coord, current):
            if index != len(moves) - 1:
                raise ValueError(f"Game {game_id} continues after a five-in-a-row at ply {index}")
            winner_color = current
            winner_model = expected_model
            termination = "five_in_row"
            break
        if board.is_full():
            if index != len(moves) - 1:
                raise ValueError(f"Game {game_id} continues after the board is full")
            termination = "full_board"
            break
        current = _other(current)

    expected_outcome = "draw" if winner_color is None else f"{winner_color.lower()}_win"
    if winner_color is None and len(board.history) != 81:
        raise ValueError(f"Game {game_id} ended before a win or a full-board draw")
    if record.get("winner_color") != winner_color or record.get("winner_model") != winner_model:
        raise ValueError(f"Game {game_id} outcome metadata disagrees with replay")
    if record.get("outcome") != expected_outcome or record.get("termination") != termination:
        raise ValueError(f"Game {game_id} termination metadata disagrees with replay")
    if int(record.get("move_count", -1)) != len(moves):
        raise ValueError(f"Game {game_id} move_count disagrees with move list")
    if int(record.get("timeout_count", -1)) != timed_out_count:
        raise ValueError(f"Game {game_id} timeout count disagrees with move records")
    expected_totals = {
        "search_nodes_total": nodes_total,
        "simulations_total": simulations_total,
        "rollout_plies_total": rollout_plies_total,
        "terminal_simulations_total": terminal_simulations_total,
        "truncated_simulations_total": truncated_simulations_total,
        "tree_depth_max": max_tree_depth,
    }
    for key, value in expected_totals.items():
        if key in record and int(record[key]) != value:
            raise ValueError(f"Game {game_id} has inconsistent aggregate {key}")
    if "thinking_seconds_total" in record and not math.isclose(float(record["thinking_seconds_total"]), thinking_seconds, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f"Game {game_id} has an inconsistent total thinking time")
    return {
        "game_id": game_id,
        "move_count": len(moves),
        "outcome": expected_outcome,
        "termination": termination,
        "thinking_seconds_total": thinking_seconds,
        "timed_out_moves": timed_out_count,
    }


def validate_records(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Check IDs, paired colors, opening layouts, and replay every complete game."""

    frozen = load_config(config)
    games = list(records)
    schedule = build_schedule(frozen)
    scheduled = {str(spec["game_id"]): spec for spec in schedule}
    by_id: dict[str, dict[str, Any]] = {}
    for game in games:
        identifier = str(game.get("game_id", ""))
        if not identifier or identifier in by_id:
            raise ValueError(f"Missing or duplicate V4B game ID: {identifier}")
        if identifier not in scheduled:
            raise ValueError(f"Unscheduled V4B game ID: {identifier}")
        validate_game_record(game, frozen, scheduled[identifier])
        by_id[identifier] = game
    missing = set(scheduled) - set(by_id)
    if require_complete and missing:
        raise ValueError(f"V4B experiment is incomplete: {len(missing)} games missing")

    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    blocks: dict[tuple[tuple[str, str], int], list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        pair = tuple(game["matchup"])
        pair_groups[pair].append(game)
        blocks[(pair, int(game["opening_seed"]))].append(game)
    for pair in _MATCHUP_ORDER(frozen):
        rows = pair_groups.get(pair, [])
        if require_complete and len(rows) != 6:
            raise ValueError(f"Matchup {pair} has {len(rows)} games; expected six")
        if require_complete and Counter(row["black_model"] for row in rows) != Counter({pair[0]: 3, pair[1]: 3}):
            raise ValueError(f"Matchup {pair} does not have three games per color")
    for (pair, seed), rows in blocks.items():
        if len(rows) not in (1, 2):
            raise ValueError(f"Opening block {pair}/{seed} has an invalid game count")
        if require_complete and len(rows) != 2:
            raise ValueError(f"Opening block {pair}/{seed} is missing its color-swapped game")
        if len(rows) == 2:
            first, second = rows
            if first["opening"] != second["opening"]:
                raise ValueError(f"Opening block {pair}/{seed} has different stones")
            if first["black_model"] != second["white_model"] or first["white_model"] != second["black_model"]:
                raise ValueError(f"Opening block {pair}/{seed} lacks a color swap")
    return {
        "complete": not missing,
        "scheduled_games": len(schedule),
        "validated_games": len(games),
        "missing_games": len(missing),
        "reused_v3_games": sum(row.get("source_type") == "reused_v3" for row in games),
        "new_games": sum(row.get("source_type") == "new" for row in games),
        "illegal_forfeits": sum(row.get("termination") == "illegal_move_forfeit" for row in games),
        "timeouts": sum(int(row.get("timeout_count", 0)) for row in games),
    }


def _MATCHUP_ORDER(config: dict[str, Any]) -> list[tuple[str, str]]:
    return [tuple(sorted(pair)) for pair in config["matchups"]]


def _matchup_summary(games: list[dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    rows = [game for game in games if set(game["matchup"]) == {first, second}]
    if not rows:
        return {}
    wins = {model: sum(game.get("winner_model") == model for game in rows) for model in (first, second)}
    draws = sum(game.get("winner_model") is None for game in rows)
    result: dict[str, Any] = {
        "matchup": [first, second],
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": {model: len(rows) - wins[model] - draws for model in (first, second)},
        "score_rate": {},
        "pure_win_rate": {},
        "paired_block_scores": {},
        "paired_wilson_95_ci": {},
        "color": {},
        "source_counts": dict(Counter(str(game.get("source_type", "unknown")) for game in rows)),
    }
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in rows:
        by_seed[int(game["opening_seed"])].append(game)
    for model in (first, second):
        points = [_score(game, model) for game in rows]
        block_scores = []
        for seed in sorted(by_seed):
            block = by_seed[seed]
            if len(block) == 2:
                block_scores.append(sum(_score(game, model) for game in block) / 2.0)
        result["score_rate"][model] = sum(points) / len(points)
        result["pure_win_rate"][model] = wins[model] / len(rows)
        result["paired_block_scores"][model] = block_scores
        result["paired_wilson_95_ci"][model] = _wilson_fractional(block_scores)
        model_color = {}
        for color in ("black", "white"):
            color_rows = [game for game in rows if game[f"{color}_model"] == model]
            model_color[color] = {
                "games": len(color_rows),
                "wins": sum(game.get("winner_model") == model for game in color_rows),
                "draws": sum(game.get("winner_model") is None for game in color_rows),
                "score_rate": sum(_score(game, model) for game in color_rows) / len(color_rows) if color_rows else None,
            }
        result["color"][model] = model_color
    return result


def _usage_summary(games: list[dict[str, Any]], model: str, source_type: str | None = None) -> dict[str, Any]:
    rows = [game for game in games if model in game.get("matchup", ()) and (source_type is None or game.get("source_type") == source_type)]
    moves = [move for game in rows for move in game.get("moves", ()) if move.get("model") == model and not move.get("opening")]
    seconds = sum(float(move.get("elapsed_seconds", 0.0)) for move in moves)
    nodes = sum(int(move.get("nodes", 0)) for move in moves)
    cutoffs = sum(int(move.get("cutoffs", 0)) for move in moves)
    simulations = sum(int(move.get("simulations", 0)) for move in moves)
    depths = [int(move.get("completed_depth", 0)) for move in moves]
    tree_depths = [int(move.get("tree_depth", 0)) for move in moves]
    node_metric_semantics = {
        "A": "non-comparable: original rule AI API counter is remaining empty cells, not searched nodes",
        "B": "Minimax visited search nodes",
        "C": "Alpha-Beta visited search nodes",
        "D": "MCTS tree selection/expansion operations plus rollout plies; simulation count reported separately",
    }[model]
    return {
        "model": model,
        "source_type": source_type or "all",
        "games": len(rows),
        "moves": len(moves),
        "thinking_seconds_total": seconds,
        "average_seconds_per_move": seconds / len(moves) if moves else 0.0,
        "average_nodes_per_move": nodes / len(moves) if moves else 0.0,
        "nodes_total": nodes,
        "node_metric_semantics": node_metric_semantics,
        "search_nodes_comparable": model in {"B", "C"},
        "average_cutoffs_per_move": cutoffs / len(moves) if moves else 0.0,
        "cutoffs_total": cutoffs,
        "cutoff_to_node_ratio": cutoffs / nodes if nodes else 0.0,
        "average_simulations_per_move": simulations / len(moves) if moves else 0.0,
        "simulations_total": simulations,
        "simulations_per_second": simulations / seconds if seconds else 0.0,
        "average_completed_depth": sum(depths) / len(depths) if depths else 0.0,
        "min_completed_depth": min(depths, default=0),
        "max_completed_depth": max(depths, default=0),
        "average_tree_depth": sum(tree_depths) / len(tree_depths) if tree_depths else 0.0,
        "max_tree_depth": max(tree_depths, default=0),
        "reused_nodes_total": sum(int(move.get("reused_nodes", 0)) for move in moves),
        "timeouts": sum(bool(move.get("timed_out", False)) for move in moves),
        "illegal_moves": sum(bool(move.get("illegal", False)) for move in moves),
    }


def summarize_records(records: Iterable[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate exact W/D/L, paired-opening Wilson intervals, color, and search costs."""

    games = list(records)
    validation = validate_records(games, config, require_complete=True)
    matchups = {
        "-".join(pair): _matchup_summary(games, pair[0], pair[1])
        for pair in _MATCHUP_ORDER(config)
    }
    model_summary: dict[str, Any] = {}
    for model in ("A", "B", "C", "D"):
        model_games = [game for game in games if model in game["matchup"]]
        wins = sum(game.get("winner_model") == model for game in model_games)
        draws = sum(game.get("winner_model") is None for game in model_games)
        blocks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for game in model_games:
            other_model = next(item for item in game["matchup"] if item != model)
            blocks[(other_model, int(game["opening_seed"]))].append(game)
        block_scores = [sum(_score(game, model) for game in block) / 2.0 for block in blocks.values() if len(block) == 2]
        colors = {}
        for color in ("black", "white"):
            color_games = [game for game in model_games if game[f"{color}_model"] == model]
            colors[color] = {
                "games": len(color_games),
                "wins": sum(game.get("winner_model") == model for game in color_games),
                "draws": sum(game.get("winner_model") is None for game in color_games),
                "score_rate": sum(_score(game, model) for game in color_games) / len(color_games) if color_games else None,
            }
        model_summary[model] = {
            "games": len(model_games),
            "wins": wins,
            "draws": draws,
            "losses": len(model_games) - wins - draws,
            "win_rate": wins / len(model_games) if model_games else 0.0,
            "score_rate": (wins + draws * 0.5) / len(model_games) if model_games else 0.0,
            "paired_block_scores": block_scores,
            "paired_wilson_95_ci": _wilson_fractional(block_scores),
            "color": colors,
        }
    usage = {model: _usage_summary(games, model) for model in ("A", "B", "C", "D")}
    usage["D_v3_reused"] = _usage_summary(games, "D", "reused_v3")
    usage["D_new_v4b"] = _usage_summary(games, "D", "new")
    b_c_savings = {
        "B_nodes_total": usage["B"]["nodes_total"],
        "C_nodes_total": usage["C"]["nodes_total"],
        "C_vs_B_node_reduction_fraction": (
            1.0 - usage["C"]["nodes_total"] / usage["B"]["nodes_total"]
            if usage["B"]["nodes_total"]
            else None
        ),
        "B_average_completed_depth": usage["B"]["average_completed_depth"],
        "C_average_completed_depth": usage["C"]["average_completed_depth"],
        "C_cutoffs": usage["C"]["cutoffs_total"],
        "note": "Game searches are not identical positions for B and C; this is descriptive tournament telemetry. See V4 fixed-position comparisons for paired search-efficiency conclusions.",
    }
    ranked = sorted(
        model_summary,
        key=lambda model: (model_summary[model]["score_rate"], model_summary[model]["wins"]),
        reverse=True,
    )
    return {
        "experiment_id": config["experiment_id"],
        "validation": validation,
        "game_count": len(games),
        "matchups": matchups,
        "models": model_summary,
        "usage": usage,
        "B_C_search_comparison": b_c_savings,
        "exploratory_ranking": ranked,
        "ranking_note": "Six games per matchup and three opening blocks make this an exploratory ordering only; overlapping paired 95% intervals and outcome discreteness preclude a stable-strength claim.",
        "interval_method": "Paired-opening block Wilson-style descriptive 95% interval: each block averages the two color-swapped game scores; n=3 per matchup. Coverage is not calibrated for this tiny paired distribution.",
    }


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _match_chart(summary: dict[str, Any], path: Path) -> None:
    width, height = 1040, 530
    left, right, top, bottom = 275, 36, 92, 84
    chart_w, chart_h = width - left - right, height - top - bottom
    bar_h, gap = 18, 11
    colors = {"A": "#3b82f6", "B": "#f97316", "C": "#8b5cf6", "D": "#0f9d72"}
    bits = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="42" font-family="sans-serif" font-size="24" font-weight="700" fill="#172033">V4B 六组配对对弈得分率</text>',
        '<text x="32" y="66" font-family="sans-serif" font-size="13" fill="#596579">每组 6 局；每个配对开局交换黑白。误差线为配对开局 Wilson 近似 95% 区间。</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + chart_w * tick / 100
        bits.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+chart_h}" stroke="#e5e9f0"/>')
        bits.append(f'<text x="{x:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#596579">{tick}%</text>')
    labels = list(summary["matchups"])
    row_h = chart_h / len(labels)
    for row_index, key in enumerate(labels):
        entry = summary["matchups"][key]
        first, second = entry["matchup"]
        y0 = top + row_index * row_h
        bits.append(f'<text x="{left-18}" y="{y0+18}" text-anchor="end" font-family="sans-serif" font-size="13" font-weight="600" fill="#172033">{first} vs {second}</text>')
        for sub, model in enumerate((first, second)):
            y = y0 + 7 + sub * (bar_h + gap)
            value = entry["score_rate"][model]
            x_end = left + chart_w * value
            bits.append(f'<rect x="{left}" y="{y}" width="{chart_w*value:.1f}" height="{bar_h}" rx="4" fill="{colors[model]}" opacity="0.88"/>')
            ci = entry["paired_wilson_95_ci"][model]
            if ci:
                x_low = left + chart_w * ci[0]
                x_high = left + chart_w * ci[1]
                center = y + bar_h / 2
                bits.append(f'<line x1="{x_low:.1f}" y1="{center:.1f}" x2="{x_high:.1f}" y2="{center:.1f}" stroke="#172033" stroke-width="2"/>')
                bits.append(f'<line x1="{x_low:.1f}" y1="{center-4:.1f}" x2="{x_low:.1f}" y2="{center+4:.1f}" stroke="#172033"/>')
                bits.append(f'<line x1="{x_high:.1f}" y1="{center-4:.1f}" x2="{x_high:.1f}" y2="{center+4:.1f}" stroke="#172033"/>')
            bits.append(f'<text x="{min(x_end+8,width-50):.1f}" y="{y+bar_h-4}" font-family="sans-serif" font-size="12" font-weight="600" fill="#172033">{model} {value:.0%}</text>')
    bits.append('<text x="32" y="500" font-family="sans-serif" font-size="12" fill="#596579">得分：胜 1，平 0.5，负 0。每组仅有 3 个配对开局，区间仅作描述。</text>')
    bits.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(bits) + "\n", encoding="utf-8")


def _model_usage_chart(summary: dict[str, Any], path: Path) -> None:
    width, height = 1000, 510
    models = ("A", "B", "C", "D")
    minimax_models = ("B", "C")
    colors = {"A": "#3b82f6", "B": "#f97316", "C": "#8b5cf6", "D": "#0f9d72"}
    max_nodes = max(summary["usage"][model]["average_nodes_per_move"] for model in minimax_models) or 1
    max_seconds = max(summary["usage"][model]["average_seconds_per_move"] for model in models) or 1
    bits = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="42" font-family="sans-serif" font-size="24" font-weight="700" fill="#172033">对局中的平均单步搜索成本</text>',
        '<text x="32" y="66" font-family="sans-serif" font-size="13" fill="#596579">左：B/C Minimax 搜索节点；右：各模型真实思考秒数。A 节点计数无可比性，D 模拟次数见表格。</text>',
    ]
    left_x, right_x, y_top, chart_h = 110, 580, 100, 315
    for panel_x, metric, maximum, title in (
        (left_x, "average_nodes_per_move", max_nodes, "B/C 搜索节点 / 步"),
        (right_x, "average_seconds_per_move", max_seconds, "实际耗时 / 步"),
    ):
        bits.append(f'<text x="{panel_x}" y="{y_top-16}" font-family="sans-serif" font-size="15" font-weight="600" fill="#172033">{title}</text>')
        for tick in range(0, 101, 25):
            x = panel_x + 330 * tick / 100
            bits.append(f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_top+chart_h}" stroke="#e5e9f0"/>')
        panel_models = minimax_models if metric == "average_nodes_per_move" else models
        for index, model in enumerate(panel_models):
            usage = summary["usage"][model]
            value = float(usage[metric])
            y = y_top + 18 + index * 70
            fraction = value / maximum if maximum else 0
            bits.append(f'<text x="{panel_x-14}" y="{y+18}" text-anchor="end" font-family="sans-serif" font-size="13" font-weight="600" fill="#172033">{model}</text>')
            bits.append(f'<rect x="{panel_x}" y="{y}" width="{330*fraction:.1f}" height="28" rx="4" fill="{colors[model]}"/>')
            if metric == "average_nodes_per_move":
                label = f'{value:,.0f}'
            else:
                label = f'{value:.3f}s'
            bits.append(f'<text x="{panel_x+min(330*fraction+8,335):.1f}" y="{y+19}" font-family="sans-serif" font-size="12" fill="#172033">{label}</text>')
        for tick in range(0, 101, 25):
            x = panel_x + 330 * tick / 100
            bits.append(f'<text x="{x:.1f}" y="{y_top+chart_h+20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#596579">{tick}%</text>')
    bits.extend([
        '<text x="32" y="484" font-family="sans-serif" font-size="12" fill="#596579">B/C 的单步耗时目标均为 200ms；完成深度和 Alpha-Beta 剪枝次数见报告表格。D 的模拟树跨回合复用。</text>',
        "</svg>",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(bits) + "\n", encoding="utf-8")


def write_analysis(
    records: Iterable[dict[str, Any]], config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Write reproducible JSON/CSV/SVG tables and return the analysis payload."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records, config)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matchup_rows = []
    for key, item in summary["matchups"].items():
        first, second = item["matchup"]
        matchup_rows.append(
            {
                "matchup": key,
                "games": item["games"],
                "first_wins": item["wins"][first],
                "draws": item["draws"],
                "second_wins": item["wins"][second],
                "first_win_rate": item["pure_win_rate"][first],
                "first_score_rate": item["score_rate"][first],
                "first_ci95_low": item["paired_wilson_95_ci"][first][0],
                "first_ci95_high": item["paired_wilson_95_ci"][first][1],
                "second_score_rate": item["score_rate"][second],
                "second_ci95_low": item["paired_wilson_95_ci"][second][0],
                "second_ci95_high": item["paired_wilson_95_ci"][second][1],
                "source_counts": json.dumps(item["source_counts"], sort_keys=True),
            }
        )
    _write_csv(root / "pairwise_summary.csv", list(matchup_rows[0]), matchup_rows)
    model_rows = []
    for model, result in summary["models"].items():
        model_rows.append({"model": model, **{key: value for key, value in result.items() if key not in {"paired_block_scores", "paired_wilson_95_ci", "color"}}, "ci95_low": result["paired_wilson_95_ci"][0], "ci95_high": result["paired_wilson_95_ci"][1]})
    _write_csv(root / "model_summary.csv", list(model_rows[0]), model_rows)
    usage_rows = list(summary["usage"].values())
    _write_csv(root / "model_usage.csv", list(usage_rows[0]), usage_rows)
    _match_chart(summary, root.parent / "figures" / "matchup_score_rates.svg")
    _model_usage_chart(summary, root.parent / "figures" / "model_search_cost.svg")
    return summary
