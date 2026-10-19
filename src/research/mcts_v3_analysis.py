"""Validation, paired statistics, and budget-trend plotting for V3."""
from __future__ import annotations

from collections import Counter
import html
from pathlib import Path
from typing import Any, Iterable

from .mcts_v2_metrics import summarize_phases
from .mcts_v2_validation import validate_records

_OPPONENTS = ("A", "B", "C")
_INTERVAL_METHOD = (
    "approximate paired-block Wilson-style 95% interval; each of only three paired "
    "opening blocks contributes its fractional mean score as a Bernoulli success count, "
    "so coverage is not calibrated for the paired-score distribution"
)


def validate_v3_records(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Replay rules, IDs, paired openings, and the registered high-compute schedule."""

    experiment_id = str(config.get("experiment_id", ""))
    if not experiment_id.startswith("wuziq-mcts-high-compute-v3"):
        raise ValueError("config is not a V3 high-compute experiment")
    phases = list(config.get("phases", ()))
    if not phases or any(str(phase.get("track", "")).lower() != "high-compute" for phase in phases):
        raise ValueError("V3 config must contain high-compute phases")
    if any(int(phase.get("board_size", 0)) != 9 for phase in phases):
        raise ValueError("V3 high-compute phases must use a 9×9 board")

    result = validate_records(records, config)
    configured_count = config.get("official_game_count")
    if configured_count is not None and int(configured_count) != result["scheduled_game_count"]:
        raise ValueError("official_game_count differs from the registered schedule")
    if require_complete and not result["complete"]:
        raise ValueError(
            f"V3 result set is incomplete: {result['missing_game_count']} games are missing"
        )
    if require_complete and result["unpaired_opening_group_count"]:
        raise ValueError("V3 complete result set contains unpaired openings")
    return result


def _paired_stats(phase: dict[str, Any], model: str, opponent: str) -> dict[str, Any]:
    pair_key = "-".join(sorted((model, opponent)))
    return phase["pairwise"][pair_key]["models"][model]


def _model_usage(
    games: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    moves = [
        move
        for game in games
        for move in game.get("moves", ())
        if move.get("model") == model
    ]
    seconds = sum(float(move.get("elapsed_seconds", 0.0)) for move in moves)
    simulations = sum(int(move.get("simulations", 0)) for move in moves)
    depths = [int(move.get("tree_depth", move.get("completed_depth", 0))) for move in moves]
    depth_counts = Counter(depths)
    over_budget = 0
    for game in games:
        budget = float(game["budget_seconds_by_model"][model])
        over_budget += sum(
            float(move.get("elapsed_seconds", 0.0)) > budget
            for move in game.get("moves", ())
            if move.get("model") == model
        )
    return {
        "moves": len(moves),
        "thinking_seconds_total": seconds,
        "average_seconds_per_move": seconds / len(moves) if moves else 0.0,
        "simulations_total": simulations,
        "average_simulations_per_move": simulations / len(moves) if moves else 0.0,
        "simulations_per_thinking_second": simulations / seconds if seconds else None,
        "average_tree_depth": sum(depths) / len(depths) if depths else 0.0,
        "max_tree_depth": max(depths, default=0),
        "tree_depth_counts": {str(depth): depth_counts[depth] for depth in sorted(depth_counts)},
        "search_timeout_flags": sum(bool(move.get("timed_out", False)) for move in moves),
        "illegal_moves": sum(bool(move.get("illegal", False)) for move in moves),
        "actual_over_budget_moves": over_budget,
    }


def summarize_v3_records(
    records: Iterable[dict[str, Any]],
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 20_261_019,
) -> dict[str, Any]:
    """Aggregate outcomes, paired 95% intervals, actual MCTS work, and budget trends."""

    games = list(records)
    identifiers = [str(game["game_id"]) for game in games]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate game_id in V3 result records")
    generic = summarize_phases(games, bootstrap_samples=bootstrap_samples, seed=seed)
    phase_results: dict[str, Any] = {}
    trend: dict[str, list[dict[str, Any]]] = {opponent: [] for opponent in _OPPONENTS}

    for phase_name in sorted(generic["phases"]):
        stats = generic["phases"][phase_name]
        phase_games = [game for game in games if str(game["phase_name"]) == phase_name]
        models = sorted(
            {str(model) for game in phase_games for model in game.get("matchup", ())}
        )
        if len(models) != 2 or "D3" not in models:
            raise ValueError(f"V3 phase {phase_name} must contain D3 and one opponent")
        opponent = next(model for model in models if model != "D3")
        if opponent not in _OPPONENTS:
            raise ValueError(f"V3 phase {phase_name} contains unsupported opponent {opponent}")
        budgets = stats["budget_seconds_by_model"]
        d3_budget = float(budgets["D3"])
        if d3_budget not in (10.0, 20.0, 50.0) or float(budgets[opponent]) != 0.05:
            raise ValueError(f"V3 phase {phase_name} has an unregistered time budget")

        d3_result = _paired_stats(stats, "D3", opponent)
        opponent_result = _paired_stats(stats, opponent, "D3")
        d3_usage = _model_usage(phase_games, "D3")
        opponent_usage = _model_usage(phase_games, opponent)
        black_games = [game for game in phase_games if game["black_model"] == "D3"]
        white_games = [game for game in phase_games if game["white_model"] == "D3"]

        def color_result(color_games: list[dict[str, Any]]) -> dict[str, Any]:
            wins = sum(game.get("winner_model") == "D3" for game in color_games)
            draws = sum(game.get("winner_model") is None for game in color_games)
            losses = len(color_games) - wins - draws
            if not color_games:
                score_rate = 0.0
            else:
                score_rate = (wins + 0.5 * draws) / len(color_games)
            return {
                "games": len(color_games),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score_rate": score_rate,
            }

        condition = {
            "phase_name": phase_name,
            "opponent_model": opponent,
            "interval_method": _INTERVAL_METHOD,
            "board_size": int(stats["board_size"]),
            "games": len(phase_games),
            "D3_budget_seconds": d3_budget,
            "opponent_budget_seconds": float(budgets[opponent]),
            "D3": {
                "wins": int(d3_result["wins"]),
                "losses": int(d3_result["losses"]),
                "draws": int(d3_result["draws"]),
                "pure_win_rate": float(d3_result["pure_win_rate"]),
                "score_rate": float(d3_result["score_rate"]),
                "pure_win_rate_paired_wilson_ci95": d3_result[
                    "pure_win_rate_paired_wilson_ci95"
                ],
                "score_rate_paired_wilson_ci95": d3_result[
                    "score_rate_paired_wilson_ci95"
                ],
                **d3_usage,
                "score_points_per_thinking_second": (
                    (int(d3_result["wins"]) + 0.5 * int(d3_result["draws"]))
                    / d3_usage["thinking_seconds_total"]
                    if d3_usage["thinking_seconds_total"]
                    else None
                ),
            },
            "opponent": {
                "wins": int(opponent_result["wins"]),
                "losses": int(opponent_result["losses"]),
                "draws": int(opponent_result["draws"]),
                "pure_win_rate": float(opponent_result["pure_win_rate"]),
                "score_rate": float(opponent_result["score_rate"]),
                "score_rate_paired_wilson_ci95": opponent_result[
                    "score_rate_paired_wilson_ci95"
                ],
                **opponent_usage,
            },
            "D3_color_split": {
                "black": color_result(black_games),
                "white": color_result(white_games),
            },
            "win_matrix": stats["win_matrix"],
            "pair_counts": stats["paired_opening_count_by_matchup"],
        }
        phase_results[phase_name] = condition
        trend[opponent].append(
            {
                "budget_seconds": d3_budget,
                "games": len(phase_games),
                "score_rate": condition["D3"]["score_rate"],
                "ci95": condition["D3"]["score_rate_paired_wilson_ci95"],
                "average_simulations_per_move": d3_usage["average_simulations_per_move"],
                "average_seconds_per_move": d3_usage["average_seconds_per_move"],
                "average_tree_depth": d3_usage["average_tree_depth"],
            }
        )

    for points in trend.values():
        points.sort(key=lambda point: point["budget_seconds"])
    return {
        "schema_version": 1,
        "game_count": len(games),
        "phase_count": len(phase_results),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "interval_method": _INTERVAL_METHOD,
        "phases": phase_results,
        "budget_trend": trend,
    }


def write_d3_budget_trend_svg(summary: dict[str, Any], destination: str | Path) -> None:
    """Write score-rate trends with paired 95% uncertainty bars as standalone SVG."""

    width, height = 960, 560
    left, right, top, bottom = 88, 36, 72, 112
    plot_width, plot_height = width - left - right, height - top - bottom
    series = summary["budget_trend"]
    budgets = sorted(
        {float(point["budget_seconds"]) for points in series.values() for point in points}
    )
    if not budgets:
        raise ValueError("Cannot draw a V3 budget chart without result points")
    x_positions = {
        budget: left + index * plot_width / max(1, len(budgets) - 1)
        for index, budget in enumerate(budgets)
    }

    def y(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="480" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">D3 不同算力预算下的得分率</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_position = y(fraction)
        pieces.extend(
            [
                f'<line x1="{left}" y1="{y_position:.1f}" x2="{width-right}" y2="{y_position:.1f}" stroke="#e2e8f0"/>',
                f'<text x="{left-12}" y="{y_position+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{fraction:.0%}</text>',
            ]
        )
    pieces.append(
        f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#64748b"/>'
    )
    for budget in budgets:
        x = x_positions[budget]
        pieces.append(
            f'<text x="{x:.1f}" y="{top+plot_height+26}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">{budget:g} 秒</text>'
        )
    pieces.append(
        f'<text x="{left+plot_width/2:.1f}" y="{height-48}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">D3 每步搜索预算（对手固定 50 毫秒）</text>'
    )
    colors = {"A": "#2563eb", "B": "#dc2626", "C": "#059669"}
    legend_x = left + 24
    for index, opponent in enumerate(_OPPONENTS):
        points = series.get(opponent, [])
        if not points:
            continue
        color = colors[opponent]
        coordinates = [
            (x_positions[float(point["budget_seconds"])], y(float(point["score_rate"])))
            for point in points
        ]
        path_data = " ".join(
            ("M" if index_point == 0 else "L") + f" {x:.1f} {y_value:.1f}"
            for index_point, (x, y_value) in enumerate(coordinates)
        )
        pieces.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for point, (x, y_value) in zip(points, coordinates):
            low, high = [float(value) for value in point["ci95"]]
            pieces.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y(high):.1f}" x2="{x:.1f}" y2="{y(low):.1f}" stroke="{color}" stroke-width="1.5"/>',
                    f'<circle cx="{x:.1f}" cy="{y_value:.1f}" r="5" fill="{color}"/>',
                    f'<text x="{x:.1f}" y="{y_value-10:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{color}">n={int(point["games"])}</text>',
                ]
            )
        x_legend = legend_x + index * 180
        pieces.extend(
            [
                f'<line x1="{x_legend}" y1="{height-20}" x2="{x_legend+24}" y2="{height-20}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x_legend+31}" y="{height-16}" font-family="sans-serif" font-size="12" fill="#334155">对战 {html.escape(opponent)}</text>',
            ]
        )
    pieces.append(
        f'<text x="{width-right}" y="{height-16}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">得分率：胜 1、平 0.5、负 0；误差线为配对块 Wilson 近似 95% 区间</text>'
    )
    pieces.append("</svg>")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(pieces) + "\n", encoding="utf-8")
