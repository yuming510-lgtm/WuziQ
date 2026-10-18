"""Summarize a V2 MCTS run and render reproducible SVG charts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.mcts_v2_metrics import summarize_phases


COLORS = {
    "A": "#64748b",
    "B": "#e76f51",
    "C": "#2563eb",
    "D": "#9333ea",
    "D1": "#0f766e",
    "D2": "#d97706",
}
MODEL_ORDER = ("A", "B", "C", "D", "D1", "D2")
BUDGETS = (0.05, 0.2, 1.0, 5.0, 10.0)


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(results_dir.glob("*_games.jsonl")):
        records.extend(load_game_records(path))
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_csv_rows(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for phase_name, phase in summary["phases"].items():
        budget_values = phase["budget_seconds_by_model"]
        distinct_budgets = set(budget_values.values())
        shared_budget = next(iter(distinct_budgets)) if len(distinct_budgets) == 1 else None
        for result in phase["ranking"]:
            model_rows.append(
                {
                    "phase": phase_name,
                    "track": phase["track"],
                    "board_size": phase["board_size"],
                    "budget_seconds": shared_budget,
                    "budget_C_seconds": budget_values.get("C"),
                    "budget_D2_seconds": budget_values.get("D2"),
                    "model": result["model"],
                    "games": result["games"],
                    "wins": result["wins"],
                    "losses": result["losses"],
                    "draws": result["draws"],
                    "pure_win_rate": result["pure_win_rate"],
                    "pure_win_rate_ci_low": result["pure_win_rate_ci95"][0],
                    "pure_win_rate_ci_high": result["pure_win_rate_ci95"][1],
                    "pure_win_rate_paired_wilson_ci_low": result[
                        "pure_win_rate_paired_wilson_ci95"
                    ][0],
                    "pure_win_rate_paired_wilson_ci_high": result[
                        "pure_win_rate_paired_wilson_ci95"
                    ][1],
                    "score_rate": result["score_rate"],
                    "score_rate_ci_low": result["score_rate_ci95"][0],
                    "score_rate_ci_high": result["score_rate_ci95"][1],
                    "score_rate_paired_wilson_ci_low": result[
                        "score_rate_paired_wilson_ci95"
                    ][0],
                    "score_rate_paired_wilson_ci_high": result[
                        "score_rate_paired_wilson_ci95"
                    ][1],
                    "average_seconds_per_move": result["average_seconds_per_move"],
                    "average_simulations_per_move": result["average_simulations_per_move"],
                    "average_nodes_per_move": result["average_nodes_per_move"],
                    "average_completed_depth": result["average_completed_depth"],
                    "timed_out_moves": result["timed_out_moves"],
                    "illegal_moves": result["illegal_moves"],
                    "model_thinking_seconds_total": result["model_thinking_seconds_total"],
                    "model_simulations_total": result["model_simulations_total"],
                    "model_nodes_total": result["model_nodes_total"],
                    "model_rollout_plies_total": result["model_rollout_plies_total"],
                    "model_timed_out_moves": result["model_timed_out_moves"],
                    "score_points_per_model_thinking_second": result[
                        "score_points_per_model_thinking_second"
                    ],
                    "simulations_per_model_thinking_second": result[
                        "simulations_per_model_thinking_second"
                    ],
                }
            )

        for pair_name, pair in phase["pairwise"].items():
            first, second = pair_name.split("-")
            first_stats, second_stats = pair["models"][first], pair["models"][second]
            pair_rows.append(
                {
                    "phase": phase_name,
                    "track": phase["track"],
                    "board_size": phase["board_size"],
                    "budget_seconds": shared_budget,
                    "budget_C_seconds": budget_values.get("C"),
                    "budget_D2_seconds": budget_values.get("D2"),
                    "model_1": first,
                    "model_2": second,
                    "games": pair["games"],
                    "wins_1": first_stats["wins"],
                    "wins_2": second_stats["wins"],
                    "draws": first_stats["draws"],
                    "model_1_pure_win_rate": first_stats["pure_win_rate"],
                    "model_1_pure_win_ci_low": first_stats["pure_win_rate_ci95"][0],
                    "model_1_pure_win_ci_high": first_stats["pure_win_rate_ci95"][1],
                    "model_1_pure_win_paired_wilson_ci_low": first_stats[
                        "pure_win_rate_paired_wilson_ci95"
                    ][0],
                    "model_1_pure_win_paired_wilson_ci_high": first_stats[
                        "pure_win_rate_paired_wilson_ci95"
                    ][1],
                    "model_1_score_rate": first_stats["score_rate"],
                    "model_1_score_ci_low": first_stats["score_rate_ci95"][0],
                    "model_1_score_ci_high": first_stats["score_rate_ci95"][1],
                    "model_1_score_paired_wilson_ci_low": first_stats[
                        "score_rate_paired_wilson_ci95"
                    ][0],
                    "model_1_score_paired_wilson_ci_high": first_stats[
                        "score_rate_paired_wilson_ci95"
                    ][1],
                    "model_2_pure_win_rate": second_stats["pure_win_rate"],
                    "model_2_pure_win_ci_low": second_stats["pure_win_rate_ci95"][0],
                    "model_2_pure_win_ci_high": second_stats["pure_win_rate_ci95"][1],
                    "model_2_pure_win_paired_wilson_ci_low": second_stats[
                        "pure_win_rate_paired_wilson_ci95"
                    ][0],
                    "model_2_pure_win_paired_wilson_ci_high": second_stats[
                        "pure_win_rate_paired_wilson_ci95"
                    ][1],
                    "model_2_score_rate": second_stats["score_rate"],
                    "model_2_score_ci_low": second_stats["score_rate_ci95"][0],
                    "model_2_score_ci_high": second_stats["score_rate_ci95"][1],
                    "model_2_score_paired_wilson_ci_low": second_stats[
                        "score_rate_paired_wilson_ci95"
                    ][0],
                    "model_2_score_paired_wilson_ci_high": second_stats[
                        "score_rate_paired_wilson_ci95"
                    ][1],
                    "model_1_thinking_seconds": first_stats[
                        "model_thinking_seconds_total"
                    ],
                    "model_2_thinking_seconds": second_stats[
                        "model_thinking_seconds_total"
                    ],
                    "model_1_simulations_total": first_stats["model_simulations_total"],
                    "model_2_simulations_total": second_stats["model_simulations_total"],
                    "model_1_score_points_per_second": first_stats[
                        "score_points_per_model_thinking_second"
                    ],
                    "model_2_score_points_per_second": second_stats[
                        "score_points_per_model_thinking_second"
                    ],
                    "score_intervals_overlap": pair["score_intervals_overlap"],
                }
            )
    return model_rows, pair_rows


def _line_svg(
    destination: Path,
    title: str,
    x_labels: list[str],
    series: dict[str, list[tuple[float, float, float] | None]],
    *,
    y_min: float,
    y_max: float,
    y_ticks: list[tuple[float, str]],
    y_label: str,
    logarithmic: bool = False,
    note: str = "",
) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 40, 72, 108
    plot_w, plot_h = width - left - right, height - top - bottom
    transform = math.log10 if logarithmic else (lambda value: value)
    lo, hi = transform(y_min), transform(y_max)

    def sx(index: int) -> float:
        return left + (plot_w / max(1, len(x_labels) - 1)) * index

    def sy(value: float) -> float:
        safe_value = max(y_min, min(y_max, value))
        return top + plot_h * (1 - (transform(safe_value) - lo) / (hi - lo))

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="600">{html.escape(title)}</text>',
    ]
    for tick, label in y_ticks:
        y = sy(tick)
        pieces.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{html.escape(label)}</text>',
            ]
        )
    pieces.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#64748b"/>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#64748b"/>',
            f'<text transform="translate(22 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    for index, label in enumerate(x_labels):
        x = sx(index)
        pieces.append(
            f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">{html.escape(label)}</text>'
        )

    for series_index, (name, values) in enumerate(series.items()):
        color = COLORS.get(name, f"hsl({(series_index*83)%360},55%,42%)")
        coordinates = []
        for index, point in enumerate(values):
            if point is None:
                continue
            value, low, high = point
            x, y = sx(index), sy(value)
            coordinates.append(f"{x:.1f},{y:.1f}")
            if low != high:
                y1, y2 = sy(low), sy(high)
                pieces.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1.5"/>',
                        f'<line x1="{x-5:.1f}" y1="{y1:.1f}" x2="{x+5:.1f}" y2="{y1:.1f}" stroke="{color}"/>',
                        f'<line x1="{x-5:.1f}" y1="{y2:.1f}" x2="{x+5:.1f}" y2="{y2:.1f}" stroke="{color}"/>',
                    ]
                )
        if coordinates:
            pieces.append(
                f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for index, point in enumerate(values):
            if point is None:
                continue
            x, y = sx(index), sy(point[0])
            pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"/>')

    legend_y = height - 48
    total_legend_w = len(series) * 110
    legend_x = max(left, (width - total_legend_w) / 2)
    for index, name in enumerate(series):
        x = legend_x + index * 110
        color = COLORS.get(name, f"hsl({(index*83)%360},55%,42%)")
        pieces.extend(
            [
                f'<line x1="{x}" y1="{legend_y}" x2="{x+22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x+29}" y="{legend_y+4}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>',
            ]
        )
    if note:
        pieces.append(
            f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{html.escape(note)}</text>'
        )
    pieces.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _budget_label(seconds: float) -> str:
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    return f"{int(seconds)}s"


def _render_score_charts(summary: dict[str, Any], figure_dir: Path) -> None:
    for board_size in (9, 15):
        phase_rows = [
            phase
            for phase in summary["phases"].values()
            if phase["track"] == "fair" and phase["board_size"] == board_size
        ]
        phase_rows.sort(key=lambda phase: next(iter(phase["budget_seconds_by_model"].values())))
        if not phase_rows:
            continue
        labels = [
            _budget_label(next(iter(phase["budget_seconds_by_model"].values())))
            for phase in phase_rows
        ]
        series: dict[str, list[tuple[float, float, float] | None]] = {}
        present_models = sorted(
            {row["model"] for phase in phase_rows for row in phase["ranking"]},
            key=lambda model: MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
        )
        for model in present_models:
            by_model = {row["model"]: row for phase in phase_rows for row in phase["ranking"]}
            values = []
            for phase in phase_rows:
                model_stats = next(
                    (row for row in phase["ranking"] if row["model"] == model), None
                )
                if model_stats is None or model_stats["games"] == 0:
                    values.append(None)
                else:
                    low, high = model_stats["score_rate_paired_wilson_ci95"]
                    values.append((model_stats["score_rate"], low, high))
            series[model] = values
        _line_svg(
            figure_dir / f"fair_score_rate_{board_size}x{board_size}.svg",
            f"{board_size}×{board_size} 同预算得分率",
            labels,
            series,
            y_min=0,
            y_max=1,
            y_ticks=[(x, f"{x:.0%}") for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)],
            y_label="得分率（平局计半分）",
            note="误差线为按配对开局块计算的 95% Wilson 区间；表格另列配对 bootstrap 区间",
        )

    extreme_rows = [
        phase for phase in summary["phases"].values() if phase["track"] == "extreme"
    ]
    extreme_rows.sort(key=lambda phase: phase["budget_seconds_by_model"].get("D2", 0))
    if extreme_rows:
        labels = [
            f"D2 {_budget_label(phase['budget_seconds_by_model']['D2'])}\nC 50ms"
            for phase in extreme_rows
        ]
        series = {}
        for model in ("C", "D2"):
            values = []
            for phase in extreme_rows:
                model_stats = next(
                    (row for row in phase["ranking"] if row["model"] == model), None
                )
                if model_stats is None or not model_stats["games"]:
                    values.append(None)
                else:
                    low, high = model_stats["score_rate_paired_wilson_ci95"]
                    values.append((model_stats["score_rate"], low, high))
            series[model] = values
        _line_svg(
            figure_dir / "extreme_d2_vs_c.svg",
            "D2 极限挑战 C（非对等时间预算）",
            labels,
            series,
            y_min=0,
            y_max=1,
            y_ticks=[(x, f"{x:.0%}") for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)],
            y_label="得分率（平局计半分）",
            note="C 固定 50ms；D2 时间预算随横轴增加",
        )


def _render_benchmark_charts(benchmark_path: Path, figure_dir: Path) -> None:
    if not benchmark_path.exists():
        return
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rows = data.get("budget_results", [])
    for board_size in (9, 15):
        relevant = [row for row in rows if int(row["board_size"]) == board_size]
        if not relevant:
            continue
        labels = [_budget_label(budget) for budget in BUDGETS]
        series = {}
        for model in ("D", "D1", "D2"):
            values: list[tuple[float, float, float] | None] = []
            for budget in BUDGETS:
                selected = [
                    int(row["actual_simulations"])
                    for row in relevant
                    if row["model"] == model and float(row["budget_seconds"]) == budget
                ]
                values.append(
                    (
                        statistics.fmean(selected),
                        float(min(selected)),
                        float(max(selected)),
                    )
                    if selected
                    else None
                )
            series[model] = values
        _line_svg(
            figure_dir / f"benchmark_simulations_{board_size}x{board_size}.svg",
            f"{board_size}×{board_size} 代表局面 MCTS 模拟量",
            labels,
            series,
            y_min=1,
            y_max=10_000,
            y_ticks=[(1, "1"), (10, "10"), (100, "100"), (1_000, "1,000"), (10_000, "10,000")],
            y_label="单步实际模拟数（对数刻度）",
            logarithmic=True,
            note="点为代表局面均值；竖线为本组位置实测最小值至最大值",
        )


def summarize(
    results_dir: Path,
    config_path: Path,
    output_path: Path,
    figure_dir: Path,
    benchmark_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    records = load_records(results_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress_path = results_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    summary = summarize_phases(records, bootstrap_samples=bootstrap_samples, seed=seed)
    summary["experiment"] = {
        "experiment_id": config.get("experiment_id"),
        "code_commit": config.get("code_commit"),
        "scheduled_games": int(progress.get("scheduled_games", len(records))),
        "games_completed": len(records),
        "remaining_games": int(progress.get("remaining_games", 0)),
        "complete": bool(progress.get("complete", False)),
        "config_path": str(config_path),
        "result_files": [path.name for path in sorted(results_dir.glob("*_games.jsonl"))],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_rows, pair_rows = _summary_csv_rows(summary)
    _write_csv(
        output_path.parent / "model_summary.csv",
        model_rows,
        list(model_rows[0]) if model_rows else ["phase", "model", "games"],
    )
    _write_csv(
        output_path.parent / "pairwise_summary.csv",
        pair_rows,
        list(pair_rows[0]) if pair_rows else ["phase", "model_1", "model_2", "games"],
    )
    _render_score_charts(summary, figure_dir)
    _render_benchmark_charts(benchmark_path, figure_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "research/v2/results")
    parser.add_argument("--config", type=Path, default=ROOT / "research/v2/results/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/v2/results/summary.json")
    parser.add_argument("--figures", type=Path, default=ROOT / "research/v2/figures")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "research/v2/pilot/performance_benchmarks_final.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_261_018)
    args = parser.parse_args()
    summary = summarize(
        args.results_dir,
        args.config,
        args.output,
        args.figures,
        args.benchmark,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "games_completed": summary["experiment"]["games_completed"],
                "scheduled_games": summary["experiment"]["scheduled_games"],
                "complete": summary["experiment"]["complete"],
                "phases": list(summary["phases"]),
                "output": str(args.output),
                "figures": str(args.figures),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
