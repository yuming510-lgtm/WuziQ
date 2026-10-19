"""Validation, metrics, plots, and Chinese report for ABC compute V4."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from ..ai import WIN_SCORE
from .abc_search_v4 import (
    EXPECTED_BUDGETS_MS,
    EXPECTED_MODELS,
    _canonical_hash,
    derive_seed,
    expected_run_ids,
    make_run_id,
)


def validate_runs(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Check the matrix, result shape, legal output moves, and shared-depth scores."""

    rows = list(records)
    ids = expected_run_ids(config)
    allowed = set(ids)
    positions = {str(row["position_id"]): row for row in config["positions"]}
    expected_config_hash = _canonical_hash(config)
    seen: set[str] = set()
    code_commits: set[str] = set()
    pair_rows: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in rows:
        run_id = str(record.get("run_id", ""))
        if not run_id:
            raise ValueError("Run record is missing run_id")
        if run_id not in allowed:
            raise ValueError(f"Run ID outside the configured matrix: {run_id}")
        if run_id in seen:
            raise ValueError(f"Duplicate run_id: {run_id}")
        seen.add(run_id)
        position_id = str(record.get("position_id", ""))
        model = str(record.get("model", ""))
        budget_ms = int(record.get("budget_ms", -1))
        if position_id not in positions:
            raise ValueError(f"Unknown position in run {run_id}")
        if run_id != make_run_id(position_id, model, budget_ms):
            raise ValueError(f"Run ID does not match record fields: {run_id}")
        if record.get("experiment_id") != config["experiment_id"]:
            raise ValueError(f"Experiment ID mismatch: {run_id}")
        if record.get("config_sha256") != expected_config_hash:
            raise ValueError(f"Config hash mismatch: {run_id}")
        if record.get("category") != positions[position_id]["category"]:
            raise ValueError(f"Category mismatch: {run_id}")
        if record.get("to_move") != positions[position_id]["to_move"]:
            raise ValueError(f"to_move mismatch: {run_id}")
        if int(record.get("seed", -1)) != derive_seed(
            int(config["base_seed"]), position_id, model
        ):
            raise ValueError(f"Seed mismatch: {run_id}")
        code_commit = str(record.get("code_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
            raise ValueError(f"Invalid code commit provenance: {run_id}")
        code_commits.add(code_commit)
        if int(record.get("board_size", -1)) != 9:
            raise ValueError(f"Non-9x9 run record: {run_id}")
        if int(record.get("plies", -1)) != len(positions[position_id]["moves"]):
            raise ValueError(f"Position ply count mismatch: {run_id}")
        if int(record.get("max_depth_limit", -1)) != 81 - int(record["plies"]):
            raise ValueError(f"Incorrect remaining-cell depth limit: {run_id}")
        elapsed = float(record.get("elapsed_seconds", -1))
        nodes_per_second = float(record.get("nodes_per_second", -1))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Invalid elapsed time: {run_id}")
        if not math.isfinite(nodes_per_second) or nodes_per_second < 0:
            raise ValueError(f"Invalid nodes_per_second: {run_id}")
        if (
            elapsed > budget_ms / 1_000.0
            and not bool(record.get("timed_out", False))
        ):
            raise ValueError(f"Over-budget run is not marked timed_out: {run_id}")
        move = record.get("move")
        if (
            not isinstance(move, list)
            or len(move) != 2
            or any(int(value) < 0 or int(value) >= 9 for value in move)
        ):
            raise ValueError(f"Illegal move coordinate in run {run_id}")
        occupied = {
            (int(item["row"]), int(item["col"])) for item in positions[position_id]["moves"]
        }
        if tuple(map(int, move)) in occupied:
            raise ValueError(f"Occupied move coordinate in run {run_id}")
        expected_tactical_moves = positions[position_id].get("expected_best_moves")
        if record.get("tactical_expected_moves") != expected_tactical_moves:
            raise ValueError(f"Tactical expectation mismatch: {run_id}")
        if expected_tactical_moves is not None:
            expected_match = list(map(int, move)) in expected_tactical_moves
            if record.get("tactical_move_matches_expected") is not expected_match:
                raise ValueError(f"Tactical result flag mismatch: {run_id}")

        depth_rows = record.get("depth_iterations", [])
        completed_depth = int(record.get("completed_depth", -1))
        max_depth_limit = int(record.get("max_depth_limit", -1))
        if completed_depth > max_depth_limit:
            raise ValueError(f"Completed depth exceeds remaining-cell limit: {run_id}")
        depths = [int(depth.get("depth", -1)) for depth in depth_rows]
        if completed_depth != len(depth_rows) or depths != list(range(1, completed_depth + 1)):
            raise ValueError(f"Incomplete or nonsequential depth trace in run {run_id}")
        for depth in depth_rows:
            if int(depth.get("nodes", -1)) < 0 or int(depth.get("cutoffs", -1)) < 0:
                raise ValueError(f"Negative depth counter in run {run_id}")
            depth_elapsed = float(depth.get("elapsed_seconds", -1))
            if not math.isfinite(depth_elapsed) or depth_elapsed < 0:
                raise ValueError(f"Invalid per-depth elapsed time in run {run_id}")
            if int(depth.get("depth", 0)) > 0 and not isinstance(depth.get("root_value"), int):
                raise ValueError(f"Missing integer root score in run {run_id}")
        if completed_depth:
            if int(record.get("root_value")) != int(depth_rows[-1]["root_value"]):
                raise ValueError(f"Final root score does not match depth trace: {run_id}")
            if list(record.get("move", [])) != list(depth_rows[-1].get("move", [])):
                raise ValueError(f"Final move does not match depth trace: {run_id}")
        if model in {"B", "C"}:
            complete_nodes = sum(int(row["nodes"]) for row in depth_rows)
            complete_cutoffs = sum(int(row["cutoffs"]) for row in depth_rows)
            partial_cutoffs = int(record.get("partial_iteration_cutoffs", -1))
            if int(record.get("completed_depth_nodes", -1)) != complete_nodes:
                raise ValueError(f"Completed-depth node total mismatch: {run_id}")
            if int(record.get("partial_iteration_nodes", -1)) < 0:
                raise ValueError(f"Invalid partial-iteration node count: {run_id}")
            if int(record.get("nodes", -1)) != complete_nodes + int(
                record.get("partial_iteration_nodes", -1)
            ):
                raise ValueError(f"Total node count does not reconcile: {run_id}")
            if int(record.get("completed_depth_cutoffs", -1)) != complete_cutoffs:
                raise ValueError(f"Completed-depth cutoff total mismatch: {run_id}")
            if partial_cutoffs < 0 or int(record.get("cutoffs", -1)) != complete_cutoffs + partial_cutoffs:
                raise ValueError(f"Cutoff total mismatch: {run_id}")
        elif (
            int(record.get("nodes", -1)) != 0
            or int(record.get("cutoffs", -1)) != 0
            or completed_depth != 0
            or depth_rows
        ):
            raise ValueError(f"Rule-only model has search counters: {run_id}")
        nodes = int(record.get("nodes", -1))
        root_candidates = int(record.get("root_candidates", -1))
        if nodes < 0 or root_candidates < 0:
            raise ValueError(f"Negative search counter in run {run_id}")
        expected_nps = nodes / elapsed if elapsed > 0 else 0.0
        if not math.isclose(nodes_per_second, expected_nps, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"nodes_per_second mismatch: {run_id}")
        expected_tactical_moves = positions[position_id].get("expected_best_moves")
        expected_tactical_match = (
            list(map(int, move)) in expected_tactical_moves
            if expected_tactical_moves is not None
            else None
        )
        if record.get("tactical_move_matches_expected") is not expected_tactical_match:
            raise ValueError(f"Tactical result flag mismatch: {run_id}")
        for depth in depth_rows:
            pair_rows[(position_id, budget_ms, model)][int(depth["depth"])] = depth

    missing = sorted(allowed - seen)
    if require_complete and missing:
        raise ValueError(f"V4 benchmark has {len(missing)} missing run IDs")
    if len(code_commits) > 1:
        raise ValueError("V4 result set contains mixed code commits")

    comparisons = 0
    for position in config["positions"]:
        position_id = str(position["position_id"])
        for budget_ms in EXPECTED_BUDGETS_MS:
            minimax = pair_rows[(position_id, budget_ms, "B")]
            alpha_beta = pair_rows[(position_id, budget_ms, "C")]
            for depth in sorted(set(minimax) & set(alpha_beta)):
                comparisons += 1
                if int(minimax[depth]["root_value"]) != int(alpha_beta[depth]["root_value"]):
                    raise ValueError(
                        "B/C root score mismatch at shared complete depth "
                        f"{depth} for {position_id}/{budget_ms}ms: "
                        f"{minimax[depth]['root_value']} != {alpha_beta[depth]['root_value']}"
                    )
    over_budget = [
        {
            "run_id": str(record["run_id"]),
            "model": str(record["model"]),
            "budget_ms": int(record["budget_ms"]),
            "elapsed_seconds": float(record["elapsed_seconds"]),
            "overrun_seconds": float(record["elapsed_seconds"])
            - int(record["budget_ms"]) / 1_000.0,
            "timed_out": bool(record.get("timed_out", False)),
        }
        for record in rows
        if float(record["elapsed_seconds"]) > int(record["budget_ms"]) / 1_000.0
    ]
    over_budget.sort(key=lambda row: row["overrun_seconds"], reverse=True)
    return {
        "checked_runs": len(rows),
        "expected_runs": len(ids),
        "missing_runs": len(missing),
        "missing_run_ids": missing,
        "shared_depth_comparisons": comparisons,
        "score_mismatch_count": 0,
        "over_budget_count": len(over_budget),
        "max_overrun_seconds": max(
            (row["overrun_seconds"] for row in over_budget), default=0.0
        ),
        "over_budget_runs": over_budget,
        "complete": not missing,
    }


def summarize_runs(records: Iterable[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-budget model metrics and paired B/C node savings by complete depth."""

    rows = list(records)
    by_model_budget: dict[str, dict[str, Any]] = {}
    for model in EXPECTED_MODELS:
        by_model_budget[model] = {}
        for budget in EXPECTED_BUDGETS_MS:
            subset = [
                row
                for row in rows
                if row.get("model") == model and int(row.get("budget_ms", -1)) == budget
            ]
            times = [float(row["elapsed_seconds"]) for row in subset]
            nodes = [int(row.get("nodes", 0)) for row in subset]
            depths = [int(row.get("completed_depth", 0)) for row in subset]
            cutoffs = [int(row.get("cutoffs", 0)) for row in subset]
            nonterminal_depths = [
                int(row.get("completed_depth", 0))
                for row in subset
                if abs(int(row.get("root_value", 0))) < WIN_SCORE
            ]
            total_nodes = sum(nodes)
            by_model_budget[model][str(budget)] = {
                "run_count": len(subset),
                "mean_elapsed_seconds": statistics.fmean(times) if times else None,
                "median_elapsed_seconds": statistics.median(times) if times else None,
                "mean_completed_depth": statistics.fmean(depths) if depths else None,
                "median_completed_depth": statistics.median(depths) if depths else None,
                "mean_nodes": statistics.fmean(nodes) if nodes else None,
                "mean_nodes_per_second": statistics.fmean(
                    float(row.get("nodes_per_second", 0)) for row in subset
                )
                if subset
                else None,
                "mean_cutoffs": statistics.fmean(cutoffs) if cutoffs else None,
                "cutoff_ratio": sum(cutoffs) / total_nodes if total_nodes else 0.0,
                "terminal_score_run_count": sum(
                    abs(int(row.get("root_value", 0))) >= WIN_SCORE for row in subset
                ),
                "nonterminal_depth_run_count": len(nonterminal_depths),
                "mean_nonterminal_completed_depth": (
                    statistics.fmean(nonterminal_depths) if nonterminal_depths else None
                ),
                "timeout_count": sum(bool(row.get("timed_out")) for row in subset),
            }

    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        lookup[(str(row["position_id"]), int(row["budget_ms"]), str(row["model"]))] = row
    alpha_beta_node_savings: dict[str, Any] = {}
    for budget in EXPECTED_BUDGETS_MS:
        comparisons: list[dict[str, float]] = []
        for position in config["positions"]:
            position_id = str(position["position_id"])
            minimax = lookup.get((position_id, budget, "B"))
            alpha_beta = lookup.get((position_id, budget, "C"))
            if minimax is None or alpha_beta is None:
                continue
            b_depths = {
                int(row["depth"]): int(row["nodes"])
                for row in minimax.get("depth_iterations", [])
            }
            c_depths = {
                int(row["depth"]): int(row["nodes"])
                for row in alpha_beta.get("depth_iterations", [])
            }
            for depth in sorted(set(b_depths) & set(c_depths)):
                b_nodes, c_nodes = b_depths[depth], c_depths[depth]
                comparisons.append(
                    {
                        "depth": depth,
                        "b_nodes": b_nodes,
                        "c_nodes": c_nodes,
                        "node_reduction_ratio": (b_nodes - c_nodes) / b_nodes if b_nodes else 0.0,
                    }
                )
        ratios = [row["node_reduction_ratio"] for row in comparisons]
        alpha_beta_node_savings[str(budget)] = {
            "shared_position_depth_pairs": len(comparisons),
            "mean_node_reduction_ratio": statistics.fmean(ratios) if ratios else None,
            "mean_b_nodes": statistics.fmean(row["b_nodes"] for row in comparisons)
            if comparisons
            else None,
            "mean_c_nodes": statistics.fmean(row["c_nodes"] for row in comparisons)
            if comparisons
            else None,
            "details": comparisons,
        }

    high_budget = [row for row in rows if row.get("model") == "C" and int(row["budget_ms"]) >= 10_000]
    tactical_choice_rates: dict[str, dict[str, Any]] = {
        model: {} for model in EXPECTED_MODELS
    }
    for model in EXPECTED_MODELS:
        for budget in EXPECTED_BUDGETS_MS:
            subset = [
                row
                for row in rows
                if row.get("model") == model
                and int(row.get("budget_ms", -1)) == budget
                and row.get("tactical_expected_moves") is not None
            ]
            hits = sum(bool(row.get("tactical_move_matches_expected")) for row in subset)
            tactical_choice_rates[model][str(budget)] = {
                "checked": len(subset),
                "matches": hits,
                "match_rate": hits / len(subset) if subset else None,
            }
    return {
        "experiment_id": config["experiment_id"],
        "board_size": 9,
        "position_count": len(config["positions"]),
        "planned_run_count": len(expected_run_ids(config)),
        "actual_run_count": len(rows),
        "budgets_ms": list(EXPECTED_BUDGETS_MS),
        "by_model_budget": by_model_budget,
        "alpha_beta_node_savings": alpha_beta_node_savings,
        "c_runs_above_depth_four_at_10s_plus": sum(
            int(row.get("completed_depth", 0)) > 4 for row in high_budget
        ),
        "c_runs_at_10s_plus": len(high_budget),
        "tactical_choice_rates": tactical_choice_rates,
    }


def make_plots(
    records: Iterable[dict[str, Any]], config: dict[str, Any], figures_dir: str | Path
) -> list[str]:
    """Write fixed-budget trend plots for depth, work, speed, and pruning benefit."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(family="Noto Sans CJK JP").get_name(),
            "axes.unicode_minus": False,
        }
    )
    rows = list(records)
    summary = summarize_runs(rows, config)
    target = Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    budgets = list(EXPECTED_BUDGETS_MS)
    labels = ["50ms", "200ms", "1s", "5s", "10s", "20s", "50s"]

    def save_series(filename: str, title: str, ylabel: str, value_key: str, *, log_y: bool = False) -> None:
        fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        plotted_values: list[float] = []
        for model in ("B", "C"):
            values = [summary["by_model_budget"][model][str(budget)][value_key] for budget in budgets]
            if log_y:
                values = [value if value is not None and float(value) > 0 else None for value in values]
            plotted_values.extend(float(value) for value in values if value is not None)
            axis.plot(labels, values, marker="o", label=model)
        if log_y and any(value > 0 for value in plotted_values):
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("单步预算")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(title="模型")
        path = target / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    save_series(
        "depth_by_budget.png",
        "B/C 平均完成迭代深度上限",
        "完成的迭代深度上限（ply）",
        "mean_completed_depth",
    )
    save_series("nodes_by_budget.png", "B/C 平均搜索节点数", "节点数", "mean_nodes", log_y=True)
    save_series(
        "nodes_per_second.png", "B/C 平均节点吞吐", "节点/秒", "mean_nodes_per_second", log_y=True
    )

    save_series(
        "alpha_beta_cutoff_ratio.png",
        "Alpha-Beta 剪枝事件占总节点比例",
        "剪枝事件 / 总节点",
        "cutoff_ratio",
    )

    fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    ratios = [summary["alpha_beta_node_savings"][str(budget)]["mean_node_reduction_ratio"] for budget in budgets]
    axis.plot(labels, ratios, marker="o", color="#6e56cf")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Alpha-Beta 在共同完整深度的节点减少比例")
    axis.set_xlabel("单步预算")
    axis.set_ylabel("(B 节点 − C 节点) / B 节点")
    axis.grid(True, alpha=0.25)
    path = target / "alpha_beta_node_savings.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for model in EXPECTED_MODELS:
        values = [summary["by_model_budget"][model][str(budget)]["mean_elapsed_seconds"] for budget in budgets]
        axis.plot(labels, values, marker="o", label=model)
    axis.set_title("实际平均单步耗时")
    axis.set_xlabel("预算标签")
    axis.set_ylabel("秒")
    axis.grid(True, alpha=0.25)
    axis.legend(title="模型")
    path = target / "actual_time_by_budget.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_report(
    config: dict[str, Any],
    records: Iterable[dict[str, Any]],
    report_path: str | Path,
    figures_dir: str | Path,
    *,
    preflight: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a complete dataset, emit plots/summary JSON, and write a Chinese report."""

    rows = list(records)
    validation = validate_runs(rows, config, require_complete=True)
    summary = summarize_runs(rows, config)
    figure_paths = make_plots(rows, config, figures_dir)
    report = [
        "# V4 ABC 搜索算力扩展实验报告",
        "",
        f"- 实验：`{config['experiment_id']}`；9×9 棋盘；固定局面：{len(config['positions'])} 个。",
        f"- 实际记录：{len(rows)}/{len(expected_run_ids(config))} 条，唯一 run ID；共同完整深度评分比较 {validation['shared_depth_comparisons']} 次，差异 0 次。",
        "- 本轮测量搜索效率和固定局面的战术选择，不包含 AI 对战胜率，也不进行模型训练。",
        "- 当前运行环境未暴露本会话的 Codex 基础模型与推理强度字段，无法独立核实“Luna / Max”配置；算法和实验数据均来自本地仓库代码实际运行。",
        "- A 使用原规则选择器，仅测量实际决策时间；没有人为等待。B 为纯 Minimax，C 为相同 Minimax 加 Alpha-Beta。",
        "- B/C 按相同候选集、排序和评价函数迭代加深；每条记录的最大深度等于局面剩余空格数。超时层不会覆盖上一个完整层结果。",
        "",
        "## 七档预算汇总",
        "",
        "| 预算 | A 平均实际耗时 | B 完成迭代上限 / 节点 / 节点每秒 | C 完成迭代上限 / 节点 / 节点每秒 | C 剪枝事件 | C 剪枝比例 | C 相同完整层节点减少 | C 超时数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in EXPECTED_BUDGETS_MS:
        a = summary["by_model_budget"]["A"][str(budget)]
        b = summary["by_model_budget"]["B"][str(budget)]
        c = summary["by_model_budget"]["C"][str(budget)]
        saving = summary["alpha_beta_node_savings"][str(budget)]["mean_node_reduction_ratio"]
        def fmt(value: Any, digits: int = 2) -> str:
            return "—" if value is None else f"{float(value):.{digits}f}"
        report.append(
            f"| {budget}ms | {fmt(a['mean_elapsed_seconds'] * 1000 if a['mean_elapsed_seconds'] is not None else None, 2)}ms | "
            f"{fmt(b['mean_completed_depth'], 2)} / {fmt(b['mean_nodes'], 0)} / {fmt(b['mean_nodes_per_second'], 0)} | "
            f"{fmt(c['mean_completed_depth'], 2)} / {fmt(c['mean_nodes'], 0)} / {fmt(c['mean_nodes_per_second'], 0)} | "
            f"{fmt(c['mean_cutoffs'], 1)} | {fmt(c['cutoff_ratio'] * 100, 2)}% | "
            f"{fmt(saving * 100 if saving is not None else None, 1)}% | {c['timeout_count']} |"
        )
    report.extend(
        [
            "",
            "### B/C 实际耗时与超时",
            "",
            "| 预算 | A 平均耗时 | B 平均耗时 | B 超时 | C 平均耗时 | C 超时 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in EXPECTED_BUDGETS_MS:
        values = {
            model: summary["by_model_budget"][model][str(budget)]
            for model in EXPECTED_MODELS
        }
        report.append(
            f"| {budget}ms | "
            f"{values['A']['mean_elapsed_seconds'] * 1000:.3f}ms | "
            f"{values['B']['mean_elapsed_seconds'] * 1000:.3f}ms | "
            f"{values['B']['timeout_count']}/12 | "
            f"{values['C']['mean_elapsed_seconds'] * 1000:.3f}ms | "
            f"{values['C']['timeout_count']}/12 |"
        )
    report.extend(
        [
            "",
            "表中的深度是已完成迭代的名义深度上限，不是所有递归分支实际走到的最大 ply；节点数与吞吐量按该模型和预算下的固定局面计算。节点数含超时不完整层的已完成工作；报告另用逐层轨迹隔离完整层节点。剪枝比例定义为该预算全部 cutoff 次数除以全部节点数。节点减少比例只比较 B/C 在同一局面、同一预算、同一完整深度的节点数。",
            "",
            "## 搜索深度与算力趋势",
            "",
            f"C 在 10 秒及以上预算中有 {summary['c_runs_above_depth_four_at_10s_plus']}/{summary['c_runs_at_10s_plus']} 个实测 run 完成超过 4 层的迭代深度上限。该数值仅覆盖固定局面，不能外推为对弈胜率。",
            "预算从 50ms 增加到 50s 后，需同时查看完成深度和有效节点；若搜索在更高预算仍常超时，说明候选树增长很快。若深度或节点增长放缓，应结合局面分支数理解，不能仅凭预算标签推断搜索已穷尽。",
            f"B 的平均已完成迭代深度上限从 50ms 的 {summary['by_model_budget']['B']['50']['mean_completed_depth']:.2f} 增加到 50s 的 {summary['by_model_budget']['B']['50000']['mean_completed_depth']:.2f}；C 从 {summary['by_model_budget']['C']['50']['mean_completed_depth']:.2f} 增加到 {summary['by_model_budget']['C']['50000']['mean_completed_depth']:.2f}。C 在 10s、20s、50s 下分别有 "
            + "、".join(
                f"{sum(int(row.get('completed_depth', 0)) > 4 for row in rows if row.get('model') == 'C' and int(row.get('budget_ms', 0)) == budget)}/12"
                for budget in (10_000, 20_000, 50_000)
            )
            + " 个局面超过 4 层。",
            f"C 的 50s 平均已完成迭代深度上限为 {summary['by_model_budget']['C']['50000']['mean_completed_depth']:.2f}，比 20s 的 {summary['by_model_budget']['C']['20000']['mean_completed_depth']:.2f} 高 {summary['by_model_budget']['C']['50000']['mean_completed_depth'] - summary['by_model_budget']['C']['20000']['mean_completed_depth']:.2f}；平均节点数则从 {summary['by_model_budget']['C']['20000']['mean_nodes']:.0f} 增至 {summary['by_model_budget']['C']['50000']['mean_nodes']:.0f}。这是固定局面集合上的收益递减信号，不能视为整体棋力饱和的证明。",
            f"C 的单节点吞吐量约为 {summary['by_model_budget']['C']['50000']['mean_nodes_per_second']:.0f} 节点/秒，低于 B 的 {summary['by_model_budget']['B']['50000']['mean_nodes_per_second']:.0f} 节点/秒；Alpha-Beta 的收益来自更少节点达到相同完整深度，而不是每个节点执行得更快。",
            "",
            "## 图表",
            "",
        ]
    )
    for path in figure_paths:
        report.append(f"- `{Path(path).name}`")
    report.extend(
        [
            "",
            "## 排除终局评分的 C 搜索深度",
            "",
            "`±WIN_SCORE` 表示根搜索已找到终局结果；这类 run 的完成迭代深度上限可能明显高于实际递归路径。下表只汇总非终局评分局面，即最终根评分未达到终局分数的局面。",
            "",
            "| 预算 | 非终局评分 run | 平均已完成迭代深度上限 | 终局评分 run |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget in EXPECTED_BUDGETS_MS:
        c = summary["by_model_budget"]["C"][str(budget)]
        mean_depth = c["mean_nonterminal_completed_depth"]
        mean_text = "—" if mean_depth is None else f"{mean_depth:.2f}"
        report.append(
            f"| {budget}ms | {c['nonterminal_depth_run_count']}/12 | "
            f"{mean_text} | {c['terminal_score_run_count']}/12 |"
        )
    report.extend(
        [
            "",
            "## 典型开局、中局和残局",
            "",
            "以下为预先指定局面在高预算下的实测；列中深度为完整迭代深度上限，耗时为秒，括号内为所选落点。终局节点可能使递归分支在到达该上限前提前返回。",
            "",
            "| 局面 | 模型 | 10s：迭代上限/秒/节点/落点 | 20s：迭代上限/秒/节点/落点 | 50s：迭代上限/秒/节点/落点 |",
            "|---|:---:|---:|---:|---:|",
        ]
    )
    selected_positions = ("opening_08", "midgame_24", "late_44", "late_58", "attack_open_four", "defense_block_four")
    record_index = {
        (str(row["position_id"]), str(row["model"]), int(row["budget_ms"])): row
        for row in rows
    }
    for position_id in selected_positions:
        for model in ("B", "C"):
            cells = []
            for budget in (10_000, 20_000, 50_000):
                row = record_index.get((position_id, model, budget))
                if row is None:
                    cells.append("—")
                else:
                    cells.append(
                        f"{int(row['completed_depth'])}/{float(row['elapsed_seconds']):.2f}/{int(row['nodes'])}/({row['move'][0]},{row['move'][1]})"
                    )
            report.append(f"| {position_id} | {model} | " + " | ".join(cells) + " |")
    report.extend(
        [
            "",
            "`late_44` 的 C 结果为迭代上限 37，`late_58` 为 23；两者在第 1 层根评分已达到 `WIN_SCORE`，其后每层节点数停在 72 和 45。原始记录不包含递归调用路径长度，因此不能据此说搜索实际展开了 37/23 个额外 ply；它记录的是这些迭代上限均已完成。",
        ]
    )
    tactical_rates = summary["tactical_choice_rates"]
    report.extend(
        [
            "",
            "## 直接攻防着法检查",
            "",
            "以下只检查两个预先定义的战术局面：黑方已有四连时立即获胜，及白方被边线封住的四连时黑方唯一落点拦截。它们是固定局面的规则检查，不是整体棋力估计。",
            "",
            "| 预算 | A 命中 | B 命中 | C 命中 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget in EXPECTED_BUDGETS_MS:
        cells = []
        for model in EXPECTED_MODELS:
            metric = tactical_rates[model][str(budget)]
            cells.append(f"{metric['matches']}/{metric['checked']}")
        report.append(f"| {budget}ms | " + " | ".join(cells) + " |")
    report.extend(["", "## 固定局面与复现", "", "局面着法序列、预算、候选半径和随机种子规则均保存在 `configs/abc_search_compute_v4.json`。原始每次测量保存在 `results/search_runs.jsonl`；运行器逐条写入并 `fsync`，按稳定 run ID 跳过已完成记录。"])
    if preflight:
        report.extend(["", "## 运行前估算", "", f"预检估算总耗时：{float(preflight.get('estimated_seconds', 0)):.1f} 秒（{float(preflight.get('estimated_minutes', 0)):.2f} 分钟）；完整实验实际耗时见 `preflight.json` 与进度记录。"])
    if execution:
        report.extend(
            [
                "",
                "## 实际运行时间与截止偏差",
                "",
                f"修正版本正式矩阵实测 {float(execution.get('actual_matrix_wall_seconds', 0)):.3f} 秒；修正版本预检实测 {float(execution.get('preflight_wall_seconds', 0)):.3f} 秒；两者合计 {float(execution.get('valid_total_wall_seconds', 0)):.3f} 秒（{float(execution.get('valid_total_wall_minutes', 0)):.2f} 分钟）。",
                f"首轮因截止边界问题而废弃的预检和矩阵共耗时 {float(execution.get('superseded_total_wall_seconds', 0)):.3f} 秒；这些数据未计入本报告统计。含复核重跑在内，所有预检与矩阵实测合计 {float(execution.get('all_measurement_wall_seconds', 0)):.3f} 秒（{float(execution.get('all_measurement_wall_minutes', 0)):.2f} 分钟）。",
                f"最终记录中有 {validation['over_budget_count']} 条实测 wall-clock 略超名义预算，最大超出 {validation['max_overrun_seconds'] * 1000:.3f} 毫秒；均保留原始耗时并计入超时统计。迭代只有在截止前完整结束才更新根结果，操作系统调度和返回路径可能造成少量 wall-clock 偏差。",
                f"最终有效矩阵相对预检的墙钟误差为 {float(execution.get('estimate_error_seconds', 0)):.2f} 秒（{float(execution.get('estimate_error_percent', 0)):.2f}%）。",
            ]
        )
    if resources:
        report.extend(["", "## 实验资源", "", f"- CPU：{resources.get('logical_cpu_count')} 个逻辑核，cgroup 配额 {resources.get('cgroup_cpu_quota_cores')} 核。", f"- 可用内存：{resources.get('memory_available_bytes')} bytes；cgroup 上限：{resources.get('cgroup_memory_limit_bytes')} bytes。", f"- 工作区可用磁盘：{resources.get('workspace_disk_free_bytes')} bytes。"])
    report.extend(["", "## 结论边界", "", "B/C 根评分一致性只验证相同完整搜索深度下的实现正确性；Alpha-Beta 的主要收益是减少达到同一深度的搜索工作量。更深搜索会改变启发式有限深度选择，但本实验没有随机抽样对手赛或统计棋力设计，不把这些搜索指标写成胜率或稳定强弱结论。"])

    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary_path = output.parent / "summary.json"
    summary_path.write_text(
        json.dumps({"validation": validation, **summary, "execution": execution}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"validation": validation, "summary": summary, "figures": figure_paths, "report": str(output)}
