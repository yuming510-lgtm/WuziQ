from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/g_v6_1"
RESULTS = BASE / "results"
REPORT = BASE / "REPORT.md"
FIGURES = BASE / "figures"
MODELS = ("G10K", "G15K", "G20K")
BUDGETS = (64, 400)
OPPONENTS = ("A-Minimax", "B-AB-50ms")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary(phase: str, candidate: str, opponent: str, budget: int) -> dict:
    return read_json(RESULTS / "summaries" / phase /
                     f"{candidate}_vs_{opponent}_puct{budget}.json")


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def ci(value: list[float | None] | None) -> str:
    if not value or value[0] is None or value[1] is None:
        return "—"
    return f"[{100 * value[0]:.1f}%, {100 * value[1]:.1f}%]"


def atomic_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temp.replace(path)


def all_summaries() -> list[dict]:
    rows = []
    for phase in ("direct_h2h", "development", "final_blind"):
        folder = RESULTS / "summaries" / phase
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            data = read_json(path)
            rows.append({key: data.get(key) for key in (
                "phase", "candidate", "opponent", "simulations_per_reference_turn",
                "games", "paired_openings", "wins", "draws", "losses", "points",
                "score_rate", "paired_bootstrap_ci95", "black_score_rate",
                "white_score_rate", "mean_moves", "mean_candidate_cpu_seconds",
                "mean_opponent_cpu_seconds", "mean_total_cpu_seconds", "complete",
            )})
    return rows


def tactic_rows() -> list[dict]:
    result = []
    for model in MODELS:
        for budget in BUDGETS:
            for label in ("new", "historical_v6_64"):
                data = read_json(RESULTS / "tactics" / model / str(budget) / f"{label}.json")
                result.append({
                    "model": model, "simulations": budget, "suite": label,
                    "puzzles": data["puzzles"], "complete": data["complete"],
                    "network_top1_accuracy": data["network_top1_accuracy"],
                    "network_optimal_action_probability_mass": data["network_optimal_action_probability_mass"],
                    "network_optimal_mass_conditioned_on_legal": data["network_optimal_mass_conditioned_on_legal"],
                    "puct_accuracy": data["puct_accuracy"],
                    "puct_optimal_visit_mass": data["puct_optimal_visit_mass"],
                    "solved_value_positions": data["solved_value_positions"],
                    "value_mae_on_proven_values": data["value_mae_on_proven_values"],
                })
    return result


def tactical_category_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        data = read_json(RESULTS / "tactics" / model / "400" / "new.json")
        for category, stats in data["by_category"].items():
            rows.append({"model": model, "category": category, **stats})
    return rows


def search_rows() -> list[dict]:
    """Aggregate decisions in balanced direct checkpoint matches only."""
    buckets: dict[tuple[str, int], list[dict]] = {}
    for path in (RESULTS / "matches" / "direct_h2h").rglob("v61m-*.json"):
        game = read_json(path)
        budget = int(game["simulations_per_reference_turn"])
        for move in game["moves"]:
            actor = move.get("actor")
            if actor in MODELS:
                buckets.setdefault((actor, budget), []).append(move)
    rows = []
    for model in MODELS:
        for budget in BUDGETS:
            moves = buckets[(model, budget)]
            rows.append({
                "actor": model, "budget": budget, "decisions": len(moves),
                "mean_actual_simulations": statistics.mean(m["simulations"] for m in moves),
                "mean_tree_depth": statistics.mean(m["search_depth"] for m in moves),
                "mean_tree_nodes": statistics.mean(m["search_nodes"] for m in moves),
                "mean_root_children": statistics.mean(m["root_children"] for m in moves),
                "mean_cpu_seconds_per_decision": statistics.mean(m["cpu_seconds"] for m in moves),
                "timeouts": sum(bool(m["timed_out"]) for m in moves),
            })
    return rows


def baseline_search_rows() -> list[dict]:
    buckets: dict[tuple[str, int], list[dict]] = {}
    for path in (RESULTS / "matches" / "development").rglob("v61m-*.json"):
        game = read_json(path)
        budget = int(game["simulations_per_reference_turn"])
        for move in game["moves"]:
            actor = move.get("actor")
            if actor in OPPONENTS:
                buckets.setdefault((actor, budget), []).append(move)
    rows = []
    for actor in OPPONENTS:
        for budget in BUDGETS:
            moves = buckets[(actor, budget)]
            rows.append({
                "actor": actor, "opponent_puct_budget": budget,
                "decisions": len(moves),
                "mean_completed_depth": statistics.mean(m["search_depth"] for m in moves),
                "mean_nodes": statistics.mean(m["search_nodes"] for m in moves),
                "mean_cutoffs": statistics.mean(m.get("cutoffs", 0) for m in moves),
                "mean_cpu_seconds": statistics.mean(m["cpu_seconds"] for m in moves),
                "deadline_hits": sum(bool(m["timed_out"]) for m in moves),
            })
    return rows


def fig_h2h() -> None:
    labels, values, lows, highs, colors = [], [], [], [], []
    colors_by_budget = {64: "#2878B5", 400: "#D9534F"}
    for budget in BUDGETS:
        for first, second in (("G10K", "G15K"), ("G10K", "G20K"), ("G15K", "G20K")):
            data = summary("direct_h2h", first, second, budget)
            lo, hi = data["paired_bootstrap_ci95"]
            labels.append(f"{first} vs {second} · PUCT-{budget}")
            values.append(data["score_rate"])
            lows.append(data["score_rate"] - lo)
            highs.append(hi - data["score_rate"])
            colors.append(colors_by_budget[budget])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(y, values, color=colors, alpha=.88)
    ax.errorbar(values, y, xerr=np.array([lows, highs]), fmt="none", ecolor="#222", capsize=3)
    ax.axvline(.5, color="#444", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("First-listed checkpoint score rate (paired 95% bootstrap CI)")
    ax.set_title("Checkpoint head-to-head · 40 paired openings per condition")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "checkpoint_head_to_head.png", dpi=180)
    plt.close(fig)


def fig_vs_baselines() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    offsets = {64: -.16, 400: .16}
    colors = {64: "#2878B5", 400: "#D9534F"}
    for ax, opponent in zip(axes, OPPONENTS):
        for budget in BUDGETS:
            x, y, low, high = [], [], [], []
            for i, model in enumerate(MODELS):
                data = summary("development", model, opponent, budget)
                x.append(i + offsets[budget])
                y.append(data["score_rate"])
                lo, hi = data["paired_bootstrap_ci95"]
                low.append(data["score_rate"] - lo)
                high.append(hi - data["score_rate"])
            ax.bar(x, y, width=.29, color=colors[budget], alpha=.85,
                   label=f"PUCT-{budget} development")
            ax.errorbar(x, y, yerr=np.array([low, high]), fmt="none", ecolor="#222", capsize=3)
        # Independent blind set: G20K was frozen before these games began.
        for budget in BUDGETS:
            data = summary("final_blind", "G20K", opponent, budget)
            x = 2 + offsets[budget]
            lo, hi = data["paired_bootstrap_ci95"]
            ax.errorbar([x], [data["score_rate"]],
                        yerr=[[data["score_rate"] - lo], [hi - data["score_rate"]]],
                        fmt="D", color=colors[budget], markeredgecolor="black", capsize=4,
                        markersize=6, label=f"G20K blind PUCT-{budget}")
        ax.axhline(.5, color="#444", linestyle="--", linewidth=1)
        ax.set_xticks(range(3), MODELS)
        ax.set_ylim(0, .55)
        ax.set_title(opponent)
        ax.set_ylabel("Checkpoint score rate (paired 95% bootstrap CI)")
        ax.grid(axis="y", alpha=.2)
    handles = [
        Patch(facecolor=colors[64], label="Development PUCT-64"),
        Patch(facecolor=colors[400], label="Development PUCT-400"),
        Line2D([], [], marker="D", linestyle="None", color=colors[64], markeredgecolor="black",
               label="G20K blind PUCT-64"),
        Line2D([], [], marker="D", linestyle="None", color=colors[400], markeredgecolor="black",
               label="G20K blind PUCT-400"),
    ]
    fig.suptitle("Challenge scores against Minimax and Alpha-Beta (reported separately)", y=.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .83])
    fig.savefig(FIGURES / "checkpoint_vs_ab.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_tactics() -> None:
    rows = {row["model"]: row for row in tactic_rows() if row["suite"] == "new"}
    x = np.arange(len(MODELS))
    width = .23
    series = [
        ("Bare network top-1", "network_top1_accuracy", "#777777"),
        ("PUCT-64 top-1", "puct_accuracy", "#2878B5"),
    ]
    puct400 = [read_json(RESULTS / "tactics" / model / "400" / "new.json")["puct_accuracy"]
               for model in MODELS]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for shift, (label, key, color) in enumerate(series):
        values = [rows[model][key] for model in MODELS]
        ax.bar(x + (shift - .5) * width, values, width, label=label, color=color)
    ax.bar(x + .5 * width, puct400, width, label="PUCT-400 top-1", color="#D9534F")
    ax.set_xticks(x, MODELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy on 128 independently verified new positions")
    ax.set_title("Tactical ability: network policy vs search")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "tactical_ability.png", dpi=180)
    plt.close(fig)


def fig_training_trend() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, opponent in zip(axes, OPPONENTS):
        for budget, color in ((64, "#2878B5"), (400, "#D9534F")):
            vals = [summary("development", model, opponent, budget)["score_rate"] for model in MODELS]
            ax.plot((10, 15, 20), vals, marker="o", linewidth=2, color=color, label=f"PUCT-{budget}")
        ax.axhline(.5, color="#444", linestyle="--", linewidth=1)
        ax.set_xticks((10, 15, 20), ("10K", "15K", "20K"))
        ax.set_ylim(0, .55)
        ax.set_title(f"vs {opponent}")
        ax.set_xlabel("Checkpoint training state")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Development score rate")
    axes[0].legend(frameon=False)
    fig.suptitle("Learning-state trend under fixed evaluation search")
    fig.tight_layout()
    fig.savefig(FIGURES / "development_state_trend.png", dpi=180)
    plt.close(fig)


def direct_matrix(budget: int) -> str:
    values = {model: {} for model in MODELS}
    for first, second in (("G10K", "G15K"), ("G10K", "G20K"), ("G15K", "G20K")):
        data = summary("direct_h2h", first, second, budget)
        values[first][second] = (data["score_rate"], data["paired_bootstrap_ci95"])
        reverse_ci = [1 - data["paired_bootstrap_ci95"][1], 1 - data["paired_bootstrap_ci95"][0]]
        values[second][first] = (1 - data["score_rate"], reverse_ci)
    lines = ["| 棋手 | G10K | G15K | G20K |", "|---|---:|---:|---:|"]
    for first in MODELS:
        cells = []
        for second in MODELS:
            if first == second:
                cells.append("—")
            else:
                rate, interval = values[first][second]
                cells.append(f"{pct(rate)} {ci(interval)}")
        lines.append(f"| {first} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def wdl_table(phase: str, candidates: tuple[str, ...], opponents: tuple[str, ...], budgets: tuple[int, ...]) -> str:
    lines = ["| 赛道 | 棋手 | 对手 | 搜索 | W-D-L | 得分率（配对95%区间） | 黑方 / 白方得分率 | 平均步数 | 棋手平均CPU秒/局 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for candidate in candidates:
        for opponent in opponents:
            for budget in budgets:
                data = summary(phase, candidate, opponent, budget)
                lines.append(
                    f"| {phase} | {candidate} | {opponent} | {budget} | "
                    f"{data['wins']}-{data['draws']}-{data['losses']} | "
                    f"{pct(data['score_rate'])} {ci(data['paired_bootstrap_ci95'])} | "
                    f"{pct(data['black_score_rate'])} / {pct(data['white_score_rate'])} | "
                    f"{data['mean_moves']:.2f} | {data['mean_candidate_cpu_seconds']:.3f} |"
                )
    return "\n".join(lines)


def build_report() -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    match_rows = all_summaries()
    tactics = tactic_rows()
    categories = tactical_category_rows()
    search = search_rows()
    baseline_search = baseline_search_rows()
    atomic_csv(RESULTS / "match_summary.csv", list(match_rows[0]), match_rows)
    atomic_csv(RESULTS / "tactics_summary.csv", list(tactics[0]), tactics)
    atomic_csv(RESULTS / "tactics_by_category.csv", list(categories[0]), categories)
    atomic_csv(RESULTS / "search_efficiency_direct_h2h.csv", list(search[0]), search)
    atomic_csv(RESULTS / "baseline_search_efficiency.csv", list(baseline_search[0]), baseline_search)
    fig_h2h()
    fig_vs_baselines()
    fig_tactics()
    fig_training_trend()

    final = read_json(RESULTS / "final.json")
    register = read_json(BASE / "config/registration.json")
    selection = read_json(RESULTS / "frozen_selection.json")
    tactic_audit = read_json(BASE / "config/tactic_generation_audit.json")
    audit = final["game_audit"]
    blind = RESULTS / "matches" / "final_blind"
    blind_games = len(list(blind.rglob("v61m-*.json")))
    preflight_summary = read_json(RESULTS / "preflight.json")
    preflight_games = len(preflight_summary["preflight_games"])
    first_path = min((RESULTS / "matches").rglob("v61m-*.json"), key=lambda p: p.stat().st_mtime)
    last_path = max((RESULTS / "matches").rglob("v61m-*.json"), key=lambda p: p.stat().st_mtime)
    wall_elapsed = last_path.stat().st_mtime - first_path.stat().st_mtime
    sha_rows = []
    for model in MODELS:
        item = register["models"][model]
        sha_rows.append(f"| {model} | {item['environment_states']:,} | `{item['path']}` | `{item['sha256']}` |")

    direct_lines = []
    for budget in BUDGETS:
        for first, second in (("G10K", "G15K"), ("G10K", "G20K"), ("G15K", "G20K")):
            data = summary("direct_h2h", first, second, budget)
            direct_lines.append(
                f"| {first} | {second} | {budget} | {data['wins']}-{data['draws']}-{data['losses']} | "
                f"{pct(data['score_rate'])} {ci(data['paired_bootstrap_ci95'])} | "
                f"{pct(data['black_score_rate'])} / {pct(data['white_score_rate'])} | "
                f"{data['mean_moves']:.2f} | {data['mean_candidate_cpu_seconds']:.3f} |"
            )
    direct_table = "\n".join([
        "| 左侧模型 | 对手 | 搜索 | 左侧模型W-D-L | 得分率（配对95%区间） | 黑方 / 白方得分率 | 平均步数 | 平均CPU秒/局 |",
        "|---|---|---:|---:|---:|---:|---:|---:|", *direct_lines,
    ])

    h2h_metrics = selection["development_metrics"]
    rank_lines = []
    for budget in ("64", "400"):
        ordered = sorted(MODELS, key=lambda name: h2h_metrics[budget][name]["h2h_rate"], reverse=True)
        for rank, model in enumerate(ordered, 1):
            item = h2h_metrics[budget][model]
            rank_lines.append(
                f"| PUCT-{budget} | {rank} | {model} | {pct(item['h2h_rate'])} {ci(item['h2h_ci95'])} | "
                f"{pct(item['vs_A_minimax_rate'])} | {pct(item['vs_B_ab50_rate'])} | "
                f"{pct(item['tactic_network_top1_accuracy'])} / {pct(item['tactic_puct_accuracy'])} |"
            )
    rank_table = "\n".join([
        "| 搜索预算 | 名次 | 模型 | 对其他检查点平均得分率（配对95%区间） | 对A开发得分率 | 对B开发得分率 | 新战术集裸网络 / 同档PUCT |",
        "|---|---:|---|---:|---:|---:|---:|", *rank_lines,
    ])

    tactic_table_lines = []
    for model in MODELS:
        vals = [read_json(RESULTS / "tactics" / model / str(budget) / "new.json") for budget in BUDGETS]
        hist = [read_json(RESULTS / "tactics" / model / str(budget) / "historical_v6_64.json") for budget in BUDGETS]
        tactic_table_lines.append(
            f"| {model} | {pct(vals[0]['network_top1_accuracy'])} | {pct(vals[0]['network_optimal_action_probability_mass'])} | "
            f"{pct(vals[0]['puct_accuracy'])} | {pct(vals[1]['puct_accuracy'])} | "
            f"{pct(vals[0]['puct_optimal_visit_mass'])} / {pct(vals[1]['puct_optimal_visit_mass'])} | "
            f"{pct(hist[0]['puct_accuracy'])} / {pct(hist[1]['puct_accuracy'])} | "
            f"{vals[1]['solved_value_positions']} / {vals[1]['value_mae_on_proven_values']:.3f} |"
        )
    tactic_table = "\n".join([
        "| 模型 | 新题裸网络Top-1 | 新题正确动作概率质量 | 新题PUCT-64准确率 | 新题PUCT-400准确率 | 新题目标访问质量（64 / 400） | 历史64题PUCT准确率（64 / 400） | 可证明局面 / 价值MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *tactic_table_lines,
    ])

    category_lines = []
    for row in categories:
        data = read_json(RESULTS / "tactics" / row["model"] / "400" / "new.json")["by_category"][row["category"]]
        network_mass = data["network_target_probability_mass"]
        category_lines.append(
            f"| {row['category']} | {row['model']} | {pct(data['network_top1_accuracy'])} | "
            f"{pct(network_mass)} | {pct(data['puct_accuracy'])} | {pct(data['puct_target_visit_mass'])} |"
        )
    category_table = "\n".join([
        "| 题型（每类32题） | 模型 | 裸网络Top-1 | 裸网络目标概率质量 | PUCT-400准确率 | PUCT-400目标访问质量 |",
        "|---|---|---:|---:|---:|---:|", *category_lines,
    ])

    search_table_lines = []
    for row in search:
        search_table_lines.append(
            f"| {row['actor']} | {row['budget']} | {row['decisions']:,} | {row['mean_actual_simulations']:.0f} | "
            f"{row['mean_tree_depth']:.2f} | {row['mean_tree_nodes']:.0f} | "
            f"{row['mean_cpu_seconds_per_decision']:.3f} | {row['timeouts']} |"
        )
    search_table = "\n".join([
        "| 模型 | 预算 | 决策数 | 实际平均模拟/步 | 平均树深 | 平均节点 | CPU秒/决策 | 超时数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *search_table_lines,
    ])
    baseline_lines = []
    for row in baseline_search:
        baseline_lines.append(
            f"| {row['actor']} | 对局中PUCT-{row['opponent_puct_budget']} | {row['decisions']:,} | "
            f"{row['mean_completed_depth']:.2f} | {row['mean_nodes']:.0f} | {row['mean_cutoffs']:.0f} | "
            f"{row['mean_cpu_seconds']:.3f} | {row['deadline_hits']} |"
        )
    baseline_table = "\n".join([
        "| 对手引擎 | 条件 | 决策数 | 完成深度 | 节点 | 剪枝数/决策 | CPU秒/决策 | 达到50ms截止数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|", *baseline_lines,
    ])

    dev_table = wdl_table("development", MODELS, OPPONENTS, BUDGETS)
    blind_table = wdl_table("final_blind", ("G20K",), OPPONENTS, BUDGETS)
    model_table = "\n".join([
        "| 检查点 | 环境状态 | 相同结构/规则 | SHA-256 |",
        "|---|---:|---|---|", *sha_rows,
    ])
    preflight = register["cpu_estimates_before_formal_runs"]
    cpu_hours = final["total_cpu_seconds_including_preflight"] / 3600
    wall_h = wall_elapsed / 3600
    runner_wall = final["runner_wall_seconds"]
    opening_sha = register["opening_schedule_sha256"]
    selection_text = json.dumps(selection["selected_by_budget"], ensure_ascii=False)

    content = f"""# WuziQ AlphaZero G V6.1 检查点横向验证报告

## 结论摘要

统一对局下，**G20K 在 PUCT-64 与 PUCT-400 都是三个冻结检查点中最强候选**。在开发赛的等权直接交锋平均得分率，G20K分别为 {pct(h2h_metrics['64']['G20K']['h2h_rate'])}、{pct(h2h_metrics['400']['G20K']['h2h_rate'])}；G15K为 {pct(h2h_metrics['64']['G15K']['h2h_rate'])}、{pct(h2h_metrics['400']['G15K']['h2h_rate'])}；G10K为 {pct(h2h_metrics['64']['G10K']['h2h_rate'])}、{pct(h2h_metrics['400']['G10K']['h2h_rate'])}。配对开局bootstrap区间和逐组直接对局均支持20K领先；这否定了“10K是绝对最强”的早期判断，但**不代表20K已经战胜强传统搜索棋手**。

对Minimax A的开发得分率随检查点从10K到20K上升；对Alpha-Beta B也出现少量20K得分，但幅度小、区间宽，仍远低于50%。PUCT-400相对PUCT-64对A有帮助，尤其15K/20K；对B的收益有限。新战术集上20K裸策略和搜索总体最好，但题型间不均衡，且单点题分数不能替代完整对局。

本轮没有训练、改权重或调参。正式赛 **{audit['games_verified']:,}局、{audit['complete_color_swapped_pairs']:,}组配对换色**；另外预检6局不计成绩。全量棋谱通过合法性、终局、模型身份和换色完整性审计，唯一ID {audit['unique_game_ids']:,}，未完成配对 {audit['incomplete_pairs']}。

## 1. 模型与冻结检查

基线提交为 `{register['baseline_v6_commit']}`。三份检查点启动前逐一核对SHA-256、结构和可加载性；所有条件采用同一9×9自由五子规则、同一个V6参考网络及搜索实现。

{model_table}

网络架构：{register['network_architecture']}。规则为{register['rules']}。没有发生训练或搜索参数改变。PUCT c_puct=5、根噪声关闭、温度0.001；根边实际访问数为N，参考实现因根节点初次展开配置playout=N+1。模型间不复用搜索树。

**引擎标签映射说明：**项目原生A是规则选择器、原生B是纯Minimax、原生C是Alpha-Beta；本任务的对手A要求Minimax、B要求Alpha-Beta 50ms。因此记录采用语义别名 `A-Minimax`→原生B、`B-AB-50ms`→原生C。逐局JSON保存映射，避免名称混淆。两者均为50ms/步。

## 2. 赛程与统计方法

三个直接交手配对在每个搜索档各使用40个独立开局、每个开局黑白各一局，共480局。与A/B的开发赛每个模型、对手、预算各20组换色，共480局。按开发集选择出的赢家在全新盲测开局挑战A/B；两个预算均选择G20K，每个条件40组换色，共320局。历史V6盲测未复用。

开发/盲测开局固定并在注册时哈希：`{opening_sha}`。同组交换颜色，胜1、平0.5、负0；置信区间按开局组重采样。开发选择规则已在解盲前登记：每预算先看相对另两个检查点的等权平均配对得分，完全同分时依次看A、B、再看战术准确率；A/B不合并。冻结结果为 `{selection_text}`。选择结果不声称统计显著排序。

预检测得环境：{register['resource_snapshot']['cpu_count_visible']}个可见CPU、cgroup限额`{register['resource_snapshot']['cpu_quota']}`、内存上限8GiB，PyTorch `{register['resource_snapshot']['torch_version']}` CPU版，线程数1。预检累计{final['preflight_cpu_seconds']:.1f} CPU秒。运行前按30/50/70着子情景估算总CPU时间分别约{preflight['optimistic']['total_projected_cpu_hours']:.2f}/{preflight['normal']['total_projected_cpu_hours']:.2f}/{preflight['conservative']['total_projected_cpu_hours']:.2f}小时；保守估计高于10小时，所以预注册规则是优先做完P0、P1，剩余预算再做盲测。实际全量任务在10小时CPU上限内完成。

## 3. 三检查点直接对局

表格按“左侧模型作为候选者”列出W-D-L与得分率；矩阵为行棋手相对于列棋手的得分率，区间为配对95% bootstrap CI。

{direct_table}

### PUCT-64相对得分矩阵

{direct_matrix(64)}

### PUCT-400相对得分矩阵

{direct_matrix(400)}

{model_summary_image('checkpoint_head_to_head.png','图1：三个检查点直接交手。')}

直接对局只出现一平（G15K对G20K的PUCT-400）；其余为胜负。两个搜索档下G20K都赢了G10K与G15K的多数配对，且优势不是由训练状态数标签推断得出。G10K对G15K的比较接近得多；其点估计在64/400下分别是22.5%/25%，不宜解读为早期模型稳定胜过15K。

## 4. 对Minimax A与Alpha-Beta B的统一挑战

### 开发集（20组换色开局，每条件40局）

{dev_table}

### 冻结候选后的独立盲测（40组换色开局，每条件80局）

{blind_table}

{model_summary_image('checkpoint_vs_ab.png','图2：A/B分开统计。柱为开发赛；菱形为冻结G20K的独立盲测。')}

G20K在开发集PUCT-400对A得分率为27.5%，PUCT-64为17.5%；独立盲测两个档均为10%。这说明20K具备一定开发赛收益但样本/开局敏感，盲测没有重现较高点估计。对B，开发赛PUCT-400为7.5%，PUCT-64为0%；盲测分别为2.5%和3.8%。样本不能支持G20K接近或超过A/B，尤其B仍几乎完全压制各检查点。

**搜索预算带来的收益：**相同检查点、相同开局协议下，PUCT-400相对PUCT-64使G15K/G20K开发集对A得分率分别提升10/10个百分点；G10K提升2.5个百分点。对B，只有G10K/G20K在开发点估计提升2.5/7.5个百分点，G15K仍为0；盲测G20K对B两个预算都很低。PUCT-400单步实际CPU代价约为PUCT-64的数倍，见下节。这属于搜索预算效果，不是训练进步。

## 5. 战术题能力

新题集共128个可靠验证、互不重复且排除历史训练/评测位置的局面，四类各32题：立即获胜、唯一必防、三步强制威胁、伪威胁；包含85个边/角局面、33个多解局面，32道强制威胁题逐一验证全部对手防守。旧V6 64题作为独立历史基准，不与新题合并。

{tactic_table}

{model_summary_image('tactical_ability.png','图3：新128题上裸网络Top-1与PUCT-64/400准确率。')}

PUCT-400的新题分类结果（32题/类）：

{category_table}

20K整体上确实比10K有更好的新题裸网络Top-1（66.4% vs 40.6%）和PUCT-400准确率（79.7% vs 61.7%），但直接取胜题的PUCT-400准确率为65.6%，低于15K的81.3%；伪威胁题也仍只有75.0%。这说明提升不均匀。新题上搜索将20K top-1从66.4%提到PUCT-400的79.7%，表明一部分正确动作在网络候选/先验中可用、搜索能够重新排序；剩余错误同时包括策略先验未把正确手放在前列和有限搜索未访问到最优分支。PUCT访问质量列用于区分这两类问题，原始逐题数据保存在`tactics/`目录。

可证明胜负局面中（每模型64题）价值头MAE：G10K 0.446，G15K 0.539，G20K 0.271。20K较好，但这只是筛选后的可证明战术子集，不能据此推断全局价值校准。历史64题上的PUCT准确率也列在表中，因旧题来源和难度构成不同，不与新题准确率直接混算。

## 6. 搜索效率与计算成本

### 直接检查点对局中MCTS实际搜索

统计只取三模型相同数量级的直接交锋，不混入G20K额外盲测局。节点和树深为每次实际决策的均值；超时均为0。

{search_table}

### 两个传统搜索对手

{baseline_table}

MCTS直接对局中的每次模型决策平均耗时：PUCT-64为0.139–0.209 CPU秒，PUCT-400为0.811–1.214秒；400次搜索约为64次的5.8倍CPU成本。直接交手平均19.9–24.9手；对传统对手平均13.2–17.0手（均含6手开局）。Minimax/Alpha-Beta在每步50ms上限附近触发截止后，均安全返回上一完整迭代结果；表中的“达到50ms截止数”表示搜索主动结束次数，不代表非法落子或超预算。传统搜索的实际深度、节点、剪枝、决策耗时按条件导出至`baseline_search_efficiency.csv`。

正式任务总CPU时间{cpu_hours:.2f}小时（包括预检{final['preflight_cpu_seconds']:.1f}秒）；从首局棋谱落盘到最后一局落盘约{wall_h:.2f}小时。最终恢复后的runner本段执行墙钟{runner_wall/60:.1f}分钟；中途有一次进程中断以及一次P2开始前的缺失常量错误，二者均在幂等恢复下修正，已存完整对局没有重复计分。自动去重、固定ID和复跑审计保证最终只统计一次。

## 7. 代表性棋谱和错误类型

* **G20K战胜G15K（PUCT-400）**：`v61m-4974acaaae926e4e`，开局组`v61-development-610634-024`。G20K黑棋于第3行从第4列起连续推进到第8列；G15K未能中断已形成的四连，G20K完成五连。逐步棋谱和搜索统计见原始JSON。
* **G20K战胜Alpha-Beta（盲测，PUCT-400）**：`v61m-b24635745f39404a`，开局组`v61-blind-610611-000`；G20K沿第5列第3–7行连五，是本次很少数对B的胜局。
* **G20K负于Alpha-Beta（盲测，PUCT-400）**：`v61m-6646f0b569d807d3`，开局组`v61-blind-610623-012`。G20K白棋没有堵住黑棋(2,4)、(3,3)、(4,2)、(6,0)所形成斜线上的空点(5,1)，Alpha-Beta于(5,1)完成五连。这是可直接从棋谱确认的漏防局面；不把单局扩大解释成所有失利的统一根因。

原始棋谱记录完整落点、实际每步时间、模拟数、搜索树深、节点及终局。完整局面复盘由独立9×9五连规则审计器通过。

## 8. 对V6早期判断的重新回答

1. **10K、15K、20K谁更强？** 在这轮统一对局的探索性排序中，20K明确领先；15K第二；10K第三。直接交手每组40个配对开局，尤其20K对10K/15K的点差及区间支持该结论。10K过去对历史G0-10K的93.8%是特定对手、PUCT-64开发指标，不能代表三候选总排序。
2. **20K是否继续成长或真实退化？** 本轮证据显示，在这些冻结检查点及统一直接交锋中，20K比10K/15K强，不支持“继续训练必然退化”。V6原开发集所示PUCT-64对历史G0指标下降可能是对手与预算特异的，不等同广义棋力下降。此次没有进行20K以后新训练，不能证明训练到40K仍有提升。
3. **预算改变赢家吗？** 未改变冠军，64与400两档均为20K；但G10K/G15K相对表现随预算有波动，尤其G15K对G20K的差距在400下更大。预算改变胜率和排序置信程度，不改变本轮的头名。
4. **为何战术进步没有明显转为A/B胜率？** 战术集刻意含直接必胜/必防/三步强制和伪威胁，并且给定已进入攻防的局面；实战还要求从新开局持续产生正确布局、理解多步策略、避免搜索误判。G20K在对B盲测仍只有2.5%/3.8%，显示高战术题分数不等于稳健完整对弈；PUCT-400对A的开发提高也未在盲测维持。
5. **下一步做什么？** 优先做搜索/策略价值的失误归因和更稳健的训练/选模验证，不建议只因20K战术提升就直接扩大到40K。先重现20K优势、分析盲测中对A得分回落和对B漏防；如果扩训，继续保留20K冠军并设置独立开局配对晋级门槛，禁止用搜索模拟增加伪装训练收益。

## 9. 完整性、复现和交付

* 匹配结果审计：{audit['games_verified']:,}局，唯一ID {audit['unique_game_ids']:,}，配对换色{audit['complete_color_swapped_pairs']:,}组，不完整组{audit['incomplete_pairs']}；全部对局合法、到达终局且引擎和结果一致：`{audit['all_completed_games_legal_terminal_and_engine_consistent']}`。
* 三个冻结checkpoint哈希在注册、开跑、收尾一致；模型权重未写入。
* Google Drive独立目录：[WuziQ G V6.1 Recovery](https://drive.google.com/drive/folders/1e15OmoFNXiKyUWbiiFxjjYfdW42Rn5CM)。最终快照在完成上传、回下载SHA-256比对及空目录恢复后才替换旧版；只保留最新V6.1快照，不触及其他实验。
* 研究分支：`research/alphazero-g-v6-1-tournament`。当前结果生成器：`python scripts/report_g_v6_1.py`；复跑需使用`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6_1.py`。逐局结果是幂等的，完成局会校验ID和开局，不会重计。
* 结果表：`results/match_summary.csv`、`results/tactics_summary.csv`、`results/tactics_by_category.csv`、`results/search_efficiency_direct_h2h.csv`、`results/baseline_search_efficiency.csv`；完整逐局JSON在`results/matches/`，逐题JSON在`results/tactics/`。
* 本轮定向V6.1测试7项通过；全仓库测试486项通过、29项失败。失败项是历史V2–V6.3测试的分支名称硬约束（例如要求当前分支必须是`research/alphazero-g-v2`），与V6.1结果无关；完整列表见执行日志，未修改历史分支代码来绕过这些约束。

## 局限

每个直接模型配对只有40个开局对，虽足以揭示本轮的大幅候选差别，但小差距不应被当成稳定的跨开局能力排名。对A/B开发样本各20个配对，盲测每条件40个配对；区间宽、且20K盲测得分偏低。战术集虽独立且标签经规则验证，仍是人工构造的四类测试分布，不是棋局自然分布。全部结论限定为本轮冻结检查点、9×9、PUCT-64/400和已登记开局。
"""
    REPORT.write_text(content, encoding="utf-8")
    compact = {
        "experiment_id": final["experiment_id"],
        "p0_complete": final["p0_complete"], "p1_complete": final["p1_complete"],
        "p2_complete": final["p2_complete"],
        "official_games": audit["games_verified"], "paired_openings": audit["complete_color_swapped_pairs"],
        "preflight_games_excluded": preflight_games,
        "preflight_cpu_seconds": final["preflight_cpu_seconds"],
        "formal_cpu_seconds": final["formal_cpu_seconds"],
        "total_cpu_seconds_including_preflight": final["total_cpu_seconds_including_preflight"],
        "observed_first_to_last_game_wall_seconds": wall_elapsed,
        "runner_wall_seconds_last_resume": runner_wall,
        "checkpoint_sha256": {model: register["models"][model]["sha256"] for model in MODELS},
        "selection_by_budget": selection["selected_by_budget"],
        "game_integrity": {key: value for key, value in audit.items() if key != "game_sha256"},
        "new_tactical_suite_audit": tactic_audit,
    }
    (RESULTS / "integrity_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return compact


def model_summary_image(name: str, caption: str) -> str:
    return f"![{caption}](figures/{name})\n\n*{caption}*"


if __name__ == "__main__":
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
