#!/usr/bin/env python3
"""Create charts and the final Chinese G V4.1 report from verified raw data."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ci_text(interval) -> str:
    if interval is None:
        return "不可估"
    return f"[{100 * interval[0]:.1f}%, {100 * interval[1]:.1f}%]"


def compare(summary: dict, candidate: str, opponent: str, mode: str) -> dict:
    rows = [row for row in summary["comparisons"]
            if row["candidate"] == candidate and row["opponent"] == opponent and row["mode"] == mode]
    if len(rows) != 1:
        raise ValueError(f"expected one result for {candidate}/{opponent}/{mode}, found {len(rows)}")
    return rows[0]


def plot_results() -> None:
    figures = ROOT / "research/g_v4_1/results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["axes.unicode_minus"] = False

    arms = read_json("research/g_v4_1/results/training/arm_summary.json")
    arm_order = ["control_v4", "state_uniform", "winner_balanced", "low_lr"]
    arm_names = {"control_v4": "V4基线", "state_uniform": "状态均匀", "winner_balanced": "黑白平衡", "low_lr": "低学习率"}
    colors = {"control_v4": "#4472C4", "state_uniform": "#ED7D31", "winner_balanced": "#70AD47", "low_lr": "#A5A5A5"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for arm_id in arm_order:
        progress = read_json(f"research/g_v4_1/checkpoints/{arm_id}/progress.json")
        rows = progress["training_history"]
        x = np.asarray([row["states_seen"] for row in rows], dtype=float)
        for ax, key, title in ((axes[0], "policy_loss", "策略损失"), (axes[1], "value_loss", "价值损失")):
            values = np.asarray([row[key] for row in rows], dtype=float)
            window = min(20, len(values))
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(x[window - 1:], smooth, color=colors[arm_id], lw=1.7, label=arm_names[arm_id])
            ax.set_title(title, fontproperties=font)
            ax.set_xlabel("新增自我对弈状态", fontproperties=font)
            ax.set_ylabel("20局移动平均损失", fontproperties=font)
            ax.grid(alpha=.24)
    axes[0].legend(prop=font, frameon=False)
    fig.suptitle("G0四组受控训练损失（只描述优化目标，不代表棋力）", fontproperties=font, fontsize=13)
    fig.savefig(figures / "training_losses.png", dpi=170)
    plt.close(fig)

    dev = read_json("research/g_v4_1/results/evaluation/development_summary.json")
    fig, ax = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
    opponents = [("G0-V3-20K", "V3旧冠军"), ("A", "A规则AI"), ("B", "B Minimax")]
    base_x = np.arange(len(opponents), dtype=float)
    width = .17
    for index, arm_id in enumerate(arm_order):
        values, low, high = [], [], []
        for opponent, _ in opponents:
            summary = compare(dev, arm_id, opponent, "puct")["summary"]
            rate = summary["score_rate"]
            interval = summary["paired_opening_bootstrap_ci95"]
            values.append(rate)
            low.append(max(0, rate - interval[0]))
            high.append(max(0, interval[1] - rate))
        ax.bar(base_x + (index - 1.5) * width, values, width, color=colors[arm_id], label=arm_names[arm_id], zorder=3)
        ax.errorbar(base_x + (index - 1.5) * width, values, yerr=np.asarray([low, high]),
                    fmt="none", ecolor="#222222", elinewidth=1, capsize=3, zorder=4)
    ax.axhline(.5, color="#666666", ls="--", lw=1)
    ax.set_xticks(base_x, [name for _, name in opponents], fontproperties=font)
    ax.set_ylim(-.04, 1.03)
    ax.set_ylabel("得分率（胜1、平0.5）", fontproperties=font)
    ax.set_title("开发集：四组最终checkpoint的PUCT-64成绩", fontproperties=font)
    ax.grid(axis="y", alpha=.24, zorder=0)
    ax.legend(prop=font, frameon=False, ncol=4, loc="upper right")
    fig.savefig(figures / "development_puct_scores.png", dpi=170)
    plt.close(fig)

    blind = read_json("research/g_v4_1/results/evaluation/blind_summary.json")
    finalist = "winner_balanced"
    groups = [("A", "network", "A／纯网络"), ("A", "puct", "A／PUCT-64"),
              ("B", "network", "B／纯网络"), ("B", "puct", "B／PUCT-64"),
              ("G0-V3-20K", "puct", "旧G0／PUCT-64")]
    fig, ax = plt.subplots(figsize=(12.5, 5.5), constrained_layout=True)
    x = np.arange(len(groups), dtype=float)
    width = .32
    for shift, candidate, label, color in ((-width / 2, "G0-V3-20K", "V3旧冠军", "#A5A5A5"),
                                           (width / 2, finalist, "V4.1盲测候选", "#70AD47")):
        valid_x, values, lower, upper, tick_labels = [], [], [], [], []
        for index, (opponent, mode, group_label) in enumerate(groups):
            if candidate == "G0-V3-20K" and opponent == "G0-V3-20K":
                continue
            summary = compare(blind, candidate, opponent, mode)["summary"]
            interval = summary["paired_opening_bootstrap_ci95"]
            valid_x.append(x[index] + shift)
            values.append(summary["score_rate"])
            lower.append(max(0, summary["score_rate"] - interval[0]))
            upper.append(max(0, interval[1] - summary["score_rate"]))
            tick_labels.append(group_label)
        ax.bar(valid_x, values, width, label=label, color=color, zorder=3)
        ax.errorbar(valid_x, values, yerr=np.asarray([lower, upper]), fmt="none", ecolor="#222222", capsize=3, zorder=4)
    ax.axhline(.5, color="#666666", ls="--", lw=1)
    ax.set_xticks(x, [label for _, _, label in groups], fontproperties=font)
    ax.set_ylim(-.04, 1.03)
    ax.set_ylabel("候选方得分率", fontproperties=font)
    ax.set_title("独立盲测：冻结V3与开发集选出的黑白平衡候选", fontproperties=font)
    ax.grid(axis="y", alpha=.24, zorder=0)
    ax.legend(prop=font, frameon=False)
    fig.savefig(figures / "blind_scores.png", dpi=170)
    plt.close(fig)

    tactics = read_json("research/g_v4_1/results/evaluation/tactics.json")["results"]
    target_models = [("G0-V3-20K", "network", "V3裸网络", "#A5A5A5"),
                     ("G0-V3-20K", "puct", "V3 PUCT-64", "#4472C4"),
                     ("winner_balanced", "network", "V4.1裸网络", "#70AD47"),
                     ("winner_balanced", "puct", "V4.1 PUCT-64", "#ED7D31")]
    category_keys = ["direct_win", "unique_block", "edge", "false_threat"]
    category_names = ["直接取胜", "单点必防", "边缘", "伪威胁"]
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    x = np.arange(4, dtype=float)
    width = .18
    for index, (checkpoint, mode, label, color) in enumerate(target_models):
        record = next(row for row in tactics if row["checkpoint_id"] == checkpoint and row["mode"] == mode)
        values = [record["categories"][key]["accuracy"] for key in category_keys]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color, zorder=3)
        for j, key in enumerate(category_keys):
            cell = record["categories"][key]
            ax.text(x[j] + (index - 1.5) * width, values[j] + .022, f"{cell['correct']}/{cell['puzzles']}",
                    ha="center", va="bottom", fontsize=7, rotation=90 if index in (0, 3) else 0)
    ax.set_xticks(x, category_names, fontproperties=font)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("正确率（每类8题）", fontproperties=font)
    ax.set_title("规则生成战术题：旧起点与选中候选（小题集）", fontproperties=font)
    ax.grid(axis="y", alpha=.24, zorder=0)
    ax.legend(prop=font, frameon=False, ncol=2)
    fig.savefig(figures / "tactical_accuracy.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=True)
    labels, black_rates = [], []
    for row in arms:
        labels.append(arm_names[row["arm"]])
        black_rates.append(row["winner_black"] / (row["winner_black"] + row["winner_white"]))
    ax.bar(labels, black_rates, color=[colors[row["arm"]] for row in arms])
    ax.axhline(.5, color="#555555", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("自博弈中黑方胜局占比", fontproperties=font)
    ax.set_title("每臂10K训练状态的颜色分布", fontproperties=font)
    ax.set_xticks(np.arange(len(labels)), labels, fontproperties=font)
    ax.grid(axis="y", alpha=.24)
    fig.savefig(figures / "selfplay_color_balance.png", dpi=170)
    plt.close(fig)

    historical = read_json("research/g_v4_1/results/diagnostics/historical_g0_checkpoint_audit.json")
    rows = historical["historical_checkpoint_tactical_sweep"]["evaluations"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)
    cohort_colors = {"V3 seed 510901": "#4472C4", "V3 seed 510902": "#5B9BD5",
                     "V4 seed 540401": "#ED7D31", "V4 seed 540402": "#70AD47",
                     "V4 seed 540403": "#A5A5A5", "V4 continuation": "#C00000"}
    for ax, mode, title in ((axes[0], "network", "裸网络直接胜点题"),
                            (axes[1], "puct", "PUCT-64直接胜点题")):
        by_cohort = {}
        for row in rows:
            if row["mode"] != mode or row["training_states"] is None:
                continue
            by_cohort.setdefault(row["cohort_id"], []).append(
                (int(row["training_states"]), row["categories"]["direct_win"]["correct"])
            )
        for cohort, values in by_cohort.items():
            values.sort()
            ax.plot([x / 1000 for x, _ in values], [y / 8 for _, y in values], marker="o", lw=1.7,
                    label=cohort, color=cohort_colors.get(cohort))
        ax.set_title(title, fontproperties=font)
        ax.set_xlabel("该checkpoint累计训练状态（千）", fontproperties=font)
        ax.set_ylabel("正确率（8个对称题）", fontproperties=font)
        ax.set_ylim(-.04, 1.05)
        ax.grid(alpha=.24)
    axes[1].legend(prop=font, frameon=False, fontsize=8)
    fig.suptitle("历史V3/V4 G0全部checkpoint：规则直接胜点诊断", fontproperties=font, fontsize=13)
    fig.savefig(figures / "historical_checkpoint_tactics.png", dpi=170)
    plt.close(fig)


def build_csv() -> None:
    rows = []
    for phase in ("development", "blind"):
        summary = read_json(f"research/g_v4_1/results/evaluation/{phase}_summary.json")
        for result in summary["comparisons"]:
            row = {"phase": phase, "candidate": result["candidate"], "opponent": result["opponent"],
                   "mode": result["mode"], **{k: result["summary"][k] for k in
                   ("games", "wins", "draws", "losses", "points", "score_rate", "paired_opening_groups")}}
            row["paired_ci95_low"], row["paired_ci95_high"] = result["summary"]["paired_opening_bootstrap_ci95"]
            row["win_wilson_ci95_low"], row["win_wilson_ci95_high"] = result["summary"]["win_rate_wilson_ci95"]
            for key in ("mean_candidate_decision_seconds", "mean_search_depth", "mean_nodes_per_move",
                        "mean_simulations_per_move", "value_mae", "mean_policy_or_search_entropy"):
                row[key] = result["compute"][key]
            rows.append(row)
    target = ROOT / "research/g_v4_1/results/evaluation/comparisons.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report() -> None:
    config = read_json("research/g_v4_1/config/experiment.json")
    preflight = read_json("research/g_v4_1/results/preflight/preflight.json")
    arms = read_json("research/g_v4_1/results/training/arm_summary.json")
    dev = read_json("research/g_v4_1/results/evaluation/development_summary.json")
    blind = read_json("research/g_v4_1/results/evaluation/blind_summary.json")
    historical = read_json("research/g_v4_1/results/diagnostics/historical_g0_checkpoint_audit.json")
    selection = read_json("research/g_v4_1/results/evaluation/selection.json")
    audit = read_json("research/g_v4_1/results/audit.json")
    test_results = read_json("research/g_v4_1/results/test_results.json")
    drive_verification = read_json("research/g_v4_1/results/recovery/drive_verification.json")
    ledger = read_json("research/g_v4_1/results/cpu_ledger.json")
    tactics = read_json("research/g_v4_1/results/evaluation/tactics.json")
    diag_probe = read_json("research/g_v4_1/results/diagnostics/tactical_probe.json")
    ablation = read_json("research/g_v4_1/results/diagnostics/ablation_tactics.json")
    by_category: dict[str, dict[str, float]] = {}
    for e in ledger["events"]:
        category = e["category"]
        if category not in by_category:
            by_category[category] = {"cpu": 0.0, "wall": 0.0}
        by_category[category]["cpu"] += e["cpu_seconds"]
        by_category[category]["wall"] += e["wall_seconds"]
    cpu_total = sum(e["cpu_seconds"] for e in ledger["events"])
    wall_total = sum(e["wall_seconds"] for e in ledger["events"])
    used = [p.stat().st_mtime for p in [ROOT / "research/g_v4_1/results/diagnostics/crossplay.jsonl",
                                        ROOT / "research/g_v4_1/results/evaluation/blind.jsonl"]]
    elapsed_wall = max(used) - min(used)
    storage_bytes = sum(p.stat().st_size for p in (ROOT / "research/g_v4_1").rglob("*") if p.is_file())

    def puct_row(candidate: str, opponent: str, phase: str) -> dict:
        return compare(dev if phase == "development" else blind, candidate, opponent, "puct")

    def result_line(candidate: str, opponent: str, mode: str, phase: str) -> str:
        row = compare(dev if phase == "development" else blind, candidate, opponent, mode)
        s, c = row["summary"], row["compute"]
        color = s["by_color"]
        return (f"| {candidate} | {opponent} | {mode} | {s['wins']}-{s['draws']}-{s['losses']} | "
                f"{s['points']:.1f}/{s['games']}（{pct(s['score_rate'])}） | "
                f"黑 {pct(color['B']['score_rate'])} / 白 {pct(color['W']['score_rate'])} | "
                f"{ci_text(s['paired_opening_bootstrap_ci95'])} | "
                f"{c['mean_candidate_decision_seconds']:.3f}s /步 | "
                f"{c['mean_search_depth']:.2f} / {c['mean_simulations_per_move']:.0f} |")

    dev_rows = []
    for row in dev["comparisons"]:
        if row["candidate"] == "G0-V3-20K" and row["opponent"] == "G0-V3-20K":
            continue
        if row["opponent"] == "G0-V3-20K" or row["opponent"] in ("A", "B"):
            dev_rows.append(result_line(row["candidate"], row["opponent"], row["mode"], "development"))
    blind_rows = []
    for row in blind["comparisons"]:
        blind_rows.append(result_line(row["candidate"], row["opponent"], row["mode"], "blind"))

    arm_lines = []
    for row in arms:
        arm_lines.append(
            f"| `{row['arm']}` | {row['sampling_mode']} | {row['learning_rate']:.1g} | {row['new_states']:,} | "
            f"{row['games']} | {row['actual_simulations']:,} | {row['optimizer_updates']:,} | "
            f"{row['winner_black']}-{row['winner_white']}-{row['draws']} | "
            f"{row['selfplay_cpu_seconds']:.1f}s | `{row['checkpoint_sha256'][:12]}…` |"
        )

    dev_v3_lines = []
    for arm in ("control_v4", "state_uniform", "winner_balanced", "low_lr"):
        n = compare(dev, arm, "G0-V3-20K", "network")["summary"]
        p = compare(dev, arm, "G0-V3-20K", "puct")["summary"]
        dev_v3_lines.append(
            f"| {arm} | 裸网络 {n['wins']}-{n['draws']}-{n['losses']} ({pct(n['score_rate'])})，CI {ci_text(n['paired_opening_bootstrap_ci95'])} | "
            f"PUCT-64 {p['wins']}-{p['draws']}-{p['losses']} ({pct(p['score_rate'])})，CI {ci_text(p['paired_opening_bootstrap_ci95'])} |"
        )

    ablation_summ = {}
    for model in ("v3_20k", "v4_b100k"):
        ablation_summ[model] = {}
        for category in ("direct_win", "unique_block"):
            ablation_summ[model][category] = {x["setting"]: f"{x['correct']}/{x['n']}" for x in ablation
                                               if x["model"] == model and x["category"] == category}
    v3_target = next(x for x in diag_probe if x["model"] == "v3_20k" and x["category"] == "direct_win")
    v4_target = next(x for x in diag_probe if x["model"] == "v4_b100k" and x["category"] == "direct_win")
    v4_target_prior_mean = next(x["mean_target_prior_mass"] for x in ablation
                                if x["model"] == "v4_b100k" and x["category"] == "direct_win"
                                and x["setting"] == "default64")
    blind_v3 = compare(blind, "winner_balanced", "G0-V3-20K", "puct")["summary"]
    blind_a = compare(blind, "winner_balanced", "A", "puct")["summary"]
    blind_b = compare(blind, "winner_balanced", "B", "puct")["summary"]
    blind_a_net = compare(blind, "winner_balanced", "A", "network")["summary"]
    blind_b_net = compare(blind, "winner_balanced", "B", "network")["summary"]
    control_a_dev = compare(dev, "control_v4", "A", "puct")["summary"]

    historical_logs = historical["historical_selfplay_audit"]
    historical_checkpoint_rows = historical["historical_checkpoint_tactical_sweep"]["evaluations"]
    puct_direct: dict[str, dict[int, int]] = {}
    for row in historical_checkpoint_rows:
        if row["mode"] == "puct":
            puct_direct.setdefault(row["cohort_id"], {})[int(row["training_states"])] = row["categories"]["direct_win"]["correct"]
    historical_checkpoint_table = []
    for cohort in sorted(puct_direct):
        stages = puct_direct[cohort]
        cells = [f"{stage // 1000}K: {stages[stage]}/8" for stage in sorted(stages)]
        historical_checkpoint_table.append(f"| {cohort} | " + "；".join(cells) + " |")
    historical_log_table = []
    for row in historical_logs["logs"]:
        historical_log_table.append(
            f"| {row['cohort']} | `{Path(row['path']).stem}` | {row['games']:,} | {row['budgeted_states']:,} | "
            f"{row['winner_black']}-{row['winner_white']}-{row['draws']} | "
            f"{row['mean_game_moves']:.1f} | {row['mean_policy_entropy']:.2f} | {row['root_value_mae']:.3f} |"
        )

    report = f'''# WuziQ G V4.1｜G0搜索诊断与强化学习改进

## 结论摘要

本轮按预注册配置完成了 **40,000 个新增自我对弈状态、1,369 局训练自博弈、816 局开发赛和360局独立盲测**。训练只用G0网络自身的PUCT-64棋谱、根访问分布与终局胜负，不含A/B/C/E/F数据或教师标签；A、B仅作为冻结评测对手。

结果没有达到晋级标准。开发集选出的 `winner_balanced` 在盲测中以PUCT-64对旧G0@20K为 **20胜、0平、20负，得分率50.0%，配对开局95% bootstrap区间[37.5%,65.0%]**；对A与B的纯网络和PUCT-64盲测均为 **0胜、0平、40负**。开发集里V4基线曾以PUCT-64赢A一局（1-0-23），这是“首次取得一个胜分”的实测例子，但独立盲测没有复现，不能称为开始稳定战胜A。

这次最明确的诊断是：**V3网络直接落子比旧G0强，不代表网络引导的PUCT也更强。** 搜索在浅树中把策略先验和值估计放大；直接胜点被策略头严重低估时，64次搜索仍可能不访问或不选择胜点。PUCT视角切换与终局处理的针对性测试通过，未发现价值回传符号错误的复现证据。`state_uniform`在开发集裸网络对旧G0的得分率最高（20/24），但相同模型PUCT-64只有8/24；该训练采样变化没有改善正式搜索棋力。

本轮没有得到可晋级的新冠军。**保留V3 G0@20K为旧冠军；`winner_balanced`只是一次性盲测候选。** 下一步不建议直接扩到几十万状态；先解决模型训练和PUCT决策之间的错配，优先验证搜索目标与根访问目标、战术漏选和对手池多样性，并设置更多中间checkpoint以观察何时退化。

## 1. 研究边界与预注册方案

- 分支基线：`{config['base_commit']}`，实验分支：`{config['branch']}`；配置SHA-256：`{config['config_sha256']}`。
- 9×9原规则、32通道/2残差块策略价值网、PUCT-64。正式晋级和盲测双方均使用PUCT-64；纯网络仅作为独立的诊断模式。
- 统一起点为冻结 `G0-seed-510901@20000`。权重继承；历史检查点没有完整可恢复优化器状态，因此每臂重建相同的前20,000个G0自博弈状态，并统一重置优化器/调度器。不是无缝续训。
- 四组每组新增10,000状态，优化器AdamW、batch16、每16状态更新一次，所有臂使用64模拟/落子、相同随机种子、同一开局轮转和BN冻结。
- 每组仅改变一个核心因素：V4基线采样；状态均匀采样；黑白胜局平衡采样；学习率从1e-4降至3e-5。每组只跑一个随机种子，因此组间差异为探索性。
- 开局按预注册表分配，训练空盘与合法4手局面各半。开发集12组、每组换色；盲测20个全新配对开局、每组换色。盲测未用于选模或调参。
- 晋级门槛为PUCT-64得分率至少62.5%、配对95%区间下限高于50%，且黑白得分率均至少50%。四组均未通过；按开发集同分后的记录规则，`winner_balanced`作为一个候选进入一次性盲测，不代表晋级。

## 2. 资源预检和真实成本

环境为CPU-only PyTorch 2.15.0，容器CPU配额2核、torch线程1、内存上限8GB、预检峰值RSS约382MB、当时可用磁盘约24.9GB。预检给出的顺序执行wall时间估计为乐观44.6分钟、中央58.8分钟、保守129.9分钟；另一个覆盖训练与评测变动的保守资源包络为约161.1分钟、2.68 CPU小时。实际ledger记录的工作wall时长为 **{wall_total / 60:.1f}分钟**，累计进程CPU时间 **{cpu_total / 3600:.2f}小时**（{cpu_total:.1f}秒），低于6 CPU小时硬上限。诊断至盲测日志时间跨度约{elapsed_wall / 3600:.2f}小时。磁盘研究目录最终约{storage_bytes / (1024 * 1024):.1f}MiB。

| 环节 | CPU时间 | 记录wall时间 |
| --- | ---: | ---: |
| 历史/代码诊断与交叉试验 | {by_category['diagnostics']['cpu']:.1f}s | {by_category['diagnostics']['wall']:.1f}s |
| 性能预检 | {by_category['preflight']['cpu']:.1f}s | {by_category['preflight']['wall']:.1f}s |
| 四组自博弈与训练 | {by_category['train']['cpu']:.1f}s | {by_category['train']['wall']:.1f}s |
| 开发赛和盲测 | {by_category['evaluation']['cpu']:.1f}s | {by_category['evaluation']['wall']:.1f}s |
| 战术题评估 | {by_category['tactical_evaluation']['cpu']:.1f}s | {by_category['tactical_evaluation']['wall']:.1f}s |
| 编排/落盘与其他记录任务 | {by_category['orchestration']['cpu']:.1f}s | {by_category['orchestration']['wall']:.1f}s |
| 自动测试与兼容回归 | {by_category['test']['cpu']:.1f}s | {by_category['test']['wall']:.1f}s |

预检测得自博弈约13.46状态/CPU秒；每个正式训练臂约64万次PUCT模拟。实测评测比预检速度慢，尤其是纯规则AI B对局；总CPU仍明显低于上限。

## 3. 搜索退步诊断

### 3.1 网络棋力与PUCT棋力不一致

V4历史记录中，B@100K对V3 G0@20K的network/network为22-0-2（91.7%），但PUCT-64/PUCT-64仅4-0-20（16.7%）。本轮另做8组配对开局的模式交叉赛：

| 挑战者模式 | 旧冠军模式 | 战绩 | 解释 |
| --- | --- | ---: | --- |
| 裸网络 | 裸网络 | 14-0-2 | 只表示短样本里网络落子模式占优 |
| 裸网络 | PUCT-64 | 1-0-15 | 搜索方明显占优 |
| PUCT-64 | 裸网络 | 14-0-2 | 搜索方自身没有单向弱势，模式和模型存在交互 |
| PUCT-64 | PUCT-64 | 3-0-13 | 两边都搜索时该候选弱于旧模型 |

以上8组、64局为**诊断样本**，没有拿来选模型或宣传排名。其价值在于确认“网络直接落子强、搜索后弱”的现象可复现，不足以单独定位成因。

### 3.2 战术点、访问分布与搜索预算消融

直接胜点题来自规则枚举的4个母局面及其D4对称变换（8题），不含教师标签。V3 G0@20K在直接胜点题上network与默认PUCT-64均0/8。目标落点的网络先验平均名次约45.4/81；PUCT-64虽然每题模拟64次，但平均只给目标约1次访问，最终没有选中胜点。提高`c_puct`到5时7/8，设为10时8/8；把价值头置零时8/8；默认128次只有2/8，256次8/8。单纯加模拟量不是稳定修复，128次结果仍弱。

对V4 B@100K，目标直接胜点平均先验质量仅约{v4_target_prior_mean:.3g}，目标落点搜索排名均值约9.75；默认64/128/256、`c_puct` 3/5/10及置零价值头都为0/8。说明训练后的策略先验分布甚至可能让战术目标不可达；V3中改变价值权重帮助较大，但V4该消融没有作用，不能把单一原因推广到所有checkpoint。

V3 `unique_block`题中默认PUCT-64为5/8，128/256为8/8；将价值头置零为0/8、`c_puct=10`为0/8。V4 B默认64为8/8。误防与漏攻受策略先验、价值估计和探索常数共同影响，并非单纯增加搜索深度即可解决。32道题涵盖4类、每类8题；对称位置不等于32个完全独立战术局面。

**对代码缺陷的检查：** `backup_path`在每条父子边切换当前方视角，终局值按当前待走方视角编码。针对测试覆盖了连续回传符号、终局胜负、合法动作遮罩、棋盘状态恢复、终局子节点不调用网络、噪声可复现。相关PUCT测试全部通过，未找到可复现的符号或终局实现错误。证据更支持模型输出/搜索交互问题，但不排除对其他位置仍有缺陷。

诊断中PUCT-64平均最大树深约1.7–2.1层；每步约4,000–4,700树节点，约0.09–0.17 CPU秒。树宽较大、深度浅，访问受81个动作的先验分配显著影响。V3 direct-win先验名次低；V4甚至有约1e-6先验质量，64次模拟难以克服。PUCT并非Minimax，不能从有限模拟保证战术完备。

### 3.3 历史数据偏差检查

额外逐条复用了V3/V4六份G0历史自博弈JSONL（两份V3 G0 seed、三份V4新seed、一份V4冠军续训），共 **{historical_logs['total_games']:,}局、{historical_logs['total_budgeted_states']:,}个预算状态、{historical_logs['total_states']:,}条完整状态记录**；9,154个全局唯一对局ID，无跨文件重复，逐局按规则回放合法。F系G-F2日志未用于G0诊断或训练。

| 历史队列 | 文件/seed | 局数 | 预算状态 | 黑胜-白胜-平 | 平均总着数 | 策略熵 | 训练分布内根价值MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(historical_log_table)}

历史V4冠军续训样本中，黑方胜3,234/4,208（76.9%），平均总着数25.8、策略访问熵2.46；V3 seed510901黑胜542/1,002（54.1%），平均总着数{historical_logs['logs'][0]['mean_game_moves']:.1f}、熵3.45。其他V3/V4初始G0队列的颜色比例接近均衡。按局等权抽样可能让短局状态权重增大，这是待验证的机制假设，不是已证实的唯一根因。本轮四个新臂各自黑胜约53.6%–56.4%，没有重现历史76.9%的偏斜；每臂样本仅约336–350局。V4.1训练未把A/B作为对手或训练输入。

### 3.4 历史全部G0 checkpoint战术轨迹

本轮复核了V3/V4目录中的39个G0模型权重文件，按SHA-256去重后对37个唯一checkpoint做同一32题network与PUCT-64诊断，共74个checkpoint/模式评估。两个文件对只是逐字节重复的 `final.pt` / `node_*.pt`，所以不重复计算。结果呈现明显的随机种子差异与非单调轨迹：

| G0路线 | PUCT-64直接胜点题结果（每阶段8题） |
| --- | --- |
{chr(10).join(historical_checkpoint_table)}

例如V3 seed510901的PUCT-64从0–10K阶段的6–8/8，降至20K与30K的0/8；V4 seed540402则在0–30K各检查点保持8/8；V4冠军续训从0到100K均0/8。说明战术能力退化并非所有种子共有的固定规律，更不是“只要训练更多就会更好”。这些题由4个母局面做对称扩展，适合作为缺陷信号，不足以形成广泛战术棋力排名。

![V3/V4全部G0 checkpoint直接胜点轨迹](results/figures/historical_checkpoint_tactics.png)

## 4. 受控训练结果

| 训练臂 | 采样方式 | 学习率 | 新增状态 | 完整对局 | 实际模拟数 | 更新数 | 黑胜-白胜-平 | 自博弈CPU秒 | 最佳权重SHA-256前缀 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(arm_lines)}

四臂共训练40,000个预算状态、2,562,304次PUCT模拟、2,500次优化更新。最终完整对局一局走完后再按预算接受前缀；末局超出预算部分只保留在完整棋谱供审计，不计入训练状态。每臂只有一个种子，不足以估计种子方差。loss曲线只反映训练目标优化程度，不能单独当作棋力证据。

![四臂策略与价值损失曲线](results/figures/training_losses.png)

### 4.1 开发集模型选择

| 训练臂 | 裸网络对V3旧冠军 | PUCT-64对V3旧冠军 |
| --- | --- | --- |
{chr(10).join(dev_v3_lines)}

`state_uniform`和`winner_balanced`在裸网络对V3时为20/24，但只有12个开局配对；其余臂也达到18/24。PUCT成绩却为7–10/24，全部得分率低于50%。纯网络优势未稳定转移到搜索模式。

开发集34种候选/对手/模式对照共816局，12个配对开局、每个开局交换颜色。下面表格展示基准模型和四个训练臂对V3/A/B的纯网络、PUCT-64成绩；表内CI为12组配对开局的bootstrap区间。没有比赛的比较没有推断结果。

| 候选模型 | 对手 | 模式 | W-D-L | 得分 | 黑/白得分率 | 配对95%区间 | 决策耗时/步 | 平均树深/模拟 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
{chr(10).join(dev_rows)}

开发集上只有V4基线PUCT对A拿到1胜（1-0-23，4.2%），其他臂均为0-0-24；全部G0基线和训练臂对B均没有胜/平。`winner_balanced`在PUCT对旧冠军得分10/24，与control_v4的10/24打平；按预注册同分候选规则及测得的计算效率入选盲测，但没有达到62.5%晋级门槛。

![开发集PUCT-64成绩](results/figures/development_puct_scores.png)

## 5. 独立盲测

一次性盲测共20组全新配对开局，每组分别执黑、执白各下一局；共360局、9种注册比较，所有棋谱逐局回放。候选为开发集选出的`winner_balanced`。候选和旧冠军各自以纯网络及PUCT-64对A/B；另有候选PUCT-64与V3旧冠军PUCT-64直接对战。

| 候选模型 | 对手 | 模式 | W-D-L | 得分 | 黑/白得分率 | 配对95%区间 | 决策耗时/步 | 平均树深/模拟 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
{chr(10).join(blind_rows)}

盲测关键行：选中候选network/network模式对A是{blind_a_net['wins']}-{blind_a_net['draws']}-{blind_a_net['losses']}；PUCT-64对A是{blind_a['wins']}-{blind_a['draws']}-{blind_a['losses']}。network/network对B为{blind_b_net['wins']}-{blind_b_net['draws']}-{blind_b_net['losses']}；PUCT-64对B为{blind_b['wins']}-{blind_b['draws']}-{blind_b['losses']}。对旧G0 PUCT-64直接交手为{blind_v3['wins']}-{blind_v3['draws']}-{blind_v3['losses']}，开局配对CI为{ci_text(blind_v3['paired_opening_bootstrap_ci95'])}。各局黑白各20局；候选对旧G0的黑方得分率{pct(blind_v3['by_color']['B']['score_rate'])}、白方{pct(blind_v3['by_color']['W']['score_rate'])}，存在明显颜色差，但样本不足以拆解颜色与开局效应。

对A/B候选得分为0/40；bootstrap因所有开局对都同分而退化为[0,0]，因此另报告二项Wilson非负分区间上限：0/40的95%区间约0%–8.8%。盲测未达到首次取得胜分、接近50%或统计支持超越A的标准。没有任何一条对A结果满足“得分率>50%且配对95%区间下限>50%”。

![独立盲测成绩](results/figures/blind_scores.png)

## 6. 战术能力、价值预测和计算效率

32道规则战术题的结果如下图；每类8题，直接胜点、单点必防、边缘、伪威胁。选中的`winner_balanced`裸网络是：直接胜点0/8、单点必防8/8、边缘0/8、伪威胁0/8；PUCT-64为0/8、6/8、3/8、0/8。V3旧冠军裸网络为0/8、8/8、0/8、1/8；PUCT为0/8、5/8、8/8、2/8。边缘题取值可能受对称位置与方位细节影响；直接胜点能力仍明显欠缺。该题集样本有限，不代表所有局面的战术准确率。

![战术题正确率](results/figures/tactical_accuracy.png)

盲测选中候选的纯网络平均决策耗时约1.3–1.5毫秒/步；PUCT-64约0.13–0.15秒/步，平均搜索深度约1.8–2.1、真实模拟固定64次。裸网络输出的价值MAE对A/B约1.05–1.11，对旧G0 PUCT局面约0.93；PUCT根价值MAE对A/B约1.02–1.07，对旧G0约0.93。标签取决于对局终局胜负，样本按模型生成的对局筛选，不能当作独立校准集。CPU每步搜索成本约为纯网络的百倍量级，却没有在A/B上带来盲测得分。V3/V4所有G0 checkpoint的逐题预测、访问数及历史棋谱汇总见[`historical_g0_checkpoint_audit.json`](results/diagnostics/historical_g0_checkpoint_audit.json)。

![训练局黑白胜率](results/figures/selfplay_color_balance.png)

**分开看两类收益：**训练臂对V3裸网络的开发赛结果说明策略输出确有变化；同臂与相同旧模型以PUCT-64交手、以及最终盲测都没有表现出可靠的搜索棋力提升。固定checkpoint下PUCT相对纯网络的模式差异是“搜索收益”，不能算训练收益。本轮没有发现某个训练改进可重复地提升PUCT-64的成绩。

## 7. 逐项回答研究问题

1. **为什么G0搜索棋力会退步？** 现有证据显示V3起点及V4续训模型在战术点上的策略先验排名很低，价值估计会改变PUCT访问分布，64次模拟容易形成浅而宽的树。直接落点网络在开发集能胜旧G0，而搜索组合对同一旧G0弱很多。训练数据有历史黑方、短局和熵变化，但V4.1中各臂颜色比例较平衡，因此这些属于值得继续验证的次级假设；不能断言唯一根因。
2. **哪种改进有效？** 状态均匀采样对裸网络相对V3开发表现较好；黑白平衡候选开发PUCT对V3为10/24，与V4基线打平。低学习率没有改善对旧G0的PUCT成绩。没有一个因素通过晋级门槛或在盲测中对A/B得分。
3. **是否开始接近A？** 控制组在开发赛曾赢A一局，但不是稳定结果。盲测候选与V3在纯网络和PUCT模式对A各40局均0胜、0平；PUCT对B也0胜、0平。本轮证据不支持接近A。
4. **是否超过旧冠军？** 候选PUCT盲测对V3是20-0-20，50%且区间跨越50%；没有证据证明超越。按预注册晋级规则，继续保留V3 G0@20K。
5. **下一轮最值得验证什么？** 先开发与棋力目标一致的搜索训练：监测策略头给精确规则胜点的概率质量、PUCT根访问与最终动作一致性；在训练中加入G0自对弈历史模型池以减小当前策略自博弈坍缩，但仍不得用A/B或教师标签；安排1K/2K/5K等中间节点的开发评估和多随机种子。任何参数变化先用独立战术局面诊断，再用PUCT-64配对赛验证，保持盲测冻结。

## 8. 自动审计、数据与复现

最终审计逐局回放了1,369局训练自博弈、816局开发赛与360局盲测：训练状态恰为40,000、训练局ID唯一、盲测和开发ID不相交、评估比较完整且均按开局换色、每个checkpoint SHA-256与训练摘要一致。盲测没有用于选模。完整结果见[`audit.json`](results/audit.json)，全部机器可读对阵统计见[`comparisons.csv`](results/evaluation/comparisons.csv)，逐局和逐步记录位于`results/selfplay/*.jsonl`及`results/evaluation/*.jsonl`。

复现环境和参数：[`experiment.json`](config/experiment.json)、[`openings.json`](config/openings.json)、`python scripts/preflight_g_v4_1.py`、`python scripts/run_g_v4_1.py --help`、`python scripts/audit_g_v4_1.py`。恢复快照创建与恢复：`python scripts/create_g_v4_1_snapshot.py --help`、`python scripts/restore_g_v4_1_snapshot.py --help`。本轮最优候选模型权重：[`winner_balanced/node_10000.pt`](checkpoints/winner_balanced/node_10000.pt)，SHA-256 `{next(x['checkpoint_sha256'] for x in arms if x['arm'] == 'winner_balanced')}`；该候选未通过晋级，不替代旧冠军。

相关测试：V4.1新增测试、PUCT视角/终局测试和完整数据审计共 **{test_results['focused_v4_1_and_puct']['passed']}项通过**；代码编译通过。兼容回归排除旧实验专属分支守卫后 **{test_results['repository_compatibility']['passed']}项通过、{test_results['repository_compatibility']['deselected_branch_locked']}项按其注册分支隔离未在V4.1分支运行**。不筛选的兼容试跑中有24项因V2/V3/V6旧实验检查要求各自的独立分支而失败，另1项V4快照分支守卫被单独排除；没有修改这些旧实验的分支身份检查。完整记录见[`test_results.json`](results/test_results.json)。Drive恢复目录为[WuziQ G V4.1 Recovery]({drive_verification['folder_url']})，只保留一份经回读、SHA-256和空目录恢复验证的最新快照；本地验证收据见[`drive_verification.json`](results/recovery/drive_verification.json)。

## 9. 限制

- 每臂单随机种子、10K状态；开发对局每项24局，正式盲测对A/B每模式40局；区间仍较宽。
- 所有盲测对A/B结果为零，允许的棋力上界仍不等于真实能力为零；只说明当前数据没有观察到得分。
- 战术题仅32道、由4个母局面对称扩展；消融不替代广泛战术题库。
- PUCT深度浅，且纯网络与搜索使用的策略决策机制不同。本轮只做受控参数消融，不做网络结构或PUCT算法改造。
- 棋谱、训练与搜索结果均来自传统AlphaZero式自博弈，本轮没有监督模仿学习，也没有神经网络外部教师数据。

本报告是探索性研究记录；目前最重要的否定性结论是：**40K额外G0自我对弈状态没有让选中模型在PUCT-64下开始得分于A或B，也没有可靠超过V3 G0冠军。**
'''
    (ROOT / "research/g_v4_1/REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    plot_results()
    build_csv()
    build_report()
    print("generated research/g_v4_1/REPORT.md, figures, and comparisons.csv")


if __name__ == "__main__":
    main()
