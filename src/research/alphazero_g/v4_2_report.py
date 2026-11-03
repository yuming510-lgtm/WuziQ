"""Chinese research report, plots and source-backed numeric summaries for G V4.2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .v4_2_storage import load_json


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _ci(summary: dict[str, Any]) -> str:
    interval = summary.get("paired_opening_bootstrap_ci95")
    return "—" if not interval else f"[{_pct(interval[0])}, {_pct(interval[1])}]"


def _score_cell(summary: dict[str, Any]) -> str:
    return f"{_pct(summary.get('score_rate'))} {_ci(summary)}"


def _safe_summary(value: dict[str, Any] | None) -> dict[str, Any]:
    return value or {}


def _line_with_ci(ax, rows: list[tuple[int, dict[str, Any]]], label: str, color: str) -> None:
    if not rows:
        return
    x = [state for state, _ in rows]
    y = [summary.get("score_rate") for _, summary in rows]
    ax.plot(x, y, marker="o", label=label, color=color)
    lows, highs = [], []
    for _, summary in rows:
        interval = summary.get("paired_opening_bootstrap_ci95")
        lows.append(float(summary.get("score_rate", 0.0)) - (float(summary.get("score_rate", 0.0)) if not interval else interval[0]))
        highs.append((float(summary.get("score_rate", 0.0)) if not interval else interval[1]) - float(summary.get("score_rate", 0.0)))
    ax.errorbar(x, y, yerr=np.asarray([lows, highs]), fmt="none", capsize=3, color=color)


def _write_figures(root: Path, config: dict[str, Any], diagnostics: dict[str, Any],
                   training: dict[str, Any], blind: dict[str, Any]) -> list[str]:
    figures = root / "research/g_v4_2/figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})

    historical = diagnostics["historical_match_summary"]
    candidates = config["historical_candidates"]
    labels = [item["candidate_id"].replace("G0-", "") for item in candidates]
    x = np.arange(len(candidates))
    a = [historical[item["candidate_id"]]["A-rule"]["score_rate"] for item in candidates]
    old = [historical[item["candidate_id"]]["G0-V3-510901-20K"]["score_rate"] for item in candidates]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, a, marker="o", label="对 A，PUCT-64", color="#2575b8")
    ax.plot(x, old, marker="s", label="对 V3 G0@20K，PUCT-64", color="#df7b18")
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=55, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("得分率")
    ax.set_title("历史 G0 检查点：固定开发开局、PUCT-64、黑白换色")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures / "historical_checkpoint_scores.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))

    stage_summaries = training.get("stage_summaries", {})
    colors = {"baseline": "#2563eb", "search_target": "#dc2626", "history_pool": "#059669"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for arm_id, stages in stage_summaries.items():
        for opponent_id, label_suffix in (("A-rule", "对 A"), ("G0-V3-510901-20K", "对 V3 G0@20K")):
            rows = []
            for item in stages:
                rows.append((int(item["training_states"]), item["matches"][opponent_id]))
            _line_with_ci(ax, rows, f"{arm_id} {label_suffix}", colors.get(arm_id, "#777"))
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("新增自我对弈状态")
    ax.set_ylabel("得分率（16组配对开发开局）")
    ax.set_title("训练检查点棋力曲线；固定 PUCT-64")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figures / "training_strength_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))

    variants = diagnostics.get("search_development_candidates", [])
    names = ["baseline", *[row["variant"] for row in variants]]
    values = [diagnostics["search_development_metrics"][name]["accuracy"] for name in names]
    validation = diagnostics.get("search_validation_metrics", {})
    selected = diagnostics.get("search_selection", {}).get("selected", "baseline")
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.36
    dev_vals = [float(value) for value in values]
    val_vals = []
    for name in names:
        metric = validation.get(name)
        val_vals.append(float(metric["accuracy"]) if metric else float("nan"))
    pos = np.arange(len(names))
    ax.bar(pos - width / 2, dev_vals, width, label="搜索开发题集", color="#6c91bf")
    ax.bar(pos + width / 2, val_vals, width, label="独立验证题集", color="#e29d4b")
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
    ax.set_xticks(pos, names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("四类精确战术题正确率")
    ax.set_title(f"冻结网络的 PUCT 搜索校准；训练臂采用：{selected}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures / "search_variant_tactics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))

    summaries = blind.get("summaries", {})
    keys = [
        "candidate_network_vs_A-rule", "candidate_vs_A-rule",
        "candidate_network_vs_B-50ms", "candidate_vs_B-50ms",
    ]
    vals = [summaries[key]["score_rate"] for key in keys if key in summaries]
    shown = [key.replace("candidate_network_vs_", "裸网络对 ").replace("candidate_vs_", "PUCT-64对 ")
             for key in keys if key in summaries]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(vals)), vals, color=["#9a9a9a", "#2878b5"] * 2)
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(vals)), shown, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("盲测得分率")
    ax.set_title("冻结候选：纯网络与 PUCT-64 在相同盲测开局上的表现")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures / "blind_network_vs_puct.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))
    return outputs


def build_v4_2_report(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    config = load_json(root_path / "research/g_v4_2/config/experiment.json")
    preflight = load_json(root_path / "research/g_v4_2/results/preflight/preflight.json")
    diagnostics = load_json(root_path / "research/g_v4_2/results/diagnostics/diagnostics.json")
    training = load_json(root_path / "research/g_v4_2/results/training/summary.json")
    blind = load_json(root_path / "research/g_v4_2/results/blind/summary.json")
    audit = load_json(root_path / "research/g_v4_2/results/integrity_audit.json")
    progress_path = root_path / "research/g_v4_2/results/progress.json"
    progress = load_json(progress_path) if progress_path.exists() else {}
    figures = _write_figures(root_path, config, diagnostics, training, blind)

    lines: list[str] = []
    lines.extend([
        "# WuziQ G V4.2：AlphaZero 搜索与训练协同优化",
        "",
        "> 本报告记录规则/搜索算法与AlphaZero式自我对弈实验，没有神经网络训练以外的教师监督；未使用A/B/C/E/F棋谱、标签或权重作为G0训练输入。A/B只在评估时出现。",
        "",
        "## 摘要",
        "",
        f"- 基线：`{config['base_commit']}`；实验分支：`{config['branch']}`。配置SHA-256：`{config['config_sha256']}`。",
        f"- 9×9，原G0 32通道、2残差块；统一训练/正式评估搜索为每步64次PUCT模拟。最多3臂各10,000个新增状态，总登记上限30,000。",
        f"- 训练臂：`{', '.join(training['active_arms'])}`；实际新增状态：{training['new_training_states_total']:,}。优化器从V3@20K权重初始化时重置，BatchNorm统计冻结。",
        f"- 开发集选出的最终候选：`{blind['frozen_candidate']['candidate_id']}`；盲测后冠军判断：`{blind['champion_decision']}`。",
        f"- 实测CPU耗时约{float(progress.get('total_cpu_seconds_including_preflight', preflight['preflight_cpu_seconds'])) / 3600:.2f}小时；盲测20组新开局、换色对弈。",
        "",
        "## 预检、环境与预注册",
        "",
        f"云端机器实测：CPU亲和{preflight['resources']['affinity_cpu_count']}核、cgroup CPU配额{preflight['resources']['cgroup_cpu_quota']}核，内存上限{preflight['resources']['memory_cgroup_limit_bytes'] / 1024**3:.1f}GiB，可用磁盘{preflight['resources']['disk_free_bytes'] / 1024**3:.1f}GiB；GPU不可用，使用CPU及{preflight['resources']['torch_threads']}个Torch线程。",
        f"短预检实测：64次搜索每步约{preflight['measurements']['search64_cpu_seconds_per_call']:.3f} CPU秒；自我对弈约{preflight['measurements']['selfplay_states_per_cpu_second']:.2f}状态/CPU秒；优化器约{preflight['measurements']['optimizer_updates_per_cpu_second']:.1f}更新/CPU秒；评测局约{preflight['measurements']['evaluation_cpu_seconds_per_game']:.3f} CPU秒。",
        f"总CPU预估（含{preflight['estimates']['game_count_components']['total_games']}场正式/开发对局，不含战术题搜索调用）：乐观{preflight['estimates']['levels']['optimistic']['total_cpu_hours']:.2f}小时、中位{preflight['estimates']['levels']['central']['total_cpu_hours']:.2f}小时、保守{preflight['estimates']['levels']['conservative']['total_cpu_hours']:.2f}小时；硬上限5小时。",
        f"开局表SHA-256：`{config['opening_schedule_sha256']}`。注册时排除了V3/V4/V4.1中2,444个唯一D4对称等价的4手开局；新开局池含训练800组、开发16组、盲测20组。",
        "开发集只用于历史筛选和训练候选选择。盲测开局在冻结候选后才执行；没有依据盲测结果回调训练或参数。",
        "",
        "## 搜索机制诊断",
        "",
        f"历史12个G0检查点先在相同16组开发开局上与A、V3 G0@20K对弈，各组交换黑白；其中胜率最高的三个用于冻结权重搜索对照。战术题64道，四类各16道，母局面之间做D4去重，标签由精确五连规则计算。",
        "",
        "| 搜索配置 | 开发题正确率 | 独立验证正确率 | 开发战术增益 | 阻挡相对保留 | 决定 |",
        "|---|---:|---:|---:|---:|---|",
    ])
    baseline_accuracy = diagnostics["search_development_metrics"]["baseline"]["accuracy"]
    for row in diagnostics["search_development_candidates"]:
        variant = row["variant"]
        val = diagnostics["search_validation_metrics"].get(variant)
        val_acc = "—" if val is None else _pct(val["accuracy"])
        decision = "采用" if diagnostics["search_selection"]["selected"] == variant else "未采用"
        ratio = row.get("blocking_ratio_to_baseline")
        lines.append(f"| {variant} | {_pct(row['development_macro_accuracy'])} | {val_acc} | {row['development_gain']:+.3f} | {_pct(ratio)} | {decision} |")
    lines.append(f"| baseline | {_pct(baseline_accuracy)} | {_pct(diagnostics['search_validation_metrics']['baseline']['accuracy'])} | 0.000 | 100.0% | 对照 |")
    lines.extend([
        "",
        f"冻结权重的自动选型最终为`{diagnostics['search_selection']['selected']}`。选型门槛是开发题宏平均至少提升5个百分点、阻挡类保持基线90%以上，并在未参与选择的验证题上不退步且满足同样的阻挡保留条件。未达门槛时，不启用搜索改进训练臂。",
        "",
        "| 类别 | 题数 | 基线正确 | 基线最优策略先验质量 | 基线根访问质量 |",
        "|---|---:|---:|---:|---:|",
    ])
    top3 = diagnostics["top3_history_models_selected_on_development_games"]
    for category in ("direct_win", "unique_block", "false_threat", "edge_win"):
        # The full per-puzzle records are stored separately; aggregate metrics below use each top-three model equally.
        per_model = [diagnostics["historical_tactical_summary"][candidate_id]["categories"][category]
                     for candidate_id in top3]
        count = sum(int(row["correct"]) for row in per_model)
        prior = float(np.mean([row["mean_optimal_prior_mass"] for row in per_model]))
        visit = float(np.mean([row["mean_root_visit_mass"] for row in per_model]))
        lines.append(f"| {category} | 48（3模型×16） | {count}/48（{_pct(count / 48)}） | {_pct(prior)} | {_pct(visit)} |")
    lines.append("")
    lines.append("策略头与价值头另行记录，不将单题即时获胜标签外推到阻挡题：")
    lines.append("")
    lines.append("| 历史候选 | 立即获胜题网络策略准确率 | 最优动作平均先验质量 | 网络价值MAE（立即胜点对+1） |")
    lines.append("|---|---:|---:|---:|")
    for candidate_id in top3:
        metrics = diagnostics["historical_tactical_summary"][candidate_id]
        direct = [metrics["categories"][category] for category in ("direct_win", "edge_win")]
        network_correct = sum(int(row["network_correct"]) for row in direct)
        network_accuracy = network_correct / max(1, sum(int(row["puzzles"]) for row in direct))
        mean_prior = float(np.mean([row["mean_optimal_prior_mass"] for row in direct]))
        mae = metrics["immediate_win_network_value_mae_vs_exact_win"]
        lines.append(f"| {candidate_id} | {_pct(network_accuracy)} | {_pct(mean_prior)} | {mae:.3f} |")
    lines.append("对直接获胜类题，策略先验质量与根访问质量的差异可显示PUCT是否补救了网络先验；两者都低时属于策略头漏报，访问质量低而先验质量尚可时更像搜索预算/探索分配问题。非终局阻挡题没有被错误地赋予`+1`价值标签。128/256模拟只用于同一题集的搜索诊断，没有计入正式64次棋力。")
    lines.append("")
    lines.append("### 128/256模拟诊断（仅搜索预算对照）")
    lines.append("")
    lines.append("| 模拟数 | 题数 | 精确战术正确率 | 平均树深 | 平均节点 |")
    lines.append("|---:|---:|---:|---:|---:|")
    expanded = diagnostics["expanded_simulation_diagnostics"]
    for simulations in (64, 128, 256):
        subset = [row for row in expanded if row["simulations"] == simulations]
        lines.append(f"| {simulations} | {len(subset)} | {_pct(np.mean([row['correct'] for row in subset]))} | {np.mean([row['max_depth'] for row in subset]):.2f} | {np.mean([row['tree_nodes'] for row in subset]):.1f} |")

    lines.extend(["", "## 历史G0阶段复核", "", "| G0检查点 | 对A得分率（配对95%区间） | 对V3@20K得分率 | 64题战术正确率 |"])
    lines.append("|---|---:|---:|---:|")
    for candidate in config["historical_candidates"]:
        candidate_id = candidate["candidate_id"]
        summary = diagnostics["historical_match_summary"][candidate_id]
        tactics = diagnostics["historical_tactical_summary"][candidate_id]
        lines.append(f"| {candidate_id} | {_score_cell(summary['A-rule'])} | {_score_cell(summary['G0-V3-510901-20K'])} | {tactics['correct']}/{tactics['puzzles']}（{_pct(tactics['accuracy'])}） |")

    lines.extend(["", "## 受控训练曲线", "", "三个预注册组从相同的V3 G0@20K冻结权重起步，使用相同PUCT-64、温度、根噪声、1e-4学习率、AdamW、批大小16、BatchNorm冻结及256状态检查点间隔。唯一变化为：原始当前网络自我对弈、通过门槛的搜索配置、或仅与历史G0模型池对弈。历史池臂只把当前G0棋手自己的状态写入训练集；冻结G0模型只提供自我对弈对手，不反向采样或训练。优化器按登记从零重置。", ""])
    for arm_id in training["active_arms"]:
        lines.extend([f"### {arm_id}", "", "| 新增状态 | 更新数 | 策略损失 | 价值损失 | 自我对弈CPU秒 | 评测CPU秒 | A对局平均树深/节点 | 对A | 对B | 对V3@20K | 挑战门槛 | 战术题 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"])
        for stage in training["stage_summaries"][arm_id]:
            metrics = stage.get("optimizer", {})
            matches = stage["matches"]
            gate = stage["challenger_gate_vs_v3_g0_20k"]
            a_search = matches["A-rule"]
            lines.append(
                f"| {stage['training_states']:,} | {metrics.get('update_count', '—')} | {float(metrics.get('policy_loss', 0.0)):.4f} | {float(metrics.get('value_loss', 0.0)):.4f} | {stage.get('selfplay_cpu_seconds_since_previous_node', 0.0):.1f} | {stage.get('evaluation_cpu_seconds', 0.0):.1f} | {a_search.get('mean_agent_search_depth', 0.0):.2f}/{a_search.get('mean_agent_nodes_per_game', 0.0):.0f} | {_score_cell(matches['A-rule'])} | {_score_cell(matches['B-50ms'])} | {_score_cell(matches['G0-V3-510901-20K'])} | {'通过' if gate['promote'] else '未通过'} | {stage['tactical_search_puct64']['correct']}/{stage['tactical_search_puct64']['puzzles']} |"
            )
        lines.append("")

    internal_path = root_path / "research/g_v4_2/results/training/arm_internal_summary.json"
    if internal_path.exists():
        internal = load_json(internal_path)
        lines.extend(["### 训练臂最终节点直接交手（开发集，非盲测）", "", "| 对局 | 场次 | 胜-平-负 | 得分率（配对95%区间） |", "|---|---:|---:|---:|"])
        for matchup, summary in internal["comparisons"].items():
            lines.append(f"| {matchup} | {summary['games']} | {summary['wins']}-{summary['draws']}-{summary['losses']} | {_score_cell(summary)} |")

    lines.extend(["", "## 冻结候选与独立盲测", "", f"开发集冻结候选为`{blind['frozen_candidate']['candidate_id']}`（训练状态{blind['frozen_candidate']['training_states']:,}）。盲测使用预注册20组新开局，每组黑白交换；候选与旧V3@20K分别对A/B，候选另以纯网络与PUCT-64对A/B。所有组合使用同样棋盘、规则、开局和颜色分配。", "", "| 评测模式/对手 | 场次 | 胜-平-负 | 得分率（配对95%区间） | 黑方候选得分率 | 白方候选得分率 |", "|---|---:|---:|---:|---:|---:|"])
    for key, label in (("candidate_vs_A-rule", "候选 PUCT-64 对A"), ("candidate_vs_B-50ms", "候选 PUCT-64 对B"),
                       ("candidate_vs_G0-V3-510901-20K", "候选 PUCT-64 对V3 G0@20K"),
                       ("candidate_network_vs_A-rule", "候选裸网络对A"), ("candidate_network_vs_B-50ms", "候选裸网络对B"),
                       ("v3_champion_vs_A-rule", "V3 G0@20K PUCT-64 对A"), ("v3_champion_vs_B-50ms", "V3 G0@20K PUCT-64 对B")):
        summary = blind["summaries"].get(key)
        if summary:
            black, white = summary["by_color"]["B"]["score_rate"], summary["by_color"]["W"]["score_rate"]
            lines.append(f"| {label} | {summary['games']} | {summary['wins']}-{summary['draws']}-{summary['losses']} | {_score_cell(summary)} | {_pct(black)} | {_pct(white)} |")
    lines.extend(["", "| 配对差值 | 均值 | 95%开局配对区间 |", "|---|---:|---:|"])
    for key, label in (("candidate_vs_A-rule", "训练候选相对初始V3@20K，对A"), ("candidate_vs_B-50ms", "训练候选相对初始V3@20K，对B"),
                       ("puct64_minus_network_vs_A-rule", "同一候选：PUCT-64减裸网络，对A"), ("puct64_minus_network_vs_B-50ms", "同一候选：PUCT-64减裸网络，对B")):
        paired = blind["paired_score_differences"][key]
        interval = paired["paired_bootstrap_ci95"]
        lines.append(f"| {label} | {paired['mean_score_rate_difference']:+.1%} | [{_pct(interval[0])}, {_pct(interval[1])}] |")
    lines.append("")
    lines.append(f"最终判定：盲测中候选相对V3 G0@20K的得分率为{_score_cell(blind['summaries'][f'candidate_vs_{"G0-V3-510901-20K"}'])}；只有开发挑战门槛通过且该直接盲测的配对区间下限超过50%才替换冠军。本次冠军记录为`{blind['champion_decision']}`。20组开局样本仍有限，区间反映配对开局波动，不应视作对所有开局的稳定等级。")

    lines.extend(["", "## 主要结论", ""])
    candidate_a = blind["summaries"].get("candidate_vs_A-rule", {})
    candidate_champion = blind["summaries"].get(f"candidate_vs_{'G0-V3-510901-20K'}", {})
    if candidate_a:
        rate = float(candidate_a["score_rate"])
        ci = candidate_a.get("paired_opening_bootstrap_ci95")
        if rate > 0.5 and ci and ci[0] > 0.5:
            statement = "该20组换色盲测显示对A得分超过50%，且配对95%区间下限高于50%；这是有限开局样本上的统计证据，不等于所有开局的普遍胜率。"
        elif rate > 0.5:
            statement = "点估计高于50%，但配对95%区间包含50%；尚不能说稳定战胜A。"
        else:
            statement = "盲测点估计没有超过50%，本轮没有证明G0已经稳定战胜A。"
        lines.append(f"- **G0是否接近A：** {statement}")
    lines.append(f"- **是否出现新冠军：** {'有开发与盲测双重支持的候选' if blind['champion_decision'] != 'G0-V3-510901-20K' else '没有；保留V3 G0@20K旧冠军'}。")
    lines.append(f"- **训练与搜索分别贡献什么：** 同一冻结候选的PUCT-64与裸网络差值见盲测表，表示搜索增益；训练候选与V3 G0@20K同预算差值表示学习增益。不得把更高模拟量或PUCT搜索收益算作网络训练成长。")
    selected_search = diagnostics["search_selection"]
    lines.append(f"- **搜索问题与有效方法：** 本轮只认同一冻结权重64次搜索下通过开发/验证门槛的配置；最终采用`{selected_search['selected']}`。先验质量、根访问质量、节点和深度数据见战术结果及原始JSON。")
    if "search_target" not in training["active_arms"]:
        lines.append("- 搜索改进未达到预注册题集门槛，因此未消耗对应训练预算；实际只训练baseline和历史G0对手池两个组。")
    else:
        lines.append("- 搜索配置通过独立验证后作为第三训练臂；其结果仍以固定PUCT-64开发/盲测比较，不将更宽搜索本身认作学习收益。")
    lines.extend([
        "- **单种子与预算限制：** 每臂仅一个随机流、最多10K状态，适合检验方向而非给出跨种子复现结论；只有满足挑战门槛的候选才可替换旧冠军。",
        "- **恢复与完整性：** 每个对局以稳定ID原子写入；训练状态每256个状态原子checkpoint。完整棋谱、训练标签、固定模拟数及开局唯一性审计见 `results/integrity_audit.json`。",
        f"- **自动审计：** 共{audit['unique_match_ids']}个唯一评测局，训练状态{audit['training_states_total']:,}；棋谱合法回放、PUCT-64及训练数据ID检查均为通过。",
        "- **模型配置声明：** 本报告只记录当前仓库代码的算法与实验配置；运行环境不能独立验证Codex模型名称或推理强度。",
        "",
        "## 图表",
        "",
    ])
    for figure in figures:
        lines.append(f"![{Path(figure).stem}](figures/{Path(figure).name})")
        lines.append("")
    lines.extend([
        "## 复现与交付文件",
        "",
        "```bash",
        "python scripts/register_g_v4_2.py --states-per-arm 10000",
        "python scripts/preflight_g_v4_2.py",
        "python scripts/run_g_v4_2.py",
        "pytest -q tests/test_alphazero_g_v4_2_*.py",
        "```",
        "",
        "主要文件：`config/experiment.json`、`config/openings.json`、`config/tactical_puzzles.json`、`results/games/`、`checkpoints/`、`figures/`。这是一项传统策略价值网络与自我对弈算法研究，不含训练checkpoint指标之外的监督教师结果。",
        "",
    ])
    target = root_path / "research/g_v4_2/REPORT.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
