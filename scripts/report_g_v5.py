from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/g_v5"


def read_json(path: Path, fallback: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def build_match_csv() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((BASE / "results/matches").rglob("v5m-*.json")):
        game = read_json(path)
        rows.append({
            "game_id": game["game_id"], "phase": path.parents[1].name,
            "candidate": game["candidate"], "opponent": game["opponent"],
            "opening_group_id": game["opening_group_id"], "candidate_color": game["candidate_color"],
            "winner_color": game["winner_color"], "points": game["points"],
            "move_count": game["move_count"], "total_cpu_seconds": game["total_cpu_seconds"],
            "candidate_cpu_seconds": game["candidate_cpu_seconds"],
            "opponent_cpu_seconds": game["opponent_cpu_seconds"],
        })
    output = BASE / "results/match_games.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["game_id", "phase", "candidate", "opponent"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return rows


def plot_training() -> None:
    metrics_dir = BASE / "results/training"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    any_data = False
    for path in sorted(metrics_dir.glob("*_metrics.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        any_data = True
        label = path.name.removesuffix("_metrics.jsonl")
        axes[0].plot([r["actual_states"] for r in rows], [r["loss"] for r in rows], marker="o", label=label)
        axes[1].plot([r["actual_states"] for r in rows], [r["cumulative_cpu_seconds"] / 3600 for r in rows], marker="o", label=label)
    axes[0].set_title("训练损失随实际状态数变化")
    axes[0].set_xlabel("实际自我对弈状态数")
    axes[0].set_ylabel("末次更新损失（仅诊断）")
    axes[1].set_title("训练CPU成本")
    axes[1].set_xlabel("实际自我对弈状态数")
    axes[1].set_ylabel("累计训练CPU小时")
    for axis in axes:
        axis.grid(alpha=0.25)
    if any_data:
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
    else:
        axes[0].text(0.5, 0.5, "无训练阶段数据", ha="center", va="center", transform=axes[0].transAxes)
    fig.tight_layout()
    destination = BASE / "charts/training_states_cpu.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_scores() -> None:
    historical = read_json(BASE / "results/historical_9x9.json", {})
    names, scores, low, high = [], [], [], []
    for row in historical.get("comparisons_vs_A_B", []):
        names.append(f"{row['candidate']}\nvs {row['opponent']}")
        scores.append(row["score_rate"])
        low.append(row["paired_opening_bootstrap_ci95"][0])
        high.append(row["paired_opening_bootstrap_ci95"][1])
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.65), 5))
    if names:
        x = list(range(len(names)))
        err = [[s - l for s, l in zip(scores, low)], [h - s for s, h in zip(scores, high)]]
        ax.errorbar(x, scores, yerr=err, fmt="o", capsize=3)
        ax.set_xticks(x, names, rotation=55, ha="right", fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("配对得分率（95% bootstrap区间）")
        ax.set_title("历史G0检查点：同一PUCT-64评测开局")
    else:
        ax.text(0.5, 0.5, "历史检查点复测尚无完整结果", ha="center", va="center", transform=ax.transAxes)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    destination = BASE / "charts/historical_g0_9x9.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    config = read_json(BASE / "config/experiment.json", {})
    summary = read_json(BASE / "results/experiment_summary.json", {})
    preflight = read_json(BASE / "results/preflight.json", {})
    integrity = read_json(BASE / "results/integrity_audit.json", {})
    six = summary.get("six_by_six", read_json(BASE / "results/six_by_six_summary.json", {}))
    historical = summary.get("historical_g0_remeasurement", read_json(BASE / "results/historical_9x9.json", {}))
    final = summary.get("final_9x9", read_json(BASE / "results/final_9x9.json", {}))
    arm_summary = summary.get("nine_by_nine_arms", read_json(BASE / "results/9x9_arms.json", {}))
    all_games = build_match_csv()
    plot_training(); plot_scores()

    training_rows = []
    for path in sorted((BASE / "results/training").glob("*_metrics.jsonl")):
        for row in read_jsonl(path):
            training_rows.append((path.name.removesuffix("_metrics.jsonl"), row))
    lines = [
        "# WuziQ G V5：标准 AlphaZero 复现与 G0 学习闭环验证",
        "",
        f"- 分支：`{config.get('branch', 'research/alphazero-g-v5')}`；基线：`{config.get('base_commit', '')}`。",
        f"- 当前状态：**{'完成' if summary.get('completed') else '未完成/预算或运行阶段停止'}**。",
        f"- 预注册配置 SHA-256：`{config.get('config_sha256', '尚未注册')}`。",
        f"- 正式CPU硬上限：{config.get('compute', {}).get('cpu_hour_hard_cap', '—')}小时；已记录CPU：{summary.get('cpu_seconds', 0) / 3600:.3f}小时。",
        f"- 完整对局数据：{len(all_games)}局，其中自我对弈训练局{len(list((BASE / 'raw/selfplay').rglob('*.json')))}局；训练状态{sum(int(r.get('actual_states', 0)) for _, r in training_rows)}个阶段累计计数（非去重总数）。",
        f"- 完整性审计：{integrity.get('evaluation_games_verified', '未执行')}局评测、{integrity.get('selfplay_games_verified', '未执行')}局自我对弈回放。",
        "",
        "## 参考实现来源与差异",
        "",
        "采用 `junxiaosong/AlphaZero_Gomoku`，固定上游提交 `" + str(config.get("reference", {}).get("commit", "")) + "`，MIT许可。源文件原样保存在 `third_party/AlphaZero_Gomoku/` 并逐文件校验 SHA-256。使用上游4平面、32/64/128通道CNN及原MCTS、Adam、8重几何增强、策略交叉熵+价值MSE目标。兼容层仅把过时 `loss.data[0]` 改为 `.item()`，并对推理启用 `inference_mode`；搜索、损失和训练目标未改。上游README中的学习规模/棋力宣传不视为本实验复现结果。",
        "",
        "参考实现本身没有 BatchNorm。原G0的网络为3平面、32通道、2残差块并含BatchNorm；因此两者棋力比较是独立算法/架构对照，不是只替换MCTS的同构消融。历史G0的5K/10K/20K、540402@30K和V4.2 history-pool@10K权重按预注册清单哈希冻结，A/B只作为评测对手。",
        "",
        "## 资源预检与预注册预算",
        "",
        f"环境：{preflight.get('resources', {}).get('cpu_quota_cores', '—')}核CPU配额、{(preflight.get('resources', {}).get('memory_limit_bytes') or 0) / (1024**3):.1f} GiB内存、GPU={'可用' if preflight.get('resources', {}).get('gpu_available') else '不可用'}、PyTorch {preflight.get('resources', {}).get('torch', '—')}。单局试跑测得6×6/400模拟 {next((x['selfplay_cpu_seconds'] for x in preflight.get('benchmark_games', []) if x['board_size']==6), 0):.2f}s；9×9/64、200、400的实测速度和搜索深度见 `results/preflight.json`。",
        f"预检估算：乐观{preflight.get('estimates', {}).get('optimistic_cpu_hours', 0):.2f}、正常{preflight.get('estimates', {}).get('normal_cpu_hours', 0):.2f}、保守{preflight.get('estimates', {}).get('conservative_cpu_hours', 0):.2f} CPU小时；估算使用每种预算一局自我对弈吞吐量、批512训练步耗时，并预留1.5 CPU小时给评测/审计。",
        "",
        "## 6×6连四学习闭环",
        "",
    ]
    if six:
        lines += [
            f"- 实际训练状态：{six.get('arm', {}).get('environment_states', 0)}；局数：{six.get('arm', {}).get('game_count', 0)}；优化更新：{six.get('arm', {}).get('update_count', 0)}；训练CPU：{six.get('arm', {}).get('cumulative_cpu_seconds', 0) / 3600:.3f}小时。",
            f"- 开发集选择节点：{six.get('dev_selected_node')}；是否满足进入9×9的预注册门槛：**{six.get('continue_to_9x9_gate_passed', False)}**；门槛节点：{six.get('gate_nodes', [])}。",
            f"- 独立盲测：{six.get('blind_result') or '未执行'}。",
        ]
        lines.append("| 节点 | 对冻结随机初始模型 W-D-L | 得分率 | 配对95%区间 | 立即胜/单点必防网络准确率 | PUCT-64准确率 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in six.get("stages", []):
            match = row.get("match", {})
            ci = match.get("paired_opening_bootstrap_ci95", [None, None])
            tactic = f"{row.get('network_tactics', 0):.1%}/{row.get('puct64_tactics', 0):.1%}" if row.get("network_tactics") is not None else "—"
            lines.append(f"| {row.get('node')} | {match.get('wins', 0)}-{match.get('draws', 0)}-{match.get('losses', 0)} | {pct(match.get('score_rate'))} | {pct(ci[0])}–{pct(ci[1])} | {tactic} | {pct(row.get('puct64_tactics'))} |")
    else:
        lines.append("尚无6×6阶段结果。")
    lines += [
        "",
        "战术题由独立精确五连规则生成，只用于测试，不进入训练。64题覆盖立即取胜和唯一必防，各32个不同母局面并做D4去重；训练状态仅由当前参考网络自我对弈生成。",
        "",
        "## 9×9历史G0复测与主实验",
        "",
        "评测统一使用64次有效根动作访问、`c_puct=5`、无根噪声。历史G0与参考模型使用同一开发/盲测开局池并换色；参考自我对弈训练的200/400档单列。",
        "",
        "| 冻结G0 | 对手 | W-D-L | 得分率 | 配对95%区间 | 平均步数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in historical.get("comparisons_vs_A_B", []):
        ci = row.get("paired_opening_bootstrap_ci95", [None, None])
        lines.append(f"| {row.get('candidate')} | {row.get('opponent')} | {row.get('wins', 0)}-{row.get('draws', 0)}-{row.get('losses', 0)} | {pct(row.get('score_rate'))} | {pct(ci[0])}–{pct(ci[1])} | {row.get('mean_moves', 0):.1f} |")
    for row in historical.get("direct_checkpoint_matches", []):
        ci = row.get("paired_opening_bootstrap_ci95", [None, None])
        lines.append(f"| {row.get('candidate')} | {row.get('opponent')} | {row.get('wins', 0)}-{row.get('draws', 0)}-{row.get('losses', 0)} | {pct(row.get('score_rate'))} | {pct(ci[0])}–{pct(ci[1])} | {row.get('mean_moves', 0):.1f} |")
    if not historical.get("comparisons_vs_A_B"):
        lines.append("| 尚无完整历史复测 | — | — | — | — | — |")
    lines += ["", "历史模型成绩95%区间以开局为配对单位bootstrap，属于本实验样本范围内的不确定性描述。", ""]

    arms = arm_summary.get("arms", []) if isinstance(arm_summary, dict) else []
    if arms:
        lines += ["### 参考实现训练路线", "", "| 路线 | 搜索预算 | 分布 | 实际状态 | 自我对弈局数 | CPU小时 | 完成 |", "|---|---:|---|---:|---:|---:|---|"]
        for arm in arms:
            train = arm.get("training", {})
            lines.append(f"| {arm.get('arm')} | {arm.get('simulations_per_move')} | {arm.get('opening_distribution')} | {train.get('environment_states', 0)} | {train.get('game_count', 0)} | {train.get('cumulative_cpu_seconds', 0)/3600:.3f} | {train.get('completed', False)} |")
    else:
        lines += ["### 参考实现训练路线", "", f"9×9训练={'未运行：6×6门槛未通过' if final.get('skipped') else '尚无完整结果'}。"]
    lines += [
        "",
        "若执行9×9搜索预算对照，固定同一训练checkpoint：64/200/400次评测与战术题结果见 `results/final_9x9.json`。增加模拟次数只算搜索带来的变化；只有同预算64的训练前后比较才计入网络学习收益。历史中盘重启仅改变自我对弈的起始状态分布，生成目标仍全部来自参考网络自身搜索。",
        "",
        "## 图表",
        "",
        "![训练状态与CPU成本](charts/training_states_cpu.png)",
        "",
        "![历史G0检查点配对结果](charts/historical_g0_9x9.png)",
        "",
        "## 交叉审计与结果解释",
        "",
        f"- 参考开源文件哈希：{integrity.get('vendored_reference_files_verified', '尚未核验')}个；自我对弈棋谱回放：{integrity.get('selfplay_games_verified', '尚未核验')}局；评测棋谱回放：{integrity.get('evaluation_games_verified', '尚未核验')}局。",
        "- 上游策略价值目标、当前行动方视角、终局正负号和两边独立规则判定由自动测试及逐局回放核验。兼容警告仅为上游旧版 `log_softmax` 未显式传 `dim`，其输出在当前版本可正常训练；本分支没有修改 vendored 上游文件。",
        "- 6×6只有一个预注册随机种子；若越过门槛，只能说明本次小预算闭环取得探索性证据，不能宣称跨种子稳定。9×9阶段若被跳过，不将6×6结果外推为9×9棋力。",
        "- 若历史G0检查点相对排序与旧报告不同，本报告仅称为当前统一开局/PUCT协议下的复测结果，不追溯覆盖历史冠军记录。",
        "",
        "## 结论",
        "",
    ]
    gate = bool(six.get("continue_to_9x9_gate_passed"))
    if gate:
        lines += [
            "1. **标准AlphaZero能否从零学会？** 6×6门槛通过，表明参考实现从随机权重在本次单种子任务上持续胜过其随机初始模型；证据仍是探索性。",
            f"2. **原G0为何未稳定成长？** 本实验通过同规复测历史节点与交叉实现检查搜索回传，不足以单独给出原G0的唯一根因。日志中的根搜索深度、价值误差、历史节点成绩和代码检查构成候选证据；仅报告可复现差异，不将相关性当根因。",
            f"3. **是否找到新路线？** 独立参考实现以同一PUCT-64进行9×9评估，盲测结果：{final.get('blind_results', '尚无完整结果')}。",
            "4. **下一步建议：** 若参考模型没有稳定超过冻结A或历史G0，不扩大训练，优先复核这次跨实现定位到的具体失败模式，并用多随机种子验证。",
        ]
    else:
        lines += [
            "1. **标准AlphaZero能否从零学会？** 本次6×6结果未满足预注册晋级门槛；不能据此断言方法在原则上不能学习，但本环境/预算下没有得到足够证据证明稳定成长。",
            "2. **原G0为何未稳定成长？** 当前证据不支持唯一根因断言。报告保留训练曲线、棋谱回放、交叉实现审计和历史节点复测，区分已观察事实与尚未验证假设。",
            "3. **是否找到真正有效的新路线？** 尚无达到预注册门槛的证据；未运行的9×9训练路线明确不计为负胜绩，也不把搜索次数增加称为学习。",
            "4. **下一轮应继续还是扩大训练？** 暂不扩大9×9训练；先多种子复核6×6学习闭环，检查策略访问目标和价值校准是否形成可重复提升。",
        ]
    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/preflight_g_v5.py",
        "PYTHONPATH=. python -m src.research.alphazero_g.v5_registration",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v5.py --session-cpu-seconds 3600",
        "PYTHONPATH=. python scripts/audit_g_v5.py",
        "PYTHONPATH=. python scripts/report_g_v5.py",
        "```",
        "",
        "同一注册配置可以重复运行上述实验命令；训练在完整对局边界原子保存并按唯一ID去重。`--session-cpu-seconds`只安排中断点，不改变全任务10 CPU小时硬上限。",
        "",
    ]
    (BASE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {BASE / 'REPORT.md'}; games={len(all_games)}; completed={summary.get('completed')}")


if __name__ == "__main__":
    main()
