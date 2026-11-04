from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/g_v6"


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _fmt_pct(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _fmt_wdl(row: dict[str, Any] | None) -> str:
    if not row:
        return "未完成"
    return f"{row.get('wins', 0)}胜 {row.get('draws', 0)}平 {row.get('losses', 0)}负"


def _fmt_ci(row: dict[str, Any] | None) -> str:
    if not row or not row.get("paired_opening_bootstrap_ci95"):
        return "—"
    low, high = row["paired_opening_bootstrap_ci95"]
    return f"{float(low) * 100:.1f}%–{float(high) * 100:.1f}%"


def _set_font() -> None:
    candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False


def _stage_rows() -> list[dict[str, Any]]:
    rows = []
    for milestone in (2_000, 5_000, 10_000, 15_000, 20_000):
        row = _json(BASE / f"results/development/stage_{milestone}.json")
        if row:
            rows.append(row)
    return rows


def _generate_charts(stages: list[dict[str, Any]]) -> None:
    chart_dir = BASE / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    _set_font()
    complete_rows = [row for row in stages if row.get("comparisons")]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for budget, color in ((64, "#1769aa"), (400, "#c44e52")):
        xs, ys, lo, hi = [], [], [], []
        for row in complete_rows:
            summary = row.get("comparisons", {}).get(str(budget), {}).get("G0-V3-510901-10K")
            if summary:
                xs.append(float(row["actual_states"]) / 1000)
                ys.append(float(summary["score_rate"]) * 100)
                interval = summary.get("paired_opening_bootstrap_ci95", [None, None])
                lo.append(ys[-1] - float(interval[0]) * 100 if interval[0] is not None else 0)
                hi.append(float(interval[1]) * 100 - ys[-1] if interval[1] is not None else 0)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                        linewidth=2, color=color, label=f"PUCT-{budget} vs 冻结G0 10K")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%得分线")
    ax.set(xlabel="累计真实自我对弈状态（千）", ylabel="配对得分率（%）",
           title="G V6：9×9棋力随训练状态变化（固定开发开局）", ylim=(-5, 105))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "learning_states.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for budget, color in ((64, "#1769aa"), (400, "#c44e52")):
        xs, ys = [], []
        for row in complete_rows:
            summary = row.get("comparisons", {}).get(str(budget), {}).get("G0-V3-510901-10K")
            metric = row.get("training_metric", {})
            if summary and metric.get("cumulative_cpu_seconds") is not None:
                xs.append(float(metric["cumulative_cpu_seconds"]) / 3600)
                ys.append(float(summary["score_rate"]) * 100)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"PUCT-{budget} vs 冻结G0 10K")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%得分线")
    ax.set(xlabel="从随机初始化起累计自我对弈训练CPU小时", ylabel="配对得分率（%）",
           title="G V6：棋力随累计训练CPU变化", ylim=(-5, 105))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "learning_cpu.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for budget, color in ((64, "#1769aa"), (400, "#c44e52")):
        xs, ys = [], []
        for row in complete_rows:
            tactics = row.get("tactics", {}).get(str(budget))
            if tactics:
                xs.append(float(row["actual_states"]) / 1000)
                ys.append(float(tactics["puct_accuracy"]) * 100)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"PUCT-{budget}战术正确率")
    xs, ys = [], []
    for row in complete_rows:
        tactics = row.get("tactics", {}).get("64") or row.get("tactics", {}).get("400")
        if tactics:
            xs.append(float(row["actual_states"]) / 1000)
            ys.append(float(tactics["network_accuracy"]) * 100)
    if xs:
        ax.plot(xs, ys, marker="s", linestyle="--", color="#55a868", label="裸网络策略正确率")
    ax.set(xlabel="累计真实自我对弈状态（千）", ylabel="64道独立战术题正确率（%）",
           title="G V6：战术能力随训练状态变化", ylim=(0, 100))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "tactics_states.png", dpi=160)
    plt.close(fig)


def _v5_comparison() -> list[str]:
    arm_rows = _json(ROOT / "research/g_v5/results/9x9_arms.json", []) or []
    lines = []
    arm = next((row for row in arm_rows if row.get("arm") == "reference400_empty"), None)
    if arm:
        for stage in arm.get("stages", []):
            if stage.get("candidate") == "reference400_empty@1500":
                lines.append(
                    f"| V5 reference400@1500 | PUCT-64 | {stage.get('opponent', '—')} | "
                    f"{_fmt_wdl(stage)} | {_fmt_pct(stage.get('score_rate'))} | {_fmt_ci(stage)} |"
                )
    final = _json(ROOT / "research/g_v5/results/final_9x9.json", {}) or {}
    selected = final.get("selected_candidate", {}).get("candidate_id", "reference200_empty@2000")
    for row in final.get("blind_results", []):
        lines.append(
            f"| V5 {selected} | PUCT-64 | {row.get('opponent', '—')} | "
            f"{_fmt_wdl(row)} | {_fmt_pct(row.get('score_rate'))} | {_fmt_ci(row)} |"
        )
    return lines


def generate_report(root: Path = ROOT) -> str:
    global ROOT, BASE
    ROOT = root.resolve()
    BASE = ROOT / "research/g_v6"
    registration = _json(BASE / "config/registration.json", {}) or {}
    preflight = _json(BASE / "results/preflight.json", {}) or {}
    audit = _json(BASE / "results/integrity_audit.json", {}) or {}
    progress = _json(BASE / "results/progress.json", {}) or {}
    training = _json(BASE / "results/training_summary.json", {}) or {}
    stages = _stage_rows()
    blind = _json(BASE / "results/final_blind.json", {}) or {}
    _generate_charts(stages)

    total_cpu = float(progress.get("v6_cumulative_cpu_seconds", 0.0))
    current_states = int(training.get("environment_states", progress.get("actual_states", 1514)))
    added_states = max(0, current_states - int(registration.get("resume_initial_states", 1514)))
    progress_complete = bool(blind.get("complete")) and current_states >= int(registration.get("target_cumulative_states", 20_000))
    status = "完成" if progress_complete else "部分完成/可续跑"
    metric = {int(row["milestone"]): row.get("training_metric", {}) for row in stages}
    training_table = []
    for milestone in (2_000, 5_000, 10_000, 15_000, 20_000):
        row = next((item for item in stages if int(item["milestone"]) == milestone), None)
        if not row:
            training_table.append(f"| {milestone:,} | 未到达 | — | — | — | — | — |")
            continue
        tm = row.get("training_metric", {})
        training_table.append(
            f"| {milestone:,} | {int(row['actual_states']):,} | {int(row['games_completed']):,} "
            f"(V6新增 {int(row['games_completed']) - int(registration.get('resume_initial_games', 47)):,}) | "
            f"{int(row['optimizer_updates']):,} | {float(tm.get('loss', float('nan'))):.4f} | "
            f"{float(tm.get('entropy', float('nan'))):.4f} | "
            f"{float(tm.get('cumulative_cpu_seconds', 0.0)) / 3600:.2f} |"
        )

    development_table = []
    for row in stages:
        for budget in (64, 400):
            for opponent in ("ref-v6-initial", "G0-V3-510901-5K", "G0-V3-510901-10K",
                             "G0-V3-510901-20K", "A", "B-50ms"):
                summary = row.get("comparisons", {}).get(str(budget), {}).get(opponent)
                if not summary:
                    continue
                development_table.append(
                    f"| {int(row['actual_states']):,} | {budget} | {opponent} | {summary.get('games',0)} | "
                    f"{_fmt_wdl(summary)} | {_fmt_pct(summary.get('score_rate'))} | {_fmt_ci(summary)} | "
                    f"{_fmt_pct(summary.get('black_score_rate'))}/{_fmt_pct(summary.get('white_score_rate'))} | "
                    f"{float(summary.get('mean_candidate_cpu_seconds') or 0):.2f} |"
                )

    tactics_table = []
    for row in stages:
        for budget in (64, 400):
            tactics = row.get("tactics", {}).get(str(budget))
            if tactics:
                for category, summary in tactics.get("by_category", {}).items():
                    tactics_table.append(
                        f"| {int(row['actual_states']):,} | {budget} | {category} | {summary['puzzles']} | "
                        f"{_fmt_pct(summary.get('network_accuracy'))} | {_fmt_pct(summary.get('puct_accuracy'))} |"
                    )

    blind_rows = []
    for key in ("A|puct64", "A|puct400", "B-50ms|puct64", "B-50ms|puct400"):
        summary = blind.get("matches", {}).get(key)
        if not summary:
            blind_rows.append(f"| {key.replace('|', ' / ')} | 未完成 | — | — | — | — | — |")
        else:
            blind_rows.append(
                f"| {key.replace('|', ' / ')} | {summary.get('games')} | {_fmt_wdl(summary)} | "
                f"{_fmt_pct(summary.get('score_rate'))} | {_fmt_ci(summary)} | "
                f"{_fmt_pct(summary.get('black_score_rate'))}/{_fmt_pct(summary.get('white_score_rate'))} | "
                f"{float(summary.get('mean_candidate_cpu_seconds') or 0):.2f} |"
            )

    v5_rows = _v5_comparison()
    pre = preflight.get("workload_estimate", {}).get("estimated_total_cpu_hours", {})
    resource = preflight.get("resource", {})
    v6_train_cpu = float(training.get("v6_training_cpu_seconds", 0.0)) / 3600
    report = f"""# WuziQ G V6：9×9等预算AlphaZero训练验证

- 状态：**{status}**。分支：`research/alphazero-g-v6`。
- V5基线：`{registration.get('v5_baseline_commit', '—')}`；注册代码：`{registration.get('registered_code_commit', '—')}`。
- 规则与训练：9×9自由五子棋，连五获胜；V5标准参考实现，Adam、8重旋转/反射数据增强、原策略交叉熵+价值MSE、原PUCT和Replay Buffer；每步400次配置搜索。
- V5检查点续训：从累计{registration.get('resume_initial_states', '—'):,}状态、{registration.get('resume_initial_games', '—')}局继续，训练目标是累计20,000个未增强真实局面状态；完整局边界会产生少量超出，报告实际值。棋谱增强后样本与状态数分开统计。
- 训练预算：10 CPU小时硬上限。已消耗V6实验CPU约 **{total_cpu/3600:.2f}小时**，其中续训CPU约 **{v6_train_cpu:.2f}小时**。当前累计训练状态 **{current_states:,}**（V6新增{added_states:,}）。全任务wall-clock未由runner可靠汇总，不将CPU小时当作墙钟时间。
- CPU/内存：{resource.get('cpu_quota', '—')}核配额，内存上限{float(resource.get('memory_limit_bytes') or 0)/1024**3:.1f}GiB；设备CPU，PyTorch {resource.get('torch','—')}。

## 主要发现

{('完整盲测已完成。' if progress_complete else '实验尚未全部完成；本报告汇总已可靠落盘阶段，剩余部分可从V6完整训练状态和逐局ID续跑。')}

主要结论由盲测中的PUCT-64与PUCT-400分列判断。搜索次数不同的结果不混作网络学习收益。固定checkpoint的PUCT-64/400差别属于搜索收益；同预算不同训练节点的变化才用于分析训练收益。盲测开局只在开发集模型冻结后使用。

## V5与V6预算对照

| 检查点/实验 | 评测预算 | 对手 | 局数 | 胜平负 | 得分率 | 配对95%区间 |
|---|---:|---|---:|---|---:|---|
{chr(10).join(v5_rows) if v5_rows else '| — | — | V5数据缺失 | — | — | — | — |'}

V5的9×9 reference400路线在V5中只训练至约1.5K状态；V6延续其完整训练状态至20K累计目标，算法、网络、学习目标及训练参数保持冻结。V6开发集为8组开局换色；最终盲测为20组全新开局换色，每个对手在每档预算各40局。

## 训练节点与训练损失

| 注册节点 | 实际累计状态 | 累计自博弈局数（括号为V6新增） | 优化更新 | 节点loss | 策略熵 | 从随机初始化累计训练CPU小时 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(training_table)}

loss下降不作为棋力提升证据。策略熵和价值/策略损失在阶段结果中保存；不同阶段训练状态增加并不等同于相同更新数。

## 开发集：阶段棋力变化

每个节点以同一组已注册开发开局、黑白换色，对随机初始参考网络、冻结历史G0 5K/10K/20K、A及B-50ms评测。每个节点均用PUCT-64与PUCT-400；计分为胜1、平0.5、负0，区间按配对开局bootstrap计算。

| 实际状态 | 候选预算 | 对手 | 完成局数 | 胜平负 | 得分率 | 配对95%区间 | 黑/白得分率 | 候选平均CPU秒/局 |
|---:|---:|---|---:|---|---:|---|---|---:|
{chr(10).join(development_table) if development_table else '| — | — | 尚无完整开发评测 | — | — | — | — | — | — |'}

![棋力随累计训练状态变化](charts/learning_states.png)

![棋力随累计训练CPU变化](charts/learning_cpu.png)

## 战术题变化

| 实际状态 | 搜索预算 | 题型 | 题数 | 裸网络准确率 | 搜索后准确率 |
|---:|---:|---|---:|---:|---:|
{chr(10).join(tactics_table) if tactics_table else '| — | — | 尚未评测 | — | — | — |'}

![战术能力随累计训练状态变化](charts/tactics_states.png)

题集来自冻结的V4.2独立战术评估，仅用于评测；它没有进入训练回放或奖励。64与400搜索差值是计算搜索收益，不是学习收益。

## 最终盲测

| 冻结候选 / 对手 | 局数 | 胜平负 | 得分率 | 配对95%区间 | 黑/白得分率 | 候选平均CPU秒/局 |
|---|---:|---|---:|---|---|---:|
{chr(10).join(blind_rows)}

候选选择只使用PUCT-64开发集上对冻结G0-V3-510901-10K的配对得分；并列时选较早节点。`frozen_candidate.json`必须先于最终盲测生成，盲测结果不用于模型选择。40局/条件是探索性样本，区间可能较宽。

## 检查点恢复与完整性

- V5来源验证：模型权重与node_1500完全一致；Adam槽、10,000行Replay Buffer和Python/NumPy/PyTorch RNG逐项加载校验。V5原始状态SHA-256保持不变。
- 最终可续训状态：{json.dumps(audit.get('resumable_checkpoint', {}), ensure_ascii=False, sort_keys=True)}
- V6自博弈棋谱：{audit.get('v6_selfplay_games_verified', 0)}局；回放训练状态{audit.get('v6_training_states_verified', 0):,}。评测棋谱：{audit.get('v6_evaluation_games_verified', 0)}局；完整配对开局{audit.get('paired_evaluation_openings_complete', 0)}组；未完成配对{audit.get('paired_evaluation_openings_incomplete', 0)}组。
- 所有已完成棋谱经V5独立网格规则器回放：落子合法、终局一致、唯一ID无重复；训练策略目标概率归一且非法点权重为零。
- Drive最新V6恢复快照信息由`results/snapshot_status.json`登记，最终哈希与空目录恢复结果在此保存。

## 资源估算与实耗

预检实测400模拟完整局CPU {float(preflight.get('benchmark', {}).get('game_cpu_seconds', 0)):.2f}秒、{preflight.get('benchmark', {}).get('actual_decision_states', 0)}个决策状态；单次真实optimizer update {float(preflight.get('benchmark', {}).get('optimizer_update_cpu_seconds', 0)):.3f}秒。20K目标总CPU估算：乐观{float(pre.get('optimistic', 0)):.2f}小时、正常{float(pre.get('normal', 0)):.2f}小时、保守{float(pre.get('conservative', 0)):.2f}小时；实耗{total_cpu/3600:.2f}小时。

## 结论

{_conclusions(stages, blind, current_states, registration)}

## 复现

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/register_g_v6.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/preflight_g_v6.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6.py --through-stage 2000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6.py --through-stage 5000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6.py --through-stage 10000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6.py --through-stage 15000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6.py --through-stage 20000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6.py --finalize-blind
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/audit_g_v6.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/report_g_v6.py
```

结果、开局池、完整checkpoint、自博弈棋谱、训练样本及逐局评测保存在`research/g_v6/`。本轮不涉及神经网络训练以外的教师监督或规则补丁。
"""
    (BASE / "REPORT.md").write_text(report, encoding="utf-8")
    return report


def _conclusions(stages: list[dict[str, Any]], blind: dict[str, Any],
                 states: int, registration: dict[str, Any]) -> str:
    if states < 20_000:
        return f"V6当前到达{states:,}个累计状态，尚未达到20K目标；在达标前不对20K成长趋势作结论。"
    if not blind.get("complete"):
        return "训练已达到20K累计目标，但V6最终盲测尚未全部完成；不能据此判定能否可靠超过A/B。"
    sections = []
    for budget in (64, 400):
        for opponent in ("A", "B-50ms"):
            row = blind.get("matches", {}).get(f"{opponent}|puct{budget}", {})
            if row:
                sections.append(
                    f"PUCT-{budget}对{opponent}: {_fmt_wdl(row)}，得分率{_fmt_pct(row.get('score_rate'))}，"
                    f"配对95%区间{_fmt_ci(row)}"
                )
    return "最终盲测：" + "；".join(sections) + "。应结合各阶段同预算开发曲线判断是持续学习、只优于随机初始，还是仍无稳定增长。"


if __name__ == "__main__":
    generate_report()
    print(str(BASE / "REPORT.md"))

