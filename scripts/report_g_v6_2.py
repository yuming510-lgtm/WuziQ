from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.research.alphazero_g.v6_2_training import ARTIFACT, ROOT


def _read(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _pct(value):
    return "未完成" if value is None else f"{100 * float(value):.1f}%"


def _summary_table(stages, milestone_rows=None):
    milestone_rows = milestone_rows or {}
    lines = ["| 累计局数 | 累计状态 | 更新次数 | 对G20K PUCT-64 | 对G20K PUCT-400 | 对A PUCT-400 | 对B-50ms PUCT-400 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for games, stage in stages:
        counters = stage.get("stage_checkpoint_counters", {})
        c64 = stage.get("comparisons", {}).get("64", {}).get("G20K", {})
        c400 = stage.get("comparisons", {}).get("400", {}).get("G20K", {})
        a400 = stage.get("comparisons", {}).get("400", {}).get("A", {})
        b400 = stage.get("comparisons", {}).get("400", {}).get("B-50ms", {})
        updates = milestone_rows.get(int(games), {}).get("optimizer_updates_cumulative", "—")
        lines.append(f"| {games} | {counters.get('states', '—')} | {updates} | {_pct(c64.get('score_rate'))} | {_pct(c400.get('score_rate'))} | {_pct(a400.get('score_rate'))} | {_pct(b400.get('score_rate'))} |")
    return "\n".join(lines)


def _plot_learning(stages, milestones, out: Path) -> None:
    games, cpu, scores = [], [], []
    for game, stage in stages:
        comparison = stage.get("comparisons", {}).get("400", {}).get("G20K", {})
        if comparison.get("complete"):
            games.append(int(game))
            scores.append(float(comparison["score_rate"]))
            metric = next((row for row in milestones.get("milestones", [])
                           if int(row["games_completed"]) == int(game)), None)
            cpu.append(float(metric["cumulative_training_cpu_seconds"]) / 3600 if metric else None)
    if not games:
        return
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.plot(games, scores, marker="o", label="PUCT-400 對冻结 G20K")
    lows, highs = [], []
    for game in games:
        stage = dict(stages)[game]
        ci = stage["comparisons"]["400"]["G20K"]["paired_opening_bootstrap_ci95"]
        lows.append(float(ci[0])); highs.append(float(ci[1]))
    axis.fill_between(games, lows, highs, alpha=.18, label="配对开局 95% 区间")
    axis.axhline(.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("累计自我对弈局数")
    axis.set_ylabel("得分率")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.set_title("训练局数与对 G20K 的棋力")
    figure.tight_layout()
    figure.savefig(out / "strength_by_games.png", dpi=160)
    plt.close(figure)

    available = [(game, value) for game, value in zip(games, cpu) if value is not None]
    if available:
        figure, axis = plt.subplots(figsize=(8.2, 4.6))
        axis.plot([x[1] for x in available], [scores[games.index(x[0])] for x in available], marker="o")
        axis.axhline(.5, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("V6.2 新增训练 CPU 小时")
        axis.set_ylabel("对 G20K 得分率（PUCT-400）")
        axis.set_ylim(0, 1)
        axis.set_title("棋力与累计训练 CPU")
        figure.tight_layout()
        figure.savefig(out / "strength_by_cpu.png", dpi=160)
        plt.close(figure)


def _plot_budgets(stages, out: Path) -> None:
    labels = [game for game, stage in stages if stage.get("comparisons", {}).get("64", {}).get("G20K", {}).get("complete")]
    if not labels:
        return
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    for budget in ("64", "400"):
        values = [float(dict(stages)[game]["comparisons"][budget]["G20K"]["score_rate"])
                  if dict(stages)[game].get("comparisons", {}).get(budget, {}).get("G20K", {}).get("complete")
                  else float("nan") for game in labels]
        axis.plot(labels, values, marker="o", label=f"PUCT-{budget}")
    axis.axhline(.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("累计自我对弈局数")
    axis.set_ylabel("对 G20K 得分率")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.set_title("不同搜索预算下对 G20K 的得分率")
    figure.tight_layout()
    figure.savefig(out / "puct64_vs_400.png", dpi=160)
    plt.close(figure)


def _plot_tactics(stages, out: Path) -> None:
    available = []
    for game, stage in stages:
        data = stage.get("tactics", {}).get("400", {}).get("new_independent_128")
        if data and data.get("complete"):
            result = _read(ROOT / data["path"])
            available.append((int(game), result))
    if not available:
        return
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    games = [item[0] for item in available]
    axis.plot(games, [item[1]["network_top1_accuracy"] for item in available], marker="o", label="裸网络策略Top-1")
    axis.plot(games, [item[1]["puct_accuracy"] for item in available], marker="o", label="PUCT-400")
    axis.set_xlabel("累计自我对弈局数")
    axis.set_ylabel("独立战术题正确率")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.set_title("新独立战术集的网络与搜索表现")
    figure.tight_layout()
    figure.savefig(out / "tactics_by_games.png", dpi=160)
    plt.close(figure)


def _match_section(stage: dict, label: str) -> str:
    rows = [f"### {label}", "", "| 对手 | 预算 | 胜-平-负 | 得分率 | 配对95%区间 | 黑方得分率 | 白方得分率 | 平均CPU秒/局 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for budget in ("64", "400"):
        for opponent in ("G20K", "A", "B-50ms"):
            item = stage.get("comparisons", {}).get(budget, {}).get(opponent)
            if not item:
                rows.append(f"| {opponent} | {budget} | 未运行 | — | — | — | — | — |")
                continue
            ci = item.get("paired_opening_bootstrap_ci95", [None, None])
            rows.append(f"| {opponent} | {budget} | {item.get('wins', 0)}-{item.get('draws', 0)}-{item.get('losses', 0)} | {_pct(item.get('score_rate'))} | {_pct(ci[0])}–{_pct(ci[1])} | {_pct(item.get('black_score_rate'))} | {_pct(item.get('white_score_rate'))} | {float(item.get('mean_candidate_cpu_seconds') or 0):.3f} |")
    return "\n".join(rows)


def main() -> None:
    artifact = ROOT / ARTIFACT
    registration = _read(artifact / "config/registration.json", {})
    preflight = _read(artifact / "results/preflight.json", {})
    milestones = _read(artifact / "results/milestones.json", {"milestones": []})
    completed = []
    for path in sorted((artifact / "results/development").glob("stage_games_*.json"),
                       key=lambda path: int(path.stem.split("_")[-1])):
        stage = _read(path)
        completed.append((int(stage["game_count"]), stage))
    completed.sort()
    out = artifact / "charts"
    out.mkdir(parents=True, exist_ok=True)
    _plot_learning(completed, milestones, out)
    _plot_budgets(completed, out)
    _plot_tactics(completed, out)

    current = _read(artifact / "results/integrity_audit.json", {})
    gate = _read(artifact / "results/gate_2000.json", {})
    selection = _read(artifact / "results/selection.json", {})
    blind = _read(artifact / "results/final_blind.json", {})
    full = None
    checkpoint_path = artifact / "checkpoints/reference400_empty/training_state.pt"
    if checkpoint_path.is_file():
        import torch
        full = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_cpu = (float(full["cumulative_cpu_seconds"])
                 - float(registration.get("resume_cumulative_cpu_seconds", 0))) if full else 0.0
    games_done = int(full["game_count"]) if full else 1_012
    milestone_rows = {int(row["games_completed"]): row for row in milestones.get("milestones", [])}
    match_sections = "\n\n".join(_match_section(stage, f"累计 {game} 局") for game, stage in completed)
    blind_lines = []
    for budget, opponents in blind.get("matches", {}).items():
        for opponent, item in opponents.items():
            ci = item.get("paired_opening_bootstrap_ci95", [None, None])
            blind_lines.append(f"| {opponent} | {budget} | {item.get('wins', 0)}-{item.get('draws', 0)}-{item.get('losses', 0)} | {_pct(item.get('score_rate'))} | {_pct(ci[0])}–{_pct(ci[1])} | {item.get('paired_openings', 0)} |")
    blind_table = "\n".join(["| 对手 | 预算 | 胜-平-负 | 得分率 | 配对95%区间 | 配对开局数 |", "|---|---:|---:|---:|---:|---:|", *blind_lines])
    summary = current.get("cumulative_new_cpu_seconds_accounted")
    if summary is None:
        summary = train_cpu
    passed = bool(gate.get("passed"))
    report = f"""# WuziQ AlphaZero G V6.2：9×9 长期自我对弈训练

## 研究范围与冻结条件

本轮从冻结的 G20K 完整训练状态继续自我对弈。保持V6的9×9自由五子棋、参考实现4-plane CNN 32/64/128、400次PUCT模拟、`c_puct=5.0`、batch 512、10,000 replay buffer、Adam与其余训练参数不变。训练没有使用A/B或教师棋谱、外部对局、人工奖励、战术题或推理规则补丁。A、B、历史G0和战术题仅用于独立评测。

基线来自 V6 commit `{registration.get('base_v6_commit','')}`：{registration.get('resume_initial_games',1012):,}局、{registration.get('resume_initial_states',20001):,}状态、{registration.get('resume_initial_updates',1011):,}次优化更新。完整恢复验证见 [`results/resume_validation.json`](results/resume_validation.json)。模型、Adam、replay及Python/NumPy/PyTorch随机状态均逐项校验。

## 资源预检与实际成本

- CPU：3逻辑CPU，cgroup限额2核；内存限额8 GiB；CUDA不可用；PyTorch线程数1。
- V6实测新增965局训练消耗{preflight.get('history_basis',{}).get('v6_added_training_cpu_seconds',0)/3600:.2f} CPU小时，{preflight.get('history_basis',{}).get('v6_full_incremental_cpu_seconds_per_game',0):.2f} CPU秒/局。V6.1的1,280局比赛实测{preflight.get('history_basis',{}).get('v61_evaluation_cpu_seconds_per_game',0):.2f} CPU秒/局。
- 实测完整续训预检局耗时{preflight.get('actual_benchmark',{}).get('cpu_seconds',0):.2f} CPU秒；该局已作为V6.2第1,013局保留，没有丢弃或重复计入。
- 预检估算到2,000局为乐观/中性/保守{preflight.get('estimates_cpu_hours',{}).get('optimistic',{}).get('to_2000_total_cpu_hours',0):.1f}/{preflight.get('estimates_cpu_hours',{}).get('normal',{}).get('to_2000_total_cpu_hours',0):.1f}/{preflight.get('estimates_cpu_hours',{}).get('conservative',{}).get('to_2000_total_cpu_hours',0):.1f} CPU小时；若通过晋级条件并完成3,000局，分别为{preflight.get('estimates_cpu_hours',{}).get('optimistic',{}).get('to_3000_total_cpu_hours_if_gate_passes',0):.1f}/{preflight.get('estimates_cpu_hours',{}).get('normal',{}).get('to_3000_total_cpu_hours_if_gate_passes',0):.1f}/{preflight.get('estimates_cpu_hours',{}).get('conservative',{}).get('to_3000_total_cpu_hours_if_gate_passes',0):.1f}小时。
- 当前累计训练 CPU 为{train_cpu/3600:.2f}小时；从训练、比赛及战术搜索明细累计记账{float(summary)/3600:.2f} CPU小时（总上限30小时）。

预检文件：[`results/preflight.json`](results/preflight.json)。

## 训练阶段

完成至累计{games_done:,}局、{int(full['environment_states']) if full else 0:,}状态、{int(full['update_count']) if full else 0:,}次更新。实际训练检查点：

{_summary_table(completed, milestone_rows) if completed else '| 尚无阶段检查点 |'}

损失、策略损失、价值损失、策略熵、局长、累计状态、优化器更新和CPU记录在 [`results/milestones.json`](results/milestones.json) 与每局日志中。损失变化只描述训练目标拟合，不单独作为棋力增长证据。

## 开发集对局结果

同一模型和对手分别用PUCT-64与PUCT-400。每组使用合法新开局，并交换黑白。配对区间以开局为单位bootstrap。未运行的项目明确保留为未运行。

{match_sections or '开发集结果尚未生成。'}

![对G20K得分率与训练局数](charts/strength_by_games.png)

![对G20K得分率与训练CPU](charts/strength_by_cpu.png)

![PUCT-64和PUCT-400对照](charts/puct64_vs_400.png)

## 2,000局晋级门槛与最终模型选择

预注册主指标是累计2,000局模型在PUCT-400下对冻结G20K的80组配对开局得分率。只有得分率严格高于55%，且配对95%区间下限严格高于50%，才允许训练至2,500和3,000局。本轮门槛：**{'通过' if passed else '未通过或尚未完成'}**。门槛记录：[`results/gate_2000.json`](results/gate_2000.json)。

开发集模型选择只使用对G20K的PUCT-400配对得分率；精确并列时选择训练局数较少的模型。选择冻结：{selection.get('selected',{}).get('game_count','尚未冻结')}局模型。最终盲测不得回流模型选择。

## 独立盲测

最终候选使用预先冻结的新开局，分别挑战冻结G20K、A-Minimax与B-AB-50ms；每个条件40组换色开局，单独报告PUCT-64与PUCT-400。

{blind_table if blind_table.strip() else '盲测尚未运行。'}

## 战术能力

沿用V6旧64题，并加入V6.1构建的128题独立集；两者没有加入训练。裸网络策略Top-1和正确动作概率质量与PUCT访问分布、正确落点率分开记录。价值误差只在提供已求解局面价值的题目上计算。

![128题独立战术集](charts/tactics_by_games.png)

逐题结果：[`results/tactics/`](results/tactics/)；题目及SHA见[`config/registration.json`](config/registration.json)。

## 结果解读

V6.1确认20K优于10K/15K，但此前并未证明把训练从约1,000局增加到2,000–3,000局会继续成长。本轮判定只根据G20K直接交手的配对结果与预注册门槛。若未通过门槛，训练停在2,000局；对A/B的少量胜局、PUCT预算变化或loss下降不替代主要晋级标准。对传统算法的胜率也不能等同于相同计算成本下的比较，因为双方预算不同。

阶段检查点及其完整状态保存在[`checkpoints/reference400_empty/`](checkpoints/reference400_empty/)，逐局自我对弈和逐局比赛保存在[`raw/`](raw/)及[`results/matches/`](results/matches/)。所有原始对局经规则回放核验；完整性报告为[`results/integrity_audit.json`](results/integrity_audit.json)。

## 复现

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/audit_g_v6_2.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/report_g_v6_2.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6_2.py --train-through-games 1250
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v6_2.py --evaluate-games 1250
```

如需继续训练，先遵守2,000局门槛和单次/累计CPU上限。具体已完成阶段见 [`results/progress.json`](results/progress.json)。

## Git与Drive

- 分支：`research/alphazero-g-v6-2-selfplay-scale`
- 基线：`0405b288d44e0cc35cfefb08a17f0714a1b3683e`
- Drive恢复状态：见[`results/snapshot_status.json`](results/snapshot_status.json)。每个新快照需完成回下载、SHA-256和空目录恢复验证后才可替换旧版。

## 研究边界

本轮是AlphaZero自我对弈训练与实战评测，没有神经网络教师训练，也没有把增加搜索预算解释为网络学习收益。阶段比赛和小样本区间仍有不确定性，训练成功与否需依赖预注册配对检验。
"""
    (artifact / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {artifact / 'REPORT.md'}")


if __name__ == "__main__":
    main()
