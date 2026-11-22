"""Generate the V6.7 Chinese report and reproducible trend figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import v6_7_training as training

STAGE_NAMES = {8_000: "G8000", 10_000: "G10000", 12_000: "G12000", 16_000: "G16000", 20_000: "G20000"}


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _summaries(root: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    base = training.artifact_path(root) / "results/evaluations/summaries"
    result = {}
    for path in sorted(base.glob("*.json")) if base.is_dir() else []:
        row = json.loads(path.read_text(encoding="utf-8"))
        result[(int(row["candidate"][1:]), str(row["opponent"]), str(row["pool"]))] = row
    return result


def _fmt_rate(row: dict[str, Any] | None) -> str:
    if not row:
        return "未测"
    lo, hi = row["paired_opening_bootstrap_ci95"]
    return f"{row['wins']}-{row['draws']}-{row['losses']}；{row['score_rate']:.1%} [{lo:.1%}, {hi:.1%}]"


def _write_training_csv(root: Path, stages: list[dict[str, Any]], baseline: dict[str, Any]) -> Path:
    path = training.artifact_path(root) / "results/training_curve.csv"
    rows = [{"games": 8_000, "training_states": baseline["environment_states"],
             "optimizer_updates": baseline["optimizer_updates"], "training_cpu_hours": 0.0,
             "mean_loss_since_previous": None, "mean_policy_loss_since_previous": None,
             "mean_value_loss_since_previous": None, "mean_policy_entropy_since_previous": None}]
    for stage in stages:
        rows.append({
            "games": stage["games"], "training_states": stage["training_states_cumulative"],
            "optimizer_updates": stage["optimizer_updates_cumulative"],
            "training_cpu_hours": stage["new_training_cpu_seconds"] / 3600,
            "mean_loss_since_previous": stage.get("interval_mean_loss"),
            "mean_policy_loss_since_previous": stage.get("interval_mean_policy_loss"),
            "mean_value_loss_since_previous": stage.get("interval_mean_value_loss"),
            "mean_policy_entropy_since_previous": stage.get("interval_mean_policy_entropy"),
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _charts(root: Path, stages: list[dict[str, Any]], summaries: dict[tuple[int, str, str], dict[str, Any]],
            baseline: dict[str, Any]) -> list[str]:
    artifact = training.artifact_path(root)
    chart_dir = artifact / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
    games = [8_000] + [int(row["games"]) for row in stages]
    training_hours = [0.0] + [float(row["new_training_cpu_seconds"]) / 3600 for row in stages]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(games, training_hours, marker="o", linewidth=2, color="#2864a5")
    ax.set(title="训练规模与累计新增CPU时间", xlabel="累计自我对弈局数", ylabel="新增训练 CPU 小时")
    ax.grid(alpha=.25)
    fig.tight_layout()
    p1 = chart_dir / "training_cpu_curve.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    cumulative_states = [int(baseline["environment_states"])] + [int(row["training_states_cumulative"]) for row in stages]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(games, cumulative_states, marker="o", linewidth=2, color="#3c9850")
    ax.set(title="累计训练状态数", xlabel="累计自我对弈局数", ylabel="累计训练状态")
    ax.grid(alpha=.25)
    fig.tight_layout()
    p_states = chart_dir / "training_states_curve.png"
    fig.savefig(p_states, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = [int(row["games"]) for row in stages]
    for key, label, color in (("interval_mean_loss", "总损失", "#2864a5"),
                              ("interval_mean_policy_loss", "策略损失", "#dc7f1e"),
                              ("interval_mean_value_loss", "价值损失", "#3c9850")):
        y = [row.get(key) for row in stages]
        if any(value is not None for value in y):
            ax.plot(x, y, marker="o", label=label, color=color)
    ax.set(title="各训练阶段平均损失", xlabel="累计自我对弈局数", ylabel="损失")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    p2 = chart_dir / "training_loss_curve.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    opponent_styles = {"G8000": ("G8000", "#4d4d4d"), "A": ("规则AI A", "#31965a"),
                       "C@50ms": ("C@50ms", "#dc7f1e"), "C@200ms": ("C@200ms", "#b44343")}
    for opponent, (label, color) in opponent_styles.items():
        xs, ys = [], []
        for node in (10_000, 12_000, 16_000, 20_000):
            row = summaries.get((node, opponent, "development"))
            if row:
                xs.append(node)
                ys.append(float(row["score_rate"]))
        if xs:
            ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=2)
        blind = summaries.get((20_000, opponent, "blind"))
        if blind:
            ax.scatter([20_000], [blind["score_rate"]], marker="x", s=85, color=color,
                       label=f"{label} 盲测", zorder=4)
    ax.axhline(.5, color="#777777", linestyle="--", linewidth=1)
    ax.set(title="G模型分阶段对战得分率（PUCT-400）", xlabel="训练局数", ylabel="挑战者得分率")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    p3 = chart_dir / "match_score_trends.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    return [str(p1.relative_to(root)), str(p_states.relative_to(root)),
            str(p2.relative_to(root)), str(p3.relative_to(root))]


def build_report(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = training.artifact_path(root)
    registration = _load_json(training.registration_path(root), {})
    preflight = _load_json(artifact / "results/preflight.json", {})
    stages_obj = _load_json(artifact / "results/stage_metrics.json", {"stages": []})
    stages = sorted(stages_obj.get("stages", []), key=lambda row: int(row["games"]))
    baseline = _load_json(artifact / "results/baseline.json", {})
    summaries = _summaries(root)
    charts = _charts(root, stages, summaries, baseline)
    curve_csv = _write_training_csv(root, stages, baseline)
    state = training.validate_training_state(root= root)
    progress = _load_json(training.progress_path(root), {})
    accounting = _load_json(training.accounting_path(root), {})
    completed_games = int(state["game_count"])
    complete = completed_games == training.TARGET_GAMES
    total_cpu = training.total_new_cpu_seconds(root)
    try:
        started_wall = time.strptime(progress.get("started_at_utc", ""), "%Y-%m-%dT%H:%M:%SZ")
        wall_elapsed_seconds = max(0.0, time.time() - time.mktime(started_wall))
    except (ValueError, TypeError):
        wall_elapsed_seconds = 0.0
    baseline_games = int(baseline.get("games", training.BASE_GAMES))

    stage_table = [
        "| 检查点 | 累计状态 | 累计更新 | 阶段总损失 | 策略损失 | 价值损失 | 策略熵 | 新增训练CPU小时 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| G8000 | {baseline.get('environment_states', '—')} | {baseline.get('optimizer_updates', '—')} | — | — | — | — | 0.00 |",
    ]
    for row in stages:
        def fmt(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.4f}"
        stage_table.append(
            f"| G{row['games']} | {row['training_states_cumulative']} | {row['optimizer_updates_cumulative']} | "
            f"{fmt('interval_mean_loss')} | {fmt('interval_mean_policy_loss')} | {fmt('interval_mean_value_loss')} | "
            f"{fmt('interval_mean_policy_entropy')} | {row['new_training_cpu_seconds']/3600:.2f} |"
        )

    match_rows = [
        "| 阶段 | 对手 | 胜-平-负 | 得分率与配对95%区间 | 执黑得分率 | 执白得分率 | 平均步数 | G CPU秒/局 | 对手CPU秒/局 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for node in training.MILESTONES:
        for opponent in training.EVAL_OPPONENTS:
            row = summaries.get((node, opponent, "development"))
            if not row:
                continue
            match_rows.append(
                f"| G{node} 开发 | {opponent} | {row['wins']}-{row['draws']}-{row['losses']} | {_fmt_rate(row)} | "
                f"{row['black_score_rate']:.1%} | {row['white_score_rate']:.1%} | {row['mean_moves']:.1f} | "
                f"{row['mean_candidate_cpu_seconds']:.2f} | {row['mean_opponent_cpu_seconds']:.2f} |"
            )
    for opponent in training.EVAL_OPPONENTS:
        row = summaries.get((20_000, opponent, "blind"))
        if row:
            match_rows.append(
                f"| G20000 盲测 | {opponent} | {row['wins']}-{row['draws']}-{row['losses']} | {_fmt_rate(row)} | "
                f"{row['black_score_rate']:.1%} | {row['white_score_rate']:.1%} | {row['mean_moves']:.1f} | "
                f"{row['mean_candidate_cpu_seconds']:.2f} | {row['mean_opponent_cpu_seconds']:.2f} |"
            )

    c_metrics = {}
    games_dir = artifact / "results/evaluations/games"
    for path in sorted(games_dir.glob("*.json")) if games_dir.is_dir() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        opponent = record.get("opponent")
        if opponent not in {"C@50ms", "C@200ms"}:
            continue
        for move in record.get("moves", []):
            if move.get("actor") == opponent:
                row = c_metrics.setdefault(opponent, {"moves": 0, "cpu": [], "depth": [], "nodes": [], "timeouts": 0})
                row["moves"] += 1
                row["cpu"].append(float(move.get("cpu_seconds", 0.0)))
                row["depth"].append(int(move.get("search_depth", 0)))
                row["nodes"].append(int(move.get("search_nodes", 0)))
                row["timeouts"] += int(bool(move.get("timed_out", False)))

    c_table = ["| C预算 | 实际落子数 | 平均完成深度 | 平均节点 | 平均CPU秒/步 | 超时比例 |",
               "|---|---:|---:|---:|---:|---:|"]
    for opponent in ("C@50ms", "C@200ms"):
        row = c_metrics.get(opponent)
        if not row:
            continue
        c_table.append(f"| {opponent} | {row['moves']} | {np.mean(row['depth']):.2f} | {np.mean(row['nodes']):.0f} | "
                       f"{np.mean(row['cpu']):.4f} | {row['timeouts']/row['moves']:.1%} |")

    preflight_lines = []
    if preflight:
        tf = preflight.get("training_forecast", {})
        ef = preflight.get("evaluation_forecast", {})
        total = preflight.get("total_forecast_cpu_hours", {})
        preflight_lines = [
            f"- 资源：{preflight.get('resources', {}).get('cpu_count_visible')} 个可见 CPU，配额 `{preflight.get('resources', {}).get('cpu_quota')}`；内存上限 `{preflight.get('resources', {}).get('memory_limit_bytes')}` 字节；磁盘可用 {preflight.get('resources', {}).get('disk_free_bytes', 0)/1024**3:.1f} GiB。",
            f"- G5000→G8000历史实测：{tf.get('historical_games')} 局耗费 {tf.get('historical_cpu_seconds', 0)/3600:.2f} CPU小时，即 {tf.get('historical_cpu_seconds_per_game', 0):.2f} CPU秒/局；剩余12,000局训练预估 {tf.get('optimistic_training_cpu_hours', 0):.1f}/{tf.get('expected_training_cpu_hours', 0):.1f}/{tf.get('conservative_training_cpu_hours', 0):.1f} CPU小时（乐观/中性/保守）。",
            f"- 对局预检外推{ef.get('planned_formal_games', 0)}局评测成本 {ef.get('optimistic_cpu_hours', 0):.1f}/{ef.get('normal_cpu_hours', 0):.1f}/{ef.get('conservative_cpu_hours', 0):.1f} CPU小时；连同打包/审计总预估 {total.get('optimistic_cpu_hours', 0):.1f}/{total.get('normal_cpu_hours', 0):.1f}/{total.get('conservative_cpu_hours', 0):.1f} CPU小时。",
        ]

    summary_eval = _load_json(artifact / "results/evaluation_audit.json", {})
    audit = _load_json(artifact / "results/integrity_audit.json", {})
    drive = _load_json(artifact / "results/drive_backup_status.json", {})
    total_matches = summary_eval.get("games", 0)
    report = f"""# WuziQ AlphaZero G V6.7：持续训练至20K

## 状态

- 训练基线：冻结 G8000；模型 SHA-256 `{training.SOURCE_MODEL_SHA256}`，完整训练状态 SHA-256 `{training.SOURCE_STATE_SHA256}`。
- 当前：G{completed_games}，累计 {state['environment_states']:,} 个训练状态、{state['optimizer_updates']:,} 次更新；新增训练 {progress.get('new_training_cpu_seconds', 0)/3600:.2f} CPU小时。
- 完成状态：{'完成G20000' if complete else '未完成；报告仅覆盖已验收进度'}。本轮未修改网络、PUCT-400、Adam、Replay Buffer、学习率或训练目标。
- 从 G8000 恢复登记至本次报告生成的墙钟时长约 {wall_elapsed_seconds/3600:.2f} 小时；其中包含训练、评测、备份及等待时间。
- 新增数据来源为同一G网络的纯自我对弈；未使用教师模型、人工棋谱、战术奖励或规则补丁。
- Codex 运行时的具体模型档位无法从实验容器独立核实。

## 预检与预算

{chr(10).join(preflight_lines) if preflight_lines else '未找到预检记录。'}

任务新增CPU硬上限为110小时，其中训练阶段最多100小时，为评测与验收保留余量。实际新增CPU约 {total_cpu/3600:.2f} 小时（训练状态内累计CPU、正式评测、审计与本地快照打包之和；Drive传输墙钟时间单列）。

## 检查点与训练指标

以下loss/entropy为相邻里程碑间的均值；它们描述优化过程，不单独证明棋力提升。

{chr(10).join(stage_table)}

![训练CPU曲线](charts/training_cpu_curve.png)

![累计训练状态曲线](charts/training_states_curve.png)

![训练损失曲线](charts/training_loss_curve.png)

## 配对对局

开发赛每个阶段、每个对手使用同一组10个固定六手开局并交换执黑/执白。G20000冻结后额外使用10个从未参与开发选模的开局做盲测。表中区间是按开局配对的自助法95%描述区间；每条件仅20局，属于探索性结果。

对手：G8000、规则AI A、Alpha-Beta C@50ms、C@200ms。C严格为 `max_depth=69`、`candidate_radius=2`，超时返回最后完整搜索层。

{chr(10).join(match_rows) if len(match_rows) > 2 else '尚无正式阶段对局。'}

![对局得分趋势](charts/match_score_trends.png)

### C实际搜索成本

{chr(10).join(c_table) if len(c_table) > 2 else '尚无C系列对局记录。'}

正式评测总计 {total_matches} 局。盲测仅在G20000后执行，且不用于修改权重或训练方案。

## 数据完整性

- 训练棋谱独立按9×9连五规则回放，核对唯一ID、确定性种子、状态数、终局、NPZ形状与胜负视角。
- 训练状态检查模型、Adam、10,000行Replay及Python/NumPy/PyTorch RNG；各里程碑检查点单独恢复校验。
- 评测棋谱逐手回放；验证换色成对、胜负、计分及C的搜索参数。
- 当前完整性审计：`{'通过' if audit.get('valid') else '尚未完成/未通过'}`；训练回放 {audit.get('training', {}).get('legal_terminal_games', 0)} 局；评测回放 {summary_eval.get('games', 0)} 局；评测条件完整：{summary_eval.get('complete', False)}。
- 最新Drive文件夹：[WuziQ G V6.7 Recovery](https://drive.google.com/drive/folders/{training.V67_DRIVE_FOLDER_ID})；源G8000包 SHA 与本轮恢复包均记录在 `results/drive_backup_status.json`。快照中仅保留恢复必需代码、状态、进度、棋谱和评测记录。

## 结论

由于本轮固定了网络、PUCT预算和训练算法，G8000到各里程碑的同预算配对成绩才可作为训练规模对照。10对开局的阶段评测以及最终盲测样本均有限；若配对区间覆盖50%，应报告“证据不足以区分”，不能视为胜率变化已稳定。对A或C的结果用于描述目前能力边界，不能将跨阶段小样本的偶然起伏解释为确定成长或退化。

## 复现

```bash
python scripts/prepare_g_v6_7.py --source-state <G8000-training-state.pt> --source-node <G8000-node.pt>
python scripts/preflight_g_v6_7.py
python scripts/run_g_v6_7_batch.py
python scripts/evaluate_g_v6_7.py --games 10000 --pool development
python scripts/audit_g_v6_7.py
python scripts/report_g_v6_7.py
```

Drive恢复使用 `scripts/restore_g_v6_7_snapshot.py <verified-snapshot.zip> <empty-directory>`；多段文件先按清单顺序拼接，再运行恢复和SHA校验。
"""
    report_path = artifact / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report": str(report_path), "report_bytes": report_path.stat().st_size,
            "chart_paths": charts, "training_curve_csv": str(curve_csv),
            "completed_games": completed_games, "evaluation_games": total_matches,
            "wall_elapsed_hours": wall_elapsed_seconds / 3600,
            "total_new_cpu_hours": total_cpu / 3600, "complete": complete}

