from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.research.alphazero_g.v5_evaluation import summarize_paired_games
from src.research.alphazero_g.v6_3_evaluation import _complete_pair_rows
from src.research.alphazero_g.v6_3_training import (
    ARTIFACT, ARM, MILESTONES, sha_file, validate_training_state,
)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _games_for_condition(directory: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("v63-*.json"))]


def _trajectory(root: Path) -> list[dict]:
    artifact = root / ARTIFACT
    base_state_path = artifact / "checkpoints" / ARM / "training_state_games_2500.pt"
    base = torch.load(base_state_path, map_location="cpu", weights_only=False)
    rows = [{
        "games": 2500,
        "states": int(base["environment_states"]),
        "updates": int(base["update_count"]),
        "new_training_cpu_seconds": 0.0,
        "new_training_cpu_hours": 0.0,
        "checkpoint_sha256": sha_file(base_state_path),
    }]
    current_state = artifact / "checkpoints" / ARM / "training_state.pt"
    current = torch.load(current_state, map_location="cpu", weights_only=False)
    actual_games = int(current["game_count"])
    for games in MILESTONES:
        if games > actual_games:
            continue
        path = artifact / "checkpoints" / ARM / f"training_state_games_{games}.pt"
        if not path.is_file():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        cpu = float(state["cumulative_cpu_seconds"]) - float(base["cumulative_cpu_seconds"])
        rows.append({
            "games": games,
            "states": int(state["environment_states"]),
            "updates": int(state["update_count"]),
            "new_training_cpu_seconds": cpu,
            "new_training_cpu_hours": cpu / 3600.0,
            "checkpoint_sha256": sha_file(path),
        })
    if actual_games not in {row["games"] for row in rows}:
        cpu = float(current["cumulative_cpu_seconds"]) - float(base["cumulative_cpu_seconds"])
        rows.append({"games": actual_games, "states": int(current["environment_states"]),
                     "updates": int(current["update_count"]),
                     "new_training_cpu_seconds": cpu, "new_training_cpu_hours": cpu / 3600.0,
                     "checkpoint_sha256": sha_file(current_state)})
    return sorted(rows, key=lambda row: row["games"])


def _training_metrics(root: Path, games: int) -> dict:
    base_games = 2500
    history_path = root / ARTIFACT / "results/training" / f"{ARM}.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line] if history_path.is_file() else []
    selected = [row for row in history if base_games <= int(row["game_id"].rsplit("-", 1)[1]) < games]
    if not selected:
        return {"games_in_interval": 0}
    return {
        "games_in_interval": len(selected),
        "mean_loss": float(np.mean([row["loss"] for row in selected])),
        "mean_policy_loss": float(np.mean([row.get("policy_loss", 0.0) for row in selected])),
        "mean_value_loss": float(np.mean([row.get("value_loss", 0.0) for row in selected])),
        "mean_policy_entropy": float(np.mean([row["entropy"] for row in selected])),
        "last_lr_multiplier": float(selected[-1]["lr_multiplier"]),
    }


def _summaries(root: Path) -> dict[str, dict]:
    matches = root / ARTIFACT / "results/matches"
    result = {}
    for model in ("G2500", "G5000"):
        for opponent in ("A", "B", "C"):
            condition = f"{model}_vs_{opponent}"
            rows = _games_for_condition(matches / condition) if (matches / condition).is_dir() else []
            if rows:
                paired_rows, completed_pairs = _complete_pair_rows(rows)
                summary = summarize_paired_games(
                    paired_rows, seed=int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16)
                )
                summary["complete"] = len(rows) == 80 and completed_pairs == 40
                summary["completed_pairs"] = completed_pairs
                summary["raw_completed_games"] = len(rows)
                summary["mean_total_cpu_seconds"] = float(np.mean([row["total_cpu_seconds"] for row in rows]))
                result[condition] = summary
            else:
                result[condition] = {"games": 0, "complete": False, "completed_pairs": 0}
    return result


def _write_charts(root: Path, trajectory: list[dict], summaries: dict[str, dict]) -> None:
    charts = root / ARTIFACT / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    games = [row["games"] for row in trajectory]
    states = [row["states"] for row in trajectory]
    cpu = [row["new_training_cpu_hours"] for row in trajectory]
    fig, left = plt.subplots(figsize=(8.4, 5.0))
    left.plot(games, states, marker="o", color="#2878B5", label="累计训练状态")
    left.set_xlabel("累计自我对弈局数")
    left.set_ylabel("累计训练状态数", color="#2878B5")
    left.tick_params(axis="y", labelcolor="#2878B5")
    right = left.twinx()
    right.plot(games, cpu, marker="s", color="#D9534F", label="新增训练 CPU 小时")
    right.set_ylabel("从 G2500 起新增 CPU 小时", color="#D9534F")
    right.tick_params(axis="y", labelcolor="#D9534F")
    left.set_title("G V6.3 训练进度与计算成本")
    left.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(charts / "training_games_states_cpu.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    opponents = ["A 规则AI", "B Minimax 50ms", "C Alpha-Beta 50ms"]
    positions = np.arange(3)
    width = 0.34
    for offset, model in ((-width/2, "G2500"), (width/2, "G5000")):
        rates, lows, highs = [], [], []
        for opponent in ("A", "B", "C"):
            summary = summaries[f"{model}_vs_{opponent}"]
            rate = float(summary.get("score_rate", float("nan")))
            ci = summary.get("paired_opening_bootstrap_ci95", [float("nan"), float("nan")])
            rates.append(rate * 100 if np.isfinite(rate) else np.nan)
            lows.append((rate - ci[0]) * 100 if np.isfinite(rate) and np.isfinite(ci[0]) else np.nan)
            highs.append((ci[1] - rate) * 100 if np.isfinite(rate) and np.isfinite(ci[1]) else np.nan)
        ax.bar(positions + offset, rates, width, label=model, yerr=np.asarray([lows, highs]), capsize=4)
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, opponents)
    ax.set_ylim(0, 100)
    ax.set_ylabel("G 得分率（%）")
    ax.set_title("同一批配对开局：G2500 与 G5000 对 A/B/C")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "same_opening_challenge.png", dpi=170)
    plt.close(fig)


def build_report(root: str | Path = ROOT) -> dict:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    trajectory = _trajectory(root)
    summaries = _summaries(root)
    _write_charts(root, trajectory, summaries)
    with (artifact / "results/training_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)
    fields = ("condition", "games", "completed_pairs", "wins", "draws", "losses", "points",
              "score_rate", "paired_opening_score_rate", "black_score_rate",
              "white_score_rate", "paired_opening_bootstrap_ci95", "mean_moves",
              "mean_candidate_cpu_seconds", "mean_opponent_cpu_seconds", "mean_total_cpu_seconds", "complete")
    with (artifact / "results/match_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, summary in summaries.items():
            writer.writerow({"condition": condition, **summary})

    progress = _json(artifact / "results/v6_3_progress.json", {})
    challenge = _json(artifact / "results/final_challenge.json", {})
    drive = _json(artifact / "results/snapshot_status.json", {})
    preflight = _json(artifact / "results/preflight.json", {})
    audit = _json(artifact / "results/integrity_audit.json", {})
    current_state = artifact / "checkpoints" / ARM / "training_state.pt"
    current_validation = validate_training_state(current_state)
    state_validation_ok = all(current_validation.get(key, False) for key in (
        "model_restored", "adam_restored", "replay_restored", "python_rng_restored",
        "numpy_rng_restored", "torch_rng_restored",
    ))
    games_current = int(progress.get("games_completed", trajectory[-1]["games"]))
    is_training_complete = games_current == 5000
    challenge_complete = bool(challenge.get("complete"))
    g5000 = next((row for row in trajectory if row["games"] == 5000), None)
    g2500_metrics = _training_metrics(root, 2500)
    stage_metrics = {str(row["games"]): _training_metrics(root, row["games"]) for row in trajectory[1:]}
    new_training_cpu = float(trajectory[-1]["new_training_cpu_seconds"])
    matches_cpu = float(challenge.get("evaluation_cpu_seconds", 0.0))
    total_core_hours = (new_training_cpu + matches_cpu) / 3600.0
    registration = _json(artifact / "config/registration.json", {})
    result_lines = []
    for opponent, label in (("A", "A 原规则 AI"), ("B", "B 纯 Minimax 50ms"), ("C", "C Alpha-Beta 50ms")):
        for model in ("G2500", "G5000"):
            summary = summaries[f"{model}_vs_{opponent}"]
            if not summary.get("games"):
                result_lines.append(f"| {label} | {model} | 尚未运行 | — | — | — | — | — |")
                continue
            ci = summary.get("paired_opening_bootstrap_ci95", [None, None])
            result_lines.append(
                f"| {label} | {model} | {summary['wins']}-{summary['draws']}-{summary['losses']} "
                f"({summary['games']}局) | {summary['score_rate']:.1%} | "
                f"{summary['black_score_rate']:.1%} | {summary['white_score_rate']:.1%} | "
                f"[{ci[0]:.1%}, {ci[1]:.1%}] | {summary.get('mean_total_cpu_seconds', 0):.1f} |"
            )
    trajectory_lines = []
    for row in trajectory:
        metrics = stage_metrics.get(str(row["games"]), g2500_metrics if row["games"] == 2500 else {})
        trajectory_lines.append(
            f"| {row['games']} | {row['states']:,} | {row['updates']:,} | "
            f"{row['new_training_cpu_hours']:.2f} | {metrics.get('mean_loss', float('nan')):.3f} | "
            f"{metrics.get('mean_policy_entropy', float('nan')):.3f} |"
        )
    snapshot_rows = drive.get("verified_snapshots", [])
    snapshot_links = "\n".join(
        f"- 第{item['sequence']}份，G{item['games_completed']}："
        f"[Drive 快照]({item.get('drive_url', '')})，SHA-256 `{item['sha256']}`"
        for item in snapshot_rows
    ) or "尚无已验证快照。"
    report = f"""# WuziQ AlphaZero G V6.3：从 G2500 续训至 G5000

## 实验状态

- 目标：从冻结 G2500 恢复完整 AlphaZero 训练状态，保持 V6.2 算法、网络、400 次 PUCT、优化器、回放池、随机数和自我对弈规则不变。
- 当前累计：**G{games_current:,}**，累计训练状态 **{trajectory[-1]['states']:,}**，新增训练 CPU **{new_training_cpu / 3600:.2f} 小时**。
- 训练到 G5000：**{'完成' if is_training_complete else '未完成'}**；最终挑战赛：**{'完成' if challenge_complete else '未完成/未开始'}**。
- 实验新增 CPU（训练与已完成挑战）：**{total_core_hours:.2f} 小时**；注册硬上限 50 CPU 小时。
- 棋力结论只按同一批 40 个新开局、双方换色的直接对局判断。训练 loss 仅作为训练状态，不解释为棋力提升。

## 训练参数与起点

9×9自由五子棋，连五获胜；AlphaZero_Gomoku 冻结参考实现的4-plane CNN、PUCT c=5、400次模拟、Adam、batch 512、Replay 10,000、学习率与训练目标均沿用 V6.2/V5 注册配置。每局从空盘开始，当前网络执黑和执白自我对弈。没有加入教师数据、人工战术奖励或规则落子补丁。

G2500 起始恢复校验：权重 SHA-256 `{registration.get('source_model_sha256','')}`；完整状态 SHA-256 `{registration.get('source_checkpoint_sha256','')}`；2,500局、45,284状态、2,499次更新、10,000回放样本、16个Adam状态槽。权重、Adam、Replay Buffer、Python/NumPy/PyTorch RNG 的实际加载检查均通过。

## 资源预估和实耗

预检在当前机器实测：{preflight.get('cpu_logical_count','?')} 个逻辑CPU，容器CPU配额 {preflight.get('cpu_quota_cores','?')} 核，内存限制 8 GiB。开盘位置的PUCT-400单步实际耗时 {preflight.get('single_decision_benchmark',{}).get('measured_cpu_seconds',float('nan')):.2f} CPU秒。按 V6.2 最近500局实际总训练CPU {preflight.get('actual_training_rate_basis','')}；新2,500局加480场比赛的乐观/正常/保守总估算分别为 **{preflight.get('estimates',{}).get('total_cpu_hours',{}).get('optimistic',float('nan')):.1f} / {preflight.get('estimates',{}).get('total_cpu_hours',{}).get('normal',float('nan')):.1f} / {preflight.get('estimates',{}).get('total_cpu_hours',{}).get('conservative',float('nan')):.1f} CPU小时**。

## 训练进度

| 累计对局 | 累计训练状态 | 优化更新 | G2500后训练CPU小时 | 区间平均loss | 策略熵 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(trajectory_lines)}

![训练对局、状态和CPU变化](charts/training_games_states_cpu.png)

模型棋力并不从loss推断。该曲线展示训练规模和成本；棋力结论来自下方固定开局挑战赛。

## G2500 与最终模型挑战赛

开局表在训练开始前固定，40组各6手的合法非终局局面；相对于V5、V6、V6.2已登记开局的D4等价重叠为0。同一批开局分别供G2500/G5000对A/B/C使用，每个条件开局双方各执黑白，共80局。G棋手双方均为PUCT-400；A为原规则AI自然耗时，B为纯Minimax 50ms，C为Alpha-Beta 50ms。

积分为胜1、平0.5、负0。表中95%区间按40个开局配对进行bootstrap；不同对手分别汇总，不合并其预算或胜率。

| 对手 | G模型 | 胜-平-负 | 得分率 | 执黑得分率 | 执白得分率 | 配对95%区间 | 每局平均双方CPU秒 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

![同开局对A/B/C成绩](charts/same_opening_challenge.png)

### 逐组结论

""" + "\n".join(
        f"- **{label}**：G2500 {summaries[f'G2500_vs_{opponent}'].get('score_rate', float('nan')):.1%}；"
        f"G5000 {summaries[f'G5000_vs_{opponent}'].get('score_rate', float('nan')):.1%}；"
        f"同一配对开局得分率变化 "
        f"{(summaries[f'G5000_vs_{opponent}'].get('paired_opening_score_rate', float('nan')) - summaries[f'G2500_vs_{opponent}'].get('paired_opening_score_rate', float('nan'))):+.1%}。"
        if summaries[f'G2500_vs_{opponent}'].get('games') and summaries[f'G5000_vs_{opponent}'].get('games')
        else f"- **{label}**：挑战赛未完成，当前数据不用于最终比较。"
        for opponent, label in (("A", "A"), ("B", "B@50ms"), ("C", "C@50ms"))
    ) + f"""

## 训练状态与完整性

- 本阶段新增训练局：{len(list((artifact / 'raw/selfplay' / ARM).glob('*.json')))}；逐局棋谱和策略/价值训练数组保存在 `raw/selfplay/{ARM}/`。
- 检查点：G2500基线及G3000/G3500/G4000/G4500/G5000（未完成节点在审计中标明）；每500局保存完整权重、Adam、回放池和随机状态。
- 棋谱/ID/样本一致性审计：**{'通过' if audit.get('all_new_completed_games_legal_and_terminal') else '待最终审计'}**；训练完整状态校验：**{'通过' if state_validation_ok else '待最终审计'}**。
- Drive专用目录：[WuziQ G V6.3 Recovery]({drive.get('drive_folder_url','https://drive.google.com/drive/folders/'+str(drive.get('drive_folder_id','')))}); 保留最近两份通过下载、SHA-256和空目录恢复检查的快照：
{snapshot_links}

## 结论

""" + (
        "G5000已完成且三组挑战数据齐全。" if challenge_complete else
        "目前训练或挑战赛尚未全部完成，因此不将中间成绩包装成最终胜负结论。"
    ) + " 棋力是否提升以G2500/G5000在相同40组配对开局上的对照为准；如果置信区间仍跨过50%，结论应为证据不足，而非稳定提升。\n"

    report_path = artifact / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report_path": str(report_path), "training_complete": is_training_complete,
            "challenge_complete": challenge_complete, "new_training_cpu_hours": new_training_cpu / 3600,
            "model_comparisons": summaries,
            "trajectory": trajectory,
            "audit_present": bool(audit)}


def main() -> None:
    result = build_report(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
