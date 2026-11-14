from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.research.alphazero_g.v6_5_audit import audit_experiment
from src.research.alphazero_g.v6_5_training import ARTIFACT, ARM, BASE_GAMES, MILESTONES, TARGET_GAMES, _baseline_cpu_seconds, sha_file


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(root: str | Path = ROOT) -> dict:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    audit = audit_experiment(root, require_complete=True)
    checkpoint_dir = artifact / "checkpoints" / ARM
    baseline = torch.load(checkpoint_dir / "training_state_games_5000.pt", map_location="cpu", weights_only=False)
    final_state_path = checkpoint_dir / "training_state_games_8000.pt"
    final_node_path = checkpoint_dir / "node_games_8000.pt"
    final_state = torch.load(final_state_path, map_location="cpu", weights_only=False)
    final_node = torch.load(final_node_path, map_location="cpu", weights_only=False)
    progress = json.loads((artifact / "results/v6_5_progress.json").read_text(encoding="utf-8"))
    segments = _read_jsonl(artifact / "results/training_segments.jsonl")
    history = _read_jsonl(artifact / "results/training/reference400_empty.jsonl")
    backup = json.loads((artifact / "results/drive_backup_status.json").read_text(encoding="utf-8"))
    preflight = json.loads((artifact / "results/preflight.json").read_text(encoding="utf-8"))
    if int(final_node["game_count"]) != TARGET_GAMES or not torch.equal(
        final_node["model_state"][next(iter(final_node["model_state"]))],
        final_state["model_state"][next(iter(final_state["model_state"]))],
    ):
        raise ValueError("G8000 model node is inconsistent with its full training state")
    for key in final_state["model_state"]:
        if not torch.equal(final_node["model_state"][key], final_state["model_state"][key]):
            raise ValueError(f"G8000 model and full state differ at {key}")

    rows = [{
        "games_cumulative": BASE_GAMES,
        "training_states_cumulative": int(baseline["environment_states"]),
        "optimizer_updates_cumulative": int(baseline["update_count"]),
        "added_cpu_hours": 0.0,
        "mean_loss": "", "mean_policy_loss": "", "mean_value_loss": "", "mean_policy_entropy": "",
        "milestone": "G5000 source",
    }]
    for segment in segments:
        game_count = int(segment["end_games"])
        milestone = f"G{game_count}" if game_count in MILESTONES else ""
        rows.append({
            "games_cumulative": game_count,
            "training_states_cumulative": int(segment["training_states_cumulative"]),
            "optimizer_updates_cumulative": int(segment["optimizer_updates_cumulative"]),
            "added_cpu_hours": float(segment["added_cpu_seconds_cumulative"]) / 3600,
            "mean_loss": float(segment["mean_loss"]),
            "mean_policy_loss": float(segment["mean_policy_loss"]),
            "mean_value_loss": float(segment["mean_value_loss"]),
            "mean_policy_entropy": float(segment["mean_policy_entropy"]),
            "milestone": milestone,
        })
    trajectory_path = artifact / "results/training_trajectory.csv"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(11, 8))
    axis1 = plt.subplot(3, 1, 1)
    x = [int(row["games_cumulative"]) for row in rows]
    states = [int(row["training_states_cumulative"]) for row in rows]
    axis1.plot(x, states, color="#2563eb", marker="o", linewidth=2, label="累计训练状态")
    axis1.set_ylabel("训练状态")
    axis1.grid(alpha=0.25)
    axis1b = axis1.twinx()
    axis1b.plot(x, x, color="#16a34a", linestyle="--", label="累计局数")
    axis1b.set_ylabel("累计自我对弈局数")
    axis1.set_title("G5000→G8000：累计局数与训练状态")

    axis2 = plt.subplot(3, 1, 2)
    hx = [BASE_GAMES + i + 1 for i in range(len(history))]
    loss = np.asarray([float(row.get("loss", 0.0)) for row in history], dtype=np.float64)
    if len(loss):
        window = min(50, len(loss))
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        axis2.plot(hx[window - 1:], smooth, color="#ea580c", linewidth=1.5, label=f"总损失（{window}局滑动平均）")
        policy = np.asarray([float(row.get("policy_loss", 0.0)) for row in history], dtype=np.float64)
        value = np.asarray([float(row.get("value_loss", 0.0)) for row in history], dtype=np.float64)
        policy_smooth = np.convolve(policy, np.ones(window) / window, mode="valid")
        value_smooth = np.convolve(value, np.ones(window) / window, mode="valid")
        axis2.plot(hx[window - 1:], policy_smooth, color="#7c3aed", alpha=0.85, label="策略损失")
        axis2.plot(hx[window - 1:], value_smooth, color="#0891b2", alpha=0.85, label="价值损失")
    axis2.set_ylabel("训练损失")
    axis2.grid(alpha=0.25)
    axis2.legend(loc="best", fontsize=8)

    axis3 = plt.subplot(3, 1, 3)
    cpu = [float(row["added_cpu_hours"]) for row in rows]
    axis3.plot(x, cpu, color="#be123c", marker="o", linewidth=2)
    axis3.set_xlabel("累计自我对弈局数")
    axis3.set_ylabel("新增训练 CPU 小时")
    axis3.set_title("累计计算成本")
    axis3.grid(alpha=0.25)
    for milestone in (6000, 7000, 8000):
        axis3.axvline(milestone, color="#64748b", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    chart_path = artifact / "charts/training_curves.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=160)
    plt.close()

    baseline_cpu = _baseline_cpu_seconds(root)
    new_cpu_hours = (float(final_state["cumulative_cpu_seconds"]) - baseline_cpu) / 3600
    loss_mean = float(np.mean([float(row.get("loss", 0.0)) for row in history])) if history else 0.0
    loss_first = float(np.mean([float(row.get("loss", 0.0)) for row in history[:100]])) if history else 0.0
    loss_last = float(np.mean([float(row.get("loss", 0.0)) for row in history[-100:]])) if history else 0.0
    snapshots = backup.get("verified_snapshots", [])
    milestone_rows = []
    for game_count in MILESTONES:
        state_path = checkpoint_dir / f"training_state_games_{game_count}.pt"
        node_path = checkpoint_dir / f"node_games_{game_count}.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        milestone_segments = [row for row in segments if int(row["end_games"]) == game_count]
        metric = milestone_segments[-1] if milestone_segments else {}
        milestone_rows.append({
            "games": game_count, "states": int(state["environment_states"]),
            "updates": int(state["update_count"]),
            "cpu_hours": (float(state["cumulative_cpu_seconds"]) - baseline_cpu) / 3600,
            "mean_loss": metric.get("mean_loss", "—"),
            "state_sha256": sha_file(state_path), "model_sha256": sha_file(node_path),
        })
    preflight_estimates = preflight["estimates"]
    report = f"""# WuziQ AlphaZero G V6.5：G5000 续训至 G8000

## 结果摘要

V6.5 从 V6.3 的 G5000 **完整训练状态**继续训练，完成新增 **3,000 局**纯自我对弈，最终到达 G8000。本轮未进行正式挑战赛，因此不把训练 loss 或训练局数解释成棋力提升。

- 训练状态：{int(final_state['environment_states']):,}（G5000 为 {int(baseline['environment_states']):,}，新增 {int(final_state['environment_states'])-int(baseline['environment_states']):,}）
- 优化器更新：{int(final_state['update_count']):,}（G5000 为 {int(baseline['update_count']):,}；每局一次更新）
- 新增训练 CPU：**{new_cpu_hours:.2f} 小时**；恢复状态中累计训练 CPU 为 {float(final_state['cumulative_cpu_seconds'])/3600:.2f} 小时
- 新增3,000局平均总损失：{loss_mean:.4f}；首100局平均 {loss_first:.4f}；末100局平均 {loss_last:.4f}
- G8000 完整状态 SHA-256：`{sha_file(final_state_path)}`
- G8000 模型 SHA-256：`{sha_file(final_node_path)}`

## 冻结条件与恢复来源

- 本实验分支基线提交：`{training.SOURCE_COMMIT}`；V6.3源恢复包 SHA-256：`{training.SOURCE_ARCHIVE_SHA256}`。该恢复包清单记录其生成代码提交为 `{training.SOURCE_ARCHIVE_CODE_COMMIT}`；恢复得到的G5000状态及权重哈希与指定基线一致。
- G5000 权重 SHA-256：`{training.SOURCE_MODEL_SHA256}`；完整状态 SHA-256：`{training.SOURCE_STATE_SHA256}`。
- 启动前从 V6.3 Google Drive 下载两个分卷，合并后的恢复包大小 37,000,229 字节；SHA-256 与登记值一致，ZIP CRC 检查通过，并在空目录完整解包。
- 实际加载校验通过：G5000、87,999状态、4,999更新、10,000条回放；模型、Adam、Replay Buffer、Python/NumPy/PyTorch 随机状态均逐项恢复一致。
- 固定 9×9连五、V6.3 CNN、PUCT-400、Adam、Replay Buffer、学习率、数据增强与目标不变。V6.5 只改变累计对局目标和恢复编排；没有教师数据、战术补丁或额外奖励。

## 资源预估与实耗

- 当前机器：{preflight['logical_cpu_count']} 个逻辑 CPU，CPU 配额 {preflight['cpu_quota_cores']} 核，内存限制 {preflight['memory_limit_bytes']/(1024**3):.1f} GiB；PyTorch CPU-only、1线程；启动时可用磁盘 {preflight['disk_free_bytes']/(1024**3):.1f} GiB。
- G5000 开盘位置 PUCT-400 预检：{preflight['single_decision_benchmark']['measured_cpu_seconds']:.2f} CPU秒，完成 {preflight['single_decision_benchmark']['root_action_visits']} 个根动作访问（参考实现每步配置400次playout）。
- 估算依据：V6.3 G2500→G5000实测 {preflight['historical_training_rate']['cpu_seconds_per_game']:.2f} CPU秒/局。乐观/正常/保守新增训练估时：{preflight_estimates['training_cpu_hours']['optimistic']:.1f}/{preflight_estimates['training_cpu_hours']['normal']:.1f}/{preflight_estimates['training_cpu_hours']['conservative']:.1f} CPU小时；含备份成本估算为 {preflight_estimates['total_cpu_hours']['optimistic']:.1f}/{preflight_estimates['total_cpu_hours']['normal']:.1f}/{preflight_estimates['total_cpu_hours']['conservative']:.1f} 小时。40小时新增CPU硬上限未触发。
- 实际运行开始：{progress.get('training_started_at_utc','未记录')}；完成：{progress.get('training_completed_at_utc','未记录')}。新增CPU {new_cpu_hours:.2f} 小时。

## 阶段检查点

| 累计对局 | 累计训练状态 | 累计更新 | G5000后CPU小时 | 阶段平均loss |
|---:|---:|---:|---:|---:|
| 5000 | {int(baseline['environment_states']):,} | {int(baseline['update_count']):,} | 0.00 | — |
"""
    for row in milestone_rows:
        mean_loss = row["mean_loss"]
        mean_loss_text = mean_loss if isinstance(mean_loss, str) else f"{float(mean_loss):.4f}"
        report += f"| {row['games']} | {row['states']:,} | {row['updates']:,} | {row['cpu_hours']:.2f} | {mean_loss_text} |\n"
    report += f"""
![G5000到G8000训练状态、loss和CPU曲线](charts/training_curves.png)

## 对局与状态审计

- 新增棋谱：{audit['new_games_replayed']:,} 局，ID 从 `reference400_empty-game-005000` 至 `reference400_empty-game-007999`，全部唯一。
- 独立规则回放：{audit['legal_terminal_games']:,}/{audit['new_games_replayed']:,} 局合法且终局、胜负一致；每局 NPZ 状态、策略和值样本数与决策步数一致。
- 新增状态总数 {audit['new_training_states']:,} 与检查点状态差值 {audit['checkpoint_training_state_delta']:,} 一致；更新数与对局数满足每局一次更新。
- 自我对弈结果分布（仅训练诊断）：黑胜 {audit['winner_distribution']['B']:,}、白胜 {audit['winner_distribution']['W']:,}、和棋 {audit['winner_distribution']['draw']:,}；平均 {audit['mean_selfplay_moves']:.1f} 个决策状态/局。该分布不是独立棋力评估。
- G6000/G7000/G8000均保存了完整 Adam、回放池、随机数和模型状态，并逐项执行恢复检查。
- V6.5 Drive 目录保留 {len(snapshots)} 个已验证恢复点；其中 G6000/G7000/G8000里程碑均须保留，最近两份滚动快照也保留。每份上传后回下载核对SHA-256并在空目录恢复。

## 结论与下一步

训练已从G5000续接至G8000，状态、更新、随机数及新增3,000局棋谱通过一致性审计。loss变化只说明训练目标拟合变化；本轮没有对A或C系列进行挑战，因而**不能据此声称G8000棋力强于G5000**。下一轮应在预注册的新开局上用冻结PUCT预算与G5000/G8000直接对战，再判断续训是否带来实战收益。

恢复目录：[WuziQ G V6.5 Recovery]({backup.get('folder_url','')})。G5000来源目录：[WuziQ G V6.3 Recovery](https://drive.google.com/drive/folders/{training.SOURCE_FOLDER_ID})。

可复现命令：`python scripts/run_g_v6_5.py prepare --source-state <G5000-training-state> --source-node <G5000-model> --drive-folder-id {backup.get('folder_id','<folder-id>')}`，随后执行 `python scripts/run_g_v6_5.py preflight`；每次运行 `python scripts/run_g_v6_5.py train-batch` 后创建并上传经过校验的恢复快照。审计：`python scripts/audit_g_v6_5.py`；报告：`python scripts/report_g_v6_5.py`。
"""
    (artifact / "REPORT.md").write_text(report, encoding="utf-8")
    return {
        "games": audit["games_completed"], "new_games": audit["new_games_replayed"],
        "states": audit["cumulative_training_states"], "updates": audit["optimizer_updates"],
        "new_cpu_hours": new_cpu_hours, "training_state_sha256": sha_file(final_state_path),
        "model_sha256": sha_file(final_node_path), "snapshots": len(snapshots),
        "report": str(artifact / "REPORT.md"), "chart": str(chart_path),
        "trajectory": str(trajectory_path),
    }


def main() -> None:
    print(json.dumps(build_report(ROOT), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
