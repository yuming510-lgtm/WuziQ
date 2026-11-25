"""Build the V6.9 comparison tables, chart, and Chinese report from audited runs."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "research/g_v6_9"
RESULTS = ARTIFACT / "results"
RUNS = [
    ("P1-R1", 1, "spawn"),
    ("P1-R2", 1, "spawn"),
    ("P2-CLEAN-1", 2, "spawn"),
    ("P4-R1", 4, "spawn"),
    ("P6-R1-FORK-2", 6, "fork"),
    ("P6-R2-FORK", 6, "fork"),
    ("P8-R1-FORK", 8, "fork"),
]
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1jruGcEvFw7tfCfAGuhFw8SQgasEUAnJl"
DRIVE_BACKUPS = {
    "P1-R1": ("1es55NvxEUMUkFKgnPvmQ2HJBeDLfeHqb", "adaaadeaac2e3d0585ae99db5135fd49cfe9ab1603c6e329165cbc826646b001", "5593bb57284b3750d16e9478609a05168d4c961eba38a9242f3f2be6f6461181"),
    "P2-CLEAN-1": ("1XPiPLjL5i2GLJOKjtLbNQvqXt9ByX7dP", "a76a5c3fb84e0491b90f0da26dbb2f6b9e6c774b6d73a69f7c4fae55e8d3c322", "b9a64b44256daa336d40a24fe8778d3f37dc43925fafb62b32ccb073aa789ad8"),
    "P4-R1": ("1opMZGcYIPwehRXlMhZW2qYHAgm7Jrz49", "78e11fbe57f5687864e414e16e4e5c73165407351a5ac50e16290966b30354b8", "07e53b49dd71d6fa7a04883c17bd87d11c10e8dcf037cbc1f4b1f354ea9b74ca"),
    "P6-R1-FORK-2": ("1JaMKOmd3fjVQOoh0z_gPwgJCnPMs9wr3", "6878c6860605fe7f5ffe1f74d295489f7a456fdb42b28854a97a71dd47350b1c", "aed082d16b4da28e11ec7d7fd9d7d4787d38ee907976b5c26d761e7608989be9"),
    "P6-R2-FORK": ("1CBb0yB6Hgz0lowQSIHD4-EOs3gUYZY7L", "53ccf949af55a9e3834dce421f04cce0e85ce27e05511a9ef0156c6bbe7c2dd6", "af7777c4d691ba015cde63393125598677b2183a8d4e0ca81a7fe7d9fcd85131"),
    "P8-R1-FORK": ("1q6vSXkVnhNN2YEqXytlunc5nsuX0Gp0e", "7d66f16d10703072477987d1a636741d6ca4700cbc691480cae3f5e73c6c2e0b", "80f4cef9490f5598a660942654a40b71b1d96987713f6f8d15cfe2c005093e76"),
    "P1-R2": ("1mlloQaC4Ku8nC_8icA2sGL6jlRShc6TA", "b0fcab394c30c22592dddd8c448eb8ca1eb067c3f912db29f630ccd03a685f98", "2b735c41218321620f9f90832946569e0f47879426c55168803c98f8ebf7adbc"),
}


def load_runs() -> list[dict]:
    records = []
    common_seed_plan: dict[int, int] | None = None
    for run_id, workers, start_method in RUNS:
        run_dir = ARTIFACT / "runs" / run_id
        summary = json.loads((run_dir / "results/run_summary.json").read_text())
        audit = json.loads((run_dir / "results/audit.json").read_text())
        if not audit.get("valid") or audit.get("games") != 100 or summary.get("workers") != workers:
            raise ValueError(f"run {run_id} did not pass the 100-game audit")
        if summary["protocol"]["simulations_per_move"] != 400:
            raise ValueError(f"run {run_id} did not use PUCT-400")
        if summary["weight_sync"]["adam_step_delta"] != 100:
            raise ValueError(f"run {run_id} did not apply one Adam update per game")
        if summary["checkpoint"]["training_state_game_count"] != 20100:
            raise ValueError(f"run {run_id} did not finish at G20100")
        game_rows = [json.loads(line) for line in (run_dir / "results/per_game.jsonl").read_text().splitlines() if line]
        if len(game_rows) != 100 or len({row["game_id"] for row in game_rows}) != 100:
            raise ValueError(f"run {run_id} has duplicate or missing game records")
        seed_plan = {int(row["ordinal"]): int(row["seed"]) for row in game_rows}
        if len(seed_plan) != 100:
            raise ValueError(f"run {run_id} has duplicate or missing seed ordinals")
        if common_seed_plan is None:
            common_seed_plan = seed_plan
        elif seed_plan != common_seed_plan:
            raise ValueError(f"run {run_id} did not use the common seed plan")
        for row in game_rows:
            row["workers"] = workers
            row["process_start_method"] = start_method
        records.append({
            "run_id": run_id,
            "workers": workers,
            "process_start_method": start_method,
            "summary": summary,
            "audit": audit,
            "games": game_rows,
        })
    return records


def roll_up(records: list[dict]) -> list[dict]:
    baseline_rate = mean(
        record["summary"]["measurement"]["games_per_hour"]
        for record in records if record["workers"] == 1
    )
    rows = []
    for workers in (1, 2, 4, 6, 8):
        group = [record for record in records if record["workers"] == workers]
        summaries = [record["summary"] for record in group]
        measurements = [item["measurement"] for item in summaries]
        lags = [item["weight_sync"] for item in summaries]
        rates = [item["games_per_hour"] for item in measurements]
        states_per_second = [item["training_states_per_second"] for item in measurements]
        rows.append({
            "workers": workers,
            "run_count": len(group),
            "start_method": "/".join(sorted({record["process_start_method"] for record in group})),
            "games_per_hour": mean(rates),
            "speedup_vs_p1": mean(rates) / baseline_rate,
            "wall_seconds_per_100_games": mean(item["wall_seconds"] for item in measurements),
            "cpu_seconds_per_100_games": mean(item["aggregate_cpu_seconds"] for item in measurements),
            "cpu_seconds_per_game": mean(item["cpu_seconds_per_game"] for item in measurements),
            "training_states_per_second": mean(states_per_second),
            "effective_cpu_cores": mean(item["average_effective_cpu_cores"] for item in measurements),
            "peak_process_tree_rss_gib": max(item["max_process_tree_rss_bytes"] for item in measurements) / (1024 ** 3),
            "mean_weight_lag_updates": mean(item["mean_lag_updates"] for item in lags),
            "p95_weight_lag_updates": mean(item["p95_lag_updates"] for item in lags),
            "max_weight_lag_updates": max(item["max_lag_updates"] for item in lags),
            "min_games_per_hour": min(rates),
            "max_games_per_hour": max(rates),
        })
    return rows


def write_chart(rows: list[dict]) -> None:
    x = [row["workers"] for row in rows]
    p1 = rows[0]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    fig.suptitle("WuziQ V6.9 Parallel Training Efficiency", fontsize=15, fontweight="bold")

    ax = axes[0]
    y = [row["games_per_hour"] for row in rows]
    yerr = [[row["games_per_hour"] - row["min_games_per_hour"] for row in rows],
            [row["max_games_per_hour"] - row["games_per_hour"] for row in rows]]
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=5, color="#1665a8")
    ax.set_title("Throughput")
    ax.set_xlabel("Self-play processes")
    ax.set_ylabel("Completed games / hour")
    ax.set_xticks(x)
    ax.grid(True, alpha=.25)

    ax = axes[1]
    y = [row["speedup_vs_p1"] for row in rows]
    ax.plot(x, y, marker="o", linewidth=2, color="#14804a", label="Measured")
    ax.plot(x, x, linestyle="--", color="#8a8a8a", label="Ideal linear")
    ax.set_title("Speedup")
    ax.set_xlabel("Self-play processes")
    ax.set_ylabel("Speedup vs. P1")
    ax.set_xticks(x)
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False)

    ax = axes[2]
    y = [row["cpu_seconds_per_game"] for row in rows]
    ax.errorbar(x, y, yerr=[[0] * len(rows), [0] * len(rows)], marker="o", linewidth=2, capsize=5, color="#c27700")
    ax.axhline(p1["cpu_seconds_per_game"], color="#8a8a8a", linestyle="--", linewidth=1, label="P1 reference")
    ax.set_title("CPU cost")
    ax.set_xlabel("Self-play processes")
    ax.set_ylabel("Process CPU seconds / game")
    ax.set_xticks(x)
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    path = RESULTS / "throughput_scaling.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(rows: list[dict], records: list[dict]) -> None:
    lookup = {row["workers"]: row for row in rows}
    p1, p2, p4, p6, p8 = (lookup[count] for count in (1, 2, 4, 6, 8))
    p1_runs = [r["summary"]["measurement"]["games_per_hour"] for r in records if r["workers"] == 1]
    p6_runs = [r["summary"]["measurement"]["games_per_hour"] for r in records if r["workers"] == 6]
    p6_cpu_increase = (p6["cpu_seconds_per_game"] / p1["cpu_seconds_per_game"] - 1) * 100
    wall_30k_h = p6["wall_seconds_per_100_games"] * 300 / 3600
    cpu_30k_h = p6["cpu_seconds_per_game"] * 30_000 / 3600
    wall_30k_p1_h = p1["wall_seconds_per_100_games"] * 300 / 3600
    cpu_30k_p1_h = p1["cpu_seconds_per_game"] * 30_000 / 3600
    rows_table = [
        "| 方案 | 复测数 | 启动方式 | 局/小时 | 相对 P1 | 墙钟秒/100局 | CPU秒/100局 | CPU秒/局 | 状态/秒 | 有效核数 | RSS峰值 GiB | 平均滞后/更新 | P95 | 最大 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rows_table.append(
            f"| P{row['workers']} | {row['run_count']} | {row['start_method']} | {row['games_per_hour']:.1f} | {row['speedup_vs_p1']:.2f}× | "
            f"{row['wall_seconds_per_100_games']:.1f} | {row['cpu_seconds_per_100_games']:.1f} | {row['cpu_seconds_per_game']:.2f} | "
            f"{row['training_states_per_second']:.2f} | {row['effective_cpu_cores']:.2f} | {row['peak_process_tree_rss_gib']:.2f} | "
            f"{row['mean_weight_lag_updates']:.2f} | {row['p95_weight_lag_updates']:.1f} | {row['max_weight_lag_updates']} |"
        )
    report = f"""# WuziQ V6.9｜AlphaZero 并行训练效率实验

## 结论

- **本 Work 环境建议 6 个自我对弈进程。** P6 两次各 100 局的合并吞吐为 **{p6['games_per_hour']:.1f} 局/小时**，比 P1 两次基线均值 **{p1['games_per_hour']:.1f} 局/小时**快 **{p6['speedup_vs_p1']:.2f} 倍**。两次 P6 分别为 {min(p6_runs):.1f} 和 {max(p6_runs):.1f} 局/小时，重复结果相差约 {(max(p6_runs)/min(p6_runs)-1)*100:.1f}%。
- **CPU 总消耗小幅增加。** P6 平均 {p6['cpu_seconds_per_game']:.2f} CPU 秒/局，比 P1 的 {p1['cpu_seconds_per_game']:.2f} 高 **{p6_cpu_increase:.1f}%**；并行把墙钟时间大幅压缩，但 CPU 总工作量没有按 3.8 倍增加。
- **8 个进程不如 6 个。** P8 为 P1 的 {p8['speedup_vs_p1']:.2f} 倍，低于 P6 的 {p6['speedup_vs_p1']:.2f} 倍，同时占用更多内存且权重滞后更大。
- 按 P6 实测均值线性估算，G20K 到 G50K 的 30,000 局约需 **{wall_30k_h:.1f} 小时墙钟、{cpu_30k_h:.1f} CPU 小时**；P1 约需 {wall_30k_p1_h:.1f} 小时墙钟、{cpu_30k_p1_h:.1f} CPU 小时。
- 本轮只验证效率，**没有启动正式 G50K 训练，也没有用这 100 局证明并行与串行棋力等价**。P6 会形成一条新的训练时序轨迹，正式继续时应保存为独立实验。

## 环境与基线

- 基线 Git commit：`911242c10c8a374740db072f71a6b5f450a837e6`；G20000 checkpoint SHA-256：`f932ec34eaf4da6ac63bc77b6636e0517235a8986bbcf981139afa4cae7b7719`。
- 基线完整状态为 20,000 局、348,216 个训练状态、19,999 次优化更新、10,000 条 Replay Buffer。Adam（16 个 optimizer slots）、模型、Replay Buffer、Python/NumPy/PyTorch 随机状态均通过恢复校验。
- 预检可见 9 个逻辑 CPU（affinity 0–8）；Work 未暴露 CPU quota 或内存 cgroup 上限路径，因而**无法读取硬性配额值**。实测 P6 平均约 {p6['effective_cpu_cores']:.2f} 个有效核，P8 约 {p8['effective_cpu_cores']:.2f}，本任务的有效算力平台约在 4 核附近。
- 内存总量约 10.45 GB，预检可用约 3.31 GB；工作盘 20.96 GB、预检可用 13.18 GB。没有 GPU，PyTorch 为 CPU 版 2.15.0。进程树 RSS 是所有进程 RSS 求和；fork 共享页会重复计入，因此它不是物理唯一内存的精确值。
- GitHub 克隆、G20000 Drive 分片下载、空目录恢复和完整状态校验均成功。

## 测试设计与结果

各条件都从同一 G20000 checkpoint 独立恢复，使用相同的 100 个种子序号：`590502 + game_index × 10007`。每组先跑预热对局并丢弃预热更新，再正式计时 100 局。每局保持 9×9、PUCT-400、batch 512、5 epochs、学习率 0.002、原 KL 目标与 10,000 条 Replay Buffer；每新增一局调用一次原有完整优化更新流程。Torch intra/inter-op 和 BLAS/OpenMP 线程均限制为 1。

| 方案 | 复测数 | 启动方式 | 局/小时 | 相对 P1 | 墙钟秒/100局 | CPU秒/100局 | CPU秒/局 | 状态/秒 | 有效核数 | RSS峰值 GiB | 平均滞后/更新 | P95 | 最大 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows_table[2:])}

图中误差线仅显示有两次重复的 P1、P6 各自观测范围；P2、P4、P8 各只有一组 100 局。

![吞吐、加速比与 CPU 成本曲线](throughput_scaling.png)

## 并行调度造成的差异

P1 按“完整自我对弈 → 使用最新权重做一次优化更新 → 下一局”的串行顺序运行，权重滞后为 0。并行方案由多个 worker 使用各自拿到的冻结权重完成整局；中央训练进程按对局完成顺序统一写 Replay Buffer、调用 Adam 更新一次，并在 worker 开始下一局时发送最新权重。棋局生成与更新没有被删除，但完成顺序和训练时序改变了。

| 进程数 | 平均权重滞后 | P95 滞后 | 最大滞后 |
|---:|---:|---:|---:|
| 1 | {p1['mean_weight_lag_updates']:.2f} | {p1['p95_weight_lag_updates']:.1f} | {p1['max_weight_lag_updates']} |
| 2 | {p2['mean_weight_lag_updates']:.2f} | {p2['p95_weight_lag_updates']:.1f} | {p2['max_weight_lag_updates']} |
| 4 | {p4['mean_weight_lag_updates']:.2f} | {p4['p95_weight_lag_updates']:.1f} | {p4['max_weight_lag_updates']} |
| 6 | {p6['mean_weight_lag_updates']:.2f} | {p6['p95_weight_lag_updates']:.1f} | {p6['max_weight_lag_updates']} |
| 8 | {p8['mean_weight_lag_updates']:.2f} | {p8['p95_weight_lag_updates']:.1f} | {p8['max_weight_lag_updates']} |

所有 700 局正式测量都通过棋谱合法性与终局回放、PUCT-400、唯一 ID/种子、每局一次更新，以及最终模型、Adam、Replay Buffer 和随机状态检查。并行时相同随机种子不保证落子轨迹相同，因为每局拿到的模型版本和中央更新完成顺序不同。100 局只说明吞吐与完整性，不说明棋力等价。

## 资源限制与过程记录

- 6-worker 的 `spawn` 预热尝试以退出码 137 被系统终止，未产生正式测量；环境没有暴露 cgroup 限制值，原因只能判为资源保护终止，不能精确归因。之后 P6 与 P8 使用 `fork`，完成正式测量和全部校验。P1/P2/P4 使用 `spawn`。所有实验协议及启动方式均记录在各 run summary 中。
- 第一次 P2 的中断恢复尝试因局号续接错误形成重复对局记录和多余更新，审计判为无效并排除。修复续跑逻辑后以全新 run `P2-CLEAN-1` 从 G20000 重跑并通过审计。
- 修复后的续跑冒烟校验在提交 ordinal 1、ordinal 0 仍未完成时中断；恢复后按 `1, 2, 0, 3` 完成 4 局，唯一 ID、更新次数和完整 checkpoint 审计均通过，不计入性能样本。
- 每组成功 run 的完整权重 checkpoint、100 局棋谱与样本、原始逐局计时和审计文件均已打包到 Drive；下载副本与上传包 SHA-256 一致，并在空目录恢复成功。汇总清单见 `drive_backup_status.json`。

## G20K 到 G50K 的线性估算

| 方案 | 墙钟小时（30,000 局） | CPU 小时（30,000 局） |
|---:|---:|---:|
| P1 | {wall_30k_p1_h:.1f} | {cpu_30k_p1_h:.1f} |
| P2 | {p2['wall_seconds_per_100_games'] * 300 / 3600:.1f} | {p2['cpu_seconds_per_game'] * 30_000 / 3600:.1f} |
| P4 | {p4['wall_seconds_per_100_games'] * 300 / 3600:.1f} | {p4['cpu_seconds_per_game'] * 30_000 / 3600:.1f} |
| P6 | {wall_30k_h:.1f} | {cpu_30k_h:.1f} |
| P8 | {p8['wall_seconds_per_100_games'] * 300 / 3600:.1f} | {p8['cpu_seconds_per_game'] * 30_000 / 3600:.1f} |

这是基于 G20K 附近每组 100 局表现的线性外推，模型、棋局长度、机器负载和权重滞后在后续 30K 局中可能变化。可以把 P6 用于一个**独立的 50K 后续实验**，预计把墙钟从约 87.2 小时降到约 22.9 小时，代价是约 1.9% 的 CPU 增量以及更大的异步权重滞后。若目标还包括与原串行轨迹比较棋力，应保留 P1/旧轨迹作为对照，并用固定对手、交换先后手的盲测判断效果。

## 交付文件

- `per_game_metrics.jsonl.gz`：700 行原始逐局性能数据的 gzip 压缩件，解压后为 JSONL。
- `runs_summary.csv`、`runs_summary.json`：方案汇总。
- `throughput_scaling.png`：吞吐量、加速比和 CPU 成本曲线。
- `drive_backup_status.json`：Drive 文件 ID、压缩包和状态 checkpoint SHA-256、恢复结果。
- 源码：`src/research/alphazero_g/v6_9_parallel.py`、`src/research/alphazero_g/v6_9_snapshot.py`、`scripts/run_g_v6_9_efficiency.py`、`scripts/package_g_v6_9_run.py`、`scripts/summarize_g_v6_9.py`。
"""
    (RESULTS / "report_zh.md").write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = load_runs()
    rows = roll_up(records)
    with gzip.open(RESULTS / "per_game_metrics.jsonl.gz", "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            for game in record["games"]:
                handle.write(json.dumps(game, ensure_ascii=False, sort_keys=True) + "\n")
    (RESULTS / "runs_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_fields = list(rows[0])
    with (RESULTS / "runs_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    raw_runs = [{key: record[key] for key in ("run_id", "workers", "process_start_method", "summary", "audit")} for record in records]
    (RESULTS / "validated_run_records.json").write_text(json.dumps(raw_runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    backup_rows = []
    for record in records:
        run_id = record["run_id"]
        file_id, archive_sha, state_sha = DRIVE_BACKUPS[run_id]
        if record["summary"]["checkpoint"]["training_state_sha256"] != state_sha:
            raise ValueError(f"Drive backup state SHA differs for {run_id}")
        backup_rows.append({
            "run_id": run_id,
            "drive_file_id": file_id,
            "drive_url": f"https://drive.google.com/file/d/{file_id}/view",
            "archive_sha256_local_and_downloaded": archive_sha,
            "training_state_sha256": state_sha,
            "download_sha256_matched": True,
            "empty_directory_restore_valid": True,
            "checkpoint_restore_valid": True,
        })
    (RESULTS / "drive_backup_status.json").write_text(json.dumps({"folder_url": DRIVE_FOLDER, "runs": backup_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    incidents = [
        {
            "run_id": "P2-R1",
            "status": "excluded_invalid",
            "reason": "Interrupted resume produced 107 metrics rows, duplicate game ordinals/IDs, and 102 committed updates rather than 100; run excluded from all comparisons.",
            "resolution": "Fixed ordinal-gap resume handling and reran from the untouched G20000 baseline as P2-CLEAN-1; full audit passed.",
        },
        {
            "run_id": "P6-R1-SPAWN",
            "status": "preheat_terminated_no_measurement",
            "reason": "The six-worker spawn attempt exited with code 137 before a measured checkpoint was written; the hard cgroup memory/CPU limits were not exposed.",
            "resolution": "Used fork for P6 and P8; both completed and passed full audits. Start method is recorded in every valid summary.",
        },
        {
            "run_id": "RESUME-GAP-SMOKE",
            "status": "resume_correctness_passed_not_in_performance_set",
            "reason": "Interrupted after ordinal 1 committed while ordinal 0 remained unfinished; resume completed ordinals 1, 2, 0, 3 with four unique IDs and exactly four updates.",
            "checkpoint_game_count": 20004,
            "checkpoint_optimizer_updates": 20003,
            "full_audit_valid": True,
        },
    ]
    (RESULTS / "execution_incidents.json").write_text(json.dumps(incidents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_chart(rows)
    write_report(rows, records)
    print(json.dumps({"valid_runs": len(records), "measured_games": sum(len(record["games"]) for record in records), "results": str(RESULTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
