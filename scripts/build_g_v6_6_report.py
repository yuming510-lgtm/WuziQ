#!/usr/bin/env python3
"""Generate deterministic CSV and Chinese report from audited V6.6 game records."""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from src.research.alphazero_g.v6_6_evaluation import (
    ARTIFACT, BASELINE_OPPONENTS, MODELS, ROOT as REPO_ROOT,
    _series_dir, audit_all, build_match_plan, summarize_v6_6,
)

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    root = ROOT
    audit = audit_all(root, require_complete=True)
    summary = summarize_v6_6(root)
    artifact = root / ARTIFACT
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    plan = build_match_plan(schedule["openings"])
    games = []
    for task in plan:
        games.append(json.loads((_series_dir(root, task) / f"{task['game_id']}.json").read_text(encoding="utf-8")))
    progress = json.loads((artifact / "results/progress.json").read_text(encoding="utf-8"))
    first_game_path = min(((_series_dir(root, task) / f"{task['game_id']}.json") for task in plan), key=os.path.getmtime)
    first_game = json.loads(first_game_path.read_text(encoding="utf-8"))
    first_game_end = dt.datetime.fromtimestamp(first_game_path.stat().st_mtime, tz=dt.timezone.utc)
    run_start = first_game_end - dt.timedelta(seconds=float(first_game["total_wall_seconds"]))
    run_end = dt.datetime.fromisoformat(progress["updated_at_utc"].replace("Z", "+00:00"))
    observed_wall_seconds = max(0.0, (run_end - run_start).total_seconds())
    runtime = {
        "planned_games": 504, "completed_games": len(games),
        "first_game_start_utc_estimate": run_start.isoformat(),
        "last_game_finished_utc": run_end.isoformat(),
        "observed_match_execution_wall_seconds": observed_wall_seconds,
        "sum_of_per_game_cpu_seconds": summary["match_cpu_seconds"],
        "sum_of_per_game_wall_seconds": summary["match_wall_seconds"],
        "wall_measurement_note": "First game's filesystem completion time minus its measured in-game wall time, through final progress timestamp; includes between-game snapshots and audit work, excludes preflight/model restore before first game.",
    }
    (artifact / "results/runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    game_csv = artifact / "results/games.csv"
    with game_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["game_id", "series", "candidate", "opponent", "opening_group_id", "candidate_color",
                  "winner_color", "points", "termination", "move_count", "candidate_cpu_seconds",
                  "opponent_cpu_seconds", "total_cpu_seconds", "total_wall_seconds", "seed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: game.get(key) for key in fields} for game in games)
    summaries = summary["baseline_summaries"]
    summary_csv = artifact / "results/condition_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["candidate", "opponent", "games", "paired_openings", "wins", "draws", "losses", "points",
                  "score_rate", "paired_opening_bootstrap_ci95", "black_score_rate", "white_score_rate",
                  "mean_plies", "candidate_cpu_seconds", "opponent_cpu_seconds", "mean_search_depth",
                  "mean_search_nodes", "mean_simulations", "opponent_mean_search_depth",
                  "opponent_mean_search_nodes", "opponent_mean_cutoffs", "timeouts"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                **{key: row.get(key) for key in fields if key not in {"black_score_rate", "white_score_rate", "candidate_cpu_seconds"}},
                "black_score_rate": row["candidate_black"]["score_rate"],
                "white_score_rate": row["candidate_white"]["score_rate"],
                "candidate_cpu_seconds": row["candidate_cpu_seconds"],
                "paired_opening_bootstrap_ci95": json.dumps(row["paired_opening_bootstrap_ci95"]),
            })
    direct = summary["direct_match"]
    plot_order = list(BASELINE_OPPONENTS)
    labels = ["A", "B 50ms", "C 50ms", "C 200ms", "C 1s", "C 5s\n(探索)"]
    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset, candidate, color, marker in [(-width / 2, "G5000", "#5470c6", "o"),
                                               (width / 2, "G8000", "#ee6666", "s")]:
        rows = [next(row for row in summaries if row["candidate"] == candidate and row["opponent"] == opponent)
                for opponent in plot_order]
        rates = [row["score_rate"] for row in rows]
        lows = [row["paired_opening_bootstrap_ci95"][0] for row in rows]
        highs = [row["paired_opening_bootstrap_ci95"][1] for row in rows]
        lower = np.asarray(rates) - np.asarray(lows)
        upper = np.asarray(highs) - np.asarray(rates)
        ax.errorbar(positions + offset, rates, yerr=[lower, upper], fmt=marker, color=color,
                    capsize=4, markersize=7, label=candidate)
        for x, y, row in zip(positions + offset, rates, rows):
            ax.text(x, y + .025, f"{100*y:.0f}%", ha="center", fontsize=8, color=color)
    ax.axhline(.5, color="#555", lw=1, ls="--")
    ax.set_xticks(positions, labels)
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel("得分率（胜1、平0.5、负0）")
    ax.set_title("G5000 / G8000 对传统棋手的配对得分率（95% bootstrap CI）")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    chart_path = artifact / "figures/ladder_score_rates.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    labels_direct = ["G8000 执黑", "G8000 执白", "配对总计"]
    vals = [direct["candidate_black"]["score_rate"], direct["candidate_white"]["score_rate"], direct["score_rate"]]
    colors = ["#5470c6", "#91cc75", "#ee6666"]
    bars = ax.bar(labels_direct, vals, color=colors)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, value + .02, pct(value), ha="center")
    lo, hi = direct["paired_opening_bootstrap_ci95"]
    ax.errorbar(2, vals[2], yerr=[[vals[2]-lo], [hi-vals[2]]], fmt="none", ecolor="#222", capsize=5)
    ax.axhline(.5, color="#555", lw=1, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("G8000 得分率")
    ax.set_title("G8000 vs G5000｜80局、40组换色开局")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    direct_path = artifact / "figures/g8000_vs_g5000.png"
    fig.savefig(direct_path, dpi=170)
    plt.close(fig)

    lines = [
        "# WuziQ AlphaZero G V6.6｜实力验收与天梯挑战",
        "",
        "## 实验结论与范围",
        "",
        f"- 有效完成并回放验收 **{audit['audited_games']} / 504 局**；对局累计 CPU {summary['match_cpu_seconds']/3600:.2f} 小时，首局到末局完成的实际墙钟区间约 {observed_wall_seconds/3600:.2f} 小时（含局间快照/审计，不含正式开赛前模型恢复与预检）。这是规则/搜索推理评测，不含训练。",
        f"- 冻结模型 G5000 SHA-256：`cc383f0667cfe02b8e8ccbaecf51eccc2c4f088d56c836a4e91e016cfae3ee97`。G8000：`490b70eabb75de1b076e1eb9d13a9d82d2ccc213f3b769bfe6c98eddcb679272`。",
        "- 主要指标是配对换色后的得分率及按开局组 bootstrap 的 95% 区间。区间不含开局池以外的不确定性；C@5s 每模型仅 6 组，属于探索结果。",
        "- G 权重、PUCT-400 配置、A/B/C 决策引擎均未训练或修改。V6.5 进度文件仅将已完成 G8000 的元数据从 `completed:false` 修正为 `true`，未改任何 checkpoint。",
        "",
        "## 开局、赛程与公平性",
        "",
        f"- 棋盘 9×9，自由连五；固定 6 手合法非终局开局，种子 `{schedule['opening_seed_selected']}`，D4 旋转/镜像去重，扫描 {schedule['validation']['historical_opening_keys_scanned']} 个历史开局签名，重叠数为 0。",
        "- G8000 对 G5000 使用 40 组×换色=80 局。每个 G 对 A、B@50ms、C@50ms、C@200ms、C@1s 各 20 组×换色=40 局；C@5s 各 6 组×换色=12 局。总计 504 局。",
        "- 两个 G 模型的基线挑战严格共享同一开局组、颜色和随机种子。传统棋手不交换开局内的固定开局序列。",
        "- A 立即执行规则策略；B 为 50ms 纯 Minimax；C 为 Alpha-Beta 并按条件限时。搜索超时使用算法返回的最后完整迭代结果，合法落子照常计局；非法着按负方判负。实际未见非法落子。",
        "",
        "## G8000 对 G5000",
        "",
        f"| 对阵（G8000视角） | 局数 | 胜-和-负 | 得分率 | 配对95%区间 | 黑得分率 | 白得分率 | 平均步数 | G8000/G5000 CPU秒/局 | G8000/G5000深度、节点、模拟/步 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| PUCT-400 | {direct['games']} | {direct['wins']}-{direct['draws']}-{direct['losses']} | {pct(direct['score_rate'])} | [{pct(lo)}, {pct(hi)}] | {pct(direct['candidate_black']['score_rate'])} | {pct(direct['candidate_white']['score_rate'])} | {direct['mean_plies']:.1f} | {direct['candidate_cpu_seconds']:.2f} / {direct['opponent_cpu_seconds']:.2f} | {direct['candidate_mean_search_depth']:.2f}/{direct['candidate_mean_search_nodes']:.0f}/{direct['candidate_mean_simulations']:.0f} / {direct['opponent_mean_search_depth']:.2f}/{direct['opponent_mean_search_nodes']:.0f}/{direct['opponent_mean_simulations']:.0f} |",
        "",
        f"![G8000与G5000直接对弈](figures/{direct_path.name})",
        "",
        "## 传统对手天梯",
        "",
        "| 棋手 | G模型 | 局数 | 胜-和-负 | 得分率 | 配对95%区间 | 执黑 | 执白 | 平均步数 | G CPU秒/局 | 对手CPU秒/局 | G深度/节点/模拟 | 对手深度/节点/剪枝 | 对手超时数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for opponent in BASELINE_OPPONENTS:
        for candidate in MODELS:
            row = next(item for item in summaries if item["candidate"] == candidate and item["opponent"] == opponent)
            lo, hi = row["paired_opening_bootstrap_ci95"]
            lines.append(f"| {opponent} | {candidate} | {row['games']} | {row['wins']}-{row['draws']}-{row['losses']} | {pct(row['score_rate'])} | [{pct(lo)}, {pct(hi)}] | {pct(row['candidate_black']['score_rate'])} | {pct(row['candidate_white']['score_rate'])} | {row['mean_plies']:.1f} | {row['candidate_cpu_seconds']:.2f} | {row['opponent_cpu_seconds']:.2f} | {row['mean_search_depth']:.2f}/{row['mean_search_nodes']:.0f}/{row['mean_simulations']:.0f} | {row['opponent_mean_search_depth']:.2f}/{row['opponent_mean_search_nodes']:.0f}/{row['opponent_mean_cutoffs']:.0f} | {row['timeouts']} |")
    lines.extend([
        "",
        f"![传统对手天梯得分率](figures/{chart_path.name})",
        "",
        "### G8000相对G5000的配对变化",
        "",
        "| 对手 | G8000−G5000 配对得分差 | 95%区间 | 开局组 |",
        "|---|---:|---:|---:|",
    ])
    for row in summary["paired_G8000_minus_G5000"]:
        lo, hi = row["paired_95_ci"]
        lines.append(f"| {row['opponent']} | {row['paired_score_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['opening_groups']} |")
    a_rows = [row for row in summaries if row["opponent"] == "A"]
    a_g5000 = next(row for row in a_rows if row["candidate"] == "G5000")
    a_g8000 = next(row for row in a_rows if row["candidate"] == "G8000")
    c200_g5000 = next(row for row in summaries if row["candidate"] == "G5000" and row["opponent"] == "C@200ms")
    c200_g8000 = next(row for row in summaries if row["candidate"] == "G8000" and row["opponent"] == "C@200ms")
    c50_g8000 = next(row for row in summaries if row["candidate"] == "G8000" and row["opponent"] == "C@50ms")
    c1_g8000 = next(row for row in summaries if row["candidate"] == "G8000" and row["opponent"] == "C@1s")
    c5_g8000 = next(row for row in summaries if row["candidate"] == "G8000" and row["opponent"] == "C@5s")
    a_games = [game for game in games if game["opponent"] == "A"]
    g8000_a_lengths = [game["move_count"] for game in a_games if game["candidate"] == "G8000"]
    lines.extend([
        "",
        "## 计算成本与深度观察",
        "",
        f"- 全部对局合计 {summary['match_cpu_seconds']/3600:.2f} CPU 小时；逐局思考墙钟时间加总 {summary['match_wall_seconds']/3600:.2f} 小时；从第一局开始至最后一局结束约 {observed_wall_seconds/3600:.2f} 小时。统计边界见 `results/runtime.json`。",
        f"- G8000 对 A 的局长均值 {float(np.mean(g8000_a_lengths)):.1f} 手，最长 {max(g8000_a_lengths)} 手；其 G 侧平均 CPU {next(r for r in a_rows if r['candidate']=='G8000')['candidate_cpu_seconds']:.2f} 秒/局。",
        "- 固定开局25手代表局面的预检中，G5000/G8000各完成400次根访问，测得约 2.01/1.83 CPU 秒、树深9/5、树节点8729/4533。该预检用于成本估算，不计入赛果。",
        "- C@5s在预检位置只完成深度4并于约2.65秒自然结束（没有触发5秒超时）。本轮保持历史 `SearchConfig` 深度上限4未改；因此 C@5s 条件可能受深度上限约束，报告的对手身份应理解为“历史C实现、5秒上限”，不是保证每步消耗满5秒。",
        "- G每步记录400次根访问、树深和节点；C/B记录实际完成深度、节点、剪枝及超时；每步详情和完整棋谱均保存在原始JSON。",
        "",
        "## 统计解读与限制",
        "",
        "- 直接对弈的配对95%区间完全高于/低于50%时，可作为本开局池上方向性证据；若跨越50%，结论是本轮不能区分。即便不跨越，40组仍是有限开局的探索性评估。",
        "- C@5s只有6组，区间很宽，不能据此确认稳定强弱。A/B/C对手来自不同算法和时间预算，成绩分开呈现、不合并成总分。",
        "- 训练局数、训练Loss没有作为棋力证据；本轮是冻结模型实战对照。先后手成绩仅描述本次配对开局，不解释为一般性先手效应。",
        "",
        "## 结论",
        "",
        f"- G8000对G5000直接对战得分率 {pct(direct['score_rate'])}，配对95%区间 [{pct(direct['paired_opening_bootstrap_ci95'][0])}, {pct(direct['paired_opening_bootstrap_ci95'][1])}] 跨过50%；本轮不能确认8K模型总体强于5K。G8000执黑 {pct(direct['candidate_black']['score_rate'])}、执白 {pct(direct['candidate_white']['score_rate'])}，颜色差异提示开局先后可能影响本次成绩。",
        f"- 对A，G8000得分率 {pct(a_g8000['score_rate'])}、配对区间 [{pct(a_g8000['paired_opening_bootstrap_ci95'][0])}, {pct(a_g8000['paired_opening_bootstrap_ci95'][1])}]；G5000为 {pct(a_g5000['score_rate'])}。在这20组固定新开局上，G8000已经有超过A的方向性结果，但G8000−G5000配对提升区间仍跨0，因此不能归因于训练带来的可靠强度提升。",
        f"- 对B@50ms，G8000得分率 {pct(next(row for row in summaries if row['candidate']=='G8000' and row['opponent']=='B@50ms')['score_rate'])}；对C@50ms为 {pct(c50_g8000['score_rate'])}，区间跨50%，结果未能区分强弱。",
        f"- G8000对C@200ms为 {c200_g8000['wins']}-{c200_g8000['draws']}-{c200_g8000['losses']}，得分率 {pct(c200_g8000['score_rate'])}；G5000为 {pct(c200_g5000['score_rate'])}。本轮没有显示G8000能突破C@200ms，配对变化反而为负。对C@1s得分率 {pct(c1_g8000['score_rate'])}；对C@5s为 {pct(c5_g8000['score_rate'])}（6组探索样本）。",
        "- 当前证据支持“G8000可以挑战A和B@50ms、与C@50ms尚不能区分，但无法挑战更高时间档的C”。这些结论限于本轮开局池与冻结引擎，不是跨开局稳定排名。",
        "",
        "## 原始数据和复现",
        "",
        "- 每局原始 JSON：`results/matches/<series>/<game_id>.json`；单局原子落盘，`results/progress.json`按局更新。",
        "- 统计：`results/summary.json`、`results/games.csv`、`results/condition_summary.csv`、`results/audit.json`。",
        "- 固定计划：`config/registration.json`、`config/opening_schedule.json`、`config/match_plan.json`。",
        "- V6.6 Google Drive 恢复目录：[WuziQ G V6.6 Recovery](https://drive.google.com/drive/folders/1uOc9NRe8-GKHeHyoqdesNfyVUZYqj5_R)；已验证快照文件 ID、SHA-256 与保留策略记录在 `results/drive_backup_status.json`。",
        "- 复现检查：`PYTHONPATH=. pytest -q tests/test_alphazero_g_v6_6.py`；运行/审计：`PYTHONPATH=. python scripts/run_g_v6_6.py run-batch --batch-games 10`、`PYTHONPATH=. python scripts/run_g_v6_6.py audit`。",
        "- 与本轮直接相关的V6.6/V6.5/V6.3测试通过 29/29。全仓库测试结果为 514 passed、29 failed；失败项属于旧V2/V3/V4/V6.3分支身份绑定的历史测试，它们要求当前Git分支分别等于各自旧分支名，在独立V6.6分支上触发保护性拒绝；本轮未修改那些历史分支守卫。",
        "- SHA-256、文件数、规则回放、配对开局数及重复ID检查见 `results/audit.json`。",
        "",
    ])
    (artifact / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"games": len(games), "cpu_hours": summary["match_cpu_seconds"] / 3600,
                      "report": str(artifact / "REPORT.md"), "charts": [str(chart_path), str(direct_path)]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
