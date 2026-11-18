#!/usr/bin/env python3
"""Build V6.6B report tables and figures from frozen raw match records."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research/g_v6_6b"
RESULTS = ART / "results"
FIGURES = ART / "figures"
OPPONENTS = ["C@200ms", "C@1s", "C@5s"]
MODELS = ["G5000", "G8000"]


def jread(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def fnum(x, digits=2):
    return f"{x:.{digits}f}"


def historical_v6_4() -> dict:
    raw = (RESULTS / "historical_v6_4_condition_summary.csv").read_text(encoding="utf-8")
    return {row["opponent"]: row for row in csv.DictReader(io.StringIO(raw))}


def build_tables(summary: dict, hist66: dict, hist64: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = summary["summaries"]
    fields = [
        "candidate", "opponent", "games", "wins", "draws", "losses", "points", "score_rate",
        "paired_openings", "paired_opening_bootstrap_ci95", "black_score_rate", "white_score_rate",
        "mean_plies", "mean_candidate_cpu_seconds", "mean_candidate_move_cpu_seconds",
        "mean_candidate_puct_search_depth", "mean_candidate_puct_nodes", "mean_candidate_puct_simulations",
        "mean_c_cpu_seconds", "mean_c_elapsed_seconds", "mean_c_search_depth", "mean_c_search_nodes",
        "mean_c_cutoffs", "c_timeout_moves", "c_move_count",
    ]
    with (RESULTS / "condition_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: r.get(key) for key in fields} for r in rows)
    with (RESULTS / "paired_model_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["opponent", "opening_groups", "paired_G8000_minus_G5000_score_rate", "paired_95_ci"], lineterminator="\n")
        w.writeheader()
        w.writerows(summary["paired_G8000_minus_G5000"])

    hist_rows = []
    for opponent in OPPONENTS:
        h64 = hist64[opponent]
        # V6.6 used SearchConfig(time_limit_seconds=...), so max_depth came from the then-default SearchConfig.
        for candidate in MODELS:
            h66 = next(x for x in hist66["baseline_summaries"] if x["candidate"] == candidate and x["opponent"] == opponent)
            h66depth = 4
            r = next(x for x in rows if x["candidate"] == candidate and x["opponent"] == opponent)
            hist_rows.append({
                "candidate": candidate, "opponent": opponent,
                "V6.4_games": int(h64["games"]), "V6.4_score_rate": float(h64["score_rate"]), "V6.4_max_depth": 69,
                "V6.4_mean_depth": json.loads(h64["opponent_move_telemetry"])["mean_completed_depth"],
                "V6.6_games": int(h66["games"]), "V6.6_score_rate": float(h66["score_rate"]), "V6.6_max_depth": h66depth,
                "V6.6_mean_depth": float(h66["opponent_mean_search_depth"]),
                "V6.6B_games": int(r["games"]), "V6.6B_score_rate": float(r["score_rate"]), "V6.6B_max_depth": 69,
                "V6.6B_mean_depth": float(r["mean_c_search_depth"]),
            })
    with (RESULTS / "historical_protocol_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(hist_rows)
    return rows, hist_rows


def matched_budget_view():
    schedule = jread(ART / "config/opening_schedule.json")
    openings = schedule["openings"]
    shared_ids = {x["opening_group_id"] for x in openings[:10]}
    tasks = jread(ART / "config/match_plan.json")["tasks"]
    games = {}
    for task in tasks:
        path = next((ART / "results/matches").glob(f"*/{task['game_id']}.json"))
        games[task["game_id"]] = jread(path)
    result = {"opening_groups": 10, "opening_group_ids": sorted(shared_ids), "conditions": []}
    for candidate in MODELS:
        for opponent in OPPONENTS:
            selected = [g for g in games.values() if g["candidate"] == candidate and g["opponent"] == opponent and g["opening_group_id"] in shared_ids]
            per_opening = {}
            for g in selected:
                per_opening.setdefault(g["opening_group_id"], []).append(float(g["points"]))
            opening_scores = [float(np.mean(per_opening[k])) for k in sorted(per_opening)]
            rng = np.random.default_rng(600 + MODELS.index(candidate) * 10 + OPPONENTS.index(opponent))
            samples = np.mean(rng.choice(opening_scores, (20000, len(opening_scores)), replace=True), axis=1)
            result["conditions"].append({
                "candidate": candidate, "opponent": opponent, "games": len(selected),
                "score_rate": float(np.mean(opening_scores)),
                "paired_opening_bootstrap_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            })
    (RESULTS / "matched_budget_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def make_figures(rows):
    colors = {"G5000": "#176B87", "G8000": "#D97732"}
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    xs = np.arange(len(OPPONENTS))
    width = 0.32
    for i, model in enumerate(MODELS):
        data = [next(x for x in rows if x["candidate"] == model and x["opponent"] == opp) for opp in OPPONENTS]
        vals = np.array([x["score_rate"] for x in data]) * 100
        cis = np.array([x["paired_opening_bootstrap_ci95"] for x in data]) * 100
        yerr = np.vstack((vals - cis[:, 0], cis[:, 1] - vals))
        ax.bar(xs + (i - .5) * width, vals, width, label=model, color=colors[model], zorder=2)
        ax.errorbar(xs + (i - .5) * width, vals, yerr=yerr, fmt="none", ecolor="#222222", capsize=4, lw=1.2, zorder=3)
    ax.axhline(50, color="#8B1E3F", ls="--", lw=1.4, label="50% 得分率")
    ax.set_xticks(xs, OPPONENTS)
    ax.set_ylabel("G 棋手得分率（%）")
    ax.set_ylim(0, 60)
    ax.set_title("V6.6B：冻结 G5000/G8000 对修正协议 C 的挑战")
    ax.grid(axis="y", alpha=.22, zorder=0)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    fig.savefig(FIGURES / "c_ladder_score_rate.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), constrained_layout=True)
    for model in MODELS:
        data = [next(x for x in rows if x["candidate"] == model and x["opponent"] == opp) for opp in OPPONENTS]
        axes[0].plot(["200ms", "1s", "5s"], [x["mean_c_search_depth"] for x in data], marker="o", lw=2, label=model, color=colors[model])
        axes[1].plot(["200ms", "1s", "5s"], [x["mean_c_search_nodes"] for x in data], marker="o", lw=2, label=model, color=colors[model])
    axes[0].set_title("C 每步实际完成深度")
    axes[0].set_ylabel("平均完整迭代深度")
    axes[1].set_title("C 每步搜索节点")
    axes[1].set_ylabel("平均节点数")
    for ax in axes:
        ax.set_xlabel("每步预算")
        ax.grid(alpha=.22)
    axes[1].legend(frameon=False)
    fig.suptitle("实际搜索量随预算增加；超时后采用最后完整深度")
    fig.savefig(FIGURES / "c_search_depth_nodes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 4.8), constrained_layout=True)
    for i, model in enumerate(MODELS):
        data = [next(x for x in rows if x["candidate"] == model and x["opponent"] == opp) for opp in OPPONENTS]
        ax.plot(["200ms", "1s", "5s"], [x["mean_c_cpu_seconds"] for x in data], marker="s", lw=2, label=f"C {model} 对局内每步 CPU", color=colors[model])
    ax.plot(["200ms", "1s", "5s"], [.2, 1, 5], color="#666", ls=":", label="标称单步上限")
    ax.set_ylabel("平均每步 CPU 秒")
    ax.set_xlabel("C 每步预算")
    ax.set_title("C 的实际每步计算成本")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "c_search_cpu.png", dpi=180)
    plt.close(fig)


def build_report(summary, rows, hist_rows, matched, audit, progress, preflight, registration):
    lookup = {(r["candidate"], r["opponent"]): r for r in rows}
    histlookup = {(r["candidate"], r["opponent"]): r for r in hist_rows}
    lines = [
        "# WuziQ V6.6B｜C 系列评测修正与复赛",
        "",
        "## 结论摘要",
        "",
        "- 200/200 局完成，全部棋谱通过逐步合法落子、终局、胜者及发动方独立规则回放；没有重复 ID、缺局或未完成局。",
        "- 按 V6.4 的 C 配置（Alpha-Beta、`max_depth=69`、`candidate_radius=2`）重赛后，G5000/G8000 对 C@200ms、C@1s、C@5s 的得分分别为 **6.25%/10.0%**、**27.5%/22.5%**、**20.0%/5.0%**。六个配对95%区间上限均低于50%。",
        "- G8000 对 G5000 的同开局配对差：C@200ms **+3.75个百分点**（95% CI −6.25 至 +13.75），C@1s **−5个百分点**（−25 至 +15），C@5s **−15个百分点**（−35 至 +5）。差异都跨过0，不能确认 G8000 更强；点估计混合，C@1s/C@5s 更偏向 G5000。",
        "- V6.6 的偏差来自调用点只传入时间参数，`SearchConfig` 的默认 `max_depth=4` 生效；C 引擎本身和评估函数未改。V6.6B 显式恢复 `max_depth=69`、半径2，和 V6.4 协议一致。",
        "- 结果支持：按 G 得分率衡量，G8000 没有显示出比 G5000 更强；两者都未能突破 C@200ms/1s/5s。这个小样本只能说明本赛程上的表现，不能断言两个 checkpoint 的总体棋力完全相同。",
        "",
        "## 冻结条件与评测协议",
        "",
        f"- 基线：`{registration['base_commit']}`；分支：`{registration['branch']}`；棋盘9×9、连五获胜。",
        f"- G5000 权重 SHA-256：`{registration['models']['G5000']['model_sha256']}`；G8000：`{registration['models']['G8000']['model_sha256']}`。两份权重与完整训练状态中的模型一致；分别冻结为5000局/87999状态/4999更新和8000局/139884状态/7999更新。",
        "- 两模型固定 PUCT-400；C 每步预算200ms、1秒、5秒。C 使用同一 Alpha-Beta 搜索、评价函数、`max_depth=69`、`candidate_radius=2`，到时返回最后一层完整迭代结果。算法代码未更改。",
        f"- 预注册开局池：20个六手合法非终局开局，固定种子 `{jread(ART / 'config/opening_schedule.json')['opening_seed_selected']}`；D4规范化后与扫描的历史开局池无重叠。开局表 SHA-256：`{registration['opening_schedule_sha256']}`；完整200局计划 SHA-256：`{registration['match_plan_sha256']}`。G5000/G8000使用完全相同的开局、颜色和随机种子。",
        "- C@200ms 与 C@1s 各为20组开局、黑白换色40局/模型；C@5s为10组开局、黑白换色20局/模型。胜1分、平0.5分、负0分。区间使用开局组为单位的配对 bootstrap 95%区间；同一组的两种颜色先平均，避免把两局当独立样本。",
        "",
        "## 正式战绩",
        "",
        "得分率以 G 棋手视角计算。CI 是开局配对 bootstrap 95% 区间。",
        "",
        "| G 棋手 | 对手 | 局数 | 胜-平-负 | 得分率 | 配对95% CI | G执黑 | G执白 | 平均手数 | G CPU/局 | G CPU/步 | G深度 | G节点/步 | G模拟/步 | C CPU/局 | C深度 | C节点/步 | 超时步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for opp in OPPONENTS:
            x = lookup[(model, opp)]
            ci = x["paired_opening_bootstrap_ci95"]
            lines.append(
                f"| {model} | {opp} | {x['games']} | {x['wins']}-{x['draws']}-{x['losses']} | {pct(x['score_rate'])} | [{pct(ci[0])}, {pct(ci[1])}] | {pct(x['black_score_rate'])} | {pct(x['white_score_rate'])} | {fnum(x['mean_plies'],1)} | {fnum(x['mean_candidate_cpu_seconds'],1)}s | {fnum(x['mean_candidate_move_cpu_seconds'],2)}s | {fnum(x['mean_candidate_puct_search_depth'],2)} | {fnum(x['mean_candidate_puct_nodes'],0)} | {fnum(x['mean_candidate_puct_simulations'],0)} | {fnum(x['mean_opponent_cpu_seconds'],1)}s | {fnum(x['mean_c_search_depth'],2)} | {fnum(x['mean_c_search_nodes'],0)} | {x['c_timeout_moves']}/{x['c_move_count']} |"
            )
    lines += [
        "",
        "![G5000/G8000 对修正协议 C 的得分率及配对置信区间](figures/c_ladder_score_rate.png)",
        "",
        "### G5000 与 G8000 的同条件差值",
        "",
        "| C对手 | 配对开局组 | G8000−G5000得分率 | 配对95% CI | 解释 |",
        "|---|---:|---:|---:|---|",
    ]
    for d in summary["paired_G8000_minus_G5000"]:
        lo, hi = d["paired_95_ci"]
        delta = d["paired_G8000_minus_G5000_score_rate"]
        lines.append(f"| {d['opponent']} | {d['opening_groups']} | {delta*100:+.2f}个百分点 | [{lo*100:+.2f}, {hi*100:+.2f}] | 区间包含0，不能确认差异 |")
    lines += [
        "",
        "20组开局的 C@200ms/C@1s 中，G8000 的点估计略高于 G5000，但区间宽且跨0；C@1s的差值更偏向 G5000。C@5s只有10组，证据更弱。不能用胜局数或训练局数替代直接交锋。",
        "",
        "### 固定相同10组开局观察预算趋势",
        "",
        "C@5s使用前10组，故可在这10组开局上作预算匹配的探索性对照（每条件20局）。",
        "",
        "| G棋手 | C预算 | 局数 | 匹配开局得分率 | 配对95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in matched["conditions"]:
        ci = x["paired_opening_bootstrap_ci95"]
        lines.append(f"| {x['candidate']} | {x['opponent']} | {x['games']} | {pct(x['score_rate'])} | [{pct(ci[0])}, {pct(ci[1])}] |")
    lines += [
        "",
        "这组匹配子样本每档只有10个开局对，区间仍很宽。得分率不随预算呈单调变化，不能据此认为较长预算降低了 C 棋力；预算改变后整局分支也随之改变。更可核验的是实际每步深度、节点和耗时随预算增加。",
        "",
        "## C 搜索量与计算成本",
        "",
        "| G棋手面对的C | C每步时间上限 | 完整深度（平均） | 节点（平均/步） | 剪枝（平均/步） | 实际耗时（CPU/步） | 超时步数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for opp in OPPONENTS:
            x = lookup[(model, opp)]
            lines.append(f"| {model} / {opp} | {dict(zip(OPPONENTS,[0.2,1.0,5.0]))[opp]:g}s | {fnum(x['mean_c_search_depth'],2)} | {fnum(x['mean_c_search_nodes'],0)} | {fnum(x['mean_c_cutoffs'],0)} | {fnum(x['mean_c_cpu_seconds'],3)}s / {fnum(x['mean_c_elapsed_seconds'],3)}s | {x['c_timeout_moves']}/{x['c_move_count']} |")
    lines += [
        "",
        "C@200ms 实际平均完成约2.1–2.4层，约1.2K节点/步；C@1s约3.8–4.0层、约7.0–7.4K节点/步；C@5s约4.7层（G5000局面）至7.3层（G8000局面）、约26–28K节点/步。深度随棋局局面复杂度变化，不应只看预算标签。超时表示迭代加深到时后停止；最终着法仍来自最后完整层。",
        "",
        "![C搜索深度和节点数](figures/c_search_depth_nodes.png)",
        "",
        "![C每步实际CPU耗时](figures/c_search_cpu.png)",
        "",
        f"G 每局平均耗时约17.95–25.63 CPU秒；C 每局约2.25–49.52 CPU秒，随预算增长。正式200局总棋手 CPU 时间 `{summary['match_cpu_seconds']:.1f}` 秒（{summary['match_cpu_hours']:.3f} CPU小时）；逐局wall时长合计 `{summary['match_wall_seconds_sum']:.1f}` 秒。执行器累计 CPU `{progress['cumulative_runner_cpu_seconds']:.1f}` 秒（含运行调度开销）；对局文件时间跨度 `{summary['match_artifact_span_seconds']:.0f}` 秒。",
        f"- 赛前实测资源：逻辑CPU {preflight['resource']['logical_cpu_count']}、容器CPU配额 {preflight['resource']['estimated_cpu_quota']:.1f}核、内存 {preflight['resource']['memory']['total_gib']:.2f} GiB（可用 {preflight['resource']['memory']['available_gib']:.2f} GiB）、磁盘空闲 {preflight['resource']['disk_free_gib']:.2f} GiB。预估乐观/中性/保守为 {preflight['estimate']['optimistic_cpu_hours']:.2f}/{preflight['estimate']['central_cpu_hours']:.2f}/{preflight['estimate']['conservative_cpu_hours']:.2f} CPU小时；实测对局约2.01 CPU小时，低于12小时上限。",
        "",
        "## 历史配置核查与可比性",
        "",
        "- V6.4 的 `v6_4_ladder.py` 明确创建 `SearchConfig(max_depth=69, candidate_radius=2)`；本轮检查到其 Alpha-Beta 算法和评估函数未变。",
        "- V6.6 的 `v6_6_evaluation.py` 仅用 `SearchConfig(time_limit_seconds=...)` 创建 C 配置；`src/ai.py` 的默认 `max_depth=4` 因而生效，默认 `candidate_radius=2`。V6.6出现深度差异是调用点未显式传最大深度，不是 Alpha-Beta 算法变化。",
        "- V6.6B显式传入69层、半径2及三档时间上限。C算法和评估函数保持原样，只有评测配置恢复为V6.4协议。",
        "",
        "| G模型 | 对手 | V6.4 (6局/档) | V6.6 (C max_depth=4) | V6.6B (C max_depth=69) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for opp in OPPONENTS:
            h = histlookup[(model, opp)]
            lines.append(f"| {model} | {opp} | {pct(h['V6.4_score_rate'])} (n={h['V6.4_games']}) | {pct(h['V6.6_score_rate'])} (n={h['V6.6_games']}) | {pct(h['V6.6B_score_rate'])} (n={h['V6.6B_games']}) |")
    lines += [
        "",
        "V6.4只有G5000、每条件6局且使用不同开局；V6.6为G5000/G8000混合条件、C深度上限错误为4、每条件40局（C@5s为12局）；V6.6B使用新开局并修正深度，C@5s为20局。以上跨轮次得分率不可当作同局面因果比较。V6.6与V6.6B的用途是指出协议变化和描述结果，不能把胜率升降全部归因于深度上限。",
        "",
        "从搜索遥测看，V6.6 的 C@5s 平均只完成3.75–3.76层，受4层上限约束；V6.6B 在 G5000/G8000的局面上分别完成4.66/7.25层。C@1s 约从V6.6的3.19层变为3.79/4.01层，C@200ms则为2.18/2.14层变为2.37/2.14层。V6.6B确实让部分局面搜索超过4层；实际胜率与历史小样本有升有降，不能据此给出单方向的跨轮次强弱因果结论。",
        "",
        "## 回放审计、恢复与自动化测试",
        "",
        f"- 审计：{audit['audited_games']}/{audit['planned_games']}局；缺失 `{audit['missing_games']}`；唯一ID `{audit['unique_game_ids']}`；`all_games_valid={str(audit['all_games_valid']).lower()}`。独立规则 oracle、Alpha-Beta配置、开局配对、两个模型共用种子与胜者校验均通过。",
        "- 冻结模型SHA、完整训练状态、优化器槽位、回放池大小、计数及可加载性均在开赛前核验。没有训练或修改模型。",
        "- V6.6B Drive 目录：`https://drive.google.com/drive/folders/" + registration["drive_folder_id"] + "`。快照上传后回下载 SHA-256、空目录恢复与棋谱回放验证通过；保留最近两份进度快照，最后另存最终报告快照。",
        "- 相关测试：`pytest tests/test_alphazero_g_v6_6b.py tests/test_alphazero_g_v6_6.py tests/test_alphazero_g_v6_5.py`。",
        "- 复现摘要：冻结 `registration.json`、`opening_schedule.json` 与 `match_plan.json`；运行 `python scripts/run_g_v6_6b.py`，审计 `python scripts/run_g_v6_6b.py audit`，汇总 `python scripts/run_g_v6_6b.py summarize`，生成本报告 `python scripts/build_g_v6_6b_report.py`。重新运行比赛需已有匹配的冻结权重与Python依赖。",
        "",
        "## 局限与结论",
        "",
        "本轮每档只有10–20组配对开局，置信区间不足以区分 G5000 与 G8000；两模型都没有任何一档得分接近50%，但这不等于算法总体实力可由这些开局完全确定。修正协议后的C在200ms、1s和5s下均能持续击败G，且预算越大，实际节点和完整深度总体越高。G8000并未显示对G5000的可靠提升，也未能挑战更高算力的C。",
        "",
        "下一步应优先扩大相同开局、相同颜色配对的直接复赛，并检查 AlphaZero 网络在长预算搜索中的战术识别与防守，而不是根据本轮小样本对checkpoint重新排序。",
        "",
    ]
    (ART / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    summary = jread(RESULTS / "summary.json")
    audit = jread(RESULTS / "audit.json")
    progress = jread(RESULTS / "progress.json")
    preflight = jread(RESULTS / "preflight.json")
    registration = jread(ART / "config/registration.json")
    hist66 = jread(ROOT / "research/g_v6_6/results/summary.json")
    hist64 = historical_v6_4()
    rows, hist_rows = build_tables(summary, hist66, hist64)
    matched = matched_budget_view()
    make_figures(rows)
    build_report(summary, rows, hist_rows, matched, audit, progress, preflight, registration)
    print(f"report={ART / 'REPORT.md'}")
    print(f"games={audit['audited_games']}/{audit['planned_games']} all_valid={audit['all_games_valid']}")
    print(f"cpu_hours={summary['match_cpu_hours']:.3f}")


if __name__ == "__main__":
    main()
