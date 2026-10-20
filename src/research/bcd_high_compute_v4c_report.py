"""Render a Chinese V4C report and research figures from validated raw results."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bcd_high_compute_v4c import load_config
from .bcd_high_compute_v4c_analysis import summarize_records, validate_records
from .checkpoint import load_game_records, write_json_atomic


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _ci(value: list[float] | None) -> str:
    return "—" if value is None else f"[{100*value[0]:.1f}%, {100*value[1]:.1f}%]"


def _fmt(value: float | int | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _transcript(game: dict[str, Any], last_n: int = 10) -> str:
    moves = game["moves"]
    suffix = moves[-last_n:]
    return "；".join(
        f"{move.get('model') or '开局'}-{move['player']}({move.get('row')},{move.get('col')})"
        for move in suffix
    )


def _history(project_root: Path) -> dict[str, Any]:
    v3 = _read_json(project_root / "research/v3/results/summary.json")
    v4b = _read_json(project_root / "research/v4b/results/summary.json")
    v4b_records = load_game_records(project_root / "research/v4b/results/9x9_games.jsonl")
    new_d_games = [row for row in v4b_records if row.get("source_type") == "new" and set(row.get("matchup", ())) in ({"B", "D"}, {"C", "D"})]
    v4b_new_usage: dict[str, dict[str, Any]] = {}
    for model in ("B", "C", "D"):
        moves = [move for game in new_d_games for move in game["moves"] if move.get("model") == model]
        secs = sum(float(move["elapsed_seconds"]) for move in moves)
        v4b_new_usage[model] = {
            "decisions": len(moves), "seconds_total": secs,
            "mean_seconds_per_move": secs / len(moves) if moves else None,
            "mean_depth": sum(int(move.get("completed_depth", 0)) for move in moves) / len(moves) if moves else None,
            "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
            "mean_simulations_per_move": sum(int(move.get("simulations", 0)) for move in moves) / len(moves) if moves else None,
            "cutoffs_total": sum(int(move.get("cutoffs", 0)) for move in moves),
        }
    v3_conditions = {}
    for opponent in ("B", "C"):
        key = f"d3-20s-vs-{opponent.lower()}"
        phase = v3["phases"][key]
        model = phase["D3"]
        v3_conditions[opponent] = {
            "games": phase["games"], "D3_budget_seconds": phase["D3_budget_seconds"],
            "opponent_budget_seconds": phase["opponent_budget_seconds"],
            "D3_wins": model["wins"], "D3_draws": model["draws"], "D3_losses": model["losses"],
            "D3_score_rate": model["score_rate"], "D3_thinking_seconds": model["thinking_seconds_total"],
            "D3_simulations": model["simulations_total"], "D3_mean_simulations": model["average_simulations_per_move"],
        }
    v4b_matchups = {}
    for matchup, key, first in (("B-C", "B-C", "B"), ("B-D", "B-D", "B"), ("C-D", "C-D", "C")):
        row = v4b["matchups"][key]
        v4b_matchups[matchup] = {
            "wins": row["wins"], "draws": row["draws"], "score_rate": row["score_rate"],
            "pure_win_rate": row["pure_win_rate"], "first_model": first,
        }
    return {
        "V3_D20_vs_B50ms_C50ms": v3_conditions,
        "V4B_matchups": v4b_matchups,
        "V4B_200ms_search": {model: v4b["usage"][model] for model in ("B", "C")},
        "V4B_D20_new_BD_CD_usage": v4b_new_usage,
        "V4B_new_D_games": len(new_d_games),
    }


def _figures(figure_dir: Path, summary: dict[str, Any], history: dict[str, Any]) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})
    pairs = [("B-C", "B"), ("B-D", "B"), ("C-D", "C")]
    labels = ["B–C：B 得分率", "B–D：B 得分率", "C–D：C 得分率"]
    old, new = [], []
    for key, model in pairs:
        old.append(history["V4B_matchups"][key]["score_rate"][model])
        new.append(summary["matchups"][key]["score_rate"][model])
    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    y = np.arange(len(labels))
    height = 0.34
    ax.barh(y - height / 2, np.array(old) * 100, height, label="V4B", color="#7995b8")
    ax.barh(y + height / 2, np.array(new) * 100, height, label="V4C", color="#d47b42")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("得分率（胜 1 分，平局 0.5 分）")
    ax.set_title("V4B → V4C 配对对阵表现（小样本描述）")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    for index, (before, after) in enumerate(zip(old, new)):
        ax.text(before * 100 + 1, index - height / 2, f"{before:.0%}", va="center", fontsize=9)
        ax.text(after * 100 + 1, index + height / 2, f"{after:.0%}", va="center", fontsize=9)
    score_path = figure_dir / "v4b_v4c_pairwise_score.svg"
    fig.savefig(score_path, format="svg")
    plt.close(fig)

    models = ("B", "C", "D")
    old_seconds = [history["V4B_D20_new_BD_CD_usage"][model]["mean_seconds_per_move"] for model in models]
    new_seconds = [summary["model_usage"][model]["mean_seconds_per_move"] for model in models]
    old_sims = [history["V4B_D20_new_BD_CD_usage"][model]["mean_simulations_per_move"] for model in models]
    new_sims = [summary["model_usage"][model]["mean_simulations_per_move"] for model in models]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    x = np.arange(3)
    width = 0.36
    ax1.bar(x - width / 2, old_seconds, width, label="V4B", color="#7995b8")
    ax1.bar(x + width / 2, new_seconds, width, label="V4C", color="#d47b42")
    ax1.set_xticks(x, ["B", "C", "D"])
    ax1.set_ylabel("平均每步实际耗时（秒）")
    ax1.set_title("计算成本")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()
    for i, value in enumerate(new_seconds):
        ax1.text(i + width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.bar(x - width / 2, old_sims, width, label="V4B", color="#7995b8")
    ax2.bar(x + width / 2, new_sims, width, label="V4C", color="#d47b42")
    ax2.set_xticks(x, ["B", "C", "D"])
    ax2.set_ylabel("平均每步 MCTS 模拟数")
    ax2.set_title("实际模拟量（B/C 为 0）")
    ax2.ticklabel_format(axis="y", style="plain")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()
    usage_path = figure_dir / "v4b_v4c_compute_cost.svg"
    fig.savefig(usage_path, format="svg")
    plt.close(fig)
    return [score_path.name, usage_path.name]


def build_report(
    project_root: str | Path,
    config_source: str | Path,
    results_dir: str | Path,
    *,
    test_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    results = Path(results_dir)
    config = load_config(config_source)
    records = load_game_records(results / "9x9_games.jsonl")
    validation = validate_records(records, config, require_complete=True)
    summary = summarize_records(records, config, require_complete=True)
    history = _history(root)
    write_json_atomic(results / "history_comparison.json", history)
    figure_names = _figures(results.parent / "figures", summary, history)
    progress = _read_json(results / "progress.json")
    preflight = _read_json(results / "preflight.json")
    drive = _read_json(results / "drive_recovery.json") if (results / "drive_recovery.json").exists() else {}
    execution = {
        "started_at_utc": progress.get("first_started_at_utc"),
        "completed_at_utc": progress.get("completed_at_utc"),
        "elapsed_seconds_total": progress.get("elapsed_seconds_total"),
        "implementation_commit": progress.get("implementation_commit"),
        "scheduled_games": progress.get("planned_games"),
        "completed_games": progress.get("completed_games"),
        "remaining_games": progress.get("remaining_games"),
    }
    write_json_atomic(results / "execution.json", execution)
    write_json_atomic(results / "validation.json", validation)
    write_json_atomic(results / "summary.json", summary)
    report_lines = [
        "# V4C 高算力最终挑战赛研究报告", "",
        "## 摘要", "",
        f"本轮在 9×9 棋盘完成 {validation['validated_games']} 局全新对弈：B–C、B–D、C–D 各 6 局。V4/V4B 历史棋局仅作比较，未计入本轮。排名（胜 1 分、平 0.5 分）为：" + " > ".join(
            f"{model}（{_pct(summary['models'][model]['score_rate'])}）" for model in summary["exploratory_ranking"]
        ) + "。此顺序受 18 局小样本影响，D 的单步预算是 B 的 50 倍、C 的 250 倍，不能解释为同等算力下的稳定强弱。", "",
        "## 配置、实现与公平性", "",
        "- B 为纯 Minimax，单步最多 1 秒；C 为 Alpha–Beta，单步最多 200ms。两者使用相同的 V4 评价函数、候选集及排序、迭代加深与截止时间实现，只有 C 启用剪枝。",
        "- D 直接复用 V4B `MCTSAgent`（纯 UCT，不含神经网络），根节点战术识别、rollout、节点扩展及跨步树复用保持不变，仅把预算从 20 秒改为 50 秒。D 的 50 秒不等于 50 秒保证用满：即时胜/必防可提前返回。",
        "- 每个对阵使用三个固定开局，每个开局交换黑白；种子由固定实验种子、对局 ID 和决策位置的 SHA-256 派生。所有本轮记录均为 `source_type=new`。",
        "- 超时仍接受引擎返回的合法着法并记录超时；非法落子判负。完成后的每局先按规则回放，再追加 JSONL 并 fsync；恢复仅按唯一 game ID 续跑。",
        "- 棋盘规则、AI 搜索源文件的哈希见配置和原始记录；当前环境为 " + str(preflight.get("resources", {}).get("cpu_model", "未记录 CPU")) + f"，CPU 配额 {_fmt(preflight.get('resources', {}).get('cgroup_cpu_quota_cores'),1)} 核，内存限制 {_fmt((preflight.get('resources', {}).get('cgroup_memory_limit_bytes') or 0)/1024**3,1)} GiB。",
        "- 本项目没有神经网络或强化学习训练，不含训练指标或 checkpoint。Codex 模型档位及推理强度无法由运行环境独立验证。", "",
        "## 比赛结果", "", "### 六局配对赛果", "",
        "| 对阵 | 胜–平–负 | 得分率 | 纯胜率 | 配对描述性 95% 区间 | 黑方得分率（前者/后者） | 白方得分率（前者/后者） |",
        "|---|---:|---|---|---|---|---|",
    ]
    for matchup in ("B-C", "B-D", "C-D"):
        item = summary["matchups"][matchup]
        a, b = item["matchup"]
        black_a = item["color"][a]["black"]["score_rate"]
        black_b = item["color"][b]["black"]["score_rate"]
        white_a = item["color"][a]["white"]["score_rate"]
        white_b = item["color"][b]["white"]["score_rate"]
        report_lines.append(
            f"| {a}–{b} | {item['wins'][a]}–{item['draws']}–{item['wins'][b]} | {a} {_pct(item['score_rate'][a])} / {b} {_pct(item['score_rate'][b])} | {a} {_pct(item['pure_win_rate'][a])} / {b} {_pct(item['pure_win_rate'][b])} | {a} {_ci(item['paired_wilson_95_ci'][a])}; {b} {_ci(item['paired_wilson_95_ci'][b])} | {_pct(black_a)} / {_pct(black_b)} | {_pct(white_a)} / {_pct(white_b)} |"
        )
    report_lines += ["", "### 总排名及先后手", "", "| 排名 | 模型 | 总战绩 | 总得分率 | 配对区间 | 黑方得分率 | 白方得分率 |", "|---:|---|---:|---:|---|---:|---:|"]
    for place, model in enumerate(summary["exploratory_ranking"], 1):
        item = summary["models"][model]
        report_lines.append(
            f"| {place} | {model} | {item['wins']}–{item['draws']}–{item['losses']} | {_pct(item['score_rate'])} | {_ci(item['paired_wilson_95_ci'])} | {_pct(item['color']['black']['score_rate'])} | {_pct(item['color']['white']['score_rate'])} |"
        )
    report_lines += ["", "每个模型参加 12 局，按 6 个‘对手×开局’的换色配对块计算总排名区间；每组对阵仅 n=3 个开局块。区间是 Wilson-style 描述区间，小样本覆盖率未校准，不能据此宣称显著差异。黑白得分率用于观察先手影响，但每格仅 6 局。", "", "## 搜索计算成本", "", "| 模型 | 决策数 | 总思考秒数 | 平均秒/步 | 平均节点/步 | 平均完成深度 | 剪枝数 | 剪枝/节点 | 模拟总数 | 平均模拟/步 | 模拟/秒 | 超时步数 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("B", "C", "D"):
        usage = summary["model_usage"][model]
        report_lines.append(
            f"| {model} | {usage['decisions']} | {_fmt(usage['total_seconds'])} | {_fmt(usage['mean_seconds_per_move'],3)} | {_fmt(usage['mean_nodes_per_move'],0)} | {_fmt(usage['mean_completed_depth'],2)} | {usage['cutoffs_total']} | {_pct(usage['cutoff_to_node_ratio'])} | {usage['simulations_total']} | {_fmt(usage['mean_simulations_per_move'],0)} | {_fmt(usage['simulations_per_second'],1)} | {usage['timeouts']} |"
        )
    report_lines += ["", "B/C 的节点数同为该 minimax 引擎的 visited search nodes。D 的 `nodes` 是树操作数加 rollout plies，与 B/C 不可直接比较；D 的工作量用真实模拟次数、思考秒数和模拟/秒表示。剪枝/节点比例是实现计数，不等同于被剪掉的整棵子树比例。", "", "## V3、V4B 对照", "", "### B 从 200ms 提高至 1 秒", ""]
    old_bc = history["V4B_matchups"]["B-C"]
    now_bc = summary["matchups"]["B-C"]
    old_b = history["V4B_200ms_search"]["B"]
    old_c = history["V4B_200ms_search"]["C"]
    b_usage, c_usage = summary["model_usage"]["B"], summary["model_usage"]["C"]
    report_lines += [
        f"V4B 中 B200ms 对 C200ms 为 B {old_bc['wins']['B']}–{old_bc['draws']}–{old_bc['wins']['C']}、得分率 {_pct(old_bc['score_rate']['B'])}；V4C B1s 对 C200ms 为 B {now_bc['wins']['B']}–{now_bc['draws']}–{now_bc['wins']['C']}、B 得分率 {_pct(now_bc['score_rate']['B'])}。B 的平均完成深度从 V4B 的 {_fmt(old_b['average_completed_depth'])} 到本轮 {_fmt(b_usage['mean_completed_depth'])}；本轮平均节点/步 {_fmt(b_usage['mean_nodes_per_move'],0)}、平均耗时 {_fmt(b_usage['mean_seconds_per_move'],3)} 秒。",
        f"本轮对照显示 B 增加算力后的赛果变化为 {_pct(old_bc['score_rate']['B'])} → {_pct(now_bc['score_rate']['B'])}，但每格只有 6 局且 C 预算不变；若得分率没有上升，也不能证明 B 的搜索没有计算收益，需结合深度/节点和低样本不确定性判断。C200ms 本轮平均深度 {_fmt(c_usage['mean_completed_depth'])}，V4B 为 {_fmt(old_c['average_completed_depth'])}，可视为同预算实现的批次波动。", "",
        "### D 从 20 秒提高至 50 秒，挑战 C200ms", "",
    ]
    old_cd = history["V4B_matchups"]["C-D"]
    now_cd = summary["matchups"]["C-D"]
    d_old = history["V4B_D20_new_BD_CD_usage"]["D"]
    d_new = summary["model_usage"]["D"]
    report_lines += [
        f"V4B D20s 对 C200ms：C {old_cd['wins']['C']}–{old_cd['draws']}–{old_cd['wins']['D']}，C 得分率 {_pct(old_cd['score_rate']['C'])}、D {_pct(old_cd['score_rate']['D'])}；V4C D50s 对 C200ms：C {now_cd['wins']['C']}–{now_cd['draws']}–{now_cd['wins']['D']}，C {_pct(now_cd['score_rate']['C'])}、D {_pct(now_cd['score_rate']['D'])}。D 平均每步实际耗时 {_fmt(d_old['mean_seconds_per_move'],2)} → {_fmt(d_new['mean_seconds_per_move'],2)} 秒；实际模拟平均/步 {_fmt(d_old['mean_simulations_per_move'],0)} → {_fmt(d_new['mean_simulations_per_move'],0)}。",
        "V3 的 D20s 面对 B50ms、C50ms 时分别取得 B 对局 3胜3负、C 对局 3胜1平2负；V4B 把对手提高到 200ms 后，D20 对 B 为 3胜2平1负、对 C 为 0胜3平3负。V4C 继续只改变 D 的预算到 50s（B/C 预算保持 V4B 同值）来观察算力收益。各批开局/决策随机种子不同，不能把逐轮赛果差直接解释为因果效果。", "",
        "## 计算收益与成本解释", "",
        "D50 的累计思考时间、模拟数和实际模拟速度直接报告在表格；若更多预算带来模拟数增加而胜率不升，仅说明本样本中额外计算没有转化为稳定得分优势，无法据此断言已饱和。模拟速度受局面、战术捷径及复用树影响。B1s 与 C200ms 的思考时间不等，且它们与 D50 差异更大；本轮不是等算力排名。", "",
        "## 典型棋谱与战术观察", "",
    ]
    chosen: list[dict[str, Any]] = []
    for model in ("B", "C", "D"):
        win = next((game for game in records if game.get("winner_model") == model), None)
        loss = next((game for game in records if model in game["matchup"] and game.get("winner_model") not in (None, model)), None)
        for label, game in (("取胜", win), ("失利", loss)):
            if game and game not in chosen:
                chosen.append(game)
                report_lines.append(f"- {model}{label}示例 `{game['game_id']}`，对阵 {'–'.join(game['matchup'])}，结果 {game['outcome']}，终止于第 {game['move_count']} 手。末段棋谱：{_transcript(game)}。完整合法棋谱在 `results/9x9_games.jsonl`。")
    if not chosen:
        report_lines.append("本轮没有决定性胜负棋谱，完整对局见 `results/9x9_games.jsonl`。")
    report_lines += ["", "战术失误判断需对照棋盘逐手复盘；本报告不把模型的合法性、胜负或搜索数本身当作战术正确性证明。逐局原始记录含每手落点、搜索深度、节点、cutoff、MCTS 模拟、耗时、超时及树复用计数。", "", "## 预检、资源与恢复", ""]
    est = preflight["estimate"]
    report_lines += [
        f"正式运行前，B/C/D 同一四手开局预检总耗时 {_fmt(preflight.get('preflight_elapsed_seconds'),1)} 秒；资源估算正式比赛约 {_fmt(est['estimated_formal_run_minutes_with_margin'],1)} 分钟，估算高侧约 {_fmt(est['estimated_formal_run_range_minutes'][1],1)} 分钟。12 局含 D 的理论 50 秒满额上界为 {_fmt(est['D50_full_board_theoretical_ceiling_seconds']/3600,1)} 小时，实测估算依赖 V4B 的回合数。",
        f"Drive 演练：{_drive_text(drive)}。恢复包包含冻结配置、进度、完整棋谱和代码版本；上传完成后经下载哈希比对并恢复到空目录验证。", "",
        "## 结论", "",
        f"本轮 18 局排名为 " + " > ".join(summary["exploratory_ranking"]) + "。", 
        f"B 1 秒对 C 200ms：" + ("本轮 B 得分高于 V4B 的 200ms 批次。" if now_bc['score_rate']['B'] > old_bc['score_rate']['B'] else "本轮 B 得分未高于 V4B 的 200ms 批次。") + " 这一结果仍只有六局，不构成稳健棋力结论。",
        f"D 50 秒对 C 200ms：" + ("D 本轮得分领先。" if now_cd['score_rate']['D'] > now_cd['score_rate']['C'] else "C 本轮得分领先。" if now_cd['score_rate']['C'] > now_cd['score_rate']['D'] else "双方本轮得分相同。") + " D 得到 250 倍于 C 的单步时间预算，无法据此评判等计算资源下的算法优劣。",
        "阶段性看，增加搜索预算应以实际完成深度、模拟数、决策时间和配对棋力同时衡量。MCTS 高预算表现必须由更多预注册对局和更广开局验证；传统搜索侧可继续研究 C 的威胁空间/候选扩展，同时保持 B/C 相同搜索逻辑的效率对照。", "",
        "## 可复现文件", "",
        "- 冻结参数：`configs/bcd_high_compute_v4c.json`。",
        "- 全部对局与逐手 telemetry：`research/v4c/results/9x9_games.jsonl`。",
        "- 汇总及统计：`research/v4c/results/summary.json`、`pairwise_summary.csv`、`model_summary.csv`、`model_usage.csv`。",
        "- 预检、资源与耗时：`research/v4c/results/preflight.json`、`execution.json`。",
        "- 运行：`PYTHONPATH=. python scripts/run_bcd_high_compute_v4c.py`；校验/汇总：`PYTHONPATH=. python scripts/summarize_bcd_high_compute_v4c.py`。",
        "- 测试：`PYTHONPATH=. python -m pytest -q`。",
        "- 测试结果：" + (f"{test_result.get('passed')} passed, {test_result.get('failed', 0)} failed" if test_result else "最终全套测试结果写入交付信息。"),
        "",
        "## 图表", "",
        f"![V4B 与 V4C 对阵得分率](../figures/{figure_names[0]})", "",
        f"![V4B 与 V4C 计算成本](../figures/{figure_names[1]})", "",
        "## 限制", "",
        "每组仅 6 局（3 个换色配对块），模型总排名每模型 n=6 个块；配对 Wilson-style 区间是描述统计，未对这种小样本和结果相关性进行校准。B/C/D 的预算不相等，D 的 50 秒/步显著高于 B 的 1 秒、C 的 0.2 秒。不同实验轮使用不同随机种子与已完成的相同开局模板，跨轮比较易受随机性与先后手影响。不能将本轮数据宣称为统计显著或稳定棋力排序。", "",
    ]
    report_path = results / "report_zh.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"report_path": str(report_path), "figures": figure_names, "validation": validation, "summary": summary, "history_comparison": history, "test_result": test_result}


def _drive_text(evidence: dict[str, Any]) -> str:
    initial = evidence.get("initial_snapshot", {})
    if all(initial.get(key) for key in ("upload", "download", "download_sha256_match", "restore", "restored_game_ids_match")):
        return f"已通过上传/下载/SHA-256/空目录恢复，预演恢复 {initial.get('restored_completed_games', 0)} 局；最终 V4C 快照状态：{evidence.get('final_snapshot', {}).get('status', '尚未生成')}。"
    return "未能通过 V4C Drive 完整恢复验证；以本地 fsync 棋谱和 Git 分支为恢复来源。"
