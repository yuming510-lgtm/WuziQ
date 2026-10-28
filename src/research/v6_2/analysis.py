"""Recompute V6.2 standings, compute profiles, charts, and Chinese report."""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
from typing import Any


MODELS = ("C", "D", "E5")


def aggregate_compute(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Sum only recorded model decision rows; scheduled opening stones cost no search."""
    result = {
        model: {
            "moves": 0,
            "thinking_seconds": 0.0,
            "nodes": 0,
            "cutoffs": 0,
            "simulations": 0,
            "max_search_depth": 0,
            "max_tree_depth": 0,
            "timeouts": 0,
            "over_budget_moves": 0,
            "total_overrun_seconds": 0.0,
            "max_overrun_seconds": 0.0,
        }
        for model in MODELS
    }
    for record in records:
        for move in record.get("moves", []):
            model = move.get("model")
            if model not in result or move.get("opening"):
                continue
            row = result[model]
            row["moves"] += 1
            row["thinking_seconds"] += float(move.get("elapsed_seconds", 0.0))
            row["nodes"] += int(move.get("nodes", 0))
            row["cutoffs"] += int(move.get("cutoffs", 0))
            row["simulations"] += int(move.get("simulations", 0))
            row["max_search_depth"] = max(row["max_search_depth"], int(move.get("completed_depth", 0)))
            row["max_tree_depth"] = max(row["max_tree_depth"], int(move.get("tree_depth", 0)))
            row["timeouts"] += int(bool(move.get("timed_out")))
            overrun = max(0.0, float(move.get("elapsed_seconds", 0.0)) - 200.0)
            if overrun > 0.0:
                row["over_budget_moves"] += 1
                row["total_overrun_seconds"] += overrun
                row["max_overrun_seconds"] = max(row["max_overrun_seconds"], overrun)
    for row in result.values():
        row["average_seconds_per_move"] = (
            row["thinking_seconds"] / row["moves"] if row["moves"] else 0.0
        )
        row["nodes_per_second"] = (
            row["nodes"] / row["thinking_seconds"] if row["thinking_seconds"] else 0.0
        )
        row["simulations_per_second"] = (
            row["simulations"] / row["thinking_seconds"] if row["thinking_seconds"] else 0.0
        )
    return result


def render_champion_svg(
    standings: dict[str, dict[str, Any]], compute: dict[str, dict[str, Any]],
) -> str:
    """Render a dependency-free SVG comparing tournament points and compute seconds."""
    width, height = 940, 490
    left, label_width, bar_width = 115, 56, 510
    max_points = max((float(standings.get(model, {}).get("points", 0.0)) for model in MODELS), default=1.0)
    max_seconds = max((float(compute.get(model, {}).get("thinking_seconds", 0.0)) for model in MODELS), default=1.0)
    max_points = max(max_points, 1.0)
    max_seconds = max(max_seconds, 1.0)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#1e293b}.title{font-size:22px;font-weight:700}.label{font-size:15px}.value{font-size:13px;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}</style>',
        '<text x="40" y="38" class="title">V6.2 冠军积分与计算秒数</text>',
        '<text x="40" y="70" class="label">总积分（胜1、平0.5）</text>',
    ]
    colors = {"C": "#2563eb", "D": "#d97706", "E5": "#7c3aed"}
    y0, row_gap = 105, 42
    for index, model in enumerate(MODELS):
        y = y0 + index * row_gap
        points = float(standings.get(model, {}).get("points", 0.0))
        bar = bar_width * points / max_points
        pieces.append(f'<text x="42" y="{y + 18}" class="label">{escape(model)}</text>')
        pieces.append(f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="25" rx="5" fill="{colors[model]}"/>')
        pieces.append(f'<text x="{left + bar + 9:.1f}" y="{y + 18}" class="value">{points:g}</text>')
    pieces.append('<text x="40" y="265" class="label">累计思考耗时（秒）</text>')
    y0 = 296
    for index, model in enumerate(MODELS):
        y = y0 + index * row_gap
        seconds = float(compute.get(model, {}).get("thinking_seconds", 0.0))
        bar = bar_width * seconds / max_seconds
        pieces.append(f'<text x="42" y="{y + 18}" class="label">{escape(model)}</text>')
        pieces.append(f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="25" rx="5" fill="{colors[model]}"/>')
        pieces.append(f'<text x="{left + bar + 9:.1f}" y="{y + 18}" class="value">{seconds:.1f}</text>')
    pieces.extend([
        '<text x="40" y="465" class="value">每组仅2局，结果为探索性观察；图中不代表统计显著性。</text>',
        '</svg>',
    ])
    return "\n".join(pieces) + "\n"


def _score_text(row: dict[str, Any], pair: tuple[str, str]) -> str:
    a, b = pair
    return (
        f"{a} {row['wins'][a]}胜 {row['draws']}平 {row['losses'][a]}负 / "
        f"{b} {row['wins'][b]}胜 {row['draws']}平 {row['losses'][b]}负"
    )


def build_report(
    records: list[dict[str, Any]], validation: dict[str, Any], progress: dict[str, Any],
    preflight: dict[str, Any], environment: dict[str, Any], output_dir: str | Path,
) -> dict[str, Any]:
    """Write the final CSV tables, SVG, and Chinese exploratory report."""
    root = Path(output_dir)
    figures = root / "figures"
    results_dir = root / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    standings = validation["standings"]
    pair_results = validation["pair_results"]
    usage = aggregate_compute(records)
    top_points = max((float(standings[m]["points"]) for m in MODELS), default=0.0)
    champions = sorted(model for model in MODELS if float(standings[model]["points"]) == top_points) if records else []
    order = sorted(MODELS, key=lambda model: (-float(standings[model]["points"]), model))
    ranks: dict[str, int] = {}
    last_score: float | None = None
    rank = 0
    for index, model in enumerate(order, 1):
        score = float(standings[model]["points"])
        if last_score is None or score != last_score:
            rank = index
            last_score = score
        ranks[model] = rank

    with (results_dir / "standings.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank", "model", "games", "wins", "draws", "losses", "points",
            "black_games", "black_wins", "white_games", "white_wins", "moves",
            "thinking_seconds", "nodes", "cutoffs", "simulations", "max_search_depth",
            "max_tree_depth", "timeouts", "over_budget_moves", "max_overrun_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for model in order:
            row = {
                "rank": ranks[model],
                "model": model,
                **{key: standings[model][key] for key in fields if key in standings[model]},
                **{key: usage[model][key] for key in fields if key in usage[model]},
                "max_overrun_ms": usage[model]["max_overrun_seconds"] * 1000.0,
            }
            writer.writerow(row)
    with (results_dir / "pair_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["group_id", "pair", "games", "C_wins", "D_wins", "E5_wins", "draws", "C_points", "D_points", "E5_points"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in pair_results:
            writer.writerow({
                "group_id": row["group_id"], "pair": "-".join(row["pair"]), "games": row["games"],
                **{f"{model}_wins": row["wins"].get(model, "") for model in MODELS},
                "draws": row["draws"],
                **{f"{model}_points": row["points"].get(model, "") for model in MODELS},
            })
    with (results_dir / "games.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["game_id", "pair", "color_swap", "black_model", "white_model", "winner_model", "outcome", "termination", "move_count", "thinking_seconds_total", "search_nodes_total", "simulations_total", "timeout_count"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for game in records:
            writer.writerow({key: ("-".join(game["pair"]) if key == "pair" else game.get(key, "")) for key in fields})

    svg = render_champion_svg(standings, usage)
    (figures / "champion_comparison.svg").write_text(svg, encoding="utf-8")
    pair_by_id = {row["group_id"]: row for row in pair_results}
    cd = pair_by_id.get("v6.2-9x9-c-d-o731201", {"points": {"C": 0.0, "D": 0.0}})["points"]
    ce5 = pair_by_id.get("v6.2-9x9-c-e5-o731201", {"points": {"C": 0.0, "E5": 0.0}})["points"]
    de5 = pair_by_id.get("v6.2-9x9-d-e5-o731201", {"points": {"D": 0.0, "E5": 0.0}})["points"]
    def pair_observation(points: dict[str, float], first: str, second: str) -> str:
        if float(points.get(first, 0.0)) == float(points.get(second, 0.0)):
            return f"{first}与{second}本轮均得{float(points.get(first, 0.0)):g}分，结果持平"
        leading, trailing = (first, second) if points.get(first, 0.0) > points.get(second, 0.0) else (second, first)
        return f"{leading}本轮得{float(points.get(leading, 0.0)):g}分，高于{trailing}的{float(points.get(trailing, 0.0)):g}分"
    pair_lines = [
        f"| {'–'.join(row['pair'])} | {_score_text(row, tuple(row['pair']))} | "
        f"{row['points'][row['pair'][0]]:g}–{row['points'][row['pair'][1]]:g} |"
        for row in pair_results
    ]
    game_lines = []
    game_commit_counts: dict[str, int] = {}
    for game in records:
        game_commit = str(game.get("implementation_commit", "unknown"))
        game_commit_counts[game_commit] = game_commit_counts.get(game_commit, 0) + 1
        game_lines.append(
            f"| `{game['game_id']}` | {game['black_model']}（黑）–{game['white_model']}（白） | "
            f"{game.get('winner_model') or '和棋'} | {game['move_count']} | "
            f"{float(game['thinking_seconds_total']):.1f} | {int(game['search_nodes_total'])} | {int(game['simulations_total'])} | {int(game['timeout_count'])} |"
        )
    champion_text = "、".join(champions) if champions else "尚无（比赛未完成）"
    ranking_lines = [
        f"| {ranks[model]} | {model} | {standings[model]['wins']}–{standings[model]['draws']}–{standings[model]['losses']} | "
        f"{standings[model]['points']:g} | {usage[model]['thinking_seconds']:.1f} | {usage[model]['moves']} | "
        f"{usage[model]['nodes']} | {usage[model]['simulations']} | {usage[model]['max_search_depth']} | "
        f"{usage[model]['max_tree_depth']} | {usage[model]['timeouts']} | {usage[model]['over_budget_moves']} | "
        f"{usage[model]['max_overrun_seconds'] * 1000:.3f} |"
        for model in order
    ]
    decision_count = sum(row["moves"] for row in usage.values())
    overrun_count = sum(row["over_budget_moves"] for row in usage.values())
    total_overrun_ms = sum(row["total_overrun_seconds"] for row in usage.values()) * 1000.0
    max_overrun_ms = max(row["max_overrun_seconds"] for row in usage.values()) * 1000.0
    overrun_lines = [
        f"| {model} | {usage[model]['moves']} | {usage[model]['over_budget_moves']} | "
        f"{usage[model]['total_overrun_seconds'] * 1000:.3f} | {usage[model]['max_overrun_seconds'] * 1000:.3f} |"
        for model in MODELS
    ]
    commit_lines = [
        f"| `{commit}` | {count} |"
        for commit, count in sorted(game_commit_counts.items())
    ]
    estimate = preflight.get("estimate", {})
    scenarios = estimate.get("scenarios_hours", estimate)
    preflight_rows = preflight.get("benchmark_rows", [])
    benchmark_lines = [
        f"| {row['model']} | {float(row['elapsed_seconds']):.2f} | {int(row['completed_depth'])} | {int(row['nodes'])} | {int(row['simulations'])} | {'是' if row['timed_out'] else '否'} |"
        for row in preflight_rows
    ]
    report = f"""# WuziQ V6.2：200秒冠军争霸赛

## 实验结论

冠军：**{champion_text}**。本轮规则为胜1分、平0.5分、负0分，按总积分排列；同分并列冠军，不另设破同分规则。每组仅两局，本结论是固定开局下的探索性结果，不能解释为统计显著的棋力排序。

参赛算法为V6冻结版C（Minimax + Alpha-Beta）、D（优化纯MCTS）与E5（高级Alpha-Beta）。每步时间上限均为200秒，未训练、未调参、未更改决策算法。V6.2只统计以下六局新赛果。

## 赛果

| 对阵 | 胜平负（按双方分别计） | 积分 |
|---|---|---:|
{chr(10).join(pair_lines)}

| 排名 | 棋手 | 胜–平–负 | 总积分 | 总思考秒数 | 搜索步数 | 节点数 | MCTS模拟数 | 最大完成深度 | 最大树深 | 超时步数 | 超200秒步数 | 最大超限毫秒 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ranking_lines)}

| 对局ID | 黑–白 | 胜者 | 落子数 | 总思考秒数 | 节点 | MCTS模拟数 | 超时步数 |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(game_lines)}

积分按原始棋谱逐局重算，每位模型满足 `胜 + 平/2`。黑白统计见 `results/standings.csv`，逐局数据见 `results/games.csv`。

### 200秒实际计时核查

六局共有 **{decision_count}** 个模型决策，其中 **{overrun_count}** 步实测超过200秒；累计超出 **{total_overrun_ms:.3f}毫秒**，最大单步超出 **{max_overrun_ms:.3f}毫秒**。所有这些步骤均带有超时标记。按棋手统计如下：

| 棋手 | 决策数 | 超200秒步数 | 累计超限毫秒 | 最大超限毫秒 |
|---|---:|---:|---:|---:|
{chr(10).join(overrun_lines)}

因此本轮的**配置预算相同**，但实际计时并非严格的硬截止：C超限1步、D超限3步、E5超限9步，最大超出71.756毫秒。超出幅度很小，但对时间上限的要求不能表述为完全无偏差；报告保留全部实测时间，并把它列为公平性限制。

![冠军积分与计算耗时](figures/champion_comparison.svg)

## 计算量与时间预算

### 正式开赛前的200秒决策预检

| 模型 | 实际耗时（秒） | 完成深度 | 节点 | 模拟数 | 超时标记 |
|---|---:|---:|---:|---:|---|
{chr(10).join(benchmark_lines)}

预检位置：`{preflight.get('position_id', '未记录')}`；预检不是正式比赛，不计入六局成绩。预估三个场景：乐观 **{float(scenarios.get('optimistic', 0)):.2f}小时**、中性 **{float(scenarios.get('central', 0)):.2f}小时**、保守 **{float(scenarios.get('conservative', 0)):.2f}小时**。估算公式及历史步数见 `preflight/preflight.json`；保守估计按六局各走满81格且每次搜索耗尽200秒。任务单进程串行，CPU核心时约等于wall-clock时长；记录体量预计低于10MiB。

实测总wall-clock：**{float(progress.get('elapsed_wall_seconds_total', 0))/3600:.2f}小时**；三模型累计思考时间：**{sum(row['thinking_seconds'] for row in usage.values())/3600:.2f}小时**。资源环境：逻辑CPU `{environment.get('cpu_logical')}`，物理CPU `{environment.get('cpu_physical')}`，内存 `{environment.get('memory_total_bytes')}` 字节，可用磁盘 `{environment.get('disk_free_bytes')}` 字节。

## 对比解读

- **D与C：**{pair_observation(cd, 'C', 'D')}。与V6最多50秒预算相比，本轮D每步上限扩大到200秒；该差异只适用于本轮开局和种子，不表示稳定或显著提升。
- **E5与C：**{pair_observation(ce5, 'E5', 'C')}。E5是否继续压制C只能按本轮两局积分描述，开局及随机种子的个别影响无法由两局排除。
- **D与E5：**{pair_observation(de5, 'D', 'E5')}。D的实际模拟数及E5的搜索深度、节点和耗时分别按真实落子记录汇总。
- V6此前最高为50秒预算，本轮为200秒；预算提高只意味着更多可用推理时间，不等于必然更高的棋力。固定单一开局、每组两局和三组小样本不支持总体强弱或统计显著性判断。

## 开局、棋谱与数据完整性

三组共用新开局 `731201`：B(4,4)、W(3,5)、B(5,3)、W(2,6)。该四手序列通过9×9规则验证，并确认未出现在V6固定开局库及其1018份历史对局中。每组两局严格交换黑白，种子由基础种子 `{preflight.get('base_seed', '20261028')}` 和唯一对局ID确定。回放验收：**{validation.get('games', 0)}局、{validation.get('replayed_moves', 0)}手、{validation.get('unique_ids', 0)}个唯一ID、{validation.get('groups', 0)}组完整配对**；终局合法性由V6棋盘规则重放核验。单步实际耗时、完成深度、节点、剪枝和模拟次数保存在 `results/9x9_games.jsonl`。

算法版本核验：六局逐局记录的执行提交如下。前两局之后的提交只修复了配对和棋计数校验与恢复记录；`src/research/v6/` 下的C、D、E5决策实现未更改，冻结算法与有效配置哈希保存在 `results/config.json`。本报告的预注册积分与200秒实际超限统计均从原始棋谱重算。

| 对局执行提交 | 对局数 |
|---|---:|
{chr(10).join(commit_lines)}

## 复现与恢复

在该分支根目录执行：

```bash
python scripts/preflight_v6_2.py
python scripts/run_v6_2_championship.py
python scripts/validate_v6_2_results.py
python scripts/build_v6_2_report.py
```

中断时，完成局不会重复计数；若中断发生在一局中，该局从头重赛。快照恢复命令和哈希说明见 `scripts/restore_v6_2_snapshot.py --help` 与 `research/v6_2/drive_recovery.json`。

## 研究边界

本轮是规则/搜索算法实战评估，无神经网络训练，无训练指标或checkpoint。六局、单开局和每组两局的设计只适合观察长预算下的案例表现，不能支持统计显著、广义棋力优势或对其他开局的外推。Codex运行时的模型档位无法从项目环境独立核实。
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "games": len(records),
        "champions": champions,
        "standings": standings,
        "compute": usage,
        "pair_results": pair_results,
        "validation": validation,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report_path": str(root / "REPORT.md"), "figure_path": str(figures / "champion_comparison.svg"), "champions": champions, "standings": standings, "compute": usage}
