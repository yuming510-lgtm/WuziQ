"""Build Chinese V5 report and the family-budget ladder SVG from validated data."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from .checkpoint import write_json_atomic

_MODEL_COLOR = {"A": "#2563eb", "B": "#0f766e", "C": "#d97706", "D": "#7c3aed"}
_BUDGETS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ci(value: list[float] | None) -> str:
    return "—" if value is None else f"[{value[0] * 100:.1f}%, {value[1] * 100:.1f}%]"


def _budget_index(label: str) -> int:
    return _BUDGETS.index(label)


def _make_gradient_svg(routes: dict[str, Any], output_path: Path) -> None:
    width, left, right = 1120, 205, 90
    row_h, top = 48, 98
    route_items = list(routes.items())
    height = top + row_h * len(route_items) + 72
    x0, x1 = left + 10, width - right
    xs = [x0 + (x1 - x0) * i / (len(_BUDGETS) - 1) for i in range(len(_BUDGETS))]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#18212f}.small{font-size:12px}.label{font-size:15px;font-weight:600}.title{font-size:21px;font-weight:700}.muted{fill:#667085}</style>',
        '<text x="28" y="35" class="title">V5 算法家族阶梯挑战边界</text>',
        '<text x="28" y="61" class="small muted">颜色表示挑战家族；红叉失败，空心点未运行，未画出的档位没有结果</text>',
    ]
    for index, budget in enumerate(_BUDGETS):
        x = xs[index]
        parts.append(f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="{height - 36}" stroke="#e6eaf0" stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{x:.1f}" y="84" text-anchor="middle" class="small">{budget}</text>')
    for row, (route_id, route) in enumerate(route_items):
        y = top + row * row_h
        challenger = route["challenger"]
        color = _MODEL_COLOR[challenger]
        parts.append(f'<text x="28" y="{y + 5}" class="label">{escape(route_id)}</text>')
        parts.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="#d7dde6" stroke-width="2"/>')
        passed_x: list[float] = []
        for level in route["levels"]:
            i = _budget_index(level["budget"])
            x = xs[i]
            status = level["status"]
            if status == "passed":
                passed_x.append(x)
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            elif status == "failed":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>')
                parts.append(f'<path d="M{x-4:.1f},{y-4:.1f} L{x+4:.1f},{y+4:.1f} M{x+4:.1f},{y-4:.1f} L{x-4:.1f},{y+4:.1f}" stroke="#ffffff" stroke-width="2"/>')
            elif status == "incomplete_pair":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#fff7ed" stroke="#d97706" stroke-width="2"/>')
            elif status in {"pending", "not_reached"}:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#ffffff" stroke="#98a2b3" stroke-width="2"/>')
        if len(passed_x) > 1:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x in passed_x)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" opacity="0.8"/>')
            # Repaint markers above the connecting line.
            for x in passed_x:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        status_text = route["highest_passed_budget"] or route["status"]
        parts.append(f'<text x="{width - right + 12}" y="{y + 4}" class="small">{escape(str(status_text))}</text>')
    legend_y = height - 18
    for index, model in enumerate(("A", "B", "C", "D")):
        x = left + index * 48
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{_MODEL_COLOR[model]}"/><text x="{x+10}" y="{legend_y+4}" class="small">{model}</text>')
    parts.append(f'<circle cx="{left+224}" cy="{legend_y}" r="6" fill="#dc2626"/><text x="{left+234}" y="{legend_y+4}" class="small">失败档</text>')
    parts.append(f'<circle cx="{left+302}" cy="{legend_y}" r="6" fill="#ffffff" stroke="#98a2b3" stroke-width="2"/><text x="{left+312}" y="{legend_y+4}" class="small">未运行</text>')
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _history_section(results_dir: Path) -> str:
    path = results_dir / "history_comparison.json"
    if not path.exists():
        return "历史结果保存在各自 V1–V4C 目录，未混入 V5 晋级判定。本节待补历史基线摘要。"
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = ["历史结果仅作背景对照，不并入 V5 胜负或路线统计。", "", "| 阶段 | 历史观测 |", "|---|---|"]
    for item in data.get("comparisons", []):
        lines.append(f"| {item.get('stage', '—')} | {item.get('finding', '—')} |")
    lines.extend(["", f"来源校验：`{data.get('sources', 'V1–V4C 仓库内已提交结果')}`。"])
    return "\n".join(lines)


def build_report(analysis: dict[str, Any], results_dir: str | Path, output_dir: str | Path) -> Path:
    results = Path(results_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_states = analysis["route_state"]["routes"]
    figure = output / "figures" / "ladder_gradient.svg"
    _make_gradient_svg(route_states, figure)

    route_lines = ["| 挑战路线 | 各档结果 | 最高通过 | 失败档 |", "|---|---|---:|---:|"]
    for route_id, route in route_states.items():
        level_text = " → ".join(
            f"{level['budget']} { {'passed':'✓','failed':'×','pending':'…','incomplete_pair':'半组','skipped_after_failure':'—','not_reached':'—'}.get(level['status'], '—') }"
            + (f" ({level['score']:.1f})" if level["score"] is not None else "")
            for level in route["levels"]
        )
        route_lines.append(
            f"| `{route_id}` | {level_text} | {route['highest_passed_budget'] or '无'} | {route['failed_budget'] or '无（尚未失败）'} |"
        )

    matrix = {row["pair"]: row for row in analysis["pairwise"]}
    model_order = ("A", "B", "C", "D")
    matrix_lines = ["| 行 \\ 列 | A | B | C | D |", "|---|---:|---:|---:|---:|"]
    for row_model in model_order:
        cells = []
        for col_model in model_order:
            if row_model == col_model:
                cells.append("—")
                continue
            pair = tuple(sorted((row_model, col_model)))
            info = matrix.get("-".join(pair))
            if not info or not info["games"]:
                cells.append("未运行")
                continue
            if row_model == pair[0]:
                wins, losses = info["first_wins"], info["second_wins"]
                points, rate = info["first_score"], info["first_score_rate"]
            else:
                wins, losses = info["second_wins"], info["first_wins"]
                points, rate = info["second_score"], info["second_score_rate"]
            cells.append(f"{wins}胜/{info['draws']}平/{losses}负，{points:.1f}/{info['games']}分（{_pct(rate)}）")
        matrix_lines.append(f"| {row_model} | " + " | ".join(cells) + " |")

    pair_lines = ["| 配对 | 已完成组/局 | 首模型胜–平–次模型胜 | 首模型得分率 [95% 描述区间] | 次模型得分率 [95% 描述区间] |", "|---|---:|---:|---:|---:|"]
    for row in analysis["pairwise"]:
        pair_lines.append(
            f"| {row['pair']} | {row['groups']} / {row['games']} | {row['first_wins']}–{row['draws']}–{row['second_wins']} "
            f"| {_pct(row['first_score_rate'])} {_ci(row['first_group_score_ci95'])} "
            f"| {_pct(row['second_score_rate'])} {_ci(row['second_group_score_ci95'])} |"
        )

    ranking = sorted(analysis["models"].items(), key=lambda row: (-float(row[1]["score_rate"] or 0), row[0]))
    rank_lines = ["| 排名（描述性） | 模型 | 胜–平–负 | 局数 | 得分率 | 95% 组级描述区间 |", "|---:|---|---:|---:|---:|---:|"]
    for place, (model, stats) in enumerate(ranking, start=1):
        rank_lines.append(
            f"| {place} | {model} | {stats['wins']}–{stats['draws']}–{stats['losses']} | {stats['games']} "
            f"| {_pct(stats['score_rate'])} | {_ci(stats['score_rate_ci95_paired_groups'])} |"
        )

    usage_lines = ["| 模型 | 实际预算档 | 决策数 | 平均秒/步 | 平均深度 | 峰值深度 | 节点/步 | 剪枝/节点 | 模拟/步 | 模拟/秒 | 超时数 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["model_usage"]:
        usage_lines.append(
            f"| {row['model']} | {row['budget_label']} | {row['decisions']} | {row['mean_seconds_per_decision']:.4f} "
            f"| {row['mean_completed_depth']:.2f} | {row['max_depth']} | {row['mean_nodes_per_decision']:.1f} "
            f"| {row['cutoff_per_node']:.3f} | {row['simulations']/row['decisions'] if row['decisions'] else 0:.1f} "
            f"| {row['simulations_per_second']:.1f} | {row['timeouts']} |"
        )

    group_lines = ["| 对阵 | 阶档 | 挑战路线 | 开局 ID | 黑胜 | 白胜 | 平 | 结果分数 |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for row in analysis["groups"]:
        route_scores = json.loads(row["route_scores"])
        score_text = "; ".join(f"{route}={score:.1f}" for route, score in route_scores.items())
        group_lines.append(
            f"| {row['pair']} | {row['rung_budget']} | `{row['route_ids']}` | {row['opening_seed']} "
            f"| {row['black_wins']} | {row['white_wins']} | {row['draws']} | {score_text} |"
        )

    side_lines = ["| 模型 | 黑方局/分 | 白方局/分 | 黑方得分率 | 白方得分率 |", "|---|---:|---:|---:|---:|"]
    for model in model_order:
        row = analysis["models"][model]
        black_rate = row["black_points"] / row["black_games"] if row["black_games"] else None
        white_rate = row["white_points"] / row["white_games"] if row["white_games"] else None
        side_lines.append(
            f"| {model} | {row['black_games']} / {row['black_points']:.1f} | {row['white_games']} / {row['white_points']:.1f} "
            f"| {_pct(black_rate)} | {_pct(white_rate)} |"
        )

    progress_path = results / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    drive_path = results / "drive_recovery.json"
    drive = json.loads(drive_path.read_text(encoding="utf-8")) if drive_path.exists() else {}
    elapsed = float(progress.get("elapsed_seconds_total", 0.0))
    wall_elapsed = float(progress.get("elapsed_wall_seconds_total", elapsed))
    groups_played = int(analysis["actual_groups"])
    routes_advanced = sum(route["highest_passed_budget"] is not None for route in route_states.values())
    failed_routes = sum(route["status"] == "failed" for route in route_states.values())
    completed_routes = sum(route["status"] == "completed" for route in route_states.values())
    opening_usage = ", ".join(f"{seed}: {count}" for seed, count in analysis["opening_usage"].items())
    drive_status = drive.get("status", "未写入 Drive 恢复演练结果")

    report = f"""# V5「ABCD 四大算法家族·全量程阶梯挑战赛」研究报告

实验日期：2026-10-20。该轮为传统规则和搜索算法的开发与对弈评估；没有神经网络训练、强化学习或 checkpoint。

## 摘要

本轮在 9×9 棋盘完成 **{analysis['actual_groups']} 组、{analysis['actual_games']} 局**（理论上限 60 组、120 局），完整挑战路线结束状态为：{routes_advanced} 条路线至少通过一档、{failed_routes} 条失败、{completed_routes} 条到达 50 秒档。有效对局计算耗时 {elapsed/3600:.2f} 小时，含快照与恢复操作的实验墙钟耗时 {wall_elapsed/3600:.2f} 小时。每组同一固定开局交换黑白，按两局得分至少 1 分晋级。完整逐档结果见下表和 [梯度图](figures/ladder_gradient.svg)。

本报告把“路线通过”解释为该两局配对达到规则阈值。每档只有两局、单一开局，因此路线边界是探索性现象，不能当成稳定棋力排名或显著差异。

## 1. 算法与预算

| 模型 | V5 设置 | 时间口径 |
|---|---|---|
| A | 沿用原启发式规则 AI | 自然决策耗时，无 sleep、无人工延迟、无预算超时 |
| B | 纯 Minimax，迭代加深 | 采用当前路线预算；使用 V4 已验证评价函数与候选顺序 |
| C | Minimax + Alpha-Beta | 与 B 搜索设置相同，只启用剪枝 |
| D | V4C 优化版纯 UCT MCTS | 采用当前路线预算；保留战术规则与跨步树复用，不调用 Minimax |

七档预算为 `{', '.join(_BUDGETS)}`。A 对 B/C/D 时，A 保持自然速度，对手按阶档增加预算。B/C/D 内部对阵共享一次 50ms–50ms 基线；后续 X→Y 由 X 使用当前档挑战固定 50ms 的 Y，反向路线单独运行并独立晋级。每个实际预算配对最多比赛一次，故理论上限为 21 个 A 路线组 + 39 个 B/C/D 组 = 60 组。

## 2. 挑战晋级树

“✓”表示该档两局总分至少 1；“×”表示失败并停止该路线；“—”表示未执行。结果括号内为挑战者两局得分。

{chr(10).join(route_lines)}

![V5 预算阶梯挑战边界](figures/ladder_gradient.svg)

未运行的档位没有被估算成胜负，也不计入对战成绩。共享 50ms 基线结果同时从双方模型视角判定，平局各得 1 分时双方都能晋级。

## 3. 所有实际对局

### 六类家族对阵汇总

{chr(10).join(pair_lines)}

### 完整胜负矩阵

每格为行模型胜/平/负、实际得分率。未发生的预算组合不以历史数据补齐。

{chr(10).join(matrix_lines)}

### 每组实际结果

{chr(10).join(group_lines) if group_lines else '没有完整配对组。'}

## 4. 观察到的模型排序与先后手

按本轮实际所有局的得分率排序如下。各模型面对的预算分布不同，且早停导致样本量不等，所以该表是阶梯赛的描述性汇总，不是同算力总排名。

{chr(10).join(rank_lines)}

| 模型 | 黑方局/分 | 白方局/分 | 黑方得分率 | 白方得分率 |
|---|---:|---:|---:|---:|
{chr(10).join(side_lines[2:])}

各组都交换黑白，消除了固定的单边颜色分配；但只有两局且同一开局，仍无法精确估计先手效应。颜色数据仅作描述。

## 5. 搜索成本与有效计算量

下表由实际每步记录聚合。A 的节点数来自规则选择器返回的空格扫描口径；B/C 为 Minimax 访问节点；D 的节点字段按 MCTS 实现定义，跨模型节点不能直接比较。

{chr(10).join(usage_lines)}

B/C 的深度代表完整完成的迭代加深层；C 剪枝量以 Alpha-Beta cutoff 计，`cutoff/节点` 只是实现计数比率，不等于所有未访问子树节点的比例。D 同时报告实际模拟次数与模拟/秒；搜索预算配置值没有替代实际模拟计数。A 的均值是自然耗时。

## 6. 与 V1–V4C 历史结果对照

{_history_section(results)}

历史实验的预算、开局池和样本设计不同；本轮得分和路线晋级不与历史局数合并。V5 用阶梯方式观察提高计算预算是否能让原本的路线继续通过，不能单独识别算法改进，因为本轮未修改决策逻辑。

## 7. 棋谱、错误与结论限制

完整走子与逐步耗时、深度、节点、cutoff、MCTS 模拟次数保存在 `9x9_games.jsonl`。每局都回放到胜负终止；非法落子判负，搜索超时使用算法返回的合法着并记录超时。典型局面可由 `game_summary.csv` 的 `game_id` 定位原始记录。

固定开局使用分布：{opening_usage}。对局组在多个开局之间按固定轮转分配，但路线只有两局且基于同一开局；开局分配和早停会造成路线间非均匀样本。表中 95% 区间是对“每组两局平均分”做 Wilson 风格的描述区间，仅两个观察块时覆盖率不稳定，不作显著性检验。

本轮是传统算法基线，不能据此推断 AlphaZero 等学习算法的成绩。下一步若要验证稳定边界，应在每个晋级边界扩大配对开局数，并单独报告同预算赛与阶梯赛。

## 8. 资源、复现与恢复

- 预检、环境资源和总耗时估算：`preflight.json`。
- 唯一对局、随机种子、开局、完整棋谱：`9x9_games.jsonl`。
- 路线与逐局进度：`progress.json`、`route_summary.csv`。
- 可复算统计：`summary.json`、`group_summary.csv`、`game_summary.csv`、`pairwise_summary.csv`、`model_usage.csv`。
- Drive 恢复验证：{drive_status}，详情见 `drive_recovery.json`。
- 代码基于 V4C 分支并固定 `src/ai.py`、`src/gomoku.py`、`src/research/mcts_d2.py` 的 SHA-256；执行代码提交记录于 `progress.json`。
- 配置及脚本：`configs/abcd_ladder_v5.json`、`python scripts/run_abcd_ladder_v5.py`、`python scripts/summarize_abcd_ladder_v5.py`。

"""
    report_path = output / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
