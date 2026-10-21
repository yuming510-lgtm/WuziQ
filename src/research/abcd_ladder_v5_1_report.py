"""Render the full Chinese V5.1 ladder report and route gradient SVG."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

_BUDGETS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")
_AXIS = ("natural", *_BUDGETS)
_FAMILIES = ("A", "B", "C", "D")
_COLORS = {"A": "#2563eb", "B": "#0f766e", "C": "#d97706", "D": "#7c3aed"}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _ci(value: list[float] | None) -> str:
    return "—" if value is None else f"[{value[0] * 100:.1f}%, {value[1] * 100:.1f}%]"


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def _fmt_status(level: dict[str, Any]) -> str:
    status = level["status"]
    points = level.get("score")
    points_text = f" {float(points):.1f}/2" if points is not None else ""
    labels = {
        "passed": "✓ 通过", "failed": "× 失败", "pending": "待执行",
        "incomplete_pair": "配对未完成", "not_reached": "未到该档",
        "not_run_after_failure": "本路线失败后未跑",
        "shared_result_after_failure": "共享棋局已记，但本路线此前失败",
        "inferred_pass": "◇ 按低档失利推断通过（0局）",
        "partial_observation_after_failure": "未配对观察，不计分",
    }
    return labels.get(status, status) + points_text


def _gradient_svg(routes: dict[str, Any], path: Path) -> None:
    width, left, right = 1470, 285, 205
    top, row_height = 122, 25
    height = top + row_height * len(routes) + 75
    x0, x1 = left, width - right
    xs = [x0 + (x1 - x0) * i / (len(_AXIS) - 1) for i in range(len(_AXIS))]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#17212e}.title{font-size:22px;font-weight:700}.small{font-size:12px}.route{font-size:12px;font-weight:600}.muted{fill:#667085}</style>',
        '<text x="24" y="32" class="title">V5.1 ABCD 各预算版本独立挑战边界</text>',
        '<text x="24" y="56" class="small muted">绿点为实测通过，红叉为实测失败，蓝菱形为按低档失利推断，灰方块为另一方向已有实测，空心点为未执行。</text>',
        '<text x="24" y="78" class="small muted">实测档为两局、同开局换色；推断档不算对局，不作实战胜率。</text>',
    ]
    for index, label in enumerate(_AXIS):
        x = xs[index]
        chunks.append(f'<line x1="{x:.1f}" y1="98" x2="{x:.1f}" y2="{height-48}" stroke="#e5eaf0" stroke-dasharray="3 4"/>')
        chunks.append(f'<text x="{x:.1f}" y="103" text-anchor="middle" class="small">{escape(label)}</text>')
    for row_index, (route_id, route) in enumerate(routes.items()):
        y = top + row_index * row_height
        family = str(route["challenger_family"])
        color = _COLORS[family]
        chunks.append(f'<text x="18" y="{y+4}" class="route">{escape(route_id)}</text>')
        chunks.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="#d8dee8" stroke-width="1"/>')
        for level in route["levels"]:
            label = str(level["opponent_budget_label"])
            x = xs[_AXIS.index(label)]
            status = level["status"]
            if status == "passed":
                chunks.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            elif status == "failed":
                chunks.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="#dc2626"/>')
                chunks.append(f'<path d="M{x-3:.1f},{y-3} L{x+3:.1f},{y+3} M{x+3:.1f},{y-3} L{x-3:.1f},{y+3}" stroke="#fff" stroke-width="1.6"/>')
            elif status == "shared_result_after_failure":
                chunks.append(f'<rect x="{x-5:.1f}" y="{y-5}" width="10" height="10" fill="#94a3b8" stroke="#fff"/>')
            elif status == "inferred_pass":
                chunks.append(f'<path d="M{x:.1f},{y-7} L{x+7:.1f},{y} L{x:.1f},{y+7} L{x-7:.1f},{y} Z" fill="#60a5fa" stroke="#fff" stroke-width="1.5"/>')
            elif status == "incomplete_pair":
                chunks.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="#fff7ed" stroke="#d97706" stroke-width="2"/>')
            elif status in {"pending", "not_reached"}:
                chunks.append(f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="#fff" stroke="#98a2b3" stroke-width="1.4"/>')
        highest = route.get("highest_passed_budget") or "未通过"
        failed = route.get("failed_budget")
        summary = f"过至 {highest}" + (f"；败于 {failed}" if failed else "")
        chunks.append(f'<text x="{width-right+14}" y="{y+4}" class="small">{escape(summary)}</text>')
    chunks.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks), encoding="utf-8")


def _route_table(analysis: dict[str, Any]) -> str:
    lines = ["| 挑战者版本 → 对手家族 | 各对手预算档结果 | 最高实测通过 | 最高（含推断）通过 | 首个实测失败档 |", "|---|---|---:|---:|---:|"]
    for route_id, route in analysis["route_state"]["routes"].items():
        levels = " → ".join(
            f"{level['opponent_budget_label']}：{_fmt_status(level)}"
            for level in route["levels"]
        )
        lines.append(
            f"| `{route_id}` | {levels} | {route.get('highest_actual_passed_budget') or '无'} "
            f"| {route.get('highest_passed_budget') or '无'} "
            f"| {route.get('failed_budget') or '无（路线完成或未失败）'} |"
        )
    return "\n".join(lines)


def _family_pair_table(analysis: dict[str, Any]) -> str:
    lines = ["| 对阵 | 实际组/局 | 首家族胜–平–次家族胜 | 首家族分/胜率 [95%组级描述区间] | 次家族分/胜率 [95%组级描述区间] |",
             "|---|---:|---:|---:|---:|"]
    for row in analysis["pairwise"]:
        if not row["games"]:
            lines.append(f"| {row['pair']} | 0/0 | 未运行 | — | — |")
            continue
        lines.append(
            f"| {row['pair']} | {row['groups']}/{row['games']} "
            f"| {row['first_wins']}–{row['draws']}–{row['second_wins']} "
            f"| {_number(row['first_points'], 1)}/{_pct(row['first_score_rate'])} {_ci(row['first_ci95_paired_groups'])} "
            f"| {_number(row['second_points'], 1)}/{_pct(row['second_score_rate'])} {_ci(row['second_ci95_paired_groups'])} |"
        )
    return "\n".join(lines)


def _version_matchup_table(analysis: dict[str, Any]) -> str:
    lines = ["| 精确预算组合 | 配对 ID / 开局 | 路线引用 | 状态 | 实际局数 | 胜–平–负 | 首/次模型得分（得分率） |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in analysis["version_pairwise"]:
        versions = row["model_versions"]
        first, second = tuple(versions)
        matchup = f"{first}@{versions[first]}–{second}@{versions[second]}"
        if row["games"]:
            result = f"{row['first_wins']}–{row['draws']}–{row['second_wins']}"
            scores = (
                f"{_number(row['first_points'], 1)}/{_pct(row['first_score_rate'])}；"
                f"{_number(row['second_points'], 1)}/{_pct(row['second_score_rate'])}"
            )
        else:
            result, scores = "未运行", "—；—"
        lines.append(
            f"| {matchup} | `{row['group_id']}` / {row['opening_seed']} "
            f"| {', '.join(row['route_ids'])} | {row['status']} | {row['games']} | {result} | {scores} |"
        )
    return "\n".join(lines)


def _models_table(analysis: dict[str, Any]) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    ranking = sorted(
        analysis["models"].items(),
        key=lambda item: (item[1]["score_rate"] is None,
                          -(item[1]["score_rate"] or 0.0), item[0]),
    )
    lines = ["| 描述排序 | 家族 | 胜–平–负 | 实际局数 | 得分率（得分） | 黑方得分率 | 白方得分率 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for rank, (family, row) in enumerate(ranking, start=1):
        side = analysis["side_effects"][family]
        black_rate = side["black_points"] / side["black_games"] if side["black_games"] else None
        white_rate = side["white_points"] / side["white_games"] if side["white_games"] else None
        lines.append(
            f"| {rank} | {family} | {row['wins']}–{row['draws']}–{row['losses']} | {row['games']} "
            f"| {_pct(row['score_rate'])} ({_number(row['points'],1)}) | {_pct(black_rate)} | {_pct(white_rate)} |"
        )
    return "\n".join(lines), ranking


def _version_usage_table(analysis: dict[str, Any]) -> str:
    lines = ["| 模型版本 | 局数 | 胜–平–负 | 得分率 | 决策数 | 平均秒/步 | 平均/峰值深度 | 节点/步 | 剪枝/节点 | 模拟/步 | 模拟/秒 | 实际耗时秒 | 超时数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    version_stats = {(row["model"], row["budget_label"]): row for row in analysis["versions"]}
    for row in analysis["model_usage"]:
        version = f"{row['model']}@{row['budget_label']}"
        result = version_stats[(row["model"], row["budget_label"])]
        lines.append(
            f"| {version} | {result['games']} | {result['wins']}–{result['draws']}–{result['losses']} "
            f"| {_pct(result['score_rate'])} | {row['decisions']} "
            f"| {_number(row['mean_seconds_per_decision'],4)} | "
            f"{_number(row['mean_completed_depth'],2)}/{row['max_depth']} | "
            f"{_number(row['mean_nodes_per_decision'],1)} | {_number(row['cutoffs_per_node'],3)} | "
            f"{_number(row['mean_simulations_per_decision'],1)} | {_number(row['simulations_per_second'],1)} | "
            f"{_number(row['elapsed_seconds'],2)} | {row['timeouts']} |"
        )
    return "\n".join(lines)


def _version_score_table(analysis: dict[str, Any]) -> str:
    lines = ["| 独立模型版本 | 实际胜–平–负 | 局数 | 得分率 | 平均计算秒/步 | 搜索树指标 | 每计算秒得分 |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for row in analysis["versions"]:
        version = f"{row['model']}@{row['budget_label']}"
        metric = "A 规则选择"
        if row["model"] in {"B", "C"}:
            metric = f"均深 {_number(row['mean_completed_depth'],2)}；峰深 {row['max_completed_depth']}；节点 {row['total_nodes']}；剪枝 {row['total_cutoffs']}"
        elif row["model"] == "D":
            metric = f"树深 {_number(row['mean_completed_depth'],2)}；模拟 {row['total_simulations']}；模拟/秒 {_number(row['simulations_per_second'],1)}"
        lines.append(
            f"| {version} | {row['wins']}–{row['draws']}–{row['losses']} | {row['games']} "
            f"| {_pct(row['score_rate'])} | {_number(row['mean_seconds_per_decision'],4)} | {metric} "
            f"| {_number(row['score_per_thinking_second'],4)} |"
        )
    return "\n".join(lines)


def _history_section(analysis: dict[str, Any], results_dir: Path) -> str:
    history_path = results_dir / "history_comparison.json"
    if not history_path.is_file():
        history_path = Path(__file__).resolve().parents[2] / "research/v5/results/history_comparison.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    lines = ["历史结果只作背景，不合并到 V5.1 晋级、排名或胜负矩阵。V1–V4C 的既有比较摘要如下：", "",
             "| 阶段 | 已验证历史观测 |", "|---|---|"]
    for item in history.get("comparisons", []):
        lines.append(f"| {item.get('stage','—')} | {item.get('finding','—')} |")
    lines.extend([
        "", "V5 复核：重新从 V5 `9x9_games.jsonl` 回放，并逐局重算 `wins + draws/2`。"
        "V5 原报告的部分矩阵第一列累计分被少算一半；得分率列是对的。V5.1 不改写历史报告和分支，只在下表给出由原始棋谱重算的数值。", "",
        "| 对阵 | 原报告首模型分 | 棋谱重算首模型分 | 实际局数 | 重算得分率 |", "|---|---:|---:|---:|---:|"])
    for row in analysis["history_v5_score_audit"]["pairs"]:
        lines.append(
            f"| {row['pair']} | {_number(row['reported_first_points'],1)} "
            f"| {_number(row['expected_first_points'],1)} | {row['games']} | {_pct(row['first_score_rate'])} |"
        )
    return "\n".join(lines)


def build_report(
    analysis: dict[str, Any],
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    preflight: dict[str, Any] | None = None,
) -> Path:
    """Write a full Chinese report from validated V5.1 summary data."""
    results = Path(results_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight = preflight or {}
    progress_path = results / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    drive_path = results / "drive_recovery.json"
    drive = json.loads(drive_path.read_text(encoding="utf-8")) if drive_path.exists() else {}
    _gradient_svg(analysis["route_state"]["routes"], output / "figures" / "ladder_gradient.svg")

    ranking_table, ranking = _models_table(analysis)
    runtime_seconds = float(progress.get("elapsed_seconds_total", 0.0))
    wall_seconds = float(progress.get("elapsed_wall_seconds_total", runtime_seconds))
    actual_seconds = float(analysis["actual_metrics"]["ai_thinking_seconds"])
    estimate = preflight.get("estimate", {})
    preflight_lines = [
        f"CPU 可见核心：{preflight.get('resources_before', {}).get('cpu_count_visible', '未记录')}；"
        f"预检记录内存可用量：{preflight.get('resources_before', {}).get('memory', {}).get('memavailable_kib', 0) / 1024:.0f} MiB；"
        f"磁盘可用量：{preflight.get('resources_before', {}).get('disk', {}).get('free_bytes', 0) / (1024**3):.1f} GiB。",
        f"预检逐版本局面耗时 {float(preflight.get('preflight_elapsed_seconds', 0.0)):.1f} 秒。"
        f"预计如所有独立路线都到达 50 秒，最多需新跑 {estimate.get('estimated_new_groups_if_all_routes_reach_50s','未估算')} 组 / "
        f"{estimate.get('estimated_new_games_if_all_routes_reach_50s','未估算')} 局；按 V5 同家族对局实际长度和逐预算实测速率，"
        f"单进程估计 {float(estimate.get('estimated_full_schedule_wall_hours_serial', 0.0)):.2f} 小时；"
        f"按完整 9×9 盘面预算的保守上限 {float(estimate.get('configured_full_board_budget_ceiling_hours', 0.0)):.2f} 小时。",
        f"所需棋谱空间估计 {float(estimate.get('estimated_storage_bytes_for_all_336_games', 0))/(1024**2):.1f} MiB。",
    ]
    if drive.get("status"):
        drive_status = str(drive["status"])
    elif drive.get("round_trip", {}).get("sha256_match") and drive.get("round_trip", {}).get("empty_directory_restore"):
        drive_status = "已验证 Drive 上传、下载、SHA-256 和空目录恢复"
    else:
        drive_status = "Drive 恢复演练未记录"
    if drive.get("latest_snapshot"):
        latest = drive["latest_snapshot"]
        drive_status += (
            f"；最新有效快照 `{latest.get('name','V5.1 snapshot')}`，"
            f"SHA-256 `{latest.get('sha256','未记录')}`，含 {latest.get('completed_games','?')} 局，"
            f"专用文件夹文件数 {drive.get('files_remaining','未记录')}。"
        )
    family_order = "、".join(f"{family}（{_pct(stats.get('score_rate'))}）" for family, stats in ranking)
    summary_text = (
        f"本轮目前完成 {analysis['actual_groups']} 组、{analysis['actual_games']} 局配对对局；"
        f"另有 {analysis['validation']['partial_observation_games']} 局未配对观察记录（不计入战绩）。"
        f"复用 V5 {analysis['actual_metrics']['actual_reused_groups']} 组、"
        f"{analysis['actual_metrics']['actual_reused_games']} 局；66 条独立路线中 "
        f"{sum(route['status']=='completed' for route in analysis['route_state']['routes'].values())} 条完成，"
        f"{sum(route['status']=='failed' for route in analysis['route_state']['routes'].values())} 条失败，"
        f"其余未到终态。AI 实际思考时间合计 {actual_seconds/3600:.2f} 小时；"
        f"运行器墙钟 {wall_seconds/3600:.2f} 小时，累计有效运行 {runtime_seconds/3600:.2f} 小时。"
        if analysis["actual_games"] else "当前没有已完成对局。"
    )
    completed_status = "完整完成" if analysis["complete"] else "未完成（只作进度预览，不能发布为最终成绩）"
    drive_section = f"恢复快照：{drive_status}。新快照先上传、回下载并实际恢复校验后，才会移除上一份 V5.1 快照。"
    route_summary = analysis["route_state"]["routes"]

    lines = [
        "# V5.1「ABCD 全模型独立阶梯挑战赛」研究报告", "",
        f"**状态：{completed_status}。** 本轮只使用传统规则 AI、Minimax、Alpha-Beta 与 MCTS 推理；没有神经网络或强化学习训练。此处的恢复归档是实验数据快照，不是训练 checkpoint。", "",
        "## 摘要", "", summary_text, "",
        f"四家族跨预算混合样本的描述性得分率顺序为：{family_order}。这不是统一预算下的棋力排名，原因是不同版本可打到的对手预算、实际局数和开局不同。", "",
        "## 1. 赛制、实现与公平性", "",
        "A、B、C、D 分别使用原规则 AI、纯 Minimax、Minimax + Alpha-Beta、冻结的纯 UCT MCTS 与 V4C 战术规则。A 自然耗时；B/C 使用相同候选集、顺序、评价与迭代加深，C 只增加 Alpha-Beta。没有针对对手的规则，也没有训练。",
        "B、C、D 的独立版本预算为 50ms、200ms、1s、5s、10s、20s、50s。A 为单一自然时间版本。路线固定挑战者版本，只逐档提高对手预算；A 路线逐档面对对手的七个版本。只安排跨家族比赛。",
        "同一预算配对只运行一次并可供两条方向相反的路线共享；每条实测路线只在自己的两局合计得分至少 1 分时升级。对局使用预注册的 12 个合法开局池和固定种子，每组相同开局换色。V5 精确匹配的对局只在其配对组到达可执行时才物化。",
        "依照追加的早停规则，若固定挑战者未通过某家族的较低预算版本，则该路线不再实测更高预算；反向路线中高于已败低档的版本标为“推断通过”，不计为实战胜场、得分率或对局。此推断采用“更高预算版本不弱于低预算版本”的单调性假设；已有完整棋局仍按真实结果保留，未配对观察保留原始记录但不计赛果。",
        "V1–V5 的决策引擎代码未修改；V5.1 只增加独立预算版本赛程、恢复、验证和统计代码。", "",
        "### 运行资源与耗时预估", "", *preflight_lines, "", drive_section, "",
        "运行器使用唯一 game ID、JSONL 每局 fsync、progress 原子替换；每局回放检查合法性、五连胜和计数，配对齐全后才推进路线。恢复时按棋谱重建路线，而不是信任缓存的晋级结果。", "",
        "## 2. 66 条独立挑战路线", "",
        "表中每条 B/C/D@预算路线都有独立状态。实测通过/失败来自两局配对；“按低档失利推断通过”是零对局的赛程剪枝结果；最高实测通过与最高（含推断）通过分开列示。", "",
        _route_table(analysis), "",
        "![V5.1 各固定版本的独立预算挑战边界](figures/ladder_gradient.svg)", "",
        "## 3. 六类家族对阵胜负矩阵", "",
        "以下胜负与得分率由全部实际对局计算；95% 区间以每个两局配对组的得分率为观察单位，采用 Wilson 近似，仅作描述。开局种类有限且路线按两局早停，区间不消除选择偏差。", "",
        _family_pair_table(analysis), "",
        "## 4. 168 个潜在精确预算对局格", "",
        "每一行对应唯一模型版本组合。未执行或推断的格子没有胜负、积分或得分率；完整棋谱进入赛果统计，未配对观察保存在原始 JSONL 和 CSV，只计资源消耗、不计胜负矩阵。", "",
        _version_matchup_table(analysis), "",
        "## 5. 家族和预算版本的描述性排名", "",
        "### 家族总体汇总", "", ranking_table, "",
        "### A 自然耗时与 B/C/D 各独立预算版本", "", _version_score_table(analysis), "",
        "各版本的实际算力投入不同，综合得分率受到路线存活和对手版本构成影响；这张表仅描述已发生对局，不代表每一预算档的固有棋力。", "",
        "## 6. 搜索效率与计算成本", "",
        "每项统计来自实际逐手记录；空白表示未发生该版本决策。B/C 深度是完整迭代加深层；C 剪枝列为 cutoff / visited node 比率，不等同于总省略子树节点比例。D 同时报真实模拟次数和每秒模拟数。不同家族的 `nodes` 口径不同，不作跨算法节点比较。A 平均时长是自然决策时间。", "",
        _version_usage_table(analysis), "",
        "实际资源收益可按版本对照得分率、平均每步秒数、搜索深度/节点、剪枝率、MCTS 模拟数及每计算秒得分。小样本和早停意味着单位资源得分不是稳定的边际收益估计。", "",
        "## 7. 先后手影响与统计限制", "",
        "| 家族 | 黑方局/分/胜 | 白方局/分/胜 | 黑方得分率 | 白方得分率 |", "|---|---:|---:|---:|---:|",
    ]
    for family in _FAMILIES:
        row = analysis["side_effects"][family]
        black_rate = row["black_points"] / row["black_games"] if row["black_games"] else None
        white_rate = row["white_points"] / row["white_games"] if row["white_games"] else None
        lines.append(
            f"| {family} | {row['black_games']}/{_number(row['black_points'],1)}/{row['black_wins']} "
            f"| {row['white_games']}/{_number(row['white_points'],1)}/{row['white_wins']} "
            f"| {_pct(black_rate)} | {_pct(white_rate)} |"
        )
    lines.extend([
        "", "每组都交换黑白，能控制组内先后手分配，但每条挑战档只有两局同一开局。固定开局结构、随机模拟与算法的战术选择会影响该档是否过线；通过门槛不是显著性检验，也不表示稳定胜过该对手。", "",
        "## 8. 对重点问题的直接回答", "",
    ])
    for fixed_budget in ("20s", "50s"):
        for opponent in ("B", "C"):
            route_id = f"D@{fixed_budget}>{opponent}"
            route = route_summary[route_id]
            lines.append(
                f"- `{route_id}`：最高通过对手预算 **{route.get('highest_passed_budget') or '尚无'}**；"
                f"失败档 **{route.get('failed_budget') or '无'}**；路线状态 `{route['status']}`。"
            )
    lines.extend([
        "- B 与 C 的预算边界请按上方每一条 `B@预算>C` 与 `C@预算>B` 路线逐档读取；反向路线独立计分。", "",
        "## 9. V1–V5 历史对照与 V5 积分修正", "", _history_section(analysis, results), "",
        "同一预算版本在不同对局中可能因对手预算、开局和颜色交换而出现不同胜负。V5.1 中固定预算模型对每个对手版本使用预注册开局分配，双局门槛只反映该开局的一次探索性配对，不是稳定等级。", "",
        "## 10. 结论与下一步", "",
        "V5.1 按追加规则把低档失利对更高预算对手版本的支配关系标为推断，从而省去无效高档赛局；只有完整实测配对进入战绩排名。已完成的旧逻辑对局照实保留。推断通过不等于实战验证，也不能证明预算收益严格单调。", "",
        "应以具体版本路线和实际资源统计作为结果。家族汇总排名受到预算与早停造成的不同样本构成影响，不能直接当作等预算对抗结论。两局一档、固定开局与平局计半晋级带来较高偶然性；后续若要建立稳定能力排序，应增加独立开局数并在预算层内做完整成对赛。", "",
        "## 11. 可复现材料", "",
        "- `configs/abcd_ladder_v5_1.json`：预算、开局池、确定性随机种子和 V5 引擎哈希。",
        "- `results/schedule.json` 与 `results/reuse_manifest.json`：完整 168 格赛程及每条历史复用来源。",
        "- `results/9x9_games.jsonl`：所有已完成棋谱及逐步搜索指标；每局包含唯一 ID。",
        "- `results/progress.json`：66 条路线、组状态和恢复进度。",
        "- `results/summary.json`、CSV 和 `figures/ladder_gradient.svg`：可重算统计与趋势图。",
        "- 运行脚本：`scripts/run_abcd_ladder_v5_1.py`；分析脚本：`scripts/summarize_abcd_ladder_v5_1.py`。",
        "", "本轮是传统规则与搜索算法的开发和对弈研究；没有神经网络训练，也不生成训练 checkpoint。", "",
    ])
    path = output / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
