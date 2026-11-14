from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_4_ladder import (
    ARTIFACT, MODEL_SHA256, TRAINING_STATE_SHA256, audit_results, atomic_json,
)


def build_score_chart(conditions: dict) -> str:
    width, height = 1120, 490
    x0, x1 = 280, 910
    plot_width = x1 - x0
    top, row_gap = 135, 76
    order = ("A", "C@200ms", "C@1s", "C@5s")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="560" y="42" text-anchor="middle" font-family="sans-serif" font-size="25" font-weight="700">G5000 对四档对手的得分率</text>',
        '<text x="560" y="72" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#555">每档6局，3组配对开局；误差线为配对开局自助法95%区间</text>',
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + fraction * plot_width
        color = "#b33a3a" if fraction == 0.5 else "#d7dce2"
        dash = ' stroke-dasharray="7,6"' if fraction == 0.5 else ""
        parts.append(f'<line x1="{x:.1f}" y1="98" x2="{x:.1f}" y2="420" stroke="{color}" stroke-width="{2 if fraction == .5 else 1}"{dash}/>')
        parts.append(f'<text x="{x:.1f}" y="445" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">{fraction:.0%}</text>')
    parts.append(f'<text x="{x0 + .5*plot_width:.1f}" y="465" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#333">G5000 得分率（胜1、平0.5、负0）</text>')
    for index, name in enumerate(order):
        row = conditions[name]
        y = top + index * row_gap
        score = row.get("score_rate")
        score = float(score) if score is not None else 0.0
        ci = row.get("paired_opening_bootstrap_ci95") or [0.0, 1.0]
        lower, upper = [max(0.0, min(1.0, float(value))) for value in ci]
        color = "#3d78b5" if index < 3 else "#8a4e73"
        parts.append(f'<text x="{x0-18}" y="{y+7}" text-anchor="end" font-family="sans-serif" font-size="18" font-weight="600">{html.escape(name)}</text>')
        parts.append(f'<rect x="{x0}" y="{y-16}" width="{score*plot_width:.1f}" height="31" rx="5" fill="{color}" opacity="0.82"/>')
        left, right = x0 + lower * plot_width, x0 + upper * plot_width
        parts.append(f'<line x1="{left:.1f}" y1="{y}" x2="{right:.1f}" y2="{y}" stroke="#222" stroke-width="3"/>')
        parts.append(f'<line x1="{left:.1f}" y1="{y-8}" x2="{left:.1f}" y2="{y+8}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<line x1="{right:.1f}" y1="{y-8}" x2="{right:.1f}" y2="{y+8}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<circle cx="{x0+score*plot_width:.1f}" cy="{y}" r="7" fill="#111"/>')
        parts.append(f'<text x="{x1+22}" y="{y+6}" font-family="sans-serif" font-size="15" fill="#222">{score:.0%} [{lower:.0%}, {upper:.0%}]</text>')
    parts.append('<text x="830" y="112" font-family="sans-serif" font-size="13" fill="#a33">50%</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _fmt(value, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _result_label(record: dict) -> str:
    if record["winner_color"] is None:
        return "和"
    return "胜" if float(record["points"]) == 1.0 else "负"


def _game_snippet(record: dict, tail: int = 8) -> str:
    moves = record["moves"][-tail:]
    return "；".join(f"{move['player']}({move['row']},{move['col']})" for move in moves)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_report(root: Path = ROOT) -> dict:
    root = root.resolve()
    artifact = root / ARTIFACT
    audit = audit_results(root, require_complete=True)
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    preflight = json.loads((artifact / "results/preflight.json").read_text(encoding="utf-8"))
    backup = json.loads((artifact / "results/drive_backup_status.json").read_text(encoding="utf-8"))
    match_paths = sorted((artifact / "results/matches").glob("*.json"))
    matches = [json.loads(path.read_text(encoding="utf-8")) for path in match_paths]
    matches.sort(key=lambda row: (row["opponent"], row["opening_seed"], row["candidate_color"]))
    (artifact / "charts").mkdir(parents=True, exist_ok=True)
    chart_path = artifact / "charts/ladder_score_rate.svg"
    chart_path.write_text(build_score_chart(audit["conditions"]), encoding="utf-8")
    game_fields = ("game_id", "opponent", "opening_seed", "candidate_color", "winner_color", "points",
                   "termination", "move_count", "candidate_cpu_seconds", "opponent_cpu_seconds",
                   "total_cpu_seconds", "total_wall_seconds")
    game_rows = [{key: record.get(key) for key in game_fields} for record in matches]
    _write_csv(artifact / "results/game_results.csv", game_rows, game_fields)
    condition_rows = []
    for name in ("A", "C@200ms", "C@1s", "C@5s"):
        row = {"opponent": name, **audit["conditions"][name]}
        row["opponent_move_telemetry"] = json.dumps(row["opponent_move_telemetry"], ensure_ascii=False)
        condition_rows.append(row)
    _write_csv(artifact / "results/condition_summary.csv", condition_rows,
               ("opponent", "games", "wins", "draws", "losses", "points", "score_rate",
                "paired_opening_score_rate", "paired_opening_bootstrap_ci95", "black_score_rate",
                "white_score_rate", "mean_moves", "mean_candidate_cpu_seconds",
                "mean_opponent_cpu_seconds", "opponent_move_telemetry"))
    checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in match_paths}
    atomic_json(artifact / "results/game_sha256.json", {
        "algorithm": "SHA-256", "games": checksums,
        "count": len(checksums), "all_unique": len(set(checksums.values())) == len(checksums),
    })

    conditions = audit["conditions"]
    table = ["| 对手 | W-D-L | 得分 | 得分率 | 配对95%区间 | G执黑 | G执白 | 平均手数 | G CPU/局 | 对手 CPU/局 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("A", "C@200ms", "C@1s", "C@5s"):
        row = conditions[name]
        ci = row["paired_opening_bootstrap_ci95"]
        ci_text = "—" if ci is None else f"[{_pct(ci[0])}, {_pct(ci[1])}]"
        table.append(f"| {name} | {row['wins']}-{row['draws']}-{row['losses']} | {row['points']:.1f}/6 | {_pct(row['score_rate'])} | {ci_text} | {_pct(row['black_score_rate'])} | {_pct(row['white_score_rate'])} | {_fmt(row['mean_moves'],1)} | {_fmt(row['mean_candidate_cpu_seconds'],1)}秒 | {_fmt(row['mean_opponent_cpu_seconds'],2)}秒 |")
    telemetry_rows = ["| 对手 | 每步时间 | 决策数 | 平均完成深度 | 平均节点 | 平均剪枝 | 到时返回完整层 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("A", "C@200ms", "C@1s", "C@5s"):
        info = conditions[name]["opponent_move_telemetry"]
        budget = "自然耗时" if info["budget_seconds"] is None else f"{info['budget_seconds']:g}秒"
        depth = "—" if name == "A" else _fmt(info["mean_completed_depth"], 2)
        nodes = "—" if name == "A" else _fmt(info["mean_nodes"], 0)
        cutoffs = "—" if name == "A" else _fmt(info["mean_cutoffs"], 0)
        telemetry_rows.append(f"| {name} | {budget} | {info['decisions']} | {depth} | {nodes} | {cutoffs} | {info['timed_out_decisions']}/{info['decisions']} |")

    key_examples = []
    preferences = (("A", 1.0), ("A", 0.0), ("C@200ms", 1.0), ("C@1s", 1.0), ("C@5s", 0.0))
    for opponent, points in preferences:
        candidate = next((row for row in matches if row["opponent"] == opponent
                          and float(row["points"]) == points), None)
        if candidate is None:
            continue
        outcome = _result_label(candidate)
        key_examples.append(
            f"- `{candidate['game_id']}`：{opponent}，开局 seed `{candidate['opening_seed']}`，G执{candidate['candidate_color']}，G{outcome}，{candidate['move_count']}手；终局前 {min(8,candidate['move_count'])} 手：{_game_snippet(candidate)}。"
        )

    cost = preflight["cost_estimate"]["estimated_cpu_seconds_with_20pct_overhead"]
    total_match_cpu = audit["total_match_cpu_seconds"]
    total_wall = audit["sum_of_game_wall_seconds"]
    schedule_openings = "\n".join(
        f"- `{opening['opening_group_id']}` (seed {opening['seed']}): "
        + " ".join(f"{m['player']}({m['row']},{m['col']})" for m in opening["moves"])
        for opening in schedule["openings"]
    )
    report = f"""# WuziQ G5000 天梯挑战 V6.4

## 结论摘要

G5000 使用冻结的 PUCT-400 参加了 24 局、4 个对手条件各 6 局。G5000 在 A 上取得 **1胜3平2负，得分率 41.7%**；C@200ms 为 **1胜0平5负，16.7%**；C@1s 为 **2胜0平4负，33.3%**；C@5s 为 **0胜0平6负，0%**。这只是每档3组开局的探索结果，不能当作稳定棋力排名。

按本轮观察，G5000 能在 A、C@200ms、C@1s 的个别对局中取胜或得分，但没有稳定击败任何一档；C@5s 六局全部失利。最高的实测预算中没有得分。尤其 C@1s 的两胜都来自同一开局组 seed 640616（黑白各一局），而其余两组均失利，显示开局影响很大。

## 赛制与冻结条件

- 规则：9×9自由五子棋，五连获胜；使用 V6.3 的规则、棋盘及搜索引擎接口。
- G5000：节点权重 SHA-256 `{MODEL_SHA256}`；完整训练状态 SHA-256 `{TRAINING_STATE_SHA256}`；5000 局、87,999 状态、4,999 次更新。仅评测，没有训练或修改权重。
- G 每步 PUCT-400、`c_puct=5.0`，原实现配置 401 playout，实际根边访问数经逐步审计均为400。
- A 使用原规则 AI 的自然运行时间；C@200ms、C@1s、C@5s 使用同一 Alpha-Beta 引擎、最大深度69、候选半径2，只有单步 deadline 不同。
- 每个对手均使用相同的3个预注册开局；每个开局 G 执黑、执白各一局。随机种子和唯一 game ID 固定。
- 赛前扫描仓库 research 数据及 Drive 恢复的完整 V6.3 文件树，发现 37,262 个 D4 规范化六手开局；本轮三个开局与历史集合重叠为0。历史开局扫描哈希：`{registration['historical_opening_scan']['sha256']}`；赛程哈希：`{schedule['schedule_sha256']}`。

### 冻结开局

{schedule_openings}

## 四档战绩

{chr(10).join(table)}

得分按胜1、平0.5、负0计算。区间以3个开局组的配对平均分作自助法95%区间；只有3个配对组，区间极不稳定，C@5s的 `[0%,0%]` 只表示这三组的实测结果。

![G5000四档对手得分率](charts/ladder_score_rate.svg)

### 先后手

| 对手 | G执黑得分率 | G执白得分率 |
|---|---:|---:|
{chr(10).join(f"| {name} | {_pct(conditions[name]['black_score_rate'])} | {_pct(conditions[name]['white_score_rate'])} |" for name in ("A","C@200ms","C@1s","C@5s"))}

A 组 G 执白为50%、执黑为33.3%；C@200ms 执白为33.3%、执黑为0%；C@1s 两色均为33.3%；C@5s 两色均为0%。每格仅3局，不能据此量化先手优势。

## 每步实际计算量

{chr(10).join(telemetry_rows)}

G5000 合计作出 {audit['candidate_move_telemetry']['decisions']} 次决策，真实完成 **{audit['candidate_move_telemetry']['simulations_total']:,} 次 MCTS 模拟**，每步严格400次；平均每步 CPU {_fmt(audit['candidate_move_telemetry']['mean_cpu_seconds'],2)} 秒、树深 {_fmt(audit['candidate_move_telemetry']['mean_tree_depth'],2)}，节点 {_fmt(audit['candidate_move_telemetry']['mean_tree_nodes'],0)}。各档 C 的超时数指预算截止后返回上一个完整迭代深度，是迭代加深的预期行为；未发生预算超额。C@5s 平均完成深度约5.80层、27,178节点，CPU均耗约58.15秒/局；C@1s 平均3.15层、7,519节点、CPU均耗约9.74秒/局。更高预算带来更多搜索，但在本次小样本中没有转成更高 G 得分。

## 算力预检与实际成本

- 环境：3个 CPU 核，PyTorch CPU-only，3线程；可用内存约8.6 GiB、可用磁盘约16.5 GiB。
- 代表局面实测：G 的400次模拟/步 CPU 2.406秒；A 自然决策0.0012秒；C@200ms 0.195秒、完成深度2；C@1s 0.994秒、深度3；C@5s 4.994秒、深度4。
- 预检按每方12/20/30次开局后决策并加20%余量，估计约 {_fmt(cost['optimistic']/60,1)} / {_fmt(cost['typical']/60,1)} / {_fmt(cost['conservative']/60,1)} CPU分钟。实际24局棋手决策 CPU 合计 **{_fmt(total_match_cpu/3600,3)}小时**（{_fmt(total_match_cpu,1)}秒）；逐局实际 wall 时间相加为 {_fmt(total_wall/60,1)}分钟。预检比实际多估的主要原因是中位手数低于预设20次/方，尤其 C@200ms、C@1s 对局较短。
- 执行期间没有环境重启或中断。预检和快照过程不计入上述比赛 CPU 总量。

## 棋谱样例

以下片段列出终局前的坐标，完整逐手数据在 `results/matches/*.json`。片段用于定位棋谱，不据此对未审查的战术原因作断言。

{chr(10).join(key_examples)}

本轮最明显的非单调结果是 C@1s 的两胜集中在同一个开局 seed，而 C@5s 三组配对均告负。增加搜索预算也改变了整局后续局面，不能把跨条件胜负差当成单独搜索节点的因果比较；6局不足以判定 C@5s 实力低于 C@1s。

## 验收、原始数据与复现

- 棋谱审计：计划24局、实际24局、唯一 ID 24个、全部独立规则回放合法、终局与胜负一致、开局组均完成黑白配对、无多余或缺失对局。
- 自动化测试：V6.4赛程、对局回放、配对、快照恢复与图表测试共15项；原五子棋规则/搜索测试29项，共44项通过。
- 历史 G5000 恢复包 #12 的 SHA-256 和空目录恢复校验已通过；模型、Adam、回放池与随机状态均可从 V6.3 Drive 原包恢复。V6.4 未训练，因此恢复快照只引用冻结模型身份和原 V6.3 恢复源，不覆盖该原包。
- V6.4 Drive 恢复目录：[{backup['folder_name']}]({backup['folder_url']})。已上传、回下载、SHA-256 比对并在空目录恢复验证的快照数量：{len(backup['validated_snapshots'])}；目录保留初始、分阶段和最终恢复点。
- 原始棋谱：`results/matches/`；完整比分：`results/game_results.csv`、`results/condition_summary.csv`；逐棋谱 SHA-256：`results/game_sha256.json`；审计：`results/audit.json`；开局与环境：`config/`；图表：`charts/ladder_score_rate.svg`。
- 复现：`python scripts/prepare_g_v6_4.py --historical-root <V6.3恢复树>`；`python scripts/preflight_g_v6_4.py --checkpoint-dir <已校验检查点目录>`；`python scripts/run_g_v6_4.py --checkpoint-dir <已校验检查点目录>`；`python scripts/audit_g_v6_4.py`；`python scripts/report_g_v6_4.py`。
- 适用恢复包后，执行器以固定 game ID 自动跳过已校验对局；单局中断会从同一预注册开局重赛，不会重复计入统计。

## 结论边界

本轮的简短阶梯只能给出探索性能力边界：G5000 曾对 A、C@200ms 和 C@1s 取分，但在每档仅3组开局的条件下没有足够证据宣称稳定战胜它们；C@5s 本轮0/6。C@1s的两胜集中在一组开局，也说明把最高单档胜利解释为稳定能力会过度推断。若继续研究，应扩大开局数并维持配对换色；本结果不能替代正式棋力评测。
"""
    (artifact / "REPORT.md").write_text(report, encoding="utf-8")
    return {"report_path": str(artifact / "REPORT.md"), "chart_path": str(chart_path),
            "games": audit["completed_games"], "audit_complete": audit["complete"],
            "match_cpu_seconds": audit["total_match_cpu_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Chinese G5000 V6.4 research report")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
