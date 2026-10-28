#!/usr/bin/env python3
"""Build reproducible V6.3 CSVs, SVG figures, and a Chinese report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.v6_3.analysis import summarize_v6_3


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _svg_report(summary: dict, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    palette = {"C": "#2563eb", "E5": "#ea580c"}
    groups = [
        ("V6.3 新赛", {model: summary["standings"][model]["points"] for model in ("C", "E5")}),
        ("含 V6.2 直接交手", {model: summary["combined_with_v6_2_direct"][model]["points"] for model in ("C", "E5")}),
    ]
    maximum = max(1.0, *(value for _, rows in groups for value in rows.values()))
    bits = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360">',
        '<rect width="900" height="360" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="sans-serif" font-size="22" font-weight="700" fill="#172033">C–E5 积分对照</text>',
    ]
    for index, (label, values) in enumerate(groups):
        y = 88 + index * 126
        bits.append(f'<text x="32" y="{y}" font-family="sans-serif" font-size="15" fill="#334155">{label}</text>')
        for j, model in enumerate(("C", "E5")):
            bar_y = y + 12 + j * 34
            width = 560 * values[model] / maximum
            bits.append(f'<text x="32" y="{bar_y+18}" font-family="sans-serif" font-size="14" fill="#334155">{model}</text>')
            bits.append(f'<rect x="82" y="{bar_y}" width="{width:.1f}" height="22" rx="4" fill="{palette[model]}"/>')
            bits.append(f'<text x="{92+width:.1f}" y="{bar_y+17}" font-family="sans-serif" font-size="14" font-weight="700" fill="#172033">{values[model]:.1f}</text>')
    bits.append('<text x="32" y="337" font-family="sans-serif" font-size="12" fill="#64748b">积分：胜 1，平 0.5，负 0；小样本探索性对照。</text>')
    bits.append('</svg>')
    (figure_dir / "score_comparison.svg").write_text("\n".join(bits) + "\n", encoding="utf-8")

    models = summary["model_compute"]
    max_time = max(1.0, *(float(models[m]["thinking_seconds"]) for m in ("C", "E5")))
    max_nodes = max(1.0, *(float(models[m]["nodes"]) for m in ("C", "E5")))
    bits = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390" viewBox="0 0 900 390">',
        '<rect width="900" height="390" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="sans-serif" font-size="22" font-weight="700" fill="#172033">计算量与平均耗时</text>',
    ]
    for index, model in enumerate(("C", "E5")):
        y = 100 + index * 120
        row = models[model]
        bits.append(f'<text x="32" y="{y}" font-family="sans-serif" font-size="16" font-weight="700" fill="#172033">{model}</text>')
        t_width = 470 * float(row["thinking_seconds"]) / max_time
        n_width = 470 * float(row["nodes"]) / max_nodes
        bits.append(f'<text x="32" y="{y+26}" font-family="sans-serif" font-size="12" fill="#64748b">总思考秒数</text><rect x="130" y="{y+10}" width="{t_width:.1f}" height="18" rx="3" fill="{palette[model]}"/><text x="{138+t_width:.1f}" y="{y+24}" font-family="sans-serif" font-size="12" fill="#172033">{float(row["thinking_seconds"]):.1f}</text>')
        bits.append(f'<text x="32" y="{y+58}" font-family="sans-serif" font-size="12" fill="#64748b">搜索节点</text><rect x="130" y="{y+42}" width="{n_width:.1f}" height="18" rx="3" fill="{palette[model]}" opacity=".68"/><text x="{138+n_width:.1f}" y="{y+56}" font-family="sans-serif" font-size="12" fill="#172033">{int(row["nodes"]):,}</text>')
        bits.append(f'<text x="650" y="{y+24}" font-family="sans-serif" font-size="12" fill="#334155">平均 {float(row["average_seconds_per_decision"]):.1f} 秒/步</text>')
        bits.append(f'<text x="650" y="{y+48}" font-family="sans-serif" font-size="12" fill="#334155">平均深度 {float(row["average_completed_depth"]):.2f}；最大 {int(row["max_search_depth"])}</text>')
    bits.append('<text x="32" y="365" font-family="sans-serif" font-size="12" fill="#64748b">节点规模受各自搜索结构影响；以相同 200 秒/步上限比较。</text>')
    bits.append('</svg>')
    (figure_dir / "compute_comparison.svg").write_text("\n".join(bits) + "\n", encoding="utf-8")


def _report(summary: dict, preflight: dict, games: list[dict], report_path: Path) -> None:
    c, e5 = summary["standings"]["C"], summary["standings"]["E5"]
    cc, ee = summary["combined_with_v6_2_direct"]["C"], summary["combined_with_v6_2_direct"]["E5"]
    leaders = summary["champions"]
    title_result = "、".join(leaders) + ("并列冠军" if len(leaders) > 1 else "为本轮探索性冠军")
    confidence = summary["paired_block_descriptive"]
    resource = preflight.get("resource_snapshot", {})
    over_budget = {model: 0 for model in ("C", "E5")}
    maximum_overrun = 0.0
    for game in games:
        for move in game["moves"]:
            model = move.get("model")
            if model not in over_budget or move.get("opening"):
                continue
            over = float(move["elapsed_seconds"]) - 200.0
            if over > 0:
                over_budget[model] += 1
                maximum_overrun = max(maximum_overrun, over)

    lines = [
        "# WuziQ V6.3：C vs E5 冠军决胜加赛", "",
        "## 结论", "",
        f"本轮新增 **{summary['new_games']} 局**，赛程状态为 `{summary['tiebreak_status']}`；按预注册积分规则，{title_result}。",
        f"本轮积分：C **{c['points']:.1f}**（{c['wins']}胜 {c['draws']}平 {c['losses']}负），E5 **{e5['points']:.1f}**（{e5['wins']}胜 {e5['draws']}平 {e5['losses']}负）。",
        f"与 V6.2 两局 C–E5（C 1胜1负、积分 1:1）直接交手合并后：C **{cc['points']:.1f}/{cc['games']}局**（{cc['wins']}胜 {cc['draws']}平 {cc['losses']}负），E5 **{ee['points']:.1f}/{ee['games']}局**（{ee['wins']}胜 {ee['draws']}平 {ee['losses']}负）。",
        "以上冠军仅指这组预先固定开局下的探索性赛果，不代表统计显著或对所有开局的稳定棋力排序。", "",
        "## 赛制与算法冻结", "",
        f"9×9 原规则；冻结的 V6 C（Alpha-Beta）对 E5（高级 Alpha-Beta），双方每步上限均为 200 秒，使用同一运行环境。V6.3 配置固定在 V6.2 提交 `{summary['baseline_commit']}`，算法文件哈希沿用 V6.2 的已验证值。未训练、未调参、未更改搜索实现。",
        "先使用新开局 `731301`、`731302`，各交换黑白一局，共4局。仅当4局积分相同，才启用预注册的 `731303` 两局加赛；加赛后仍平分则并列。历史对局只用于合并比较，不计入 V6.3 新赛。", "",
        "| 开局 ID | 预注册序列（行,列；从1开始） | 状态 |", "|---|---|---|",
    ]
    config = json.loads((report_path.parent / "results/config.json").read_text(encoding="utf-8"))
    for opening_id in [*config["initial_opening_ids"], str(config["tiebreak_opening_id"])]:
        seq = " → ".join(f"{m['player']}({m['row']+1},{m['col']+1})" for m in config["openings"][opening_id])
        state = "首轮固定开局" if opening_id in config["initial_opening_ids"] else ("已执行加赛" if summary["conditional_games"] else "预注册条件开局，未执行")
        lines.append(f"| {opening_id} | {seq} | {state} |")
    lines.extend([
        "", "## 每局结果", "",
        "| 对局 ID | 阶段 | 开局 | 黑棋 | 白棋 | 结果 | 落子数 | C 思考秒 | E5 思考秒 | C 节点 | E5 节点 | C/E5 最大深度 |", 
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["_game_rows"]:
        result = "和棋" if not row["winner_model"] else f"{row['winner_model']} 胜"
        lines.append(
            f"| `{row['game_id']}` | {row['stage']} | {row['opening_id']} | {row['black_model']} | {row['white_model']} | {result} | {int(row['move_count'])} | {_fmt(float(row['thinking_C_seconds']))} | {_fmt(float(row['thinking_E5_seconds']))} | {int(row['nodes_C']):,} | {int(row['nodes_E5']):,} | {row['max_depth_C']}/{row['max_depth_E5']} |"
        )
    lines.extend([
        "", "## 颜色与开局配对", "",
        "| 开局 | 局数 | C 积分 | E5 积分 | C 胜 | 和棋 | E5 胜 | C执黑局/胜 | E5执黑局/胜 | 对局合计思考秒 |", 
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["pair_blocks"]:
        opening_row = next((x for x in summary["_opening_rows"] if x["opening_id"] == row["opening_id"]), None)
        if opening_row is None:
            continue
        lines.append(
            f"| {row['opening_id']} | {int(opening_row['games'])} | {float(opening_row['C_points']):.1f} | {float(opening_row['E5_points']):.1f} | {opening_row['C_wins']} | {opening_row['draws']} | {opening_row['E5_wins']} | {opening_row['C_black_games']}/{opening_row['C_black_wins']} | {opening_row['E5_black_games']}/{opening_row['E5_black_wins']} | {_fmt(float(opening_row['sum_thinking_seconds']))} |"
        )
    lines.extend([
        "", "## 搜索成本与计时", "",
        "| 模型 | 决策数 | 总思考秒 | 平均秒/步 | 节点 | 节点/秒 | 剪枝数 | 平均完成深度 | 最大深度 | 超时标记 | 超200秒步数 |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in ("C", "E5"):
        row = summary["model_compute"][model]
        lines.append(
            f"| {model} | {row['decisions']} | {_fmt(row['thinking_seconds'])} | {_fmt(row['average_seconds_per_decision'])} | {int(row['nodes']):,} | {_fmt(row['nodes_per_second'])} | {int(row['cutoffs']):,} | {_fmt(row['average_completed_depth'])} | {row['max_search_depth']} | {row['timeouts']} | {row['over_budget_moves']} |"
        )
    lines.extend([
        "",
        f"比赛运行墙钟 {summary['elapsed_wall_seconds'] / 3600:.2f} 小时，累计算法思考时间 {sum(summary['model_compute'][m]['thinking_seconds'] for m in ('C','E5')) / 3600:.2f} 小时；含短预检的可测时间约 {summary['total_measured_seconds_including_preflight'] / 3600:.2f} 小时。超时标记按引擎返回值或超出配置预算生成；实际超过200秒的步骤：C {over_budget['C']} 步、E5 {over_budget['E5']} 步，最大超限 {_fmt(maximum_overrun * 1000, 3)} 毫秒。超过上限的实测值保留，不作截断。",
        "", "## 计算环境、预检与时间估算", "",
        f"逻辑 CPU：{resource.get('logical_cpus', '未知')}；内存：{resource.get('memory_total_bytes', 0)/1024**3:.2f} GiB；可用磁盘：{resource.get('disk_free_bytes', 0)/1024**3:.2f} GiB。赛前对 C 与 E5 各进行一次 {_fmt(preflight.get('preflight_budget_seconds_per_model', 0))} 秒预检，实测耗时分别为 "
        + "、".join(f"{m['model']} {_fmt(m['elapsed_seconds'])} 秒（深度 {m['completed_depth']}，节点 {m['nodes']:,}，合法落点 {m['move']}）" for m in preflight.get("measurements", [])) + ". 预检不计入正式对局。",
        "V6.2 已完成的 C–E5 两局实际思考时间为 "
        + "、".join(_fmt(x) + " 秒" for x in preflight.get("duration_estimates", {}).get("basis", {}).get("v6_2_direct_games_thinking_seconds", []))
        + "。据此，4局乐观/中性预估约 "
        + f"{preflight.get('duration_estimates',{}).get('optimistic',{}).get('four_games_seconds',0)/3600:.2f}/{preflight.get('duration_estimates',{}).get('central',{}).get('four_games_seconds',0)/3600:.2f} 小时；若平分、跑满6局约 "
        + f"{preflight.get('duration_estimates',{}).get('optimistic',{}).get('six_games_seconds',0)/3600:.2f}/{preflight.get('duration_estimates',{}).get('central',{}).get('six_games_seconds',0)/3600:.2f} 小时。按6局各剩余77步均耗尽200秒的保守上界约 {preflight.get('duration_estimates',{}).get('conservative',{}).get('six_games_seconds',0)/3600:.2f} 小时。",
        "", "## 配对不确定性与综合冠军判断", "",
        f"新增赛的 {confidence['opening_blocks']} 个换色开局块中，C领先 {confidence['C_ahead']} 块、平 {confidence['tied']} 块、E5领先 {confidence['E5_ahead']} 块；对非平分开局块的 C 领先比例 Wilson 95% 描述区间为 `{confidence['C_ahead_wilson_95_among_decisive_blocks']}`。这个区间只作小样本描述，不是显著性检验。",
        f"按本轮规则，当前结果为 **{title_result}**；按 V6.2 直接交手与本轮全部实测合并，积分为 C {cc['points']:.1f}、E5 {ee['points']:.1f}。这提供了多开局的补充证据，但样本仍只有 {cc['games']} 局，而且部分 V6.2/V6.3 环境计时有轻微超限。不能将冠军标签解释为普遍棋力或统计显著差异。",
        "", "## 重要棋局与复现边界", "",
    ])
    for game in games:
        result = "和棋" if game.get("winner_model") is None else f"{game['winner_model']} 胜"
        tail = [m for m in game["moves"] if not m.get("opening")][-8:]
        seq = "，".join(f"{m['model']}({int(m['row'])+1},{int(m['col'])+1})" for m in tail)
        lines.append(f"- `{game['game_id']}`：{result}，终局方式 `{game['termination']}`；最后至多8手：{seq}。棋谱只支持本局落子与终局事实，不证明较早局面不存在其他必胜选择。")
    lines.extend([
        "",
        "开局块按双方交换先后手成对，能直接呈现这两个具体开局上的颜色变化；两局同一开局仍不足以分离先手效应、引擎随机差异和局面因素。V6.2 的 731201 与本轮三个开局均按原始棋谱分别记录。",
        "", "## 图表与数据", "",
        "![积分对比](figures/score_comparison.svg)", "",
        "![搜索成本](figures/compute_comparison.svg)", "",
        "- `results/9x9_games.jsonl`：完整原始棋谱和逐步搜索指标。",
        "- `results/games.csv`、`results/moves.csv`、`results/opening_summary.csv`：可分析表格。",
        "- `results/summary.json`、`results/integrity_check.json`：汇总与验收。",
        "- `configs/v6_3/championship.json`：冻结配置、随机种子、开局与 Drive 目录。",
        "- `results/drive_recovery.json`：Drive 最新快照 ID、哈希及恢复验证记录。",
        "", "## 复现", "",
        "```bash",
        "python scripts/preflight_v6_3.py",
        "python scripts/run_v6_3_championship.py",
        "python scripts/validate_v6_3_results.py",
        "python scripts/build_v6_3_report.py",
        "```",
        "", "本项目仅进行传统规则/搜索算法对弈评估，没有神经网络训练、训练指标或 checkpoint。", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_3/results")
    args = parser.parse_args()
    result_dir = ROOT / args.output_dir
    config_path = result_dir / "config.json"
    preflight_path = ROOT / "research/v6_3/preflight/preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    summary = summarize_v6_3(config_path, result_dir, preflight=preflight)
    games = load_game_records(result_dir / "9x9_games.jsonl")
    summary["_game_rows"] = [
        {**row, "winner_model": row["winner_model"] or None, "color_swap": row["color_swap"] == "True"}
        for row in csv_dicts(result_dir / "games.csv")
    ]
    summary["_opening_rows"] = list(csv_dicts(result_dir / "opening_summary.csv"))
    _svg_report(summary, ROOT / "research/v6_3/figures")
    report_path = ROOT / "research/v6_3/REPORT.md"
    _report(summary, preflight, games, report_path)
    print(json.dumps({"report": str(report_path), "games": summary["new_games"], "leaders": summary["champions"]}, ensure_ascii=False, indent=2))


def csv_dicts(path: Path):
    import csv
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


if __name__ == "__main__":
    main()
