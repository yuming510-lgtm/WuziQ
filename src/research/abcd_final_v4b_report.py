"""Build the Chinese V4B report from durable results and V1–V4 artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from .abcd_final_v4b import load_config
from .abcd_final_v4b_analysis import validate_records, write_analysis
from .checkpoint import load_game_records, write_json_atomic


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _ci(interval: list[float] | None) -> str:
    return "—" if interval is None else f"[{100*interval[0]:.1f}%, {100*interval[1]:.1f}%]"


def _historical_summary(project_root: Path) -> dict[str, Any]:
    v1_summary = _load(project_root / "research/results/summary.json")["9"]["results"]
    v1 = {}
    for pair, row in v1_summary["pairwise"].items():
        v1[pair] = {
            model: {key: value for key, value in stats.items() if key in {"games", "wins", "losses", "draws", "pure_win_rate", "score_rate", "score_rate_ci95"}}
            for model, stats in row["models"].items()
        }

    v2_summary = _load(project_root / "research/v2/results/summary.json")
    v2 = []
    for name, phase in v2_summary["phases"].items():
        if int(phase.get("board_size", 0)) != 9 or "D2" not in phase.get("models", ()):
            continue
        opponents = [model for model in phase["models"] if model != "D2"]
        if len(opponents) != 1:
            continue
        opponent = opponents[0]
        key = "-".join(sorted(("D2", opponent)))
        stats = phase.get("pairwise", {}).get(key, {}).get("models", {}).get("D2", {})
        v2.append(
            {
                "phase": name,
                "track": phase.get("track"),
                "opponent": opponent,
                "budget_seconds_by_model": phase.get("budget_seconds_by_model"),
                "games": phase.get("game_count"),
                "wins": stats.get("wins"),
                "draws": stats.get("draws"),
                "losses": stats.get("losses"),
                "score_rate": stats.get("score_rate"),
                "mean_seconds_per_move": stats.get("average_seconds_per_move"),
                "mean_simulations_per_move": stats.get("average_simulations_per_move"),
            }
        )

    v3_summary = _load(project_root / "research/v3/results/summary.json")
    v3 = []
    for name, phase in v3_summary["phases"].items():
        if float(phase.get("D3_budget_seconds", 0)) != 20.0:
            continue
        stats = phase["D3"]
        v3.append(
            {
                "phase": name,
                "opponent": phase["opponent_model"],
                "D3_budget_seconds": phase["D3_budget_seconds"],
                "opponent_budget_seconds": phase["opponent_budget_seconds"],
                "games": phase["games"],
                "wins": stats["wins"],
                "draws": stats["draws"],
                "losses": stats["losses"],
                "score_rate": stats["score_rate"],
                "average_seconds_per_move": stats["average_seconds_per_move"],
                "average_simulations_per_move": stats["average_simulations_per_move"],
            }
        )

    v4_summary = _load(project_root / "research/v4/results/summary.json")
    v4 = {
        model: {
            budget: {
                key: v4_summary["by_model_budget"][model][budget][key]
                for key in (
                    "run_count", "mean_elapsed_seconds", "mean_completed_depth", "mean_nodes",
                    "mean_nodes_per_second", "mean_cutoffs", "cutoff_ratio", "timeout_count",
                )
            }
            for budget in ("50", "200")
        }
        for model in ("B", "C")
    }
    return {
        "V1_9x9_pairwise_50ms": v1,
        "V2_9x9_D2_conditions": sorted(v2, key=lambda row: row["phase"]),
        "V3_D3_20s_conditions": sorted(v3, key=lambda row: row["opponent"]),
        "V4_BC_50_vs_200ms_fixed_positions": v4,
    }


def _short_game_transcript(game: dict[str, Any], tail_plies: int = 12) -> str:
    moves = [move for move in game["moves"] if not move.get("illegal")]
    start = max(0, len(moves) - tail_plies)
    prefix = "；".join(
        f"{move.get('model') or '开局'}-{move['player']}({move['row']},{move['col']})"
        for move in moves[start:]
    )
    return f"前段完整棋谱见原始数据；末 {len(moves)-start} 手：{prefix}"


def _historical_markdown(history: dict[str, Any]) -> str:
    lines = [
        "| 轮次 | 条件 | D 类 MCTS 结果 | 计算条件 |",
        "|---|---|---:|---|",
    ]
    for pair, value in history["V1_9x9_pairwise_50ms"].items():
        d = value.get("D", {})
        if d:
            lines.append(
                f"| V1 | D vs {pair.replace('D-', '').replace('-D', '')} | {d['wins']} 胜 / {d['draws']} 平 / {d['losses']} 负（得分率 {_pct(d['score_rate'])}） | A/B/C/D 原配置；D 50ms；每组 200 局 |"
            )
    for row in history["V2_9x9_D2_conditions"]:
        if row["opponent"] in {"B", "C"}:
            budget = row["budget_seconds_by_model"]
            lines.append(
                f"| V2 | D2 vs {row['opponent']} | {row['wins']} 胜 / {row['draws']} 平 / {row['losses']} 负（得分率 {_pct(row['score_rate'])}） | D2 {budget['D2']:.3g}s，对手 {budget[row['opponent']]:.3g}s；{row['games']} 局 |"
            )
    for row in history["V3_D3_20s_conditions"]:
        lines.append(
            f"| V3 | D3 vs {row['opponent']} | {row['wins']} 胜 / {row['draws']} 平 / {row['losses']} 负（得分率 {_pct(row['score_rate'])}） | D3 20s，对手 50ms；6 局 |"
        )
    lines.append("| V4 | B/C 固定局面性能 | 无正式对局结果 | 比较 50ms 与 200ms 搜索深度、节点和剪枝 |")
    return "\n".join(lines)


def build_report(results_dir: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Write tables, history comparison, and the complete V4B Chinese report."""

    root = Path(project_root or Path(__file__).resolve().parents[2])
    results = Path(results_dir)
    config = load_config(_load(results / "config.json"))
    records = load_game_records(results / "9x9_games.jsonl")
    validation = validate_records(records, config, require_complete=True)
    summary = write_analysis(records, config, results)
    history = _historical_summary(root)
    write_json_atomic(results / "history_comparison.json", history)
    reuse_audit = _load(results / "reuse_audit.json")
    preflight = _load(results / "preflight.json")
    execution_path = results / "execution.json"
    execution = _load(execution_path) if execution_path.exists() else _load(results / "progress.json")
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    matchup_lines = [
        "| 对阵 | 胜-平-负 | 纯胜率 | 得分率（配对 95% CI） | 黑棋/白棋得分率 | 来源 |",
        "|---|---|---|---|---|---|",
    ]
    for key, item in summary["matchups"].items():
        first, second = item["matchup"]
        first_wdl = f"{item['wins'][first]}-{item['draws']}-{item['losses'][first]}"
        second_wdl = f"{item['wins'][second]}-{item['draws']}-{item['losses'][second]}"
        first_color, second_color = item["color"][first], item["color"][second]
        source = ", ".join(f"{name} {count}" for name, count in sorted(item["source_counts"].items()))
        matchup_lines.append(
            f"| {first}–{second} | {first}: {first_wdl}; {second}: {second_wdl} | {first} {_pct(item['pure_win_rate'][first])}; {second} {_pct(item['pure_win_rate'][second])} | {first} {_pct(item['score_rate'][first])} {_ci(item['paired_wilson_95_ci'][first])}; {second} {_pct(item['score_rate'][second])} {_ci(item['paired_wilson_95_ci'][second])} | {first} 黑 {_pct(first_color['black']['score_rate'])} / 白 {_pct(first_color['white']['score_rate'])}; {second} 黑 {_pct(second_color['black']['score_rate'])} / 白 {_pct(second_color['white']['score_rate'])} | {source} |"
        )

    model_lines = [
        "| 排名 | 模型 | 总战绩 W-D-L | 纯胜率 | 得分率（配对 95% CI） |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, model in enumerate(summary["exploratory_ranking"], 1):
        item = summary["models"][model]
        model_lines.append(
            f"| {rank} | {model} | {item['wins']}-{item['draws']}-{item['losses']} | {_pct(item['win_rate'])} | {_pct(item['score_rate'])} {_ci(item['paired_wilson_95_ci'])} |"
        )

    usage_lines = [
        "| 模型 | 决策数 | 平均耗时/步 | 平均节点/步 | 完成深度均值（范围） | 剪枝数 | 模拟数/步 | 模拟/秒 | 树深最大 | 超时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("A", "B", "C", "D"):
        item = summary["usage"][model]
        depth_range = f"{item['average_completed_depth']:.2f}（{item['min_completed_depth']}–{item['max_completed_depth']}）"
        usage_lines.append(
            f"| {model} | {item['moves']} | {item['average_seconds_per_move']:.4f}s | {item['average_nodes_per_move']:.1f} | {depth_range} | {item['cutoffs_total']} | {item['average_simulations_per_move']:.1f} | {item['simulations_per_second']:.1f} | {item['max_tree_depth']} | {item['timeouts']} |"
        )

    v4_lines = [
        "| 模型 | 预算 | 平均耗时 | 平均深度 | 平均节点 | 节点/秒 | 平均剪枝 | 超时决策 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("B", "C"):
        for budget in ("50", "200"):
            item = history["V4_BC_50_vs_200ms_fixed_positions"][model][budget]
            v4_lines.append(
                f"| {model} | {budget}ms | {item['mean_elapsed_seconds']:.4f}s | {item['mean_completed_depth']:.2f} | {item['mean_nodes']:.1f} | {item['mean_nodes_per_second']:.0f} | {item['mean_cutoffs']:.1f} | {item['timeout_count']}/12 |"
            )

    d_win = next(
        (game for game in records if game["source_type"] == "new" and game["winner_model"] == "D" and set(game["matchup"]) & {"B", "C"}),
        next((game for game in records if game["winner_model"] == "D"), None),
    )
    d_loss = next(
        (game for game in records if "D" in game["matchup"] and game["winner_model"] in {"B", "C"}),
        None,
    )
    examples = []
    for label, game in (("D 获胜", d_win), ("D 失利", d_loss)):
        if game is None:
            examples.append(f"- **{label}**：本次没有观察到此类对局。")
        else:
            examples.append(
                f"- **{label}**：`{game['game_id']}`，{game['black_model']}执黑、{game['white_model']}执白，{game['move_count']} 手，{game['outcome']}（{game['termination']}）。{_short_game_transcript(game)}"
            )

    pred = preflight["estimate"]
    resource = preflight["resources"]
    d_usage = summary["usage"]["D_new_v4b"]
    d_vs_c = summary["matchups"].get("C-D", {})
    d_vs_b = summary["matchups"].get("B-D", {})
    if d_vs_b:
        d_b = d_vs_b["score_rate"]["D"]
    else:
        d_b = 0.0
    if d_vs_c:
        d_c = d_vs_c["score_rate"]["D"]
    else:
        d_c = 0.0
    if d_c > 0.5:
        conclusion = "在这 6 局探索性样本里，D20 对 C200ms 的得分率超过一半；这说明当前样本出现挑战成功，但不足以证明 MCTS 稳定强于 Alpha-Beta。"
    elif d_c == 0.5:
        conclusion = "在这 6 局探索性样本里，D20 与 C200ms 得分持平；目前不能判断谁更强。"
    else:
        conclusion = "在这 6 局探索性样本里，D20 对 C200ms 的得分率低于一半；这组样本没有显示 MCTS 突破 Alpha-Beta。"
    report = f"""# V4B：ABCD 最终对弈总结

## 执行结论

本轮完成 **{validation['validated_games']} 局**：新增 **{validation['new_games']} 局**，复用 V3 **{validation['reused_v3_games']} 局**。六组对阵均各 6 局，每个固定开局交换黑白一次。正式运行实际耗时 **{execution.get('wall_seconds_this_invocation', execution.get('elapsed_seconds_this_run', 0)):.1f} 秒**（另有预检 {preflight.get('preflight_elapsed_seconds', 0):.1f} 秒）。

V4B 条件下 D20 对 B200ms 得分率为 **{_pct(d_b)}**，对 C200ms 为 **{_pct(d_c)}**。{conclusion} D20 每步实际完成 **{d_usage['average_simulations_per_move']:.0f} 次模拟**（共 {d_usage['simulations_total']:,} 次；新跑局平均 {d_usage['simulations_per_second']:.0f} 次/秒），平均耗时 {d_usage['average_seconds_per_move']:.2f} 秒。MCTS每步约为 B/C 预算的 100 倍；结果属于非等算力挑战。

本报告没有神经网络训练；A/B/C/D 都是规则或搜索算法。本环境无法读取 Codex 会话的模型档位，因此不能独立核验 Codex 6.0 Luna / Max 配置。

## 实验方法与公平性

- 9×9 棋盘；A 使用原规则策略，决策后立即落子，不人为等待。
- B 为纯 Minimax，C 为 Minimax + Alpha-Beta；每步均 200ms。两者使用 V4 相同的候选半径、排序、评价函数与迭代加深逻辑；每步深度上限为剩余空位数（初始开局后 77 层），仅 C 启用剪枝。超时返回已完成层的结果。
- D 显示为 V4B 的 D，实际调用 V3 D3 的 `MCTSAgent`，每局持久化一个代理，保留 UCT、战术根筛选、战术 rollout 与跨步树复用，每步 20 秒。没有调用 Minimax，也没有神经网络。
- 对阵：A–B、A–C、B–C、A–D、B–D、C–D。开局种子 731091、731092、731093；每个开局两局交换黑白。普通新局种子由配置种子 `{config['base_seed']}` 和唯一 ID 哈希派生。
- 超时仍采用搜索器返回的合法落子并记超时；非法落子按负方判负。每局在 JSONL 落盘并 fsync 后才更新进度。

## 环境、预检与恢复

| 项目 | 实测值 |
|---|---:|
| CPU | {resource.get('cpu_model', '未知')}；逻辑 CPU {resource.get('logical_cpus')}；容器 CPU 配额 {resource.get('cgroup_cpu_quota_cores')} 核 |
| 内存 | cgroup 上限 {resource.get('cgroup_memory_limit_bytes', 0)/1024**3:.1f} GiB；可用内存 {resource.get('memory_available_bytes', 0)/1024**3:.1f} GiB |
| 工作区可用磁盘 | {resource.get('workspace_disk_free_bytes', 0)/1024**3:.1f} GiB |
| 正式任务预估 | {pred['estimated_formal_run_minutes']:.1f} 分钟；含 25% MCTS 对局长度余量范围约 {pred['estimated_formal_run_range_minutes'][0]:.1f}–{pred['estimated_formal_run_range_minutes'][1]:.1f} 分钟 |
| 12 局 D20 全部打满盘的松上界 | {pred['D20_fullboard_theoretical_ceiling_seconds_for_12_new_D_games']/3600:.2f} 小时 |
| 结果与恢复包预计存储 | {preflight['storage_estimate']['estimated_final_recovery_bundle_bytes']/1024**2:.2f} MiB |

预检在 V3 注册开局 731091 的同一 4 手局面上各测一次 A/B/C/D，不计入正式对局。D3 测得 {next(row['simulations'] for row in preflight['benchmarks'] if row['model']=='D'):,} 次模拟、{next(row['elapsed_seconds_wall'] for row in preflight['benchmarks'] if row['model']=='D'):.2f} 秒；A/B/C 的实测深度、节点、剪枝和落点保存在 `preflight.json`。时间估算使用 V3 已记录的 D20 对 B/C 各 6 局真实耗时和走子数，并对更强的 200ms 对手增加 25% 余量。

Google Drive 已完成本轮专属恢复包的上传、下载、SHA-256 校验和隔离目录恢复演练。恢复包只含恢复必需的配置、进度、棋谱、复用审计和代码版本；Drive 上只保留最新有效 V4B 恢复包，不触碰 V1–V4 的快照。

## V3 六局复用核验

复用源：`{reuse_audit.get('source_games')}`，V3 源 commit `{reuse_audit.get('source_code_commit')}`。V3 整套 54 局重放验证通过；D3 引擎源码 SHA-256 为 `{reuse_audit.get('current_mcts_d2_sha256')}`，与 V3 commit 中源码一致。六局各自保持原始棋谱、时间、模拟数、搜索树深、种子和超时数据；规范化副本将历史模型名 D3 映射为 V4B 的 D，并通过 `source_game_id`、逐局 SHA-256 和 `v3_reused_games.jsonl` 保留来源证据。A–D 的六局是历史复用，不计入新增。

## 六组完整胜负结果

得分率定义为胜 1、平 0.5、负 0。区间基于 3 个配对开局块，每块汇总交换黑白的两局后计算 Wilson-style 95% 描述区间；小样本下该区间覆盖率未经校准，只作为不确定性提示。

{chr(10).join(matchup_lines)}

### 完整胜负矩阵（行方视角）

| 行方 \\ 对手 | A | B | C | D |
|---|---|---|---|---|
| A | — | {summary['matchups']['A-B']['wins']['A']}-{summary['matchups']['A-B']['draws']}-{summary['matchups']['A-B']['losses']['A']} | {summary['matchups']['A-C']['wins']['A']}-{summary['matchups']['A-C']['draws']}-{summary['matchups']['A-C']['losses']['A']} | {summary['matchups']['A-D']['wins']['A']}-{summary['matchups']['A-D']['draws']}-{summary['matchups']['A-D']['losses']['A']} |
| B | {summary['matchups']['A-B']['wins']['B']}-{summary['matchups']['A-B']['draws']}-{summary['matchups']['A-B']['losses']['B']} | — | {summary['matchups']['B-C']['wins']['B']}-{summary['matchups']['B-C']['draws']}-{summary['matchups']['B-C']['losses']['B']} | {summary['matchups']['B-D']['wins']['B']}-{summary['matchups']['B-D']['draws']}-{summary['matchups']['B-D']['losses']['B']} |
| C | {summary['matchups']['A-C']['wins']['C']}-{summary['matchups']['A-C']['draws']}-{summary['matchups']['A-C']['losses']['C']} | {summary['matchups']['B-C']['wins']['C']}-{summary['matchups']['B-C']['draws']}-{summary['matchups']['B-C']['losses']['C']} | — | {summary['matchups']['C-D']['wins']['C']}-{summary['matchups']['C-D']['draws']}-{summary['matchups']['C-D']['losses']['C']} |
| D | {summary['matchups']['A-D']['wins']['D']}-{summary['matchups']['A-D']['draws']}-{summary['matchups']['A-D']['losses']['D']} | {summary['matchups']['B-D']['wins']['D']}-{summary['matchups']['B-D']['draws']}-{summary['matchups']['B-D']['losses']['D']} | {summary['matchups']['C-D']['wins']['D']}-{summary['matchups']['C-D']['draws']}-{summary['matchups']['C-D']['losses']['D']} | — |

![六组配对得分率和区间](../figures/matchup_score_rates.svg)

## 四模型总排名与先后手

每个模型共 18 局；排名仅按本轮样本得分率排序，不能视为稳定棋力定论。

{chr(10).join(model_lines)}

| 模型 | 黑棋得分率（局数） | 白棋得分率（局数） |
|---|---:|---:|
{chr(10).join(f"| {model} | {_pct(item['color']['black']['score_rate'])}（{item['color']['black']['games']}） | {_pct(item['color']['white']['score_rate'])}（{item['color']['white']['games']}） |" for model, item in summary['models'].items())}

## 搜索深度与计算成本

{chr(10).join(usage_lines)}

![各模型平均单步成本](../figures/model_search_cost.svg)

V4 固定局面基准中，B 从 50ms 到 200ms 的平均深度由 {history['V4_BC_50_vs_200ms_fixed_positions']['B']['50']['mean_completed_depth']:.2f} 增至 {history['V4_BC_50_vs_200ms_fixed_positions']['B']['200']['mean_completed_depth']:.2f}，平均节点由 {history['V4_BC_50_vs_200ms_fixed_positions']['B']['50']['mean_nodes']:.0f} 增至 {history['V4_BC_50_vs_200ms_fixed_positions']['B']['200']['mean_nodes']:.0f}；C 的平均深度由 {history['V4_BC_50_vs_200ms_fixed_positions']['C']['50']['mean_completed_depth']:.2f} 增至 {history['V4_BC_50_vs_200ms_fixed_positions']['C']['200']['mean_completed_depth']:.2f}，节点由 {history['V4_BC_50_vs_200ms_fixed_positions']['C']['50']['mean_nodes']:.0f} 增至 {history['V4_BC_50_vs_200ms_fixed_positions']['C']['200']['mean_nodes']:.0f}。这是相同局面上的搜索预算变化，不代表胜率本身的因果变化。

| V4 固定局面搜索 | 预算 | 平均耗时 | 深度 | 节点 | 节点/秒 | 剪枝 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(v4_lines)}

V4B 实际对局中，B/C 节点、深度和 C 剪枝数如上表。两者不在同一条独立固定局面序列上逐状态逐节点比较，所以对局汇总只能描述实际使用量；B/C 算法效率因果结论沿用 V4 的相同局面验证。D20s 新跑局实际完成 {d_usage['simulations_total']:,} 次模拟、平均 {d_usage['average_simulations_per_move']:.0f} 次/步；树深最大 {d_usage['max_tree_depth']}，树复用节点累计 {d_usage['reused_nodes_total']:,}。历史复用的六局另见 `model_usage.csv` 中 `D_v3_reused` 行。

## 与 V1–V4 历史结果的关系

{_historical_markdown(history)}

V3 对 B/C 的对手计算预算为 50ms，V4B 为 200ms；V4B 的 B/C 又都显著低于 D 的 20 秒。应将结果解释为“搜索预算提升条件下的探索性挑战”，不能把对局差异单独归因于算法改进。V3 D3 和 V4B D 使用相同 MCTS 源代码与搜索设置。

## 典型棋谱与失利观察

{chr(10).join(examples)}

原始 JSONL 保留全部 36 局的完整走子和每步指标；上面的例子只列尾段以便阅读。棋谱坐标使用从 0 开始的 `(行,列)`。胜局按真实五连判定，平局按棋盘填满判定；非法落子则单独按弃权判负。

## 结论与限制

1. A–D 历史六局与 V3 D20 对 A 的来源、开局、颜色、预算及棋谱完全匹配，因此合法复用；其余 30 局是 V4B 新运行。
2. D 能否挑战 B/C，应分别看 B200ms 和 C200ms 两组的 6 局配对结果；C 是最强传统搜索对手。本轮 {conclusion}
3. D 的每步计算时间约为 B/C 的 100 倍。即便 D20 获得更多模拟，也不能直接称其性价比更高；按每秒模拟数及每局思考总时长一起看。
4. 仅三个开局、每组六局，且 A–D 来自 V3，统计区间宽、配对块数少。没有对先后手、开局或偶然战术做种子筛选；结果不支持稳定等级宣称。
5. 本轮只做 9×9。没有 15×15 验证，也没有训练或 checkpoint。

## 复现与文件

- 固定配置：`configs/abcd_final_v4b.json`
- 对弈运行：`python scripts/run_abcd_final_v4b.py --config configs/abcd_final_v4b.json --output-dir research/v4b/results`
- 全局校验与报告：`python scripts/summarize_abcd_final_v4b.py --results research/v4b/results`
- 原始数据：`research/v4b/results/9x9_games.jsonl`；V3 源记录副本：`research/v4b/results/v3_reused_games.jsonl`
- 统计表：`pairwise_summary.csv`、`model_summary.csv`、`model_usage.csv`；机器可读汇总：`summary.json`、`history_comparison.json`
- 恢复包校验：`python scripts/restore_abcd_final_v4b_snapshot.py <snapshot.zip> <empty-destination>`
- 分支提交：`{current_commit}`（最终数据报告提交会更新此值）
"""
    (results / "report_zh.md").write_text(report, encoding="utf-8")
    return {
        "validation": validation,
        "summary": summary,
        "historical": history,
        "report_path": str(results / "report_zh.md"),
    }
