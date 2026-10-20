"""Measure V5 resource use and estimate the full ladder before official play."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
from typing import Any

from ..ai import choose_move
from ..gomoku import GomokuBoard
from .abcd_ladder_v5 import build_schedule, derive_seed, load_config, settings_for_model
from .abcd_ladder_v5 import _project_root, _sha256
from .checkpoint import write_json_atomic
from .mcts_d2 import MCTSAgent


def resource_snapshot(directory: str | Path) -> dict[str, Any]:
    cpu_model = "unknown"
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    memory: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(errors="replace").splitlines():
            key, _, value = line.partition(":")
            if value.strip().endswith("kB"):
                memory[key] = int(value.strip().split()[0]) * 1024
    cpu_quota = None
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        cpu_quota = None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    memory_limit = None
    try:
        value = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        memory_limit = None if value == "max" else int(value)
    except (OSError, ValueError):
        pass
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    disk = shutil.disk_usage(Path(directory))
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": os.cpu_count(), "cpu_model": cpu_model,
        "cgroup_cpu_quota_cores": cpu_quota,
        "load_average_1_5_15m": load,
        "memory_total_bytes": memory.get("MemTotal"), "memory_available_bytes": memory.get("MemAvailable"),
        "cgroup_memory_limit_bytes": memory_limit,
        "workspace_disk_total_bytes": disk.total, "workspace_disk_free_bytes": disk.free,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def _benchmark_board(config: dict[str, Any]) -> GomokuBoard:
    board = GomokuBoard(9)
    for player, row, col in config["opening_catalog"]["731101"]:
        board.place_stone(str(player), (int(row), int(col)))
    return board


def _historical_decisions(root: Path) -> dict[str, float]:
    games_path = root / "research/v4c/results/9x9_games.jsonl"
    counts = {"B": [], "C": [], "D": []}
    if games_path.exists():
        for line in games_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            for model in counts:
                count = sum(move.get("model") == model for move in game.get("moves", []))
                if count:
                    counts[model].append(float(count))
    return {model: sum(values) / len(values) if values else 30.0 for model, values in counts.items()}


def _historical_elapsed(root: Path) -> dict[str, float]:
    path = root / "research/v4c/results/model_usage.csv"
    result: dict[str, float] = {}
    if path.exists():
        import csv
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                result[str(row["model"])] = float(row["mean_seconds_per_move"])
    return result


def estimate_full_schedule(config: dict[str, Any], benchmarks: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Estimate a full all-routes-through-50s run and a strict full-board ceiling."""
    benchmark_time = {
        (row["model"], row["budget_label"]): float(row["elapsed_seconds_wall"])
        for row in benchmarks if row["model"] in {"B", "C", "D"}
    }
    historical_moves = _historical_decisions(root)
    historical_elapsed = _historical_elapsed(root)
    estimate_seconds = 0.0
    theoretical_seconds = 0.0
    model_estimate_seconds = {model: 0.0 for model in ("B", "C", "D")}
    group_counts = {"A-B": 0, "A-C": 0, "A-D": 0, "B-C": 0, "B-D": 0, "C-D": 0}
    schedule = build_schedule(config)
    longest_game_ceiling = 0.0
    longest_group_ceiling = 0.0
    for group in schedule:
        pair_name = "-".join(group["pair"])
        group_counts[pair_name] += 1
        game_budget_ceiling = []
        for game in group["games"]:
            one_game = 0.0
            for model in game["pair"]:
                budget = game["budget_seconds_by_model"].get(model)
                if model == "A" or budget is None:
                    continue
                one_game += 39.0 * float(budget)
                rung = next(item["label"] for item in config["budget_ladder"] if float(item["seconds"]) == float(budget))
                per_move = benchmark_time.get((model, rung), float(budget))
                estimated_decisions = historical_moves.get(model, 30.0)
                estimate_seconds += estimated_decisions * per_move
                model_estimate_seconds[model] += estimated_decisions * per_move
                theoretical_seconds += 39.0 * float(budget)
            game_budget_ceiling.append(one_game)
            longest_game_ceiling = max(longest_game_ceiling, one_game)
        longest_group_ceiling = max(longest_group_ceiling, sum(game_budget_ceiling))

    n_games = sum(group_counts.values()) * 2
    jsonl = root / "research/v4c/results/9x9_games.jsonl"
    historical_jsonl_bytes = jsonl.stat().st_size if jsonl.exists() else 0
    estimated_storage = max(1_000_000, int(historical_jsonl_bytes * n_games / 18 * 1.5))
    return {
        "maximum_groups": len(schedule), "maximum_games": n_games,
        "groups_by_family_pair": group_counts,
        "preflight_model_decisions_per_game_v4c": historical_moves,
        "v4c_historical_mean_seconds_per_move": historical_elapsed,
        "estimated_full_schedule_wall_seconds_from_rung_benchmarks": estimate_seconds,
        "estimated_full_schedule_wall_minutes": estimate_seconds / 60.0,
        "estimated_model_thinking_seconds": model_estimate_seconds,
        "configured_budget_sum_full_board_ceiling_seconds_using_39_moves_each": theoretical_seconds,
        "configured_budget_sum_full_board_ceiling_hours": theoretical_seconds / 3600.0,
        "longest_single_game_budget_ceiling_seconds_using_39_moves_each": longest_game_ceiling,
        "longest_single_game_budget_ceiling_minutes": longest_game_ceiling / 60.0,
        "longest_paired_group_budget_ceiling_seconds_using_39_moves_each": longest_group_ceiling,
        "longest_paired_group_budget_ceiling_minutes": longest_group_ceiling / 60.0,
        "estimated_storage_bytes": estimated_storage,
        "estimate_note": "Full 60-group route continuation is unlikely because routes can fail early. Wall estimate uses V4C observed decisions per game and current-run per-rung measurements. The configured-budget ceiling assumes all 120 games play all remaining moves (at most 39 decisions per non-A model per game) at their scheduled budgets. A is excluded because it has no artificial time budget.",
    }


def history_comparison(root: Path) -> dict[str, Any]:
    """Extract compact V1–V4C context without merging it into V5 statistics."""
    comparisons: list[dict[str, str]] = []
    sources: list[str] = []
    v1_path = root / "research/results/summary.json"
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))["9"]["results"]
        bc = v1["win_matrix"]["B"]["C"]
        comparisons.append({
            "stage": "V1（9×9、50ms 基线）",
            "finding": f"完成 {v1['game_count']} 局；B–C 为 B {bc['wins']} 胜、{bc['draws']} 平、C {bc['losses']} 胜；报告综合排序为 C、B、A、D。",
        })
        sources.append("research/results/summary.json")
    v2_path = root / "research/v2/results/summary.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        for key, label in (
            ("extreme-9x9-d2-200ms-vs-c-50ms", "200ms"),
            ("extreme-9x9-d2-1s-vs-c-50ms", "1s"),
            ("extreme-9x9-d2-5s-vs-c-50ms", "5s"),
        ):
            phase = v2["phases"].get(key)
            if not phase:
                continue
            d2 = phase["pairwise"]["C-D2"]["models"]["D2"]
            comparisons.append({
                "stage": f"V2 D2 vs C50ms（{label}）",
                "finding": f"{phase['game_count']} 局；D2 {d2['wins']} 胜、{d2['draws']} 平、{d2['losses']} 负，得分率 {d2['score_rate']:.1%}。",
            })
        sources.append("research/v2/results/summary.json")
    v3_path = root / "research/v3/results/summary.json"
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        for budget in (10, 20, 50):
            phase = v3["phases"].get(f"d3-{budget}s-vs-c")
            if not phase:
                continue
            d3 = phase["D3"]
            comparisons.append({
                "stage": f"V3 D3 {budget}s vs C50ms",
                "finding": f"{phase['games']} 局；D3 {d3['wins']} 胜、{d3['draws']} 平、{d3['losses']} 负，得分率 {d3['score_rate']:.1%}。",
            })
        sources.append("research/v3/results/summary.json")
    v4_path = root / "research/v4/results/summary.json"
    if v4_path.exists():
        v4 = json.loads(v4_path.read_text(encoding="utf-8"))
        comparisons.append({
            "stage": "V4 ABC 搜索预算性能",
            "finding": f"完成 {v4['actual_run_count']} 个搜索测试；C 在 10–50 秒档中有 {v4['c_runs_above_depth_four_at_10s_plus']}/{v4['c_runs_at_10s_plus']} 次超过深度 4，显示加时可越过原层数限制，但后续迭代的收益取决于局面。",
        })
        sources.append("research/v4/results/summary.json")
    v4b_path = root / "research/v4b/results/summary.json"
    if v4b_path.exists():
        v4b = json.loads(v4b_path.read_text(encoding="utf-8"))
        comparisons.append({
            "stage": "V4B（ABCD 最终对弈）",
            "finding": f"{v4b['game_count']} 局探索赛，观测排序 {', '.join(v4b['exploratory_ranking'])}；C200ms、B200ms、D20s 的对阵只各 6 局，样本小且预算不等。",
        })
        sources.append("research/v4b/results/summary.json")
    v4c_path = root / "research/v4c/results/summary.json"
    if v4c_path.exists():
        v4c = json.loads(v4c_path.read_text(encoding="utf-8"))
        d_usage = v4c.get("model_usage", {}).get("D", {})
        records = []
        for pair in ("B-C", "B-D", "C-D"):
            row = v4c["matchups"].get(pair, {})
            first, second = pair.split("-")
            records.append(f"{pair} {row.get('wins', {}).get(first, 0)}–{row.get('draws', 0)}–{row.get('wins', {}).get(second, 0)}")
        comparisons.append({
            "stage": "V4C（B1s / C200ms / D50s）",
            "finding": f"{v4c['game_count']} 局；三个对阵均为一胜、四平、一负（{'; '.join(records)}），每组 6 局。D50s 实测 {d_usage.get('decisions', 0)} 步、均值 {d_usage.get('mean_seconds_per_move', 0):.2f}s/步、{d_usage.get('mean_simulations_per_move', 0):.0f} 次模拟/步。",
        })
        sources.append("research/v4c/results/summary.json")
    return {
        "comparisons": comparisons,
        "sources": ", ".join(sources),
        "historical_data_in_v5_score": False,
        "note": "V1–V4C are historical context only; the V5 schedule and results are evaluated separately.",
    }


def run_preflight(config_source: str | Path, output_path: str | Path) -> dict[str, Any]:
    preflight_started = time.perf_counter()
    config = load_config(config_source)
    root = _project_root()
    resource_before = resource_snapshot(root)
    benchmarks: list[dict[str, Any]] = []
    base_board = _benchmark_board(config)
    for model in ("A", "B", "C", "D"):
        budgets = [None] if model == "A" else [float(item["seconds"]) for item in config["budget_ladder"]]
        for budget in budgets:
            board = GomokuBoard.from_dict(base_board.to_dict())
            label = "natural" if budget is None else next(item["label"] for item in config["budget_ladder"] if float(item["seconds"]) == budget)
            settings = settings_for_model(config, model, len(board.history), budget)
            seed = derive_seed(int(config["base_seed"]), "v5-preflight", model, label)
            started = time.perf_counter()
            if model == "D":
                result = MCTSAgent().choose_move(board, "B", settings, seed)
            else:
                result = choose_move(board, "B", model, settings, seed)
            wall = time.perf_counter() - started
            benchmarks.append({
                "model": model, "budget_label": label, "budget_seconds": budget,
                "move": list(result.move), "completed_depth": int(result.completed_depth),
                "tree_depth": int(result.completed_depth), "nodes": int(result.nodes),
                "cutoffs": int(result.cutoffs), "simulations": int(result.simulations),
                "elapsed_seconds_reported": float(result.elapsed_seconds), "elapsed_seconds_wall": wall,
                "timed_out": bool(result.timed_out), "seed": seed,
            })
    resource_after = resource_snapshot(root)
    result = {
        "experiment_id": config["experiment_id"],
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight_only": True,
        "official_games_started": 0,
        "base_code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
        ).stdout.strip(),
        "source_sha256": {relative: _sha256((root / relative).read_bytes()) for relative in config["engine_sha256"]},
        "resources_before": resource_before,
        "resources_after": resource_after,
        "position": {
            "board_size": 9, "opening_seed": 731101,
            "opening_moves": config["opening_catalog"]["731101"], "to_move": "B",
        },
        "benchmarks": benchmarks,
        "estimate": estimate_full_schedule(config, benchmarks, root),
        "preflight_elapsed_seconds": time.perf_counter() - preflight_started,
    }
    write_json_atomic(Path(output_path), result)
    write_json_atomic(Path(output_path).with_name("history_comparison.json"), history_comparison(root))
    return result
