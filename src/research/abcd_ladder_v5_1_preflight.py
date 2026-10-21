"""Resource check, representative search benchmark and V5.1 runtime estimate."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from ..ai import choose_move
from ..gomoku import GomokuBoard
from .abcd_ladder_v5 import (
    _project_root, derive_seed, load_config as load_v5_config, settings_for_model,
)
from .abcd_ladder_v5_1 import build_routes, build_schedule, load_config
from .abcd_ladder_v5_1_reuse import prepare_v5_reuse
from .checkpoint import load_game_records, write_json_atomic
from .mcts_d2 import MCTSAgent

_BUDGETS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")


def _resource_snapshot(path: str | Path) -> dict[str, Any]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, raw = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key.lower() + "_kib"] = int(raw.strip().split()[0])
    usage = shutil.disk_usage(path)
    return {
        "cpu_count_visible": os.cpu_count(),
        "memory": memory,
        "disk": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
    }


def _benchmark_opening(config: dict[str, Any], opening_seed: int) -> GomokuBoard:
    board = GomokuBoard(9)
    for player, row, col in config["opening_catalog"][str(opening_seed)]:
        board.place_stone(str(player), (int(row), int(col)))
    return board


def benchmark_searches(
    config: dict[str, Any],
    *,
    before_move: Callable[[str, str, float | None], None] | None = None,
) -> dict[str, Any]:
    """Run one real, identical opening-position decision per version budget."""
    frozen_v5 = load_v5_config(_project_root() / "configs/abcd_ladder_v5.json")
    rows: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    cases: list[tuple[str, str, float | None]] = [("A", "natural", None)]
    for model in ("B", "C", "D"):
        cases.extend((model, label, float(rung["seconds"]))
                     for label, rung in zip(_BUDGETS, config["budget_ladder"], strict=True))
    for index, (model, label, budget) in enumerate(cases):
        if before_move:
            before_move(model, label, budget)
        board = _benchmark_opening(config, int(config["opening_seeds"][0]))
        current = "B"
        settings = settings_for_model(frozen_v5, model, len(board.history), budget)
        seed = derive_seed(int(config["base_seed"]), "v5.1-preflight", model, label, index)
        started = time.perf_counter()
        if model == "D":
            result = MCTSAgent().choose_move(board, current, settings, seed)
        else:
            result = choose_move(board, current, model, settings, seed)
        wall_seconds = time.perf_counter() - started
        rows.append({
            "model": model, "budget_label": label,
            "budget_seconds": budget,
            "measured_seconds": float(result.elapsed_seconds),
            "measured_wall_seconds": wall_seconds,
            "completed_depth": int(result.completed_depth),
            "nodes": int(result.nodes), "cutoffs": int(result.cutoffs),
            "simulations": int(result.simulations), "tree_depth": int(result.completed_depth),
            "timed_out": bool(result.timed_out),
            "legal_move": (
                isinstance(result.move, (tuple, list)) and len(result.move) == 2
                and board.in_bounds(int(result.move[0]), int(result.move[1]))
                and board.get((int(result.move[0]), int(result.move[1]))) is None
            ),
        })
    return {"cases": rows, "benchmark_wall_seconds": time.perf_counter() - wall_started}


def _historical_turns_by_pair(v5_results_dir: Path, v5_config_path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    v5_config = load_v5_config(v5_config_path)
    records = load_game_records(v5_results_dir / "9x9_games.jsonl")
    from .abcd_ladder_v5_analysis import validate_records
    validation = validate_records(records, v5_config, require_complete=True)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in records:
        grouped["-".join(row["pair"])].append(max(1, int(row["move_count"]) - int(v5_config["opening_plies"])))
    turns = {pair: sum(counts) / len(counts) for pair, counts in grouped.items()}
    return turns, {"games": len(records), "groups": validation["groups"], "validation": validation}


def estimate_runtime(
    schedule: list[dict[str, Any]],
    reuse_manifest: dict[str, Any],
    benchmark: dict[str, Any],
    turns_by_pair: dict[str, float],
    *,
    storage_bytes_per_game: int,
) -> dict[str, Any]:
    measured: dict[tuple[str, str], float] = {
        (row["model"], row["budget_label"]): max(float(row["measured_seconds"]), 0.0)
        for row in benchmark["cases"]
    }
    v5_reuse_ids = {row["group_id"] for row in reuse_manifest.get("groups", [])}
    new_groups = [group for group in schedule if group["group_id"] not in v5_reuse_ids]
    estimated_seconds = 0.0
    budget_ceiling_seconds = 0.0
    group_estimates: list[dict[str, Any]] = []
    for group in new_groups:
        first, second = group["pair"]
        pair = "-".join(group["pair"])
        mean_nonopening_plies = float(turns_by_pair.get(pair, 30.0))
        measured_times: dict[str, float] = {}
        budget_seconds: dict[str, float | None] = group["budget_seconds_by_model"]
        for model in (first, second):
            if budget_seconds[model] is None:
                label = "natural"
            else:
                label = next(
                    row["budget_label"] for row in benchmark["cases"]
                    if row["model"] == model and float(row["budget_seconds"]) == float(budget_seconds[model])
                )
            measured_times[model] = measured[(model, label)]
        group_seconds = mean_nonopening_plies * sum(measured_times.values())
        ceiling_seconds = 78.0 * sum(
            0.0 if budget_seconds[model] is None else float(budget_seconds[model])
            for model in (first, second)
        )
        estimated_seconds += group_seconds
        budget_ceiling_seconds += ceiling_seconds
        group_estimates.append({
            "group_id": group["group_id"], "pair": pair,
            "model_versions": dict(group["model_versions"]),
            "mean_nonopening_plies_per_game": mean_nonopening_plies,
            "estimated_compute_seconds_two_games": group_seconds,
        })
    eligible_groups = int(reuse_manifest.get("reused_groups", 0))
    return {
        "eligible_v5_reused_groups": eligible_groups,
        "estimated_new_groups_if_all_routes_reach_50s": len(new_groups),
        "estimated_new_games_if_all_routes_reach_50s": 2 * len(new_groups),
        "estimated_full_schedule_wall_seconds_serial": estimated_seconds,
        "estimated_full_schedule_wall_hours_serial": estimated_seconds / 3600.0,
        "configured_full_board_budget_ceiling_hours": budget_ceiling_seconds / 3600.0,
        "maximum_single_game_budget_seconds": 39.0 * max(
            sum(float(value or 0.0) for value in group["budget_seconds_by_model"].values())
            for group in schedule
        ),
        "estimated_storage_bytes_for_all_336_games": int(storage_bytes_per_game) * 336,
        "group_estimates": group_estimates,
        "estimation_method": (
            "每个未复用配对组按V5同家族对局的实际平均非开局着数估计；"
            "双方各自使用本次相同开局实测每步耗时；另列77个剩余格的配置时间极端上界。"
        ),
    }


def run_preflight(
    config_source: str | Path | dict[str, Any],
    output_path: str | Path,
    *,
    benchmarker: Callable[[dict[str, Any]], dict[str, Any]] = benchmark_searches,
) -> dict[str, Any]:
    frozen = load_config(config_source)
    root = _project_root()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    resources_before = _resource_snapshot(output.parent)
    schedule, reuse_manifest = prepare_v5_reuse(
        build_schedule(frozen), root / "research/v5/results", frozen
    )
    routes = build_routes(frozen)
    if len(routes) != 66 or len(schedule) != 168:
        raise ValueError("V5.1 route generation preflight failed its independent-route cardinality")
    group_ids = [group["group_id"] for group in schedule]
    game_ids = [game["game_id"] for group in schedule for game in group["games"]]
    if len(group_ids) != len(set(group_ids)) or len(game_ids) != len(set(game_ids)):
        raise ValueError("V5.1 route generation preflight found duplicate game/group IDs")
    for route in routes.values():
        positions = {group_id: index for index, group_id in enumerate(group_ids)}
        route_positions = [positions[level["group_id"]] for level in route["levels"]]
        if route_positions != sorted(route_positions):
            raise ValueError(f"V5.1 route dependency order is cyclic or non-monotone: {route['route_id']}")
    started = time.perf_counter()
    benchmark = benchmarker(frozen)
    if len(benchmark["cases"]) != 22 or any(not row["legal_move"] for row in benchmark["cases"]):
        raise ValueError("V5.1 search preflight did not produce all 22 legal representative decisions")
    v5_results = root / "research/v5/results"
    turns, history = _historical_turns_by_pair(v5_results, root / "research/v5/results/config.json")
    v5_raw_bytes = (v5_results / "9x9_games.jsonl").stat().st_size
    v5_games = len(load_game_records(v5_results / "9x9_games.jsonl"))
    bytes_per_game = max(1, int(v5_raw_bytes / max(v5_games, 1)))
    estimate = estimate_runtime(schedule, reuse_manifest, benchmark, turns,
                                storage_bytes_per_game=bytes_per_game)
    result = {
        "experiment_id": frozen["experiment_id"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "code_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "resources_before": resources_before,
        "resources_after": _resource_snapshot(output.parent),
        "schedule_audit": {
            "independent_routes": len(routes), "groups": len(schedule),
            "game_ids": len(game_ids), "duplicate_ids": False,
            "eligible_v5_reuse_groups": reuse_manifest["reused_groups"],
            "eligible_v5_reuse_games": reuse_manifest["reused_games"],
            "v5_validation": history["validation"],
        },
        "benchmark": benchmark,
        "preflight_elapsed_seconds": time.perf_counter() - started,
        "historical_turn_basis": {"games": history["games"], "groups": history["groups"],
                                  "mean_nonopening_plies_by_family_pair": turns},
        "estimate": estimate,
        "human_readable_estimate": (
            f"預估全部仍需新跑约 {estimate['estimated_new_groups_if_all_routes_reach_50s']} 组 / "
            f"{estimate['estimated_new_games_if_all_routes_reach_50s']} 局；按V5平均对局长度与本次逐档"
            f"实测速度，单进程约 {estimate['estimated_full_schedule_wall_hours_serial']:.2f} 小时；"
            f"配置预算的全棋盘上界约 {estimate['configured_full_board_budget_ceiling_hours']:.2f} 小时。"
        ),
    }
    write_json_atomic(output, result)
    return result
