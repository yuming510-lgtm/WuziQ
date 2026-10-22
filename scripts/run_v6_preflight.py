#!/usr/bin/env python3
"""Measure representative V6 move costs and freeze a task-size estimate."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import choose_move
from src.gomoku import GomokuBoard
from src.research.abcd_ladder_v5 import derive_seed, settings_for_model
from src.research.checkpoint import load_game_records
from src.research.mcts_d2 import MCTSAgent
from src.research.v6.config import BUDGET_LABELS, BUDGET_SECONDS, load_config, v5_1_engine_settings
from src.research.v6.engine import V6Agent
from src.research.v6.puzzles import board_for_position, load_tactical_positions
from src.research.v6.reuse import prepare_v5_1_reuse
from src.research.v6.schedule import build_schedule
from src.research.v6.state import derive_state


def _resources() -> dict[str, object]:
    disk = shutil.disk_usage(ROOT)
    result: dict[str, object] = {
        "cpu_logical": os.cpu_count(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }
    try:
        import psutil
        result.update({
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_total_bytes": psutil.virtual_memory().total,
            "memory_available_bytes": psutil.virtual_memory().available,
        })
    except ImportError:
        result["memory_total_bytes"] = None
        result["memory_available_bytes"] = None
    return result


def _decision(model: str, board: GomokuBoard, budget: float | None, engine: dict, config: dict):
    player = "B"
    seed = 6042026
    if model in {"A", "B", "C", "D"}:
        settings = settings_for_model(engine, model, len(board.history), budget)
        if model == "D":
            return MCTSAgent().choose_move(board, player, settings, seed)
        return choose_move(board, player, model, settings, seed)
    return V6Agent(
        model,
        max_attack_depth=int(config["tactical_limits"]["max_attack_depth"]),
        max_nodes=int(config["tactical_limits"]["pns_max_nodes"] if model == "E4" else config["tactical_limits"]["max_nodes"]),
    ).choose_move(
        board, player, time_limit_seconds=float(budget), seed=seed,
        max_depth=int(config["tactical_limits"]["e5_max_depth"]),
    )


def _time_for(model: str, budget_index: int | None, measured: dict[tuple[str, str], dict], scenario: str) -> float:
    if model == "A":
        return float(measured[("A", "natural")]["elapsed_seconds"])
    label = BUDGET_LABELS[int(budget_index)]
    seconds = BUDGET_SECONDS[int(budget_index)]
    row = measured.get((model, label))
    if row is not None:
        return float(row["elapsed_seconds"])
    if scenario == "optimistic":
        return min(seconds, float(measured[(model, "5s")]["elapsed_seconds"]))
    if scenario == "central":
        five = measured[(model, "5s")]
        if five["timed_out"]:
            return seconds
        return min(seconds, float(five["elapsed_seconds"]) * (seconds / 5.0) ** 0.6)
    return seconds


def _estimate(
    schedule: list[dict], state: dict, reused_groups: int,
    measured: dict[tuple[str, str], dict], historic_plies: float,
) -> dict[str, object]:
    reachable: set[str] = set()
    for route_id, route in state["routes"].items():
        if route["status"] != "active":
            continue
        for level in route["levels"]:
            if level["status"] in {"pending", "not_reached", "incomplete_pair"}:
                reachable.add(str(level["group_id"]))
    schedule_by_id = {str(group["group_id"]): group for group in schedule}
    frontier = set(state["pending_group_ids"])
    scenarios = {
        "optimistic": (frontier, 20.0, 1.0),
        "central": (reachable, historic_plies, 0.5),
        "conservative": (reachable, 81.0, 1.0),
    }
    results: dict[str, object] = {}
    for name, (group_ids, total_plies, work_fraction) in scenarios.items():
        engine_plies = max(0.0, float(total_plies) - 4.0)
        work = 0.0
        for group_id in group_ids:
            group = schedule_by_id[group_id]
            for model in group["pair"]:
                version = group["model_versions"][model]
                if model == "A":
                    cost = _time_for(model, None, measured, name)
                else:
                    index = BUDGET_LABELS.index(version)
                    cost = _time_for(model, index, measured, name)
                work += engine_plies * cost
        work *= work_fraction
        results[name] = {
            "groups_assumed": round(len(group_ids) * work_fraction),
            "avg_total_plies_per_game_assumed": total_plies,
            "estimated_wall_seconds_serial": work,
            "estimated_wall_hours_serial": work / 3600.0,
            "assumption": {
                "optimistic": "only the current route frontier runs; historical 9x9 game length is shortened to 20 plies; >5s decisions are capped at the measured 5s cost",
                "central": "half of currently reachable groups run; V5.1 reused-game average length is used; 10-50s searches scale from observed 5s behavior or consume their full budget if 5s timed out",
                "conservative": "all currently reachable groups run to a full 81-ply board; every search move consumes its full configured budget",
            }[name],
        }
    return {
        "reachable_unique_groups_from_current_routes": len(reachable),
        "current_frontier_groups": len(frontier),
        "current_reused_groups": reused_groups,
        "current_reused_games": 2 * reused_groups,
        "theoretical_max_groups": len(schedule),
        "theoretical_max_games": 2 * len(schedule),
        "max_additional_games_after_reuse": 2 * (len(schedule) - reused_groups),
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6/preflight")
    args = parser.parse_args()
    config = load_config()
    engine = v5_1_engine_settings(config)
    schedule, route_specs = build_schedule(config)
    frozen_schedule, manifest, reused = prepare_v5_1_reuse(schedule, config)
    state = derive_state(reused, frozen_schedule, route_specs)
    positions = {row["position_id"]: row for row in load_tactical_positions()}
    representative = positions["vct_cross_three"]
    models = ["B", "C", "D", "E1", "E2", "E3", "E4", "E5"]
    rows: list[dict] = []
    measured: dict[tuple[str, str], dict] = {}
    started = time.perf_counter()
    board = board_for_position(representative)
    result = _decision("A", board, None, engine, config)
    row = {
        "model": "A", "budget_label": "natural", "budget_seconds": None,
        "position_id": representative["position_id"], "move": list(result.move),
        "elapsed_seconds": float(result.elapsed_seconds), "nodes": int(result.nodes),
        "completed_depth": int(result.completed_depth), "cutoffs": int(result.cutoffs),
        "simulations": int(result.simulations), "timed_out": bool(result.timed_out),
        "stats": dict(getattr(result, "stats", {}) or {}),
    }
    rows.append(row)
    measured[("A", "natural")] = row
    for model in models:
        for label, budget in zip(BUDGET_LABELS[:4], BUDGET_SECONDS[:4]):
            board = board_for_position(representative)
            result = _decision(model, board, budget, engine, config)
            stats = dict(getattr(result, "stats", {}) or {})
            row = {
                "model": model, "budget_label": label, "budget_seconds": budget,
                "position_id": representative["position_id"], "move": list(result.move),
                "elapsed_seconds": float(result.elapsed_seconds), "nodes": int(result.nodes),
                "completed_depth": int(result.completed_depth), "tree_depth": int(stats.get("max_attack_depth", result.completed_depth)),
                "cutoffs": int(result.cutoffs), "simulations": int(result.simulations),
                "timed_out": bool(result.timed_out), "stats": stats,
            }
            rows.append(row)
            measured[(model, label)] = row
            print(json.dumps({
                "event": "preflight_move", "model": model, "budget": label,
                "seconds": row["elapsed_seconds"], "nodes": row["nodes"],
                "depth": row["completed_depth"], "simulations": row["simulations"],
                "timeout": row["timed_out"],
            }), flush=True)

    source_path = ROOT / str(config["reuse_source"]["results_dir"]) / "9x9_games.jsonl"
    historical_records = load_game_records(source_path)
    source_record_bytes = source_path.stat().st_size
    average_source_record_bytes = source_record_bytes / max(1, len(historical_records))
    estimated_v6_record_bytes = average_source_record_bytes * 4.0
    estimated_raw_results_bytes = estimated_v6_record_bytes * int(config["maximum_games"])
    estimated_schedule_bytes = len(json.dumps(schedule, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    historic_plies = sum(int(item["move_count"]) for item in reused) / max(1, len(reused))
    estimate = _estimate(schedule, state, int(manifest["v6_reused_groups"]), measured, historic_plies)
    target = ROOT / args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "benchmark_position_id": representative["position_id"],
        "benchmark_position_fixture_sha256": representative["fixture_sha256"],
        "seed": 6042026,
        "environment": _resources(),
        "benchmark_wall_seconds": time.perf_counter() - started,
        "benchmark_rows": rows,
        "reuse_manifest_counts": {
            "source_records": manifest["source_record_count"],
            "full_groups": manifest["source_full_groups"],
            "reused_groups": manifest["v6_reused_groups"],
            "reused_games": manifest["v6_reused_games"],
            "partial_games_excluded": manifest["source_partial_games_excluded"],
        },
        "historical_v5_1_game_count": len(historical_records),
        "historical_v6_reused_average_game_plies": historic_plies,
        "storage_estimate": {
            "v5_1_raw_bytes": source_record_bytes,
            "v5_1_average_record_bytes": average_source_record_bytes,
            "v6_record_size_multiplier": 4.0,
            "v6_conservative_average_record_bytes": estimated_v6_record_bytes,
            "maximum_v6_games": int(config["maximum_games"]),
            "maximum_raw_game_records_bytes": estimated_raw_results_bytes,
            "full_schedule_json_bytes": estimated_schedule_bytes,
            "estimated_uncompressed_outputs_bytes": estimated_raw_results_bytes + estimated_schedule_bytes + 20_000_000,
            "estimated_uncompressed_outputs_mib": (estimated_raw_results_bytes + estimated_schedule_bytes + 20_000_000) / (1024 * 1024),
            "assumption": "four times the measured V5.1 average raw game record size to allow for V6 per-move metrics and verified tactical certificates, plus schedule and 20 MB for config/progress/report/CSV/plots",
        },
        "estimate": estimate,
    }
    json_path = target / "preflight.json"
    csv_path = target / "benchmark_moves.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["model", "budget_label", "budget_seconds", "position_id", "move", "elapsed_seconds", "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations", "timed_out"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            writer.writerow({key: json.dumps(item.get(key), ensure_ascii=False) if isinstance(item.get(key), list) else item.get(key) for key in fields})
    print(json.dumps({
        "event": "preflight_complete",
        "resource": payload["environment"],
        "benchmark_wall_seconds": payload["benchmark_wall_seconds"],
        "reused_groups": manifest["v6_reused_groups"],
        "reused_games": manifest["v6_reused_games"],
        "excluded_partial_games": manifest["source_partial_games_excluded"],
        "estimate": estimate,
        "json": str(json_path.relative_to(ROOT)),
        "csv": str(csv_path.relative_to(ROOT)),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
