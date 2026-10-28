#!/usr/bin/env python3
"""Run one real 200-second decision per frozen model and estimate task cost."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import choose_move
from src.gomoku import GomokuBoard
from src.research.abcd_ladder_v5 import settings_for_model
from src.research.checkpoint import write_json_atomic
from src.research.mcts_d2 import MCTSAgent
from src.research.v6.config import load_config as load_v6_config, v5_1_engine_settings
from src.research.v6.engine import V6Agent
from src.research.v6.puzzles import board_for_position, load_tactical_positions
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.schedule import derive_seed


def _resources() -> dict[str, int | None]:
    usage = shutil.disk_usage(ROOT)
    result: dict[str, int | None] = {
        "cpu_logical": os.cpu_count(),
        "cpu_physical": None,
        "memory_total_bytes": None,
        "memory_available_bytes": None,
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
    }
    try:
        import psutil
        result.update({
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_total_bytes": psutil.virtual_memory().total,
            "memory_available_bytes": psutil.virtual_memory().available,
        })
    except ImportError:
        pass
    return result


def _decision(model: str, board: GomokuBoard, player: str, budget: float, seed: int, engine: dict, v6: dict):
    if model == "D":
        return MCTSAgent().choose_move(board, player, settings_for_model(engine, model, len(board.history), budget), seed)
    if model == "C":
        settings = settings_for_model(engine, model, len(board.history), budget)
        return choose_move(board, player, "C", settings, seed)
    if model == "E5":
        return V6Agent(
            "E5",
            max_attack_depth=int(v6["tactical_limits"]["max_attack_depth"]),
            max_nodes=int(v6["tactical_limits"]["max_nodes"]),
        ).choose_move(
            board,
            player,
            time_limit_seconds=budget,
            seed=seed,
            max_depth=int(v6["tactical_limits"]["e5_max_depth"]),
        )
    raise ValueError(f"Unsupported V6.2 preflight model: {model}")


def _historical_game_data() -> tuple[list[dict], float, dict[str, float]]:
    path = ROOT / "research/v6/results/9x9_games.jsonl"
    games = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_plies = [int(game["move_count"]) for game in games]
    pair_plies: dict[str, list[int]] = {"C-D": [], "C-E5": [], "D-E5": []}
    for game in games:
        pair = tuple(game.get("pair", ()))
        if len(pair) == 2 and set(pair) <= {"C", "D", "E5"}:
            pair_plies["-".join(pair)].append(int(game["move_count"]))
    averages = {
        pair: statistics.mean(values) if values else statistics.mean(all_plies)
        for pair, values in pair_plies.items()
    }
    return games, statistics.mean(all_plies), averages


def _estimates(config: dict, rows: list[dict], overall_plies: float, pair_plies: dict[str, float]) -> dict:
    benchmark = {str(row["model"]): float(row["elapsed_seconds"]) for row in rows}
    opening_plies = len(config["opening"])
    cap = float(config["time_limit_seconds"])
    pairs = config["pairs"]

    def scenario(plies_by_pair: dict[str, float], cost_fraction: float) -> float:
        total = 0.0
        for pair in pairs:
            key = "-".join(pair)
            decision_pairs = max(1.0, float(plies_by_pair[key]) - opening_plies)
            for model in pair:
                # Across two games, each model takes approximately one half of the
                # decision plies, so the model-specific count is game_plies - opening.
                one_move_seconds = min(cap, max(cap * 0.25, benchmark[model] * cost_fraction))
                total += decision_pairs * one_move_seconds
        return total / 3600.0

    optimistic = scenario(pair_plies, 0.50)
    central = scenario({"-".join(pair): overall_plies for pair in pairs}, 0.75)
    conservative = (6 * (81 - opening_plies) * cap) / 3600.0
    return {
        "optimistic": round(optimistic, 2),
        "central": round(central, 2),
        "conservative": round(conservative, 2),
        "assumptions": {
            "optimistic": "Use V6's pair-specific mean game length; each model move consumes 50% of its measured 200-second preflight time, with a 50-second floor.",
            "central": "Use V6's overall mean game length; each model move consumes 75% of its measured 200-second preflight time, with a 50-second floor.",
            "conservative": "All six games reach 81 plies; four registered opening plies are free, and every remaining decision consumes the full 200-second limit.",
            "wall_clock": "Games and decisions run serially on one process; no parallel speedup is assumed.",
        },
        "historical_v6_average_plies_per_game": round(overall_plies, 2),
        "historical_v6_pair_average_plies": {key: round(value, 2) for key, value in pair_plies.items()},
        "theoretical_max_decision_calls": 6 * (81 - opening_plies),
        "theoretical_max_wall_hours": round(conservative, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6_2/preflight")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_v6_2_config()
    v6 = load_v6_config()
    engine = v5_1_engine_settings(v6)
    positions = {row["position_id"]: row for row in load_tactical_positions()}
    position = positions["vct_cross_three"]
    board = board_for_position(position)
    player = str(position["to_move"])
    before = (tuple(tuple(row) for row in board.grid), tuple(board.history))
    seed = derive_seed(int(config["base_seed"]), "preflight", position["position_id"])
    rows: list[dict] = []
    for model in config["models"]:
        model_board = board_for_position(position)
        move_started = time.perf_counter()
        result = _decision(
            model, model_board, player, float(config["time_limit_seconds"]), seed, engine, v6
        )
        wall_seconds = time.perf_counter() - move_started
        move = tuple(result.move)
        legal = model_board.in_bounds(*move) and model_board.get(move) is None
        if not legal:
            raise RuntimeError(f"V6.2 preflight {model} returned illegal move {move}")
        stats = dict(getattr(result, "stats", {}) or {})
        elapsed = float(result.elapsed_seconds)
        row = {
            "model": model,
            "position_id": position["position_id"],
            "position_fixture_sha256": position["fixture_sha256"],
            "to_move": player,
            "budget_seconds": float(config["time_limit_seconds"]),
            "move": list(move),
            "legal": legal,
            "elapsed_seconds": elapsed,
            "wall_seconds": wall_seconds,
            "nodes": int(result.nodes),
            "completed_depth": int(result.completed_depth),
            "tree_depth": int(stats.get("max_attack_depth", result.completed_depth)),
            "cutoffs": int(result.cutoffs),
            "simulations": int(result.simulations),
            "timed_out": bool(result.timed_out or elapsed > float(config["time_limit_seconds"])),
            "over_budget_seconds": max(0.0, elapsed - float(config["time_limit_seconds"])),
            "board_unchanged": (
                tuple(tuple(row) for row in model_board.grid), tuple(model_board.history)
            ) == before,
            "stats": stats,
        }
        if not row["board_unchanged"]:
            raise RuntimeError(f"V6.2 preflight {model} mutated the board")
        rows.append(row)
        print(json.dumps({
            "event": "v6_2_preflight_move", "model": model,
            "elapsed_seconds": elapsed, "depth": row["completed_depth"],
            "nodes": row["nodes"], "simulations": row["simulations"],
            "timed_out": row["timed_out"],
        }, ensure_ascii=False), flush=True)

    historical, overall_plies, pair_plies = _historical_game_data()
    estimate = _estimates(config, rows, overall_plies, pair_plies)
    disk = shutil.disk_usage(ROOT)
    sample_bytes = (ROOT / "research/v6/results/9x9_games.jsonl").stat().st_size
    average_record = sample_bytes / max(1, len(historical))
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "base_seed": config["base_seed"],
        "source_baseline_commit": config["baseline_commit"],
        "position_id": position["position_id"],
        "position_fixture_sha256": position["fixture_sha256"],
        "benchmark_wall_seconds": time.perf_counter() - started,
        "environment": _resources(),
        "benchmark_rows": rows,
        "historical_v6_game_count": len(historical),
        "storage_estimate": {
            "historical_v6_average_record_bytes": round(average_record),
            "six_game_raw_estimate_bytes": round(average_record * 6 * 4),
            "disk_free_bytes": disk.free,
            "estimated_total_artifact_bytes": 5_000_000,
        },
        "estimate": estimate,
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "preflight.json", payload)
    with (output / "benchmark_moves.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "position_id", "budget_seconds", "move", "legal", "elapsed_seconds", "wall_seconds", "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations", "timed_out", "over_budget_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "move": json.dumps(row["move"]), "legal": int(row["legal"]), "timed_out": int(row["timed_out"])})
    print(json.dumps({
        "event": "v6_2_preflight_complete",
        "wall_seconds": payload["benchmark_wall_seconds"],
        "environment": payload["environment"],
        "estimate_hours": estimate,
        "json": str((output / "preflight.json").relative_to(ROOT)),
        "csv": str((output / "benchmark_moves.csv").relative_to(ROOT)),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
