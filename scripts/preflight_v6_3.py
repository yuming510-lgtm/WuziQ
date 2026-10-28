#!/usr/bin/env python3
"""Short C/E5 runtime check and data-based V6.3 duration estimate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from src.research.abcd_ladder_v5 import settings_for_model
from src.research.v6.config import v5_1_engine_settings
from src.research.v6.engine import V6Agent
from src.research.v6.config import load_config as load_v6_config
from src.research.v6_3.config import load_v6_3_config
from src.research.v6_3.schedule import derive_seed


def _resource_snapshot() -> dict[str, int]:
    memory = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory = int(line.split()[1]) * 1024
            break
    usage = shutil.disk_usage(ROOT)
    return {
        "logical_cpus": os.cpu_count() or 0,
        "memory_total_bytes": memory,
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output", default="research/v6_3/preflight/preflight.json")
    args = parser.parse_args()
    config = load_v6_3_config()
    v6_config = load_v6_config()
    engine_config = v5_1_engine_settings(v6_config)
    opening = config["openings"][config["initial_opening_ids"][0]]
    measurements = []
    started_all = time.perf_counter()
    for model in ("C", "E5"):
        board = GomokuBoard(9)
        for move in opening:
            board.place_stone(move["player"], (int(move["row"]), int(move["col"])))
        seed = derive_seed(int(config["base_seed"]), "preflight", model)
        started = time.perf_counter()
        if model == "C":
            settings = settings_for_model(engine_config, model, len(board.history), float(args.seconds))
            result = choose_move(board, "B", model, settings, seed)
        else:
            result = V6Agent("E5").choose_move(
                board, "B", time_limit_seconds=float(args.seconds), seed=seed,
                max_depth=int(v6_config["tactical_limits"]["e5_max_depth"]),
            )
        elapsed = time.perf_counter() - started
        legal = isinstance(result.move, (tuple, list)) and len(result.move) == 2 and board.in_bounds(*result.move) and board.get(tuple(result.move)) is None
        if not legal:
            raise RuntimeError(f"{model} preflight returned an illegal move: {result.move}")
        measurements.append({
            "model": model, "budget_seconds": float(args.seconds),
            "elapsed_seconds": elapsed, "move": list(result.move),
            "legal": legal, "timed_out": bool(result.timed_out),
            "completed_depth": int(result.completed_depth), "nodes": int(result.nodes),
            "cutoffs": int(result.cutoffs), "simulations": int(result.simulations),
        })

    old_path = ROOT / "research/v6_2/results/9x9_games.jsonl"
    old_games = [json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    direct = [row for row in old_games if row.get("pair") == ["C", "E5"]]
    if len(direct) != 2:
        raise ValueError("Expected two V6.2 direct C–E5 games for duration estimation")
    direct_seconds = [float(game["thinking_seconds_total"]) for game in direct]
    historical_min, historical_mean = min(direct_seconds), sum(direct_seconds) / len(direct_seconds)
    # The original four-ply opening leaves 77 model decisions if a 9x9 game fills.
    worst_four = 4 * 77 * float(config["time_limit_seconds"])
    worst_six = 6 * 77 * float(config["time_limit_seconds"])
    elapsed_preflight = time.perf_counter() - started_all
    estimate = {
        "optimistic": {"four_games_seconds": 4 * historical_min, "six_games_seconds": 6 * historical_min},
        "central": {"four_games_seconds": 4 * historical_mean, "six_games_seconds": 6 * historical_mean},
        "conservative": {"four_games_seconds": worst_four, "six_games_seconds": worst_six},
        "basis": {
            "v6_2_direct_games_thinking_seconds": direct_seconds,
            "v6_2_direct_games_moves": [int(game["move_count"]) for game in direct],
            "per_new_game_optimistic": historical_min,
            "per_new_game_central": historical_mean,
            "max_model_decisions_per_full_game": 77,
            "per_move_budget_seconds": config["time_limit_seconds"],
            "four_and_six_game_decision": "2 V6.3 openings (4 games), plus 1 additional paired opening only if the first four scores tie.",
        },
    }
    payload = {
        "experiment_id": config["experiment_id"],
        "preflight_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preflight_budget_seconds_per_model": float(args.seconds),
        "preflight_elapsed_seconds": elapsed_preflight,
        "measurements": measurements,
        "resource_snapshot": _resource_snapshot(),
        "drive_precheck": {
            "folder_id": config["drive_folder_id"],
            "upload_download_sha_restore_delete": "passed via independent probe; archive-level restore will be verified before games",
        },
        "historical_v6_2_200s_reference": {
            "direct_games": len(direct), "both_models_budget_seconds": 200.0,
            "C_E5_points": {"C": 1.0, "E5": 1.0},
        },
        "duration_estimates": estimate,
        "official_games_excluded_from_preflight": True,
    }
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
