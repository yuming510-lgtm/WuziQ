"""Run reproducible move-budget probes and a small 9x9/15x15 pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import SearchConfig, choose_move
from src.gomoku import GomokuBoard
from src.research.efficiency import generate_position_suite, run_fixed_depth_comparison
from src.research.runner import run_tournament


def _mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _cpu_affinity_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def run_preflight(config_path: Path, output_dir: Path, pilot_games: int = 12) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = SearchConfig(**config["search"])
    output_dir.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(output_dir)
    environment_before = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_affinity_count": _cpu_affinity_count(),
        "memory_available_bytes": _mem_available_bytes(),
        "disk_available_bytes": disk_before.free,
    }

    probes = []
    for size in (9, 15):
        positions = generate_position_suite(
            count=3,
            board_size=size,
            seed=int(config["base_seed"]) + size,
        )
        for position in positions:
            board = GomokuBoard.from_dict(position["board"])
            for model in config["models"]:
                result = choose_move(
                    board,
                    position["to_move"],
                    model,
                    settings,
                    seed=int(position["seed"]) + ord(model),
                )
                probes.append(
                    {
                        "board_size": size,
                        "position_id": position["position_id"],
                        "position_seed": position["seed"],
                        "stone_count": len(board.history),
                        "to_move": position["to_move"],
                        "model": model,
                        "move": list(result.move),
                        "elapsed_seconds": result.elapsed_seconds,
                        "nodes": result.nodes,
                        "completed_depth": result.completed_depth,
                        "cutoffs": result.cutoffs,
                        "simulations": result.simulations,
                        "timed_out": result.timed_out,
                    }
                )
    (output_dir / "move_budget_microbenchmark.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pilot_runs = []
    for size in (9, 15):
        phase_dir = output_dir / f"{size}x{size}"
        started = time.perf_counter()
        games = run_tournament(
            config,
            phase_dir,
            max_games=pilot_games,
            board_sizes=[size],
        )
        wall_seconds = time.perf_counter() - started
        raw_path = phase_dir / f"{size}x{size}_games.jsonl"
        byte_count = raw_path.stat().st_size if raw_path.exists() else 0
        pilot_runs.append(
            {
                "board_size": size,
                "new_games": len(games),
                "wall_seconds": wall_seconds,
                "average_seconds_per_game": wall_seconds / len(games) if games else None,
                "jsonl_bytes": byte_count,
                "average_bytes_per_game": byte_count / len(games) if games else None,
                "average_moves_per_game": (
                    sum(record["move_count"] for record in games) / len(games) if games else None
                ),
                "timeout_count": sum(record["timeout_count"] for record in games),
            }
        )

    fixed_positions = generate_position_suite(
        count=8,
        board_size=9,
        seed=int(config["base_seed"]) + 91,
    )
    fixed_started = time.perf_counter()
    fixed_depth = run_fixed_depth_comparison(
        fixed_positions,
        depth=2,
        time_limit_seconds=60.0,
        seed=int(config["base_seed"]),
    )
    fixed_wall_seconds = time.perf_counter() - fixed_started

    phase_specs = {int(phase["board_size"]): phase for phase in config["phases"]}
    game_estimate = 0.0
    storage_estimate = 0.0
    for run in pilot_runs:
        phase = phase_specs[run["board_size"]]
        per_game = run["average_seconds_per_game"] or 0.0
        game_estimate += per_game * 6 * int(phase["games_per_matchup"])
        storage_estimate += (run["average_bytes_per_game"] or 0.0) * 6 * int(
            phase["games_per_matchup"]
        )
    estimated_serial_seconds = game_estimate + fixed_wall_seconds
    estimated_with_50_percent_buffer = estimated_serial_seconds * 1.5
    estimated_storage_bytes = int(storage_estimate * 1.25 + 2_000_000)
    disk = shutil.disk_usage(output_dir)
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config": config,
        "environment": {
            **environment_before,
            "memory_available_bytes_after_pilot": _mem_available_bytes(),
            "process_max_rss_kib": int(max_rss),
            "disk_available_bytes_after_pilot": disk.free,
            "move_budget_seconds": settings.time_limit_seconds,
        },
        "microbenchmark": {
            "probe_positions": 6,
            "model_decisions": len(probes),
            "records": probes,
        },
        "pilot_runs": pilot_runs,
        "fixed_depth_pilot": {
            "positions": len(fixed_depth),
            "depth": 2,
            "wall_seconds": fixed_wall_seconds,
            "average_nodes_minimax": sum(x["minimax"]["nodes"] for x in fixed_depth)
            / len(fixed_depth),
            "average_nodes_alpha_beta": sum(x["alpha_beta"]["nodes"] for x in fixed_depth)
            / len(fixed_depth),
            "average_prune_ratio": sum(x["prune_ratio"] for x in fixed_depth)
            / len(fixed_depth),
        },
        "estimate": {
            "total_games": sum(
                6 * int(phase["games_per_matchup"]) for phase in config["phases"]
            ),
            "serial_seconds_without_buffer": estimated_serial_seconds,
            "serial_hours_without_buffer": estimated_serial_seconds / 3600,
            "serial_hours_with_50_percent_buffer": estimated_with_50_percent_buffer / 3600,
            "estimated_research_storage_bytes": estimated_storage_bytes,
            "storage_basis": "pilot JSONL bytes/game × scheduled games, 25% data margin, 2 MB for summaries/config/report",
            "time_basis": "measured wall seconds/game per board size × six matchups × configured games per matchup, plus fixed-depth study; 50% execution margin",
        },
    }
    (output_dir / "preflight_estimate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark and estimate the WuziQ AI study.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/four_ai_experiment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pilot")
    parser.add_argument("--pilot-games", type=int, default=12)
    args = parser.parse_args(argv)
    report = run_preflight(args.config, args.output, args.pilot_games)
    print(json.dumps(report["estimate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
