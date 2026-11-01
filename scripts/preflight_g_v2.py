#!/usr/bin/env python3
"""Measure current-worker throughput for the frozen G V2 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v2_preflight import _atomic_json, run_v2_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--schedule", type=Path, default=Path("research/g_v2/config/openings.json"))
    parser.add_argument("--output", type=Path, default=Path("research/g_v2/results/preflight/preflight.json"))
    parser.add_argument("--seed", type=int, default=510702)
    args = parser.parse_args()
    root = args.root.resolve()
    schedule_path = args.schedule if args.schedule.is_absolute() else root / args.schedule
    output_path = args.output if args.output.is_absolute() else root / args.output
    schedule = json.loads(schedule_path.read_text())
    result = run_v2_preflight(root, schedule, args.seed)
    _atomic_json(output_path, result)
    print(json.dumps({
        "output": str(output_path),
        "wall_seconds": result["measurements_wall_seconds"],
        "cpu_seconds": result["measurements_cpu_seconds"],
        "resources": result["resources"],
        "simulation_mean_seconds": {
            str(sims): sum(row["mean_seconds"] for row in result["simulation_benchmarks"] if row["simulations"] == sims) / 3
            for sims in (16, 32, 64)
        },
        "selfplay_states_per_second": {
            row["lr_id"]: row["budgeted_states"] / row["cpu_seconds"] for row in result["selfplay_games"]
        },
        "training_updates_per_second": {
            row["lr_id"]: row["updates_per_second"] for row in result["training_updates"]
        },
        "evaluation_games": len(result["evaluation_games"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
