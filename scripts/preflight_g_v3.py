#!/usr/bin/env python3
"""Measure G V3 self-play, learning, evaluation, and machine throughput."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_preflight import _atomic_json, run_v3_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=510901)
    parser.add_argument("--output", type=Path, default=Path("research/g_v3/results/preflight/preflight.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = run_v3_preflight(root, seed=args.seed)
    _atomic_json(output, result)
    print(json.dumps({
        "output": str(output),
        "wall_seconds": result["measurements_wall_seconds"],
        "cpu_seconds": result["measurements_cpu_seconds"],
        "resources": result["resources"],
        "simulation_seconds_per_search": {
            str(simulations): sum(row["mean_cpu_seconds"] for row in result["simulation_benchmarks"]
                                  if row["simulations"] == simulations) / 3
            for simulations in (16, 32, 64)
        },
        "selfplay_states_per_cpu_second": result["measured_state_rate_median"],
        "update_cpu_seconds": result["measured_update_cpu_seconds_median"],
        "evaluation_cpu_seconds_per_game": result["measured_evaluation_cpu_seconds_median"],
        "estimate_10k_each_route": result["runtime_estimate_10k_per_seed_each_route"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
