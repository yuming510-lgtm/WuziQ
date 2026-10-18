"""Complete selected fixed-simulation targets and persist after each target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import SearchConfig, choose_move
from src.research.checkpoint import write_json_atomic
from src.research.mcts_v2 import _current_code_commit, derive_seed
from src.research.mcts_v2_benchmark import representative_positions


JOBS = (
    ("D1", 1_000, 30.0),
    ("D2", 10_000, 120.0),
    ("D2", 100_000, 750.0),
)


def run(config_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    position = next(
        row
        for row in representative_positions(int(config["base_seed"]))
        if row["position_id"] == "opening-9x9"
    )
    rows: list[dict] = []
    payload = {
        "code_commit": _current_code_commit(),
        "position_id": position["position_id"],
        "board_size": 9,
        "seed_basis": int(config["base_seed"]),
        "jobs": [
            {"model": model, "requested_simulations": target, "max_seconds": limit}
            for model, target, limit in JOBS
        ],
        "results": rows,
    }
    write_json_atomic(output_path, payload)
    for model, target, limit in JOBS:
        seed = derive_seed(
            int(config["base_seed"]), "fixed-target", position["position_id"], model, target
        )
        started = time.perf_counter()
        result = choose_move(
            position["board"],
            position["player"],
            model,
            SearchConfig(time_limit_seconds=limit, mcts_simulations=target),
            seed,
        )
        rows.append(
            {
                "model": model,
                "requested_simulations": target,
                "actual_simulations": result.simulations,
                "nodes": result.nodes,
                "tree_depth": result.completed_depth,
                "rollout_plies": result.rollout_plies,
                "terminal_simulations": result.terminal_simulations,
                "truncated_simulations": result.truncated_simulations,
                "elapsed_seconds": result.elapsed_seconds,
                "wall_seconds": time.perf_counter() - started,
                "timed_out": result.timed_out,
                "move": list(result.move),
                "seed": seed,
                "max_seconds": limit,
            }
        )
        write_json_atomic(output_path, payload)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mcts_potential_v2.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/v2/pilot/fixed_targets_final.json"
    )
    args = parser.parse_args()
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
