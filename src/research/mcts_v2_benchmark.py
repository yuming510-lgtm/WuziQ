"""Resource, wall-clock, and fixed-simulation benchmarks for MCTS V2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable

from ..ai import SearchConfig, choose_move
from ..gomoku import GomokuBoard, Player
from .mcts_v2 import derive_seed, paired_opening


def resource_snapshot(path: str | Path = ".") -> dict[str, Any]:
    """Capture non-secret CPU, memory, and disk capacity for the report."""

    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            amount = value.strip().split()[0]
            memory[key] = int(amount) * 1024
    except (OSError, ValueError, IndexError):
        memory = {}
    disk = shutil.disk_usage(path)
    return {
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": memory.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
    }


def _board_copy(board: GomokuBoard) -> GomokuBoard:
    cloned = GomokuBoard(board.size)
    for player, coord in board.history:
        cloned.place_stone(player, coord)
    return cloned


def representative_positions(seed: int = 5102026) -> list[dict[str, Any]]:
    """Return fixed opening, middle-game, tactical and sparse-board states."""

    positions: list[dict[str, Any]] = []
    for size in (9, 15):
        opening = GomokuBoard(size)
        sequence = paired_opening(size, derive_seed(seed, "opening", size))
        for player, coord in sequence:
            opening.place_stone(player, coord)
        positions.append(
            {
                "position_id": f"opening-{size}x{size}",
                "board": opening,
                "player": _other(sequence[-1][0]),
            }
        )

        middle = GomokuBoard(size)
        rng = random.Random(derive_seed(seed, "middle", size))
        player: Player = "B"
        target_stones = 16 if size == 9 else 24
        while len(middle.history) < target_stones:
            choices = [
                (row, col)
                for row in range(size)
                for col in range(size)
                if middle.get((row, col)) is None
            ]
            coord = rng.choice(choices)
            middle.place_stone(player, coord)
            if middle.check_victory(coord, player):
                middle.undo()
                continue
            player = _other(player)
        positions.append(
            {"position_id": f"midgame-{size}x{size}", "board": middle, "player": player}
        )

        for name, threat_player in (("forced-win", "B"), ("forced-block", "W")):
            tactical = GomokuBoard(size)
            row = size // 2
            start = max(0, size // 2 - 2)
            for col in range(start, start + 4):
                tactical.place_stone(threat_player, (row, col))
            positions.append(
                {
                    "position_id": f"{name}-{size}x{size}",
                    "board": tactical,
                    "player": "B",
                    "expected_move": (row, start + 4),
                }
            )
    return positions


def benchmark_position(
    board: GomokuBoard,
    player: Player,
    models: Iterable[str] = ("D", "D1", "D2"),
    target_simulations: Iterable[int] = (100, 1_000, 10_000, 100_000),
    max_seconds: float = 30.0,
    base_seed: int = 5102026,
) -> list[dict[str, Any]]:
    """Measure actual completed simulations at fixed configured caps."""

    records: list[dict[str, Any]] = []
    for model in models:
        for target in target_simulations:
            started = time.perf_counter()
            result = choose_move(
                _board_copy(board),
                player,
                model,
                SearchConfig(
                    time_limit_seconds=max_seconds,
                    mcts_simulations=int(target),
                ),
                seed=derive_seed(base_seed, "simulations", model, target),
            )
            records.append(
                {
                    "model": model,
                    "requested_simulations": int(target),
                    "actual_simulations": int(result.simulations),
                    "nodes": int(result.nodes),
                    "tree_depth": int(result.completed_depth),
                    "rollout_plies": int(result.rollout_plies),
                    "terminal_simulations": int(result.terminal_simulations),
                    "truncated_simulations": int(result.truncated_simulations),
                    "elapsed_seconds": float(result.elapsed_seconds),
                    "wall_seconds": time.perf_counter() - started,
                    "timed_out": bool(result.timed_out),
                    "move": list(result.move),
                    "max_seconds": max_seconds,
                    "seed": derive_seed(base_seed, "simulations", model, target),
                }
            )
            if result.simulations < target and result.timed_out:
                break
    return records


def benchmark_budgets(
    board: GomokuBoard,
    player: Player,
    budgets_seconds: Iterable[float],
    models: Iterable[str] = ("D", "D1", "D2"),
    simulation_cap: int = 1_000_000,
    base_seed: int = 5102026,
    position_id: str = "position",
) -> list[dict[str, Any]]:
    """Compare completed work at equal per-move wall-clock budgets."""

    records: list[dict[str, Any]] = []
    for budget in budgets_seconds:
        for model in models:
            seed = derive_seed(base_seed, "budget", position_id, budget, model)
            result = choose_move(
                _board_copy(board),
                player,
                model,
                SearchConfig(
                    time_limit_seconds=float(budget),
                    mcts_simulations=simulation_cap,
                ),
                seed=seed,
            )
            records.append(
                {
                    "position_id": position_id,
                    "model": model,
                    "budget_seconds": float(budget),
                    "actual_simulations": int(result.simulations),
                    "nodes": int(result.nodes),
                    "tree_depth": int(result.completed_depth),
                    "rollout_plies": int(result.rollout_plies),
                    "terminal_simulations": int(result.terminal_simulations),
                    "truncated_simulations": int(result.truncated_simulations),
                    "elapsed_seconds": float(result.elapsed_seconds),
                    "timed_out": bool(result.timed_out),
                    "move": list(result.move),
                    "seed": seed,
                }
            )
    return records


def run_benchmarks(config: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Run the pre-registered position and simulation-throughput benchmarks."""

    resources_before = resource_snapshot(Path(output_path).parent)
    positions = representative_positions(int(config["base_seed"]))
    budget_rows: list[dict[str, Any]] = []
    budgets = config.get("benchmark_budgets_seconds", [0.05, 0.2, 1, 5, 10])
    for position in positions:
        # Tactical states verify response at 50ms; larger budgets use non-tactical
        # states to avoid returning early without exercising the search.
        selected_budgets = [0.05] if "forced-" in position["position_id"] else budgets
        budget_rows.extend(
            {
                **row,
                "board_size": position["board"].size,
                "position_id": position["position_id"],
            }
            for row in benchmark_budgets(
                position["board"],
                position["player"],
                selected_budgets,
                base_seed=int(config["base_seed"]),
                position_id=position["position_id"],
            )
        )

    sim_position = next(
        position for position in positions if position["position_id"] == "opening-9x9"
    )
    simulation_rows = benchmark_position(
        sim_position["board"],
        sim_position["player"],
        models=("D1", "D2"),
        target_simulations=config.get(
            "simulation_targets", [100, 1_000, 10_000, 100_000]
        ),
        max_seconds=float(config.get("max_seconds_per_simulation_target", 30)),
        base_seed=int(config["base_seed"]),
    )
    payload = {
        "resources_before": resources_before,
        "resources_after": resource_snapshot(Path(output_path).parent),
        "benchmark_budgets_seconds": list(budgets),
        "budget_results": budget_rows,
        "simulation_results": simulation_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mcts_potential_v2.json")
    parser.add_argument(
        "--output", default="research/v2/pilot/performance_benchmarks.json"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_benchmarks(config, args.output)
    print(
        json.dumps(
            {
                "resources_before": result["resources_before"],
                "budget_rows": len(result["budget_results"]),
                "simulation_rows": len(result["simulation_results"]),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
