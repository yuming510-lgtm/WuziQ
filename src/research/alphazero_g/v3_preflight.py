"""Measured resource preflight and transparent G V3 runtime estimates."""
from __future__ import annotations

import json
import os
import resource
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...gomoku import GomokuBoard
from .evaluation import play_evaluation_match
from .replay import ReplayBuffer
from .selfplay import play_selfplay_game
from .training import GTrainer, initialize_arm
from .v3_schedule import build_v3_schedule


def _resource_snapshot(root: Path) -> dict[str, Any]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(value.strip().split()[0]) * 1024
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "logical_cpus": os.cpu_count() or 1,
        "affinity_cpus": affinity,
        "memory_bytes": memory,
        "max_rss_bytes": int(usage.ru_maxrss * (1024 if os.name == "posix" else 1)),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _positions() -> dict[str, tuple[GomokuBoard, str]]:
    definitions = {
        "opening": [("B", (4, 4)), ("W", (3, 4)), ("B", (3, 5)), ("W", (5, 3))],
        "middle": [
            ("B", (4, 4)), ("W", (3, 4)), ("B", (4, 5)), ("W", (5, 4)),
            ("B", (3, 5)), ("W", (5, 5)), ("B", (3, 3)), ("W", (5, 3)),
            ("B", (2, 4)), ("W", (6, 4)), ("B", (4, 3)), ("W", (4, 6)),
        ],
        "tactical": [
            ("B", (4, 1)), ("W", (0, 0)), ("B", (4, 2)), ("W", (0, 1)),
            ("B", (4, 3)), ("W", (0, 2)), ("B", (4, 4)), ("W", (1, 0)),
        ],
    }
    result: dict[str, tuple[GomokuBoard, str]] = {}
    for name, moves in definitions.items():
        board = GomokuBoard(9)
        for player, coordinate in moves:
            board.place_stone(player, coordinate)
        result[name] = (board, "B")
    return result


def estimate_v3_runtime(
    *,
    states_per_cpu_second: float,
    update_cpu_seconds: float,
    evaluation_cpu_seconds_per_game: float,
    states_per_seed: int,
    seed_count: int,
    route_count: int,
    total_evaluation_games: int,
) -> dict[str, float | int]:
    """Estimate CPU seconds from measured state/update/game rates.

    The central estimate charges every route and seed for its full state quota;
    updates are budgeted at one per 16 states, matching the V3 registration.
    Low/high scenarios are explicit multipliers, not confidence intervals.
    """
    positive_rates = (states_per_cpu_second, update_cpu_seconds, evaluation_cpu_seconds_per_game)
    if any(not np.isfinite(value) or value <= 0 for value in positive_rates):
        raise ValueError("throughput and measured CPU costs must be finite and positive")
    counts = (states_per_seed, seed_count, route_count, total_evaluation_games)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
        raise ValueError("runtime estimate counts must be nonnegative integers")
    if states_per_seed < 1 or seed_count < 1 or route_count < 1:
        raise ValueError("states, seed count, and route count must be positive")
    total_states = int(states_per_seed) * int(seed_count) * int(route_count)
    total_updates = total_states / 16.0
    selfplay = total_states / float(states_per_cpu_second)
    optimizer = total_updates * float(update_cpu_seconds)
    evaluation = int(total_evaluation_games) * float(evaluation_cpu_seconds_per_game)
    central = selfplay + optimizer + evaluation
    return {
        "total_budgeted_states": total_states,
        "estimated_optimizer_updates": total_updates,
        "selfplay_cpu_seconds": selfplay,
        "optimizer_cpu_seconds": optimizer,
        "evaluation_cpu_seconds": evaluation,
        "central_cpu_seconds": central,
        "optimistic_cpu_seconds": central * 0.70,
        "conservative_cpu_seconds": central * 1.75,
    }


def run_v3_preflight(root: str | Path, *, seed: int) -> dict[str, Any]:
    """Exercise real PUCT-64 games, optimizer updates, and both inference modes."""
    root_path = Path(root).resolve()
    torch.set_num_threads(1)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    cpu_started = time.process_time()
    resources = _resource_snapshot(root_path)

    schedule = build_v3_schedule(
        [int(seed)], train_groups_per_seed=4, development_groups=2, blind_groups=2,
        excluded_openings=[],
    )
    random_start = next(group for group in schedule["train_openings_by_seed"][str(seed)] if group["kind"] == "random4")
    dev_opening = schedule["development_openings"][0]
    from .puct import PUCTSearch

    simulation_benchmarks: list[dict[str, Any]] = []
    benchmark_model = initialize_arm("G-F2", seed=int(seed), device=resources["device"])
    for position_id, (board, to_move) in _positions().items():
        PUCTSearch(benchmark_model).search(board, to_move, 1, seed=seed)
        for simulations in (16, 32, 64):
            samples: list[float] = []
            search_result = None
            for repeat in range(2):
                search_result = PUCTSearch(benchmark_model).search(
                    board, to_move, simulations, add_root_noise=False, temperature=0.0,
                    seed=int(seed) + 1000 * simulations + repeat,
                )
                samples.append(float(search_result.elapsed_seconds))
            assert search_result is not None
            simulation_benchmarks.append({
                "position_id": position_id,
                "simulations": simulations,
                "repeats": len(samples),
                "elapsed_cpu_seconds": samples,
                "mean_cpu_seconds": float(statistics.mean(samples)),
                "simulations_per_cpu_second": simulations / max(float(statistics.mean(samples)), 1e-9),
                "network_evaluations": int(search_result.network_evaluations),
                "tree_nodes": int(search_result.tree_nodes),
                "max_depth": int(search_result.max_depth),
            })

    selfplay_games: list[dict[str, Any]] = []
    training_updates: list[dict[str, Any]] = []
    evaluation_games: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(("G0", "G-F2")):
        arm_seed = int(seed) + arm_index * 37
        model = initialize_arm(arm, seed=arm_seed, device=resources["device"])
        game_id = f"g-v3-preflight-{arm}-seed-{int(seed)}"
        game = play_selfplay_game(
            model, random_start["opening"], arm_id=arm, game_id=game_id,
            opening_group_id=random_start["opening_group_id"], simulations=64,
            seed=arm_seed + 10_000, device=resources["device"],
        )
        rows = len(game["experience_rows"])
        selfplay_games.append({
            "arm_id": arm,
            "game_id": game_id,
            "completed": game["completed"],
            "sample_positions": rows,
            "wall_seconds": float(game["stats"]["wall_seconds"]),
            "cpu_seconds": float(game["stats"]["cpu_seconds"]),
            "simulations": int(game["stats"]["simulations"]),
            "moves": int(game["move_count"]),
            "winner_color": game["winner_color"],
        })
        replay = ReplayBuffer(max(128, rows + 1), seed=arm_seed)
        replay.add_complete_game(game)
        trainer = GTrainer(
            initialize_arm(arm, seed=arm_seed, device=resources["device"]),
            seed=arm_seed + 20_000, learning_rate=1e-4, expected_updates=20,
            device=resources["device"],
        )
        update = trainer.train_updates(replay, updates=20, batch_size=16)
        training_updates.append({
            "arm_id": arm,
            "completed": True,
            "updates": 20,
            "batch_size": 16,
            "wall_seconds": float(update["wall_seconds"]),
            "cpu_seconds": float(update["cpu_seconds"]),
            "updates_per_cpu_second": 20 / max(float(update["cpu_seconds"]), 1e-9),
            "selfplay_rows": rows,
        })
        for mode in ("network", "puct"):
            eval_id = f"g-v3-preflight-{arm}-{mode}-A-seed-{int(seed)}"
            result = play_evaluation_match(
                model,
                game_id=eval_id,
                opening_group_id=dev_opening["opening_group_id"],
                arm_id=arm,
                mode=mode,
                opponent="A",
                opening=dev_opening["opening"],
                agent_color="B" if mode == "network" else "W",
                seed=arm_seed + 30_000 + (0 if mode == "network" else 1),
                simulations=64,
                device=resources["device"],
            )
            evaluation_games.append({
                "arm_id": arm,
                "mode": mode,
                "completed": result["completed"],
                "wall_seconds": float(result["stats"]["total_wall_seconds"]),
                "cpu_seconds": float(result["stats"]["total_cpu_seconds"]),
                "simulations": int(result["stats"]["agent_simulations"]),
                "winner_color": result["winner_color"],
            })

    state_rates = [row["sample_positions"] / max(row["cpu_seconds"], 1e-9) for row in selfplay_games]
    update_costs = [row["cpu_seconds"] / row["updates"] for row in training_updates]
    evaluation_costs = [row["cpu_seconds"] for row in evaluation_games]
    estimate = estimate_v3_runtime(
        states_per_cpu_second=float(statistics.median(state_rates)),
        update_cpu_seconds=float(statistics.median(update_costs)),
        evaluation_cpu_seconds_per_game=float(statistics.median(evaluation_costs)),
        states_per_seed=10_000,
        seed_count=2,
        route_count=2,
        total_evaluation_games=3_520,
    )
    return {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v3-preflight",
        "seed": int(seed),
        "resources": resources,
        "simulation_benchmarks": simulation_benchmarks,
        "selfplay_games": selfplay_games,
        "training_updates": training_updates,
        "evaluation_games": evaluation_games,
        "measured_state_rate_median": float(statistics.median(state_rates)),
        "measured_update_cpu_seconds_median": float(statistics.median(update_costs)),
        "measured_evaluation_cpu_seconds_median": float(statistics.median(evaluation_costs)),
        "runtime_estimate_10k_per_seed_each_route": estimate,
        "evaluation_games_estimate_basis": {
            "development": "4 groups x 2 colors x 5 opponents x 2 modes x 6 milestones x 4 arm-seed combinations = 1920",
            "blind": "20 groups x 2 colors x 5 opponents x 2 modes x 4 arm-seed combinations = 1600",
            "total": 3520,
            "note": "Exploratory CPU estimate; later registered schedule may reduce optional opponents or milestones within the finite cap.",
        },
        "measurements_wall_seconds": time.perf_counter() - started,
        "measurements_cpu_seconds": time.process_time() - cpu_started,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
