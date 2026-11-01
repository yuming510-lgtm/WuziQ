#!/usr/bin/env python3
"""Measure real PUCT, complete self-play and optimizer rates on this worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import resource
import shutil
import time

import numpy as np
import torch

from src.gomoku import GomokuBoard
from src.research.alphazero_g.puct import PUCTSearch
from src.research.alphazero_g.experiment import build_g_schedule, validate_g_schedule
from src.research.alphazero_g.replay import ReplayBuffer, append_complete_game, load_game_log
from src.research.alphazero_g.selfplay import play_selfplay_game
from src.research.alphazero_g.training import GTrainer, initialize_arm


def _resource_snapshot(root: Path) -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            mem[key] = int(value.strip().split()[0]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "memory_bytes": mem,
        "max_rss_bytes": int(usage.ru_maxrss * (1024 if os.name == "posix" else 1)),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _positions() -> dict[str, tuple[GomokuBoard, str]]:
    opening_moves = [((4, 4), "B"), ((3, 4), "W"), ((3, 5), "B"), ((5, 3), "W")]
    opening = GomokuBoard(9)
    for coord, player in opening_moves:
        opening.place_stone(player, coord)
    middle = GomokuBoard(9)
    sequence = [
        (4, 4), (3, 4), (4, 5), (5, 4), (3, 5), (5, 5), (3, 3), (5, 3),
        (2, 4), (6, 4), (4, 3), (4, 6), (2, 5), (6, 3), (2, 3), (6, 5),
    ]
    for index, coord in enumerate(sequence):
        player = "B" if index % 2 == 0 else "W"
        middle.place_stone(player, coord)
    tactical = GomokuBoard(9)
    tactical_moves = [
        ("B", (4, 1)), ("W", (0, 0)), ("B", (4, 2)), ("W", (0, 1)),
        ("B", (4, 3)), ("W", (0, 2)), ("B", (4, 4)), ("W", (1, 0)),
    ]
    for player, coord in tactical_moves:
        tactical.place_stone(player, coord)
    return {"opening": (opening, "B"), "middle": (middle, "B"), "tactical": (tactical, "B")}


def _history(root: Path) -> tuple[set[str], list[list[dict]]]:
    excluded_ids: set[str] = set()
    excluded_openings: list[list[dict]] = []
    config_roots = [root / "research/f_v1/config", root / "research/f_v2/config", root / "research/f_v2/results/precorrection/config"]
    for config_root in config_roots:
        if not config_root.exists():
            continue
        for path in config_root.rglob("*.json"):
            try:
                content = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            def visit(value):
                if isinstance(value, dict):
                    group_id = value.get("opening_group_id")
                    if isinstance(group_id, str):
                        excluded_ids.add(group_id)
                    opening = value.get("opening")
                    if isinstance(opening, list) and len(opening) == 4 and all(
                        isinstance(move, dict) and {"player", "row", "col"} <= set(move)
                        for move in opening
                    ):
                        excluded_openings.append(opening)
                    for nested in value.values():
                        visit(nested)
                elif isinstance(value, list):
                    for nested in value:
                        visit(nested)

            visit(content)
    generation = root / "research/f_v1/config/generation.json"
    if generation.exists():
        data = json.loads(generation.read_text())
        excluded_ids.update(data.get("excluded_opening_group_ids", []))
    return excluded_ids, excluded_openings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--schedule", type=Path, default=Path("research/g_v1/config/openings.json"))
    parser.add_argument("--output", type=Path, default=Path("research/g_v1/results/preflight.json"))
    parser.add_argument("--games-log", type=Path, default=Path("research/g_v1/results/preflight_selfplay.jsonl"))
    parser.add_argument("--seed", type=int, default=510600)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--train-groups", type=int, default=8)
    parser.add_argument("--development-groups", type=int, default=2)
    parser.add_argument("--blind-groups", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    schedule_path = args.schedule if args.schedule.is_absolute() else root / args.schedule
    output_path = args.output if args.output.is_absolute() else root / args.output
    games_path = args.games_log if args.games_log.is_absolute() else root / args.games_log
    excluded_ids, excluded_openings = _history(root)
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text())
        validate_g_schedule(schedule, excluded_openings=excluded_openings)
        schedule_ids = {
            group["opening_group_id"]
            for section in ("train_openings", "development_openings", "blind_openings")
            for group in schedule[section]
        }
        if schedule_ids & excluded_ids:
            raise ValueError("saved G schedule reuses a historical opening group id")
    else:
        schedule = build_g_schedule(
            args.seed, args.train_groups, args.development_groups, args.blind_groups,
            excluded_group_ids=excluded_ids, excluded_openings=excluded_openings,
        )
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    if not schedule.get("train_openings"):
        raise ValueError("preflight schedule needs at least one training opening")
    torch.set_num_threads(max(1, min(int(args.torch_threads), len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started_all = time.perf_counter()
    resources = _resource_snapshot(root)
    bench_model = initialize_arm("G0", args.seed)
    search = PUCTSearch(bench_model)
    positions = _positions()
    simulation_benchmarks = []
    for position_id, (board, to_move) in positions.items():
        search.search(board, to_move, 1, seed=args.seed)  # warm-up, excluded
        for simulations in (16, 32, 64):
            repeats = []
            for repeat in range(2):
                result = search.search(
                    board, to_move, simulations, add_root_noise=False,
                    temperature=0, seed=args.seed + repeat,
                )
                repeats.append(result.elapsed_seconds)
            simulation_benchmarks.append({
                "position_id": position_id,
                "simulations": simulations,
                "repeats": len(repeats),
                "elapsed_seconds": repeats,
                "mean_seconds": float(np.mean(repeats)),
                "simulations_per_second": simulations / float(np.mean(repeats)),
                "network_evaluations_last_repeat": result.network_evaluations,
                "tree_nodes_last_repeat": result.tree_nodes,
                "max_depth_last_repeat": result.max_depth,
            })
    prior_games, seen = load_game_log(games_path)
    recorded_by_id = {game["game_id"]: game for game in prior_games}
    selfplay_measurements = []
    training_measurements = []
    arm_games: dict[str, list[dict]] = {"G0": [], "G-F2": []}
    opening_group = schedule["train_openings"][0]
    for arm_index, arm_id in enumerate(("G0", "G-F2")):
        initial_seed = args.seed + arm_index * 1000
        model = initialize_arm(arm_id, initial_seed)
        for game_index in range(2):
            game_id = f"preflight-{arm_id}-s{args.seed}-g{game_index:02d}"
            if game_id in recorded_by_id:
                game = recorded_by_id[game_id]
            else:
                game_started = time.perf_counter()
                game = play_selfplay_game(
                    model,
                    opening_group["opening"],
                    arm_id=arm_id,
                    game_id=game_id,
                    opening_group_id=opening_group["opening_group_id"],
                    simulations=16,
                    seed=args.seed + arm_index * 10000 + game_index,
                    temperature_moves=12,
                    device=resources["device"],
                )
                wall = time.perf_counter() - game_started
                game["stats"]["preflight_wall_seconds"] = wall
                append_complete_game(games_path, game, seen)
            arm_games[arm_id].append(game)
            selfplay_measurements.append({
                "arm_id": arm_id,
                "game_id": game_id,
                "completed": game["completed"],
                "sample_positions": len(game["experience_rows"]),
                "wall_seconds": float(game["stats"].get("preflight_wall_seconds", game["stats"]["cpu_wall_seconds"])),
                "cpu_seconds": float(game["stats"].get("cpu_seconds", game["stats"]["cpu_wall_seconds"])),
                "simulations": int(game["stats"]["simulations"]),
                "move_count": game["move_count"],
                "completion_overrun_positions": 0,
            })
        replay = ReplayBuffer(capacity=10000, seed=args.seed + arm_index)
        for game in arm_games[arm_id]:
            replay.add_complete_game(game)
        trainer = GTrainer(
            model, seed=args.seed + arm_index + 12345, expected_updates=25,
            learning_rate=1e-3, device=resources["device"],
        )
        trainer.train_updates(replay, updates=5, batch_size=16)
        metrics = trainer.train_updates(replay, updates=20, batch_size=16)
        training_measurements.append({
            "arm_id": arm_id,
            "completed": True,
            "updates": 20,
            "batch_size": 16,
            "wall_seconds": float(metrics["wall_seconds"]),
            "cpu_seconds": float(metrics["cpu_seconds"]),
            "updates_per_second": 20 / float(metrics["wall_seconds"]),
            "selfplay_rows": replay.position_count,
        })
    output = {
        "format_version": 1,
        "preflight_seed": args.seed,
        "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "resources": resources,
        "simulation_benchmarks": simulation_benchmarks,
        "selfplay_games": selfplay_measurements,
        "training_updates": training_measurements,
        "measurements_wall_seconds": time.perf_counter() - started_all,
        "note": "All rates are measured with the 9x9 32-channel/2-block network on this worker; self-play uses 16 simulations per move.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(json.dumps({
        "preflight": str(output_path),
        "wall_seconds": output["measurements_wall_seconds"],
        "resources": resources,
        "simulations": {str(s): float(np.mean([item["mean_seconds"] for item in simulation_benchmarks if item["simulations"] == s])) for s in (16, 32, 64)},
        "selfplay_games": len(selfplay_measurements),
        "optimizer_updates_per_second": {item["arm_id"]: item["updates_per_second"] for item in training_measurements},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
