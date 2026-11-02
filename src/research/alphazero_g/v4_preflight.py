"""Measured, non-training G V4 throughput and resource preflight."""
from __future__ import annotations

import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...gomoku import GomokuBoard
from ..imitation_f.model import PolicyValueNet
from .puct import PUCTSearch
from .replay import ReplayBuffer
from .selfplay import play_selfplay_game
from .training import GTrainer
from .v3_evaluation import play_v3_match
from .v3_schedule import validate_v3_schedule


def _positions() -> dict[str, tuple[GomokuBoard, str]]:
    opening = GomokuBoard(9)
    opening.place_stone("B", (4, 4))
    midgame = GomokuBoard(9)
    sequence = [
        ("B", (4, 4)), ("W", (4, 5)), ("B", (3, 4)), ("W", (5, 5)),
        ("B", (5, 3)), ("W", (3, 6)), ("B", (6, 5)), ("W", (2, 3)),
    ]
    for player, coord in sequence:
        midgame.place_stone(player, coord)
    return {"opening": (opening, "W"), "midgame": (midgame, "B")}


def _resource_snapshot() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            mem[key] = int(value.strip().split()[0]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "affinity_cpus": len(affinity) or os.cpu_count() or 1,
        "affinity_cpu_ids": affinity,
        "logical_cpus": os.cpu_count(),
        "memory_total_bytes": mem.get("MemTotal"),
        "memory_available_bytes": mem.get("MemAvailable"),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "peak_rss_bytes": int(usage * (1024 if os.name == "posix" else 1)),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "torch_version": torch.__version__,
    }


def run_v4_preflight(root: str | Path, *, seed: int = 540401) -> dict[str, Any]:
    root_path = Path(root).resolve()
    schedule_path = root_path / "research/g_v4/config/openings.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    torch.set_num_threads(1)
    model = PolicyValueNet(board_size=9, channels=32, blocks=2, seed=int(seed))
    model.eval()

    simulation_rows: list[dict[str, Any]] = []
    for position_id, (board, player) in _positions().items():
        for simulations in (16, 32, 64):
            samples = []
            result = None
            for repeat in range(2):
                search = PUCTSearch(model, device="cpu")
                started = time.process_time()
                result = search.search(board, player, simulations, seed=seed + repeat, add_root_noise=False)
                samples.append(time.process_time() - started)
            assert result is not None
            simulation_rows.append({
                "position_id": position_id,
                "simulations": simulations,
                "cpu_seconds": samples,
                "mean_cpu_seconds": float(np.mean(samples)),
                "simulations_per_cpu_second": float(simulations / np.mean(samples)),
                "tree_nodes": result.tree_nodes,
                "network_evaluations": result.network_evaluations,
                "max_depth": result.max_depth,
            })

    train_groups = schedule["train_openings_by_seed"][str(seed)]
    selected = [train_groups[0], train_groups[1], train_groups[2]]
    games: list[dict[str, Any]] = []
    for index, group in enumerate(selected):
        game = play_selfplay_game(
            model, group["opening"], arm_id="G0-v4-preflight", game_id=f"g-v4-preflight-s{seed}-{index}",
            opening_group_id=group["opening_group_id"], simulations=64,
            seed=seed + 10_000 + index * 97,
        )
        games.append(game)
    replay = ReplayBuffer(20_000, seed=seed + 20_000)
    replay.add_complete_game(games[0])
    trainer = GTrainer(
        model, seed=seed + 30_000, learning_rate=1e-4, weight_decay=1e-4,
        expected_updates=64, bn_mode="freeze",
    )
    training = trainer.train_updates(replay, updates=20, batch_size=16)

    evaluation_rows = []
    for mode in ("network", "puct"):
        for opponent in ("A", "B"):
            group = schedule["development_openings"][0]
            match = play_v3_match(
                model, None, game_id=f"g-v4-preflight-{mode}-vs-{opponent}",
                opening_group=group, agent_color="B", mode=mode, opponent=opponent,
                seed=seed + 40_000 + (0 if mode == "network" else 100) + (0 if opponent == "A" else 10),
                simulations=64, candidate_checkpoint_id="G0-preflight", phase="preflight",
            )
            evaluation_rows.append({
                "mode": mode, "opponent": opponent, "completed": True,
                "cpu_seconds": float(match["stats"]["total_cpu_seconds"]),
                "wall_seconds": float(match["stats"]["total_wall_seconds"]),
                "simulations": int(match["stats"].get("agent_simulations", 0)),
                "moves": len(match.get("moves", [])),
            })

    prior_progress = json.loads((root_path / "research/g_v3/checkpoints/G0/seed-510901/progress.json").read_text())
    prior_rate = int(prior_progress["budgeted_environment_steps"]) / float(prior_progress["selfplay_cpu_seconds"])
    payload = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4-preflight",
        "formal_training_started": False,
        "seed": int(seed),
        "resources": _resource_snapshot(),
        "simulation_benchmarks": simulation_rows,
        "selfplay_games": [{
            "game_id": game["game_id"], "completed": bool(game["completed"]),
            "sample_positions": len(game["experience_rows"]),
            "cpu_seconds": float(game["stats"]["cpu_seconds"]),
            "wall_seconds": float(game["stats"]["wall_seconds"]),
            "moves": int(game["move_count"]), "simulations": int(game["stats"]["simulations"]),
            "opening_kind": selected[index]["kind"],
        } for index, game in enumerate(games)],
        "training_updates": [{
            "completed": True, "updates": int(training["updates"]),
            "cpu_seconds": float(training["cpu_seconds"]),
            "updates_per_cpu_second": float(training["updates"] / training["cpu_seconds"]),
            "batch_size": 16,
        }],
        "evaluation_games": evaluation_rows,
        "historical_g0_30k_state_rate_per_cpu_second": prior_rate,
        "historical_v3_seed510901_selfplay_cpu_seconds_for_30k": float(prior_progress["selfplay_cpu_seconds"]),
        "measurement_cpu_seconds": float(sum(row["stats"]["cpu_seconds"] for row in games)
                                           + training["cpu_seconds"]
                                           + sum(row["cpu_seconds"] for row in evaluation_rows)
                                           + sum(sum(row["cpu_seconds"]) for row in simulation_rows)),
    }
    return payload

