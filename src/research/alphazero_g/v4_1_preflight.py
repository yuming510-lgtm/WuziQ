"""Measured CPU/resource preflight before G V4.1 formal training."""
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
from .training import GTrainer, load_g_checkpoint
from .v3_evaluation import play_v3_match
from .v4_1_config import load_v4_1_config
from .v4_1_budget import estimate_runtime


def _positions() -> dict[str, tuple[GomokuBoard, str]]:
    opening = GomokuBoard(9)
    opening.place_stone("B", (4, 4))
    midgame = GomokuBoard(9)
    sequence = [
        ("B", (4, 4)), ("W", (4, 5)), ("B", (3, 4)), ("W", (5, 5)),
        ("B", (5, 3)), ("W", (3, 6)), ("B", (6, 5)), ("W", (2, 3)),
    ]
    for player, coordinate in sequence:
        midgame.place_stone(player, coordinate)
    return {"opening": (opening, "W"), "midgame": (midgame, "B")}


def _resources() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    mem: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            mem[key] = int(value.strip().split()[0]) * 1024
    cpu_quota = None
    for quota_path in (Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        if quota_path.is_file():
            values = quota_path.read_text().strip().split()
            try:
                if len(values) == 2 and values[0] != "max":
                    cpu_quota = float(values[0]) / float(values[1])
                elif len(values) == 1 and float(values[0]) > 0:
                    period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
                    cpu_quota = float(values[0]) / float(period)
            except (ValueError, ZeroDivisionError):
                pass
            if cpu_quota is not None:
                break
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "affinity_cpu_count": len(affinity) or os.cpu_count() or 1,
        "affinity_cpu_ids": affinity,
        "logical_cpu_count": os.cpu_count(),
        "cgroup_cpu_quota": cpu_quota,
        "memory_total_bytes": mem.get("MemTotal"),
        "memory_available_bytes": mem.get("MemAvailable"),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "peak_rss_bytes": int(usage * (1024 if os.name == "posix" else 1)),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "torch_threads": 1,
    }


def run_v4_1_preflight(root: str | Path, *, seed: int = 741101) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_1_config(root_path)
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text())
    torch.set_num_threads(1)
    clock_start = time.process_time()
    seed_model = PolicyValueNet(board_size=9, channels=32, blocks=2, seed=seed)
    seed_model.eval()

    simulations_rows: list[dict[str, Any]] = []
    for position_id, (board, player) in _positions().items():
        for simulations in (16, 32, 64, 128, 256):
            times = []
            result = None
            for repeat in range(2):
                search = PUCTSearch(seed_model, device="cpu")
                started = time.process_time()
                result = search.search(board, player, simulations,
                                       add_root_noise=False, seed=seed + repeat)
                times.append(time.process_time() - started)
            assert result is not None
            simulations_rows.append({
                "position_id": position_id,
                "simulations": simulations,
                "cpu_seconds_each": times,
                "mean_cpu_seconds": float(np.mean(times)),
                "simulations_per_cpu_second": float(simulations / max(np.mean(times), 1e-9)),
                "network_evaluations": int(result.network_evaluations),
                "tree_nodes": int(result.tree_nodes),
                "max_depth": int(result.max_depth),
            })

    initial_model = load_g_checkpoint(root_path / config["initial_checkpoint"], device="cpu")
    initial_model.eval()
    opening_seed = int(schedule["seeds"][0])
    train_groups = schedule["train_openings_by_seed"][str(opening_seed)]
    sample_games: list[dict[str, Any]] = []
    for index, group in enumerate(train_groups[:2]):
        game = play_selfplay_game(
            initial_model, group["opening"], arm_id="G0-v4-1-preflight",
            game_id=f"g-v4-1-preflight-{index:02d}",
            opening_group_id=str(group["opening_group_id"]), simulations=64,
            seed=seed + 10_000 + index, temperature_moves=12, late_temperature=0.1,
        )
        sample_games.append(game)
    training_start_cpu = time.process_time()
    replay = ReplayBuffer(20_000, seed=seed + 20_000)
    replay.add_complete_game(sample_games[0])
    trainer = GTrainer(initial_model, seed=seed + 30_000, learning_rate=1e-4,
                       weight_decay=1e-4, expected_updates=64, device="cpu", bn_mode="freeze")
    update_metrics = trainer.train_updates(replay, updates=20, batch_size=16,
                                           sampling_mode="uniform_game")
    training_measurement_cpu = time.process_time() - training_start_cpu

    evaluation_rows: list[dict[str, Any]] = []
    group = schedule["development_openings"][0]
    for mode in ("network", "puct"):
        for opponent in ("A", "B"):
            row = play_v3_match(
                initial_model, None,
                game_id=f"g-v4-1-preflight-{mode}-vs-{opponent}",
                opening_group=group, agent_color="B", mode=mode, opponent=opponent,
                seed=seed + 40_000 + (100 if mode == "puct" else 0) + (10 if opponent == "B" else 0),
                simulations=64, candidate_checkpoint_id="G0-v4-1-preflight", phase="preflight",
            )
            evaluation_rows.append({
                "mode": mode, "opponent": opponent,
                "completed": bool(row.get("completed", False)),
                "cpu_seconds": float(row["stats"]["total_cpu_seconds"]),
                "wall_seconds": float(row["stats"]["total_wall_seconds"]),
                "moves": len(row["moves"]),
                "simulations": int(row["stats"].get("agent_simulations", 0)),
            })

    selfplay_cpu = [float(game["stats"]["cpu_seconds"]) for game in sample_games]
    state_count = sum(len(game["experience_rows"]) for game in sample_games)
    update_cpu = float(update_metrics["cpu_seconds"])
    measured = {
        "training_states": 40_000,
        "update_every_states": 16,
        "selfplay_states_per_cpu_second": state_count / max(sum(selfplay_cpu), 1e-9),
        "optimizer_cpu_seconds_per_update": update_cpu / max(int(update_metrics["updates"]), 1),
        "evaluation_cpu_seconds_per_network_game": float(np.mean([
            row["cpu_seconds"] for row in evaluation_rows if row["mode"] == "network"
        ])),
        "evaluation_cpu_seconds_per_puct_game": float(np.mean([
            row["cpu_seconds"] for row in evaluation_rows if row["mode"] == "puct"
        ])),
        # 104.76 seconds were measured in the V4.1 diagnostic runs before the
        # full resource preflight. The current preflight is added below.
        "diagnostic_cpu_seconds": 104.76,
    }
    estimate = estimate_runtime(measured)
    preflight_cpu = time.process_time() - clock_start
    for key in ("total_cpu_seconds",):
        for level in ("central", "optimistic", "conservative"):
            estimate[key][level] += preflight_cpu
    for level in ("central", "optimistic", "conservative"):
        estimate["total_wall_minutes_sequential"][level] += preflight_cpu / 60.0
    payload = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4-1",
        "formal_training_started": False,
        "seed": int(seed),
        "resources": _resources(),
        "simulation_benchmarks": simulations_rows,
        "selfplay_games": [{
            "game_id": game["game_id"],
            "completed": bool(game["completed"]),
            "states": len(game["experience_rows"]),
            "cpu_seconds": float(game["stats"]["cpu_seconds"]),
            "wall_seconds": float(game["stats"]["wall_seconds"]),
            "move_count": int(game["move_count"]),
            "simulations": int(game["stats"]["simulations"]),
            "opening_kind": train_groups[index]["kind"],
        } for index, game in enumerate(sample_games)],
        "optimizer_benchmark": {
            "updates": int(update_metrics["updates"]),
            "cpu_seconds_reported": update_cpu,
            "wall_seconds_reported": float(update_metrics["wall_seconds"]),
            "cpu_seconds_inclusive": training_measurement_cpu,
            "updates_per_cpu_second": int(update_metrics["updates"]) / max(update_cpu, 1e-9),
            "batch_size": 16,
        },
        "evaluation_games": evaluation_rows,
        "measurements": measured,
        "estimate": estimate,
        "preflight_cpu_seconds": preflight_cpu,
    }
    return payload
