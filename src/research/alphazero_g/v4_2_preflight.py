"""Measured throughput and resource preflight before any V4.2 formal training."""
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
from .evaluation import play_evaluation_match
from .puct import PUCTConfig, PUCTSearch
from .replay import ReplayBuffer
from .selfplay import play_selfplay_game
from .training import GTrainer, load_g_checkpoint
from .v4_2_budget import estimate_v4_2_runtime
from .v4_2_config import load_v4_2_config


def _resource_snapshot(root: Path) -> dict[str, Any]:
    cgroup_cpu = None
    cpu_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_path.is_file():
        quota, period = cpu_path.read_text().strip().split()
        if quota != "max":
            cgroup_cpu = float(quota) / float(period)
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(value.strip().split()[0]) * 1024
    return {
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "cgroup_cpu_quota": cgroup_cpu,
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": memory.get("MemAvailable"),
        "memory_cgroup_limit_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text())
        if Path("/sys/fs/cgroup/memory.max").is_file() and Path("/sys/fs/cgroup/memory.max").read_text().strip().isdigit() else None,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def _representative_positions() -> dict[str, tuple[GomokuBoard, str]]:
    opening = GomokuBoard(9)
    opening.place_stone("B", (4, 4))
    midgame = GomokuBoard(9)
    moves = [("B", (4, 4)), ("W", (4, 5)), ("B", (3, 4)), ("W", (5, 5)),
             ("B", (5, 3)), ("W", (3, 6)), ("B", (6, 5)), ("W", (2, 3))]
    for player, coordinate in moves:
        midgame.place_stone(player, coordinate)
    return {"opening": (opening, "W"), "midgame": (midgame, "B")}


def _estimates(config: dict[str, Any], measurements: dict[str, Any]) -> dict[str, Any]:
    candidates = len(config["historical_candidates"])
    variants = len(config["search_variants"]) - 1
    arms = len(config["arms"])
    development_groups = int(config["evaluation"]["development_groups"])
    blind_groups = int(config["evaluation"]["blind_groups"])
    historical_games = candidates * 2 * development_groups * 2
    search_variant_games = variants * 3 * 8 * 2
    stage_groups = int(config["evaluation"]["stage_groups"])
    stage_games = arms * 4 * 3 * stage_groups * 2
    final_games = 7 * blind_groups * 2
    internal_games = (arms * (arms - 1) // 2) * 8 * 2
    evaluation_games = historical_games + search_variant_games + stage_games + final_games + internal_games
    history_per_opponent = candidates * development_groups * 2
    stage_per_opponent = arms * 4 * stage_groups * 2
    blind_per_mode = blind_groups * 2
    evaluation_games_by_opponent = {
        "A": history_per_opponent + stage_per_opponent + 3 * blind_per_mode,
        "B": stage_per_opponent + 3 * blind_per_mode,
        "MODEL": history_per_opponent + search_variant_games + stage_per_opponent
                 + blind_per_mode + internal_games,
    }
    if sum(evaluation_games_by_opponent.values()) != evaluation_games:
        raise ValueError("preflight matchup accounting does not cover every registered evaluation game")
    measured_by_opponent = measurements["evaluation_cpu_seconds_by_opponent"]
    weighted_eval_cost = sum(
        evaluation_games_by_opponent[name] * measured_by_opponent[name]
        for name in evaluation_games_by_opponent
    ) / evaluation_games
    diagnostic_calls = (
        candidates * int(config["diagnostics"]["tactical_per_category"]) * 4
        + min(3, candidates) * variants * int(config["diagnostics"]["tactical_per_category"]) * 4
        + 2 * 64 * 2
    )
    levels = {
        "optimistic": {"states_rate": 1.25, "update_rate": 1.15, "evaluation_cost": 0.8,
                       "search_cost": 0.8, "fixed": 180.0},
        "central": {"states_rate": 1.0, "update_rate": 1.0, "evaluation_cost": 1.0,
                    "search_cost": 1.0, "fixed": 300.0},
        "conservative": {"states_rate": 0.65, "update_rate": 0.67, "evaluation_cost": 1.6,
                         "search_cost": 1.6, "fixed": 500.0},
    }
    states_per_arm = int(config["training"]["states_per_arm"])
    history_pool_states = (states_per_arm if "history_pool" in config["arms"] else 0)
    results = {}
    for label, factors in levels.items():
        estimate = estimate_v4_2_runtime(
            states_per_arm=states_per_arm,
            arms=arms,
            training_states_per_cpu_second=(measurements["selfplay_states_per_cpu_second"] * factors["states_rate"]),
            optimizer_updates_per_cpu_second=(measurements["optimizer_updates_per_cpu_second"] * factors["update_rate"]),
            evaluation_games=evaluation_games,
            evaluation_cpu_seconds_per_game=(weighted_eval_cost * factors["evaluation_cost"]),
            diagnostic_search_calls=diagnostic_calls,
            diagnostic_cpu_seconds_per_search=(measurements["search64_cpu_seconds_per_call"] * factors["search_cost"]),
            fixed_overhead_cpu_seconds=factors["fixed"],
            update_every_states=int(config["training"]["update_every_states"]),
        )
        # History-pool games run a frozen G0 search on the non-learner side,
        # while only learner decisions enter replay, so their CPU/state cost
        # is conservatively counted twice.
        history_extra_seconds = history_pool_states / (
            measurements["selfplay_states_per_cpu_second"] * factors["states_rate"]
        )
        estimate["components_cpu_seconds"]["history_pool_opponent_search_extra"] = history_extra_seconds
        estimate["total_cpu_seconds"] += history_extra_seconds
        estimate["total_cpu_hours"] = estimate["total_cpu_seconds"] / 3600.0
        results[label] = estimate
    return {
        "game_count_components": {
            "historical_checkpoint_vs_A_and_old_champion": historical_games,
            "frozen_search_variant_matches": search_variant_games,
            "training_stage_evaluations": stage_games,
            "final_blind_and_control_matches": final_games,
            "training_arm_internal_matches": internal_games,
            "total_games": evaluation_games,
            "evaluation_games_by_opponent": evaluation_games_by_opponent,
        },
        "diagnostic_search_calls_estimated": diagnostic_calls,
        "levels": results,
        "cpu_hard_cap_hours": float(config["compute"]["cpu_hour_cap"]),
    }


def run_v4_2_preflight(root: str | Path, *, seed: int = 742_209) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_2_config(root_path)
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    candidate = next(row for row in config["historical_candidates"]
                     if row["candidate_id"] == "G0-V3-510901-20K")
    model = load_g_checkpoint(root_path / candidate["checkpoint_path"], device="cpu")
    model.eval()
    started_cpu, started_wall = time.process_time(), time.perf_counter()

    search_rows = []
    configs = {
        "baseline": PUCTConfig(),
        "c_puct_5": PUCTConfig(c_puct=5.0),
        "c_puct_10": PUCTConfig(c_puct=10.0),
        "prior_smoothing_10pct": PUCTConfig(prior_smoothing_fraction=0.1),
        "value_neutral": PUCTConfig(leaf_value_scale=0.0),
    }
    for position_id, (board, player) in _representative_positions().items():
        for variant_id, search_config in configs.items():
            for simulations in (64, 128, 256):
                costs, result = [], None
                for repeat in range(2):
                    process_start = time.process_time()
                    result = PUCTSearch(model, search_config).search(
                        board, player, simulations, add_root_noise=False, seed=seed + repeat
                    )
                    costs.append(time.process_time() - process_start)
                assert result is not None
                search_rows.append({
                    "position_id": position_id, "variant_id": variant_id,
                    "simulations": simulations, "cpu_seconds_each": costs,
                    "mean_cpu_seconds": float(np.mean(costs)),
                    "tree_nodes": int(result.tree_nodes), "max_depth": int(result.max_depth),
                    "network_evaluations": int(result.network_evaluations),
                })

    opening = schedule["train_openings"][0]
    game = play_selfplay_game(
        model, opening["opening"], arm_id="g-v4-2-preflight", game_id="g-v4-2-preflight-0001",
        opening_group_id=opening["opening_group_id"], simulations=64, seed=seed,
        config=PUCTConfig(), device="cpu",
    )
    replay = ReplayBuffer(20_000, seed=seed + 1)
    replay.add_complete_game(game)
    trainer = GTrainer(model, seed=seed + 2, learning_rate=1e-4, weight_decay=1e-4,
                       expected_updates=32, device="cpu", bn_mode="freeze")
    update = trainer.train_updates(replay, updates=32, batch_size=16, sampling_mode="uniform_game")

    eval_rows = []
    for opponent in ("A", "B", "MODEL"):
        row = play_evaluation_match(
            model, game_id=f"g-v4-2-preflight-vs-{opponent}",
            opening_group_id="g-v4-2-preflight-opening", arm_id="G0-V3-510901-20K",
            mode="puct", opponent=opponent,
            opening=list(opening["opening"]), agent_color="B", seed=seed + (100 if opponent == "B" else 0),
            simulations=64, config=PUCTConfig(), device="cpu",
            opponent_model=model if opponent == "MODEL" else None,
            opponent_mode="puct" if opponent == "MODEL" else "network",
            opponent_simulations=64,
        )
        eval_rows.append({
            "opponent": opponent, "completed": bool(row.get("completed")),
            "cpu_seconds": float(row.get("stats", {}).get("total_cpu_seconds", 0.0)),
            "wall_seconds": float(row.get("stats", {}).get("total_wall_seconds", 0.0)),
            "moves": len(row.get("moves", [])),
            "agent_simulations": int(row.get("stats", {}).get("agent_simulations", 0)),
        })

    state_count = len(game["experience_rows"])
    game_cpu = float(game["stats"]["cpu_seconds"])
    search64 = [row["mean_cpu_seconds"] for row in search_rows if row["simulations"] == 64]
    measurements = {
        "selfplay_states_per_cpu_second": state_count / max(game_cpu, 1e-9),
        "optimizer_updates_per_cpu_second": int(update["updates"]) / max(float(update["cpu_seconds"]), 1e-9),
        "evaluation_cpu_seconds_by_opponent": {
            name: float(next(row["cpu_seconds"] for row in eval_rows if row["opponent"] == name))
            for name in ("A", "B", "MODEL")
        },
        "search64_cpu_seconds_per_call": float(np.median(search64)),
    }
    measurements["evaluation_cpu_seconds_per_game"] = float(np.mean(list(
        measurements["evaluation_cpu_seconds_by_opponent"].values()
    )))
    estimates = _estimates(config, measurements)
    return {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "formal_training_started": False,
        "seed": int(seed),
        "resources": _resource_snapshot(root_path),
        "search_benchmarks": search_rows,
        "selfplay_benchmark": {
            "completed": bool(game["completed"]), "states": state_count,
            "games": 1, "simulations": int(game["stats"]["simulations"]),
            "cpu_seconds": game_cpu, "wall_seconds": float(game["stats"]["wall_seconds"]),
            "moves": int(game["move_count"]),
        },
        "optimizer_benchmark": {
            "updates": int(update["updates"]), "cpu_seconds": float(update["cpu_seconds"]),
            "wall_seconds": float(update["wall_seconds"]), "batch_size": 16,
        },
        "evaluation_benchmarks": eval_rows,
        "measurements": measurements,
        "estimates": estimates,
        "preflight_cpu_seconds": time.process_time() - started_cpu,
        "preflight_wall_seconds": time.perf_counter() - started_wall,
    }
