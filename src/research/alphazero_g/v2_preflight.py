"""Measured CPU preflight and finite equal-budget registration for G V2."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...gomoku import GomokuBoard
from .evaluation import append_evaluation_match, load_evaluation_log, play_evaluation_match
from .puct import PUCTSearch
from .replay import ReplayBuffer, append_complete_game, load_game_log
from .selfplay import play_selfplay_game
from .training import F2_SHA256, GTrainer, initialize_arm
from .v2_schedule import validate_v2_schedule

LR_IDS = {"lr_1e4", "lr_3e5", "lr_1e3"}
LEARNING_RATES = {"lr_1e4": 1e-4, "lr_3e5": 3e-5, "lr_1e3": 1e-3}
SIMULATIONS = (16, 32, 64)
POSITIONS = ("opening", "middle", "tactical")
MILESTONES = (1000, 2000, 5000, 10000, 20000)
OPPONENTS = ("A", "B", "F2")
MODES = ("network", "puct")


def _resource_snapshot(root: Path) -> dict[str, Any]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(value.strip().split()[0]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
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
    boards: dict[str, list[tuple[str, tuple[int, int]]]] = {
        "opening": [
            ("B", (4, 4)), ("W", (3, 4)), ("B", (3, 5)), ("W", (5, 3)),
        ],
        "middle": [
            ("B", (4, 4)), ("W", (3, 4)), ("B", (4, 5)), ("W", (5, 4)),
            ("B", (3, 5)), ("W", (5, 5)), ("B", (3, 3)), ("W", (5, 3)),
            ("B", (2, 4)), ("W", (6, 4)), ("B", (4, 3)), ("W", (4, 6)),
            ("B", (2, 5)), ("W", (6, 3)), ("B", (2, 3)), ("W", (6, 5)),
        ],
        "tactical": [
            ("B", (4, 1)), ("W", (0, 0)), ("B", (4, 2)), ("W", (0, 1)),
            ("B", (4, 3)), ("W", (0, 2)), ("B", (4, 4)), ("W", (1, 0)),
        ],
    }
    result: dict[str, tuple[GomokuBoard, str]] = {}
    for name, moves in boards.items():
        board = GomokuBoard(9)
        for player, coord in moves:
            board.place_stone(player, coord)
        result[name] = (board, "B")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_v2_preflight(
    root: Path,
    schedule: dict[str, Any],
    seed: int,
    *,
    games_log: Path | None = None,
    evaluation_log: Path | None = None,
) -> dict[str, Any]:
    """Measure the real worker on three PUCT positions, self-play, updates and matches."""
    root = Path(root).resolve()
    validate_v2_schedule(schedule)
    torch.set_num_threads(1)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    cpu_started = time.process_time()
    resources = _resource_snapshot(root)
    opening = schedule.get("train_openings", [None])[0]
    if not opening:
        raise ValueError("V2 preflight requires at least one registered training opening")

    model = initialize_arm("G-F2", seed=int(seed), device=resources["device"])
    simulation_benchmarks: list[dict[str, Any]] = []
    for position_id, (board, to_move) in _positions().items():
        PUCTSearch(model).search(board, to_move, 1, seed=seed)  # warm-up, excluded
        for simulations in SIMULATIONS:
            repeats: list[float] = []
            for repeat in range(3):
                search = PUCTSearch(model)
                result = search.search(
                    board, to_move, simulations, add_root_noise=False, temperature=0.0,
                    seed=seed + simulations * 100 + repeat,
                )
                repeats.append(float(result.elapsed_seconds))
            simulation_benchmarks.append({
                "position_id": position_id,
                "simulations": simulations,
                "repeats": len(repeats),
                "elapsed_seconds": repeats,
                "mean_seconds": float(statistics.mean(repeats)),
                "simulations_per_second": simulations / float(statistics.mean(repeats)),
                "network_evaluations": int(result.network_evaluations),
                "tree_nodes": int(result.tree_nodes),
                "max_depth": int(result.max_depth),
            })

    selfplay_path = games_log or root / "research/g_v2/results/preflight/selfplay.jsonl"
    existing_games, game_ids = load_game_log(selfplay_path)
    games_by_id = {str(game["game_id"]): game for game in existing_games}
    selfplay_games: list[dict[str, Any]] = []
    training_updates: list[dict[str, Any]] = []
    for index, lr_id in enumerate(sorted(LR_IDS)):
        arm_id = f"G-F2-{lr_id}"
        game_id = f"g-v2-preflight-{lr_id}-seed-{int(seed)}"
        game = games_by_id.get(game_id)
        if game is None:
            model = initialize_arm("G-F2", seed=int(seed) + index, device=resources["device"])
            game = play_selfplay_game(
                model, opening["opening"], arm_id=arm_id, game_id=game_id,
                opening_group_id=opening["opening_group_id"], simulations=64,
                seed=int(seed) + 1000 + index, device=resources["device"],
            )
            game["candidate_lr_id"] = lr_id
            append_complete_game(selfplay_path, game, game_ids)
            games_by_id[game_id] = game
        if game.get("candidate_lr_id") != lr_id:
            raise ValueError("preflight game ID collided with a different learning-rate candidate")
        cpu_seconds = float(game["stats"].get("cpu_seconds", game["stats"].get("wall_seconds", 0)))
        selfplay_games.append({
            "lr_id": lr_id,
            "game_id": game_id,
            "completed": bool(game["completed"]),
            "budgeted_states": len(game["experience_rows"]),
            "cpu_seconds": cpu_seconds,
            "wall_seconds": float(game["stats"]["wall_seconds"]),
            "simulations": int(game["stats"]["simulations"]),
            "move_count": int(game["move_count"]),
        })
        replay = ReplayBuffer(4096, seed=int(seed) + index)
        replay.add_complete_game(game)
        trainer_model = initialize_arm("G-F2", seed=int(seed) + index, device=resources["device"])
        trainer = GTrainer(
            trainer_model, seed=int(seed) + index + 10_000, learning_rate=LEARNING_RATES[lr_id],
            expected_updates=20, device=resources["device"],
        )
        measurement = trainer.train_updates(replay, updates=20, batch_size=16)
        training_updates.append({
            "lr_id": lr_id,
            "completed": True,
            "updates": 20,
            "batch_size": 16,
            "cpu_seconds": float(measurement["cpu_seconds"]),
            "wall_seconds": float(measurement["wall_seconds"]),
            "updates_per_second": 20 / max(float(measurement["cpu_seconds"]), 1e-9),
            "selfplay_rows": replay.position_count,
        })

    eval_path = evaluation_log or root / "research/g_v2/results/preflight/evaluation.jsonl"
    evaluation_records, evaluation_ids = load_evaluation_log(eval_path)
    recorded_evaluation = {str(item["game_id"]): item for item in evaluation_records}
    evaluation_games: list[dict[str, Any]] = []
    eval_opening = schedule.get("development_openings", [None])[0]
    if not eval_opening:
        raise ValueError("V2 preflight requires at least one registered development opening")
    for mode in MODES:
        for opponent in OPPONENTS:
            game_id = f"g-v2-preflight-eval-{mode}-{opponent}-seed-{int(seed)}"
            match = recorded_evaluation.get(game_id)
            if match is None:
                model = initialize_arm("G-F2", seed=int(seed), device=resources["device"])
                match = play_evaluation_match(
                    model, game_id=game_id, opening_group_id=eval_opening["opening_group_id"],
                    arm_id="F2-preflight", mode=mode, opponent=opponent,
                    opening=eval_opening["opening"], agent_color="B", seed=int(seed) + len(evaluation_games),
                    simulations=64, device=resources["device"],
                )
                match["preflight"] = True
                append_evaluation_match(eval_path, match, evaluation_ids)
            stats = match["stats"]
            agent_moves = sum(
                1 for item in match["moves"] if item.get("actor") == "agent"
            )
            evaluation_games.append({
                "mode": mode,
                "opponent": opponent,
                "completed": bool(match["completed"]),
                "simulations_per_move": int(match["simulations_per_agent_move"]),
                "total_cpu_seconds": float(stats["total_cpu_seconds"]),
                "agent_seconds": float(stats["agent_seconds"]),
                "agent_moves": agent_moves,
                "opponent_seconds": float(stats["opponent_seconds"]),
                "post_opening_moves": int(match["move_count"] - len(match["opening"])),
            })

    result = {
        "format_version": 1,
        "seed": int(seed),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "schedule_sha256": schedule["schedule_sha256"],
        "resources": resources,
        "torch_threads": 1,
        "simulation_benchmarks": simulation_benchmarks,
        "selfplay_games": selfplay_games,
        "training_updates": training_updates,
        "evaluation_games": evaluation_games,
        "measurements_wall_seconds": time.perf_counter() - started,
        "measurements_cpu_seconds": time.process_time() - cpu_started,
        "note": "Measured on the connected worker using the frozen 9x9 32-channel/2-block F2 model, PUCT-64 self-play, and independent 1-thread CPU updates.",
    }
    return result


def register_v2_budget(
    preflight: dict[str, Any], proposed: dict[str, Any], schedule: dict[str, Any]
) -> dict[str, Any]:
    """Choose the largest registered milestone fitting a conservative 2-CPU-hour cap."""
    rates: dict[str, float] = {}
    update_rates: dict[str, float] = {}
    for name, rows in (
        ("simulation_benchmarks", preflight.get("simulation_benchmarks")),
        ("selfplay_games", preflight.get("selfplay_games")),
        ("training_updates", preflight.get("training_updates")),
        ("evaluation_games", preflight.get("evaluation_games")),
    ):
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"preflight is missing measured {name}")
    benchmark_rows = preflight["simulation_benchmarks"]
    observed = {(int(item["simulations"]), str(item["position_id"])) for item in benchmark_rows}
    for simulation_count in SIMULATIONS:
        if any((simulation_count, position) not in observed for position in POSITIONS):
            raise ValueError("preflight needs 16, 32 and 64 simulation measurements on opening, middle and tactical positions")
    if float(proposed.get("hard_cpu_hour_cap", 0)) <= 0 or float(proposed["hard_cpu_hour_cap"]) > 2.0:
        raise ValueError("V2 compute cap must be positive and no more than two CPU hours")
    lrs = proposed.get("candidate_learning_rates")
    if lrs != LEARNING_RATES:
        raise ValueError("V2 candidate learning rates must match the three registered candidates")
    if int(proposed.get("puct_simulations", 64)) != 64:
        raise ValueError("G V2 is registered at exactly 64 PUCT simulations per move")
    if int(proposed.get("update_every_budgeted_states", 0)) != 16 or int(proposed.get("batch_size", 0)) != 16:
        raise ValueError("G V2 uses one batch-16 update per 16 budgeted states")
    if len(schedule.get("development_openings", [])) != 8 or len(schedule.get("blind_openings", [])) < 20:
        raise ValueError("G V2 needs eight development groups and at least twenty sealed blind groups")
    by_lr_selfplay = {str(item.get("lr_id")): item for item in preflight["selfplay_games"]}
    by_lr_updates = {str(item.get("lr_id")): item for item in preflight["training_updates"]}
    if set(by_lr_selfplay) != LR_IDS or set(by_lr_updates) != LR_IDS:
        raise ValueError("preflight must measure every candidate learning rate")
    for lr_id in LR_IDS:
        game = by_lr_selfplay[lr_id]
        update = by_lr_updates[lr_id]
        if game.get("completed") is not True or update.get("completed") is not True:
            raise ValueError("preflight contains an incomplete candidate measurement")
        states = int(game.get("budgeted_states", 0))
        cpu = float(game.get("cpu_seconds", 0))
        updates = int(update.get("updates", 0))
        update_cpu = float(update.get("cpu_seconds", 0))
        if states < 1 or cpu <= 0 or updates < 1 or update_cpu <= 0 or int(game.get("simulations", 0)) < states * 64:
            raise ValueError("preflight has invalid self-play or update throughput")
        rates[lr_id] = states / cpu
        update_rates[lr_id] = updates / update_cpu
    eval_rows = preflight["evaluation_games"]
    evaluation_cost: dict[tuple[str, str], float] = {}
    for mode in MODES:
        for opponent in OPPONENTS:
            rows = [row for row in eval_rows if row.get("mode") == mode and row.get("opponent") == opponent]
            if not rows or any(row.get("completed") is not True for row in rows):
                raise ValueError("preflight must measure complete network/PUCT matches against A, B and F2")
            if mode == "puct" and any(int(row.get("simulations_per_move", 64)) != 64 for row in rows):
                raise ValueError("PUCT evaluation preflight must use 64 simulations per move")
            if any(float(row.get("total_cpu_seconds", 0)) <= 0 for row in rows):
                raise ValueError("preflight has invalid evaluation CPU timings")
            evaluation_cost[(mode, opponent)] = statistics.median(float(row["total_cpu_seconds"]) for row in rows)
    mode_turn_cost: dict[str, float] = {}
    for mode in MODES:
        values = [
            float(row["agent_seconds"]) / max(int(row["agent_moves"]), 1)
            for row in eval_rows if row.get("mode") == mode
        ]
        if not values or min(values) <= 0:
            raise ValueError("preflight is missing per-move evaluation timings")
        mode_turn_cost[mode] = statistics.median(values)
    post_opening_moves = statistics.median(int(row.get("post_opening_moves", 0)) for row in eval_rows)
    if post_opening_moves < 1:
        raise ValueError("preflight evaluation matches did not make a post-opening move")

    if preflight.get("schedule_sha256", schedule.get("schedule_sha256")) != schedule.get("schedule_sha256"):
        raise ValueError("preflight and registered opening schedule checksums differ")
    cap_seconds = float(proposed["hard_cpu_hour_cap"]) * 3600
    starting_quota = min(int(proposed.get("states_per_candidate", proposed.get("proposed_states_per_candidate", 20_000))), 20_000)
    if starting_quota < 2_000:
        raise ValueError("proposed budget must include at least the 1k and 2k evaluation nodes")

    def estimate(quota: int) -> tuple[dict[str, float], list[int], list[int]]:
        nodes = [value for value in MILESTONES if value <= quota]
        if not {1000, 2000}.issubset(nodes):
            raise ValueError("registered budget must retain the 1k and 2k checkpoints")
        stages = [nodes[0], *[right - left for left, right in zip(nodes, nodes[1:])]]
        selfplay_cpu = sum(quota / rates[lr_id] for lr_id in sorted(LR_IDS))
        updates_per_candidate = quota // int(proposed["update_every_budgeted_states"])
        optimizer_cpu = sum(updates_per_candidate / update_rates[lr_id] for lr_id in sorted(LR_IDS))
        node_count = len(nodes)
        dev_candidate_cpu = (
            48 * node_count * sum(evaluation_cost[(mode, opp)] for mode in MODES for opp in OPPONENTS)
        )
        dev_champion_cpu = 48 * node_count * 2 * evaluation_cost[("puct", "F2")]
        dev_f2_cpu = 16 * sum(evaluation_cost[(mode, opp)] for mode in MODES for opp in ("A", "B"))
        blind_ab_cpu = 80 * sum(evaluation_cost[(mode, opp)] for mode in MODES for opp in ("A", "B"))
        turns_per_side = float(post_opening_moves) / 2.0
        blind_head_to_head_cpu = turns_per_side * (
            80 * mode_turn_cost["network"]
            + 160 * (mode_turn_cost["network"] + mode_turn_cost["puct"])
            + 80 * mode_turn_cost["puct"]
        )
        evaluation_cpu = dev_candidate_cpu + dev_champion_cpu + dev_f2_cpu + blind_ab_cpu + blind_head_to_head_cpu
        central = (selfplay_cpu + optimizer_cpu + evaluation_cpu) * 1.05
        return {
            "central_cpu_seconds": central,
            "optimistic_cpu_seconds": central * 0.75,
            "conservative_cpu_seconds": central * 1.25,
            "selfplay_cpu_seconds": selfplay_cpu,
            "optimizer_cpu_seconds": optimizer_cpu,
            "development_cpu_seconds": dev_candidate_cpu + dev_champion_cpu + dev_f2_cpu,
            "development_champion_cpu_seconds": dev_champion_cpu,
            "blind_cpu_seconds": blind_ab_cpu + blind_head_to_head_cpu,
        }, nodes, stages

    quota = starting_quota
    while True:
        estimates, nodes, stage_steps = estimate(quota)
        if estimates["conservative_cpu_seconds"] <= cap_seconds:
            break
        lower = [value for value in MILESTONES if 2000 <= value < quota]
        if not lower:
            raise ValueError("even the 2k-per-candidate budget exceeds the registered CPU cap")
        quota = lower[-1]
    stage_updates = [
        (nodes[index] // 16) - (nodes[index - 1] // 16 if index else 0)
        for index in range(len(nodes))
    ]
    config = {
        "format_version": 1,
        "registered": True,
        "experiment_id": str(proposed.get("experiment_id", "wuziq-alphazero-g-v2")),
        "base_commit": str(proposed.get("base_commit", "6e2bf48aa6b630c0fa4f20284ceb3b5aab295080")),
        "f2_sha256": F2_SHA256,
        "seed": int(proposed.get("seed", preflight.get("seed", schedule.get("seed", 510701)))),
        "candidate_learning_rates": dict(LEARNING_RATES),
        "candidates": {
            lr_id: {"learning_rate": LEARNING_RATES[lr_id], "states_per_candidate": quota,
                    "updates_per_candidate": quota // 16, "simulations_per_move": 64}
            for lr_id in sorted(LR_IDS)
        },
        "states_per_candidate": quota,
        "updates_per_candidate": quota // 16,
        "batch_size": int(proposed["batch_size"]),
        "update_every_budgeted_states": int(proposed["update_every_budgeted_states"]),
        "stage_nodes": nodes,
        "stage_environment_steps": stage_steps,
        "stage_updates": stage_updates,
        "replay_capacity": max(int(proposed["replay_capacity"]), quota),
        "simulation_budget": 64,
        "hard_cpu_hour_cap": float(proposed["hard_cpu_hour_cap"]),
        "evaluation": {
            "development_groups": len(schedule["development_openings"]),
            "development_nodes": nodes,
            "development_matches_per_candidate_node": 112,
            "blind_groups": len(schedule["blind_openings"]),
            "blind_conditions": ["F2-network", "F2-PUCT64", "G-F2-network", "G-F2-PUCT64"],
            "blind_games": 560,
            "b_time_limit_seconds": 0.05,
        },
        "estimates": estimates,
        "rates": {
            "selfplay_states_per_cpu_second_by_candidate": rates,
            "optimizer_updates_per_cpu_second_by_candidate": update_rates,
            "evaluation_cpu_seconds_per_full_game": {
                f"{mode}:{opp}": evaluation_cost[(mode, opp)] for mode in MODES for opp in OPPONENTS
            },
            "simulation_seconds_per_search": {
                str(sims): float(statistics.median(
                    float(row["mean_seconds"]) for row in benchmark_rows if int(row["simulations"]) == sims
                )) for sims in SIMULATIONS
            },
        },
        "schedule_sha256": schedule.get("schedule_sha256"),
        "preflight_commit": str(preflight.get("code_commit", "unknown")),
        "data_source": "own terminally labelled self-play only; no F labels, no teacher labels, no artificial rewards",
    }
    return config


def register_budget_in_config(preflight_path: Path, proposed_path: Path, schedule_path: Path, output_path: Path) -> dict[str, Any]:
    preflight = json.loads(Path(preflight_path).read_text())
    proposed = json.loads(Path(proposed_path).read_text())
    schedule = json.loads(Path(schedule_path).read_text())
    normalized = dict(proposed)
    normalized["states_per_candidate"] = proposed.get("proposed_states_per_candidate", proposed.get("states_per_candidate", 20_000))
    normalized["puct_simulations"] = proposed.get("puct_simulations", proposed.get("simulations_per_move", 64))
    registered = register_v2_budget(preflight, normalized, schedule)
    registered = {**proposed, **registered}
    registered["opening_schedule_path"] = proposed.get("opening_schedule_path", "research/g_v2/config/openings.json")
    registered["opening_schedule_sha256"] = schedule["schedule_sha256"]
    registered["requested_states_per_candidate"] = int(normalized["states_per_candidate"])
    registered["preflight_sha256"] = hashlib.sha256(Path(preflight_path).read_bytes()).hexdigest()
    registered["preflight_path"] = "research/g_v2/results/preflight/preflight.json"
    _atomic_json(Path(output_path), registered)
    return registered
