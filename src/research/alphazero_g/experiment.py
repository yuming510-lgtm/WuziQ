"""Pre-registered, deterministic opening schedules with history exclusions."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from ...gomoku import GomokuBoard


def _opening_digest(opening: list[dict[str, Any]]) -> str:
    normalized = [{"player": move["player"], "row": int(move["row"]), "col": int(move["col"])} for move in opening]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _random_opening(rng: np.random.Generator) -> list[dict[str, Any]]:
    board = GomokuBoard(9)
    opening: list[dict[str, Any]] = []
    player = "B"
    for _ in range(4):
        legal = [(row, col) for row in range(9) for col in range(9) if board.get((row, col)) is None]
        row, col = legal[int(rng.integers(0, len(legal)))]
        board.place_stone(player, (row, col))
        if board.check_victory((row, col), player):
            raise RuntimeError("generated four-ply opening unexpectedly terminated")
        opening.append({"player": player, "row": row, "col": col})
        player = "W" if player == "B" else "B"
    return opening


def build_g_schedule(
    seed: int,
    train_groups: int,
    development_groups: int,
    blind_groups: int,
    excluded_group_ids: Iterable[str] = (),
    excluded_openings: Iterable[list[dict[str, Any]]] = (),
) -> dict[str, Any]:
    """Create distinct legal four-ply train/dev/blind openings and hash them."""
    counts = (train_groups, development_groups, blind_groups)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
        raise ValueError("opening group counts must be nonnegative integers")
    if development_groups < 1 or blind_groups < 1:
        raise ValueError("development and blind schedules each need at least one group")
    excluded_openings = tuple(list(opening) for opening in excluded_openings)
    excluded_ids = set(str(value) for value in excluded_group_ids)
    excluded_digests = {_opening_digest(list(opening)) for opening in excluded_openings}
    rng = np.random.default_rng(int(seed))
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "development": [], "blind": []}
    seen_digests = set(excluded_digests)
    total = sum(counts)
    attempts = 0
    while sum(len(items) for items in groups.values()) < total:
        attempts += 1
        if attempts > max(10000, total * 100):
            raise RuntimeError("could not create a collision-free G opening schedule")
        opening = _random_opening(rng)
        digest = _opening_digest(opening)
        if digest in seen_digests:
            continue
        seed_value = int(rng.integers(0, 2**63 - 1))
        group_id = "g-open-" + hashlib.sha256(f"{int(seed)}:{seed_value}:{digest}".encode()).hexdigest()[:20]
        if group_id in excluded_ids:
            continue
        seen_digests.add(digest)
        group = {"opening_group_id": group_id, "opening_seed": seed_value, "opening": opening}
        for section, limit in zip(groups, counts, strict=True):
            if len(groups[section]) < limit:
                groups[section].append(group)
                break
    payload = {
        "format_version": 1,
        "seed": int(seed),
        "excluded_group_ids": sorted(excluded_ids),
        "train_openings": groups["train"],
        "development_openings": groups["development"],
        "blind_openings": groups["blind"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["schedule_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    validate_g_schedule(payload, excluded_openings=excluded_openings)
    return payload


def validate_g_schedule(
    schedule: dict[str, Any], *, excluded_openings: Iterable[list[dict[str, Any]]] = ()
) -> None:
    seen_ids: set[str] = set()
    seen_digests = {_opening_digest(list(opening)) for opening in excluded_openings}
    for key in ("train_openings", "development_openings", "blind_openings"):
        for group in schedule.get(key, []):
            group_id = group.get("opening_group_id")
            if not isinstance(group_id, str) or not group_id or group_id in seen_ids:
                raise ValueError("opening group ids must be unique and nonempty")
            seen_ids.add(group_id)
            board = GomokuBoard(9)
            expected = "B"
            opening = group.get("opening")
            if not isinstance(opening, list) or len(opening) != 4:
                raise ValueError("each G opening must contain exactly four plies")
            for move in opening:
                if move.get("player") != expected:
                    raise ValueError("G opening colors must alternate from black")
                coord = (int(move["row"]), int(move["col"]))
                board.place_stone(expected, coord)
                if board.check_victory(coord, expected):
                    raise ValueError("G opening is terminal")
                expected = "W" if expected == "B" else "B"
            digest = _opening_digest(opening)
            if digest in seen_digests:
                raise ValueError("G opening overlaps another or an excluded opening")
            seen_digests.add(digest)
    if set(schedule.get("excluded_group_ids", [])) & seen_ids:
        raise ValueError("G opening id collides with excluded historical groups")
    stored_hash = schedule.get("schedule_sha256")
    if stored_hash is not None:
        payload = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if stored_hash != expected:
            raise ValueError("G schedule hash does not match its contents")


def register_g_budget(
    preflight: dict[str, Any],
    schedule: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a matched finite budget only after real end-to-end measurements."""
    required_measurements = ("simulation_benchmarks", "selfplay_games", "training_updates")
    for name in required_measurements:
        values = preflight.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"preflight is missing measured {name}")
    validate_g_schedule(schedule)
    benchmark_counts = {int(item.get("simulations", 0)) for item in preflight["simulation_benchmarks"]}
    if not {16, 32, 64}.issubset(benchmark_counts):
        raise ValueError("preflight must measure 16, 32, and 64 PUCT simulations")
    positions_by_simulations: dict[int, set[str]] = {}
    for item in preflight["simulation_benchmarks"]:
        positions_by_simulations.setdefault(int(item["simulations"]), set()).add(str(item.get("position_id")))
    if any(len(positions_by_simulations.get(value, set())) < 3 for value in (16, 32, 64)):
        raise ValueError("each preflight simulation budget needs opening, middle, and tactical positions")
    arms_measured = {str(item.get("arm_id")) for item in preflight["selfplay_games"]}
    update_arms = {str(item.get("arm_id")) for item in preflight["training_updates"]}
    if arms_measured != {"G0", "G-F2"} or update_arms != {"G0", "G-F2"}:
        raise ValueError("preflight must include complete self-play and updates for both arms")
    for name in ("selfplay_games", "training_updates"):
        for item in preflight[name]:
            if item.get("completed", True) is not True:
                raise ValueError(f"preflight contains an incomplete measurement in {name}")
            elapsed = item.get("wall_seconds", item.get("elapsed_seconds"))
            if elapsed is None or not math.isfinite(float(elapsed)) or float(elapsed) <= 0:
                raise ValueError(f"preflight contains an invalid runtime in {name}")
    simulations = int(parameters.get("simulations_per_move", 0))
    environment_steps = int(parameters.get("environment_steps_per_arm", 0))
    updates = int(parameters.get("updates_per_arm", 0))
    batch_size = int(parameters.get("batch_size", 0))
    replay_capacity = int(parameters.get("replay_capacity", 0))
    stage_steps = parameters.get("stage_environment_steps")
    stage_updates = parameters.get("stage_updates")
    cpu_cap = parameters.get("cpu_hour_cap")
    if simulations not in benchmark_counts:
        raise ValueError("registered simulations must have been directly benchmarked")
    if min(environment_steps, updates, batch_size, replay_capacity) < 1:
        raise ValueError("environment steps, updates, batch size, and replay capacity must be positive")
    if not isinstance(stage_steps, list) or not stage_steps or sum(map(int, stage_steps)) != environment_steps:
        raise ValueError("stage environment steps must sum to the exact per-arm quota")
    if not isinstance(stage_updates, list) or len(stage_steps) != len(stage_updates) or sum(map(int, stage_updates)) != updates:
        raise ValueError("stage updates must match stages and sum to the per-arm update quota")
    if cpu_cap is None or not math.isfinite(float(cpu_cap)) or float(cpu_cap) <= 0:
        raise ValueError("a finite positive hard CPU-hour cap is required")
    rates = []
    for item in preflight["selfplay_games"]:
        steps = int(item.get("sample_positions", 0))
        if steps < 1:
            raise ValueError("self-play preflight measurement needs sample_positions")
        rates.append(steps / float(item.get("wall_seconds", item.get("elapsed_seconds"))))
    step_rate = float(np.median(rates))
    update_rates = [
        float(item.get("updates", 0)) / float(item.get("wall_seconds", item.get("elapsed_seconds")))
        for item in preflight["training_updates"]
    ]
    if any(rate <= 0 or not math.isfinite(rate) for rate in update_rates):
        raise ValueError("preflight update throughput must be positive")
    update_rate = float(np.median(update_rates))
    benchmark_reference = float(np.mean([
        item["mean_seconds"] for item in preflight["simulation_benchmarks"]
        if int(item["simulations"]) == 16
    ]))
    benchmark_selected = float(np.mean([
        item["mean_seconds"] for item in preflight["simulation_benchmarks"]
        if int(item["simulations"]) == simulations
    ]))
    simulation_cost_scale = benchmark_selected / max(benchmark_reference, 1e-9)
    development_nodes = int(parameters.get("development_checkpoint_nodes", 3))
    if development_nodes < 2:
        raise ValueError("at least two pre-registered development checkpoint nodes are required")
    games_per_development_node = 2 * len(schedule["development_openings"]) * 2 * 4 * 2
    blind_game_count = 2 * len(schedule["blind_openings"]) * 2 * 4 * 2
    development_game_count = games_per_development_node * development_nodes
    evaluation_games = development_game_count + blind_game_count
    # Scale full-game self-play throughput to the selected, directly benchmarked sim count.
    avg_steps = float(np.median([int(item["sample_positions"]) for item in preflight["selfplay_games"]]))
    avg_game_seconds = float(np.median([float(item.get("wall_seconds", item.get("elapsed_seconds"))) for item in preflight["selfplay_games"]]))
    full_game_seconds = avg_game_seconds / max(avg_steps, 1.0) * environment_steps * 2 * simulation_cost_scale
    updates_seconds = updates * 2 / update_rate
    evaluation_seconds = evaluation_games * avg_game_seconds * 0.5 * simulation_cost_scale
    central_seconds = full_game_seconds + updates_seconds + evaluation_seconds
    conservative_seconds = central_seconds * 2.0
    optimistic_seconds = central_seconds * 0.65
    if conservative_seconds > float(cpu_cap) * 3600:
        raise ValueError("conservative registered run estimate exceeds the hard CPU-hour cap")
    return {
        "format_version": 1,
        "registered": True,
        "seed": int(parameters.get("seed", schedule["seed"])),
        "arms": {
            "G0": {"simulations_per_move": simulations, "environment_steps": environment_steps, "updates": updates},
            "G-F2": {"simulations_per_move": simulations, "environment_steps": environment_steps, "updates": updates},
        },
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "stage_environment_steps": list(map(int, stage_steps)),
        "stage_updates": list(map(int, stage_updates)),
        "evaluation": {
            "opponents": ["A", "B", "F1", "F2"],
            "modes": ["network", "puct"],
            "development_opening_groups": len(schedule["development_openings"]),
            "blind_opening_groups": len(schedule["blind_openings"]),
        "development_checkpoint_nodes": development_nodes,
        "development_games_per_checkpoint": games_per_development_node,
        "development_games_total": development_game_count,
            "blind_games_per_selected_checkpoint": blind_game_count,
            "total_games_per_checkpoint_node": evaluation_games,
            "puct_simulations": simulations,
            "blind_runs": 1,
        },
        "hard_cpu_hour_cap": float(cpu_cap),
        "estimates": {
            "optimistic_wall_seconds": optimistic_seconds,
            "central_wall_seconds": central_seconds,
            "conservative_wall_seconds": conservative_seconds,
            "selfplay_sample_positions_per_second": step_rate,
            "optimizer_updates_per_second": update_rate,
            "simulation_cost_scale_from_16": simulation_cost_scale,
            "development_plus_blind_games": evaluation_games,
        },
        "schedule_sha256": schedule["schedule_sha256"],
        "f2_sha256": "50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257",
        "training_data_source": "own completed self-play only; no F supervised samples or teacher labels",
    }
