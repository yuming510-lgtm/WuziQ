"""Measured resource and evaluation-cost preflight for the registered V6.7 plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
from . import v5_evaluation as v5_eval
from . import v6_7_evaluation as evaluation
from . import v6_7_training as training


def run_preflight(root: str | Path = training.ROOT) -> dict:
    root = Path(root).resolve()
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    resources = training.preflight_resources(root)
    forecast = training.historical_training_forecast(root)
    pools = evaluation.create_opening_pools(root)
    preflight_seed = 670_700
    attempt = 0
    while True:
        attempt += 1
        opening = v5_eval.generate_opening_pool(size=9, n_in_row=5, groups=1,
                                                plies=6, seed=preflight_seed + attempt - 1)[0]
        key = v5_eval.opening_key(opening["moves"], 9)
        registered = {v5_eval.opening_key(row["moves"], 9)
                      for row in pools["development"] + pools["blind"]}
        if key not in registered:
            break
        if attempt > 100:
            raise RuntimeError("unable to find a disjoint preflight opening")
    baseline_node = evaluation._model_path(root, training.BASE_GAMES)
    baseline_sha = training.sha_file(baseline_node)
    opponents = ("G8000", "A", "C@50ms", "C@200ms")
    pilots = []
    pilot_cpu_started = time.process_time()
    for index, opponent in enumerate(opponents):
        record = evaluation.play_match(
            root=root,
            game_id=f"PREFLIGHT_G8000_vs_{opponent.replace('@', '').replace('-', '')}",
            candidate_games=training.BASE_GAMES,
            opponent_name=opponent,
            opening=opening,
            candidate_color="B",
            seed=670_710 + index,
            candidate_sha256=baseline_sha,
            opponent_sha256=baseline_sha if opponent == "G8000" else "not_applicable",
        )
        pilots.append(record)
        training.record_aux_cpu(root, "evaluation_cpu_seconds", float(record["total_cpu_seconds"]),
                                unique_id=f"PREFLIGHT1_{record['game_id']}")
    per_opponent = {row["opponent"]: float(row["total_cpu_seconds"]) for row in pilots}
    eval_games_per_opponent = 100
    estimated_eval_cpu = sum(per_opponent.values()) / len(per_opponent) * eval_games_per_opponent
    # One sample per opponent is intentionally widened to show match-length variance.
    evaluation_forecast = {
        "games_per_opponent": eval_games_per_opponent,
        "planned_formal_games": eval_games_per_opponent * len(opponents),
        "pilot_cpu_seconds_per_game": per_opponent,
        "optimistic_cpu_hours": estimated_eval_cpu * 0.75 / 3600,
        "normal_cpu_hours": estimated_eval_cpu * 1.5 / 3600,
        "conservative_cpu_hours": estimated_eval_cpu * 2.5 / 3600,
        "pilot_opening_disjoint_from_registered_pools": True,
        "pilot_games_excluded_from_formal_results": True,
    }
    packaging_reserve = 4.0
    audit_reserve = 2.0
    totals = {
        "optimistic_cpu_hours": forecast["optimistic_training_cpu_hours"] + evaluation_forecast["optimistic_cpu_hours"] + packaging_reserve + audit_reserve,
        "normal_cpu_hours": forecast["expected_training_cpu_hours"] + evaluation_forecast["normal_cpu_hours"] + packaging_reserve + audit_reserve,
        "conservative_cpu_hours": forecast["conservative_training_cpu_hours"] + evaluation_forecast["conservative_cpu_hours"] + packaging_reserve + audit_reserve,
    }
    opening_hash = hashlib.sha256(
        json.dumps(pools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    registration_path = training.registration_path(root)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("opening_pools_sha256") not in (None, opening_hash):
        raise ValueError("pre-registered opening pool changed")
    registration["opening_pools_sha256"] = opening_hash
    registration["preflight"] = {"resources": resources, "training_forecast": forecast,
                                  "evaluation_forecast": evaluation_forecast,
                                  "total_forecast_cpu_hours": totals,
                                  "pilot_opening": opening,
                                  "pilot_cpu_seconds": time.process_time() - pilot_cpu_started}
    training.atomic_json(registration_path, registration)
    result = {
        "resources": resources,
        "training_forecast": forecast,
        "evaluation_forecast": evaluation_forecast,
        "total_forecast_cpu_hours": totals,
        "hard_cap_cpu_hours": 110,
        "opening_pool_sha256": opening_hash,
        "pilot_games": [{"game_id": row["game_id"], "opponent": row["opponent"],
                         "move_count": row["move_count"], "candidate_cpu_seconds": row["candidate_cpu_seconds"],
                         "opponent_cpu_seconds": row["opponent_cpu_seconds"],
                         "total_cpu_seconds": row["total_cpu_seconds"],
                         "candidate_score": row["points"]} for row in pilots],
        "preflight_cpu_seconds": time.process_time() - started_cpu,
        "preflight_wall_seconds": time.perf_counter() - started_wall,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    training.atomic_json(training.artifact_path(root) / "results/preflight.json", result)
    return result

