"""Pre-registration, schedule isolation and continuation audit for G V4."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .v3_schedule import build_v3_schedule, validate_v3_schedule
from .training import F2_SHA256

BASE_COMMIT = "2401e97b62b887fe66c9137815ac44b440d0fe9e"
BRANCH = "research/alphazero-g-v4"
V3_CHAMPION_ID = "G0-seed-510901@20000"
V3_CHAMPION_REL = "research/g_v3/checkpoints/G0/seed-510901/node_20000.pt"
V3_CHAMPION_SHA = "ed6e2e6b5a25f355db5b31a9c58ba22158af2b345f86b3bcd3add9c3a49da7f9"
SEEDS = [540401, 540402, 540403]
JOB_SEEDS = [*SEEDS, 540404, 540405]
DEFAULT_TRAIN_GROUPS = 100
DEVELOPMENT_GROUPS = 12
BLIND_GROUPS = 20
CPU_HOUR_CAP = 12.0
F2_REL = "research/f_v2/checkpoints/best_f2.pt"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def config_digest(config: dict[str, Any]) -> str:
    return _sha(_canonical({key: value for key, value in config.items() if key != "config_sha256"}))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)
    fd = __import__("os").open(path.parent, __import__("os").O_RDONLY)
    try:
        __import__("os").fsync(fd)
    finally:
        __import__("os").close(fd)


def _all_openings(schedule: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for groups in schedule.get("train_openings_by_seed", {}).values():
        rows.extend([dict(move) for move in group["opening"]] for group in groups)
    for key in ("development_openings", "blind_openings"):
        rows.extend([dict(move) for move in group["opening"]] for group in schedule.get(key, []))
    return rows


def build_v4_opening_schedule(root: str | Path) -> dict[str, Any]:
    """Reserve distinct training/development/blind openings away from all V3 sets."""
    root_path = Path(root).resolve()
    old = json.loads((root_path / "research/g_v3/config/openings.json").read_text(encoding="utf-8"))
    validate_v3_schedule(old)
    schedule = build_v3_schedule(
        JOB_SEEDS,
        train_groups_per_seed=DEFAULT_TRAIN_GROUPS,
        development_groups=DEVELOPMENT_GROUPS,
        blind_groups=BLIND_GROUPS,
        excluded_openings=_all_openings(old),
    )
    for groups in schedule["train_openings_by_seed"].values():
        for group in groups:
            group["opening_group_id"] = group["opening_group_id"].replace("g-v3-", "g-v4-")
    for key in ("development_openings", "blind_openings"):
        for group in schedule[key]:
            group["opening_group_id"] = group["opening_group_id"].replace("g-v3-", "g-v4-")
    schedule["source_experiment_id"] = "wuziq-alphazero-g-v4"
    schedule["excluded_v3_opening_count"] = len(_all_openings(old))
    schedule["schedule_sha256"] = _sha(_canonical({k: v for k, v in schedule.items() if k != "schedule_sha256"}))
    validate_v3_schedule(schedule)
    return schedule


def audit_v3_continuation(root: str | Path) -> dict[str, Any]:
    """Confirm the old 20K champion and whether its exact train state exists."""
    root_path = Path(root).resolve()
    checkpoint = root_path / V3_CHAMPION_REL
    if not checkpoint.is_file():
        raise FileNotFoundError(V3_CHAMPION_REL)
    digest = _sha(checkpoint.read_bytes())
    if digest != V3_CHAMPION_SHA:
        raise ValueError("V3 G0 @20K champion checkpoint SHA-256 mismatch")
    progress_path = root_path / "research/g_v3/checkpoints/G0/seed-510901/progress.json"
    state_path = root_path / "research/g_v3/checkpoints/G0/seed-510901/training_state.pt"
    if not progress_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("V3 continuation audit needs progress and latest train state")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    import torch
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state_progress = state.get("progress", {})
    has_exact_20k_state = (
        int(state_progress.get("budgeted_environment_steps", -1)) == 20_000
        and int(state.get("trainer", {}).get("update_count", -1)) == 1_250
    )
    if has_exact_20k_state:
        mode = "exact_optimizer_replay_restore"
    else:
        mode = "weight_inheritance_optimizer_reset_replay_rebuilt_from_first_20k_v3_states"
    result = {
        "v3_champion_id": V3_CHAMPION_ID,
        "checkpoint": V3_CHAMPION_REL,
        "checkpoint_sha256": digest,
        "checkpoint_verified": True,
        "latest_v3_training_state": "research/g_v3/checkpoints/G0/seed-510901/training_state.pt",
        "latest_training_state_budgeted_states": int(state_progress.get("budgeted_environment_steps", -1)),
        "latest_training_state_optimizer_updates": int(state.get("trainer", {}).get("update_count", -1)),
        "latest_training_state_replay_positions": int(state_progress.get("replay_positions", -1)),
        "latest_training_state_replay_games": int(state_progress.get("replay_games", -1)),
        "node_20000_training_state_available": has_exact_20k_state,
        "v4_continuation_mode": mode,
        "replay_source": "research/g_v3/results/selfplay/G0-seed-510901.jsonl",
        "replay_reconstruction_budgeted_states": 20_000,
        "v3_completed_states": int(progress.get("budgeted_environment_steps", -1)),
        "v3_optimizer_updates": int(progress.get("optimizer_updates", -1)),
        "no_exact_20k_state_claim": not has_exact_20k_state,
    }
    return result


def _estimate_runtime(
    preflight: dict[str, Any],
    *,
    states_max: int = 300_000,
    evaluation_games: int = 11_424,
) -> dict[str, Any]:
    selfplay = [row for row in preflight["selfplay_games"] if row.get("completed") is True]
    updates = [row for row in preflight["training_updates"] if row.get("completed") is True]
    evaluations = [row for row in preflight["evaluation_games"] if row.get("completed") is True]
    state_rates = [float(row["sample_positions"]) / float(row["cpu_seconds"])
                   for row in selfplay if float(row.get("cpu_seconds", 0)) > 0]
    update_costs = [float(row["cpu_seconds"]) / int(row["updates"])
                    for row in updates if int(row.get("updates", 0)) > 0]
    eval_costs = [float(row["cpu_seconds"]) for row in evaluations if float(row.get("cpu_seconds", 0)) > 0]
    if not state_rates or not update_costs or not eval_costs:
        raise ValueError("G V4 preflight lacks self-play, optimizer, or evaluation throughput")
    rate = statistics.median(state_rates)
    update_cost = statistics.median(update_costs)
    eval_cost = statistics.median(eval_costs)
    updates_total = states_max // 16
    states_cpu = states_max / rate
    optimizer_cpu = updates_total * update_cost
    evaluation_cpu = evaluation_games * eval_cost
    fixed_overhead = max(600.0, 0.06 * (states_cpu + optimizer_cpu + evaluation_cpu))
    central = states_cpu + optimizer_cpu + evaluation_cpu + fixed_overhead
    optimistic = states_max / max(state_rates) + updates_total * min(update_costs) + evaluation_games * min(eval_costs) + fixed_overhead
    conservative = states_max / (rate * 0.72) + updates_total * max(update_costs) * 1.5 + evaluation_games * max(eval_costs) * 1.5 + fixed_overhead * 1.4
    return {
        "maximum_budgeted_states": states_max,
        "evaluation_game_estimate": evaluation_games,
        "measured_state_rate_median_per_cpu_second": rate,
        "measured_optimizer_cpu_seconds_per_update_median": update_cost,
        "measured_evaluation_cpu_seconds_per_game_median": eval_cost,
        "estimated_optimizer_updates": updates_total,
        "cpu_seconds": {
            "optimistic": optimistic,
            "central": central,
            "conservative": conservative,
        },
        "hours": {key: value / 3600 for key, value in {
            "optimistic": optimistic, "central": central, "conservative": conservative,
        }.items()},
        "fixed_overhead_cpu_seconds": fixed_overhead,
        "basis": "Actual V4 preflight medians; conservative assumes 28% lower self-play throughput, 1.5x measured update/evaluation costs, and 40% overhead margin. Includes up to 300K new self-play states, all maximum registered development checkpoints, and 720 final blind games.",
    }


def register_v4_config(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    import subprocess
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root_path, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise ValueError(f"V4 registration requires {BRANCH}")
    schedule_path = root_path / "research/g_v4/config/openings.json"
    preflight_path = root_path / "research/g_v4/results/preflight/preflight.json"
    audit_path = root_path / "research/g_v4/results/preflight/v3_continuation_audit.json"
    if not all(path.is_file() for path in (schedule_path, preflight_path, audit_path)):
        raise FileNotFoundError("schedule, measured preflight and V3 continuation audit are required")
    schedule_bytes, preflight_bytes, audit_bytes = (
        path.read_bytes() for path in (schedule_path, preflight_path, audit_path)
    )
    schedule, preflight, audit = (json.loads(data) for data in (schedule_bytes, preflight_bytes, audit_bytes))
    validate_v3_schedule(schedule)
    if schedule.get("seeds") != JOB_SEEDS:
        raise ValueError("opening schedule does not reserve every V4 job seed")
    resources = preflight.get("resources", {})
    if int(resources.get("affinity_cpus", 0)) < 1 or int(resources.get("memory_available_bytes", 0)) < 1:
        raise ValueError("resource snapshot is incomplete")
    if {int(row.get("simulations", 0)) for row in preflight.get("simulation_benchmarks", [])} < {16, 32, 64}:
        raise ValueError("preflight must measure PUCT-16/32/64")
    if not preflight.get("selfplay_games") or not preflight.get("training_updates") or not preflight.get("evaluation_games"):
        raise ValueError("preflight lacks actual self-play, optimizer or evaluation measurements")
    if not audit.get("checkpoint_verified"):
        raise ValueError("V3 champion weight verification is required")
    if not (root_path / V3_CHAMPION_REL).is_file() or _sha((root_path / V3_CHAMPION_REL).read_bytes()) != V3_CHAMPION_SHA:
        raise ValueError("V3 continuation checkpoint changed after audit")
    estimate = _estimate_runtime(preflight)
    if estimate["hours"]["conservative"] > CPU_HOUR_CAP:
        raise ValueError("conservative estimate exceeds the registered 12 CPU-hour cap")
    payload: dict[str, Any] = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4",
        "registered": True,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "board_size": 9,
        "initialization": "G0 random weights; three new seeds, plus V3 G0@20K warm continuation",
        "network": {"channels": 32, "residual_blocks": 2, "policy_actions": 81, "value_head": True},
        "training": {
            "algorithm": "AlphaZero-style self-play reinforcement learning",
            "simulation_budget": 64,
            "policy_target": "PUCT root visit distribution",
            "value_target": "terminal outcome from side-to-move perspective",
            "teacher_labels": False,
            "imitation_samples": False,
            "handcrafted_tactical_rewards": False,
            "bn_policy": "freeze",
            "batch_size": 16,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "scheduler": "CosineAnnealingLR",
            "update_every_budgeted_states": 16,
            "replay_capacity_positions": 20_000,
            "temperature_moves": 12,
            "late_temperature": 0.1,
            "opening_mix": {"empty": 0.5, "random4": 0.5},
        },
        "stages": {
            "A": {"new_random_seeds": SEEDS, "states_each": 30_000, "states_total": 90_000},
            "B": {"start_checkpoint": V3_CHAMPION_ID, "new_states": 100_000,
                  "continuation_mode": audit["v4_continuation_mode"]},
            "C": {"maximum_new_states": 110_000, "requires_A_candidate_to_pass_v3_champion_gate": True,
                  "early_stop_after_consecutive_below_50pct_challenges": 3},
            "maximum_total_new_states": 300_000,
            "minimum_planned_new_states": 190_000,
        },
        "evaluation": {
            "development_opening_groups": DEVELOPMENT_GROUPS,
            "blind_opening_groups": BLIND_GROUPS,
            "paired_color_games_per_group": 2,
            "candidate_modes": ["network", "puct64"],
            "opponents": ["own_start", "stage_champion", "G0-V3-20K", "A", "B", "F2"],
            "B_time_limit_seconds": 0.05,
            "promotion": {"score_rate_at_least": 0.625, "paired_ci95_lower_above": 0.5,
                          "each_color_at_least": 0.5, "groups": DEVELOPMENT_GROUPS},
            "blind_used_for_selection": False,
            "estimated_development_and_blind_games": 11_424,
        },
        "seed_plan": {"new_random_model_seeds": SEEDS, "continuation_job_seeds": JOB_SEEDS[3:],
                      "opening_schedule_seeds": JOB_SEEDS},
        "f2_sha256": F2_SHA256,
        "v3_champion": audit,
        "opening_schedule_path": "research/g_v4/config/openings.json",
        "opening_schedule_sha256": _sha(schedule_bytes),
        "preflight_path": "research/g_v4/results/preflight/preflight.json",
        "preflight_sha256": _sha(preflight_bytes),
        "continuation_audit_path": "research/g_v4/results/preflight/v3_continuation_audit.json",
        "continuation_audit_sha256": _sha(audit_bytes),
        "hard_cpu_hour_cap": CPU_HOUR_CAP,
        "runtime_estimate": estimate,
        "runtime_estimate_phase_breakdown": {
            "maximum_development_games": 10_704,
            "maximum_blind_games": 720,
            "tactical_checkpoint_probes": "at most two 32-position exact-rule evaluations per initial/training milestone checkpoint; separately timed and charged in the CPU ledger",
        },
        "estimated_resources": {
            "cpu": f"{resources.get('affinity_cpus')} available CPU cores, single-threaded training/search",
            "peak_rss_bytes_measured_preflight": preflight.get("peak_rss_bytes"),
            "disk_free_bytes_measured_preflight": resources.get("disk_free_bytes"),
            "storage_reserve_bytes": 2_000_000_000,
        },
    }
    payload["config_sha256"] = config_digest(payload)
    _atomic_json(root_path / "research/g_v4/config/experiment.json", payload)
    return payload


def load_v4_config(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = json.loads((root_path / "research/g_v4/config/experiment.json").read_text(encoding="utf-8"))
    if config.get("registered") is not True or config.get("branch") != BRANCH:
        raise ValueError("G V4 experiment is not registered on its branch")
    if config.get("base_commit") != BASE_COMMIT or config.get("config_sha256") != config_digest(config):
        raise ValueError("G V4 registration/base commit hash mismatch")
    for key in ("opening_schedule", "preflight", "continuation_audit"):
        path_key = {"opening_schedule": "opening_schedule_path", "preflight": "preflight_path",
                    "continuation_audit": "continuation_audit_path"}[key]
        sha_key = {"opening_schedule": "opening_schedule_sha256", "preflight": "preflight_sha256",
                   "continuation_audit": "continuation_audit_sha256"}[key]
        path = root_path / config[path_key]
        if not path.is_file() or _sha(path.read_bytes()) != config[sha_key]:
            raise ValueError(f"registered V4 {key} evidence changed")
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    return config

