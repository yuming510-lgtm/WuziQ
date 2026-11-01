"""Hash-locked G V3 training registration."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Literal

from .v3_schedule import validate_v3_schedule
from .v3_preflight import estimate_v3_runtime

BASE_COMMIT = "c95eb5aa919c769af11ea6b7fc22df2594ae1c28"
F2_SHA256 = "50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257"
BRANCH = "research/alphazero-g-v3"
VALID_STATES = (10_000, 30_000, 50_000)
STAGE_CANDIDATES = (1_000, 2_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def evaluation_game_counts(
    *, seed_count: int, stage_count: int, development_groups: int,
    promotion_groups: int, blind_groups: int,
) -> dict[str, int]:
    """Count real distinct games; G-F2's initial checkpoint is frozen F2."""
    counts = (seed_count, stage_count, development_groups, promotion_groups, blind_groups)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
        raise ValueError("evaluation plan counts must be nonnegative integers")
    if min(seed_count, stage_count, development_groups, promotion_groups, blind_groups) < 1:
        raise ValueError("evaluation plan counts must be positive")
    distinct_matchups_per_route = {"G0": 5, "G-F2": 4}
    paired_modes = 2
    color_games = 2
    milestone_matchups = sum(distinct_matchups_per_route.values())
    baseline_matchups = 4 + 4
    development = (
        int(seed_count) * (int(stage_count) * milestone_matchups + baseline_matchups)
        * int(development_groups) * paired_modes * color_games
    )
    promotion = (
        int(seed_count) * int(stage_count) * 2 * int(promotion_groups) * color_games
    )
    blind = (
        int(seed_count) * sum(distinct_matchups_per_route.values()) * int(blind_groups)
        * paired_modes * color_games
    )
    return {"development": development, "promotion": promotion, "blind": blind,
            "total": development + promotion + blind}


def register_v3_budget(
    root: str | Path,
    *,
    seeds: list[int],
    states_per_seed: int,
    cpu_hour_cap: float,
    bn_policy: Literal["update", "freeze"],
) -> dict[str, Any]:
    """Write the only formal G V3 config after schedule and BN decisions are fixed."""
    root_path = Path(root).resolve()
    repository = _repo_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != BRANCH:
        raise ValueError(f"G V3 registration requires branch {BRANCH}")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("seeds must be a nonempty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if isinstance(states_per_seed, bool) or states_per_seed not in VALID_STATES:
        raise ValueError(f"states_per_seed must be one of {VALID_STATES}")
    if not math.isfinite(float(cpu_hour_cap)) or cpu_hour_cap <= 0:
        raise ValueError("CPU hour cap must be finite and positive")
    if bn_policy not in {"update", "freeze"}:
        raise ValueError("bn_policy must be 'update' or 'freeze'")

    schedule_path = root_path / "research/g_v3/config/openings.json"
    if not schedule_path.is_file():
        raise FileNotFoundError("registered G V3 opening schedule is missing")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    if list(schedule["seeds"]) != seeds:
        raise ValueError("V3 config seeds differ from the opening schedule")
    schedule_bytes = schedule_path.read_bytes()
    schedule_sha = _sha(schedule_bytes)

    preflight_path = root_path / "research/g_v3/results/preflight/preflight.json"
    bn_probe_path = root_path / "research/g_v3/results/bn_probe/probe.json"
    if not preflight_path.is_file() or not bn_probe_path.is_file():
        raise FileNotFoundError("measured G V3 preflight and BN probe results are required before registration")
    preflight_bytes = preflight_path.read_bytes()
    bn_probe_bytes = bn_probe_path.read_bytes()
    preflight = json.loads(preflight_bytes)
    bn_probe = json.loads(bn_probe_bytes)
    if int(preflight.get("resources", {}).get("affinity_cpus", 0)) < 1:
        raise ValueError("preflight does not contain a measured CPU resource snapshot")
    if not {16, 32, 64}.issubset({int(row.get("simulations", 0)) for row in preflight.get("simulation_benchmarks", [])}):
        raise ValueError("preflight must measure 16/32/64 simulation PUCT searches")
    if {str(row.get("arm_id")) for row in preflight.get("selfplay_games", [])} != {"G0", "G-F2"}:
        raise ValueError("preflight must measure self-play throughput on both routes")
    if {str(row.get("arm_id")) for row in preflight.get("training_updates", [])} != {"G0", "G-F2"}:
        raise ValueError("preflight must measure training throughput on both routes")
    if not preflight.get("evaluation_games"):
        raise ValueError("preflight must contain completed evaluation-speed samples")
    if bn_probe.get("formal_training") is not False:
        raise ValueError("BN selection evidence is missing or was incorrectly marked as formal training")
    if bn_probe.get("selection_rule", {}).get("selected_common_mode") != bn_policy:
        raise ValueError("registered BN policy differs from the measured selection decision")
    state_rates = [
        float(row["sample_positions"]) / float(row["cpu_seconds"])
        for row in preflight["selfplay_games"]
        if int(row.get("sample_positions", 0)) > 0 and float(row.get("cpu_seconds", 0)) > 0
    ]
    update_costs = [
        float(row["cpu_seconds"]) / int(row["updates"])
        for row in preflight["training_updates"]
        if int(row.get("updates", 0)) > 0 and float(row.get("cpu_seconds", 0)) > 0
    ]
    evaluation_costs = [
        float(row["cpu_seconds"])
        for row in preflight["evaluation_games"]
        if row.get("completed") is True and float(row.get("cpu_seconds", 0)) > 0
    ]
    if not state_rates or not update_costs or not evaluation_costs:
        raise ValueError("preflight throughput samples are incomplete")
    estimate = estimate_v3_runtime(
        states_per_cpu_second=float(statistics.median(state_rates)),
        update_cpu_seconds=float(statistics.median(update_costs)),
        evaluation_cpu_seconds_per_game=float(statistics.median(evaluation_costs)),
        states_per_seed=states_per_seed,
        seed_count=len(seeds),
        route_count=2,
        total_evaluation_games=4000,
    )
    counts = evaluation_game_counts(
        seed_count=len(seeds), stage_count=6, development_groups=4,
        promotion_groups=12, blind_groups=20,
    )
    estimate["evaluation_games"] = counts["total"]
    estimate["evaluation_games_by_phase"] = counts
    estimate["estimate_basis"] = (
        "development: 6 trained nodes x 2 seeds x (5 G0 + 4 G-F2 distinct matchups), "
        "plus node0 x 2 seeds x 8 baseline matchups, all x 4 groups x 2 modes x 2 colors; "
        "promotion: 6 nodes x 2 routes x 2 seeds x "
        "12 groups x 2 colors; blind: 2 seeds x (5 G0 + 4 G-F2 matchups) x "
        "20 groups x 2 modes x 2 colors. G-F2 initial is the same frozen F2 checkpoint."
    )
    if float(estimate["conservative_cpu_seconds"]) > float(cpu_hour_cap) * 3600:
        raise ValueError("conservative measured runtime estimate exceeds the registered CPU-hour cap")

    f2_path = root_path / "research/f_v2/checkpoints/best_f2.pt"
    if not f2_path.is_file():
        raise FileNotFoundError("frozen F2 checkpoint is missing")
    f2_sha = _sha(f2_path.read_bytes())
    if f2_sha != F2_SHA256:
        raise ValueError(f"F2 checkpoint hash SHA-256 mismatch: {f2_sha}")

    payload: dict[str, Any] = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v3",
        "registered": True,
        "base_commit": BASE_COMMIT,
        "branch": BRANCH,
        "f2_sha256": F2_SHA256,
        "opening_schedule_path": "research/g_v3/config/openings.json",
        "opening_schedule_sha256": schedule_sha,
        "preflight_sha256": _sha(preflight_bytes),
        "bn_probe_sha256": _sha(bn_probe_bytes),
        "seeds": list(seeds),
        "routes": ["G0", "G-F2"],
        "board_size": 9,
        "channels": 32,
        "residual_blocks": 2,
        "simulation_budget": 64,
        "bn_policy": bn_policy,
        "states_per_seed": int(states_per_seed),
        "stage_nodes": [node for node in STAGE_CANDIDATES if node <= states_per_seed],
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "update_every_budgeted_states": 16,
        "replay_capacity": 20_000,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "training_opening_mix": {"empty": 0.5, "random4": 0.5},
        "evaluation": {
            "milestone_opening_groups_per_opponent": 4,
            "promotion_opening_groups": 12,
            "blind_opening_groups_per_matchup_seed": 20,
            "blind_games_per_matchup_seed": 40,
            "b_time_limit_seconds": 0.05,
            "modes": ["network", "puct64"],
            "opponents": ["A", "B", "F2", "own_initial", "other_route"],
            "opponent_aliases": {"G-F2:own_initial": "G-F2:F2@frozen"},
        },
        "promotion": {
            "min_score_rate": 0.625,
            "paired_bootstrap_lower_95_gt": 0.5,
            "min_color_score_rate": 0.5,
            "opening_groups": 12,
        },
        "bn_probe_selection": {
            "states_per_route": 256,
            "updates": 16,
            "opening_groups": 8,
            "tie_score_rate_margin": 0.125,
            "tie_default": "freeze",
        },
        "hard_cpu_hour_cap": float(cpu_hour_cap),
        "runtime_estimate": estimate,
    }
    config_sha = _sha(_canonical(payload))
    payload["config_sha256"] = config_sha
    target = root_path / "research/g_v3/config/experiment.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != payload:
            amended = dict(old)
            amended.pop("pre_training_amendments", None)
            old_digest = amended.pop("config_sha256", None)
            if old_digest == _sha(_canonical(amended)) and amended == payload:
                return old
            raise ValueError("G V3 config is already registered with different values")
        return old
    _atomic_json(target, payload)
    return payload


def amend_v3_evaluation_estimate_before_training(root: str | Path) -> dict[str, Any]:
    """Correct the evaluation count before training, without changing train budget."""
    root_path = Path(root).resolve()
    config_path = root_path / "research/g_v3/config/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_sha = config.get("config_sha256")
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    if old_sha != _sha(_canonical(unsigned)):
        raise ValueError("cannot amend a G V3 config with an invalid registered hash")
    formal_paths = (
        root_path / "research/g_v3/results/selfplay",
        root_path / "research/g_v3/results/evaluation",
        root_path / "research/g_v3/checkpoints/G0",
        root_path / "research/g_v3/checkpoints/G-F2",
    )
    if any(path.exists() and any(path.rglob("*.jsonl")) for path in formal_paths[:2]):
        raise ValueError("evaluation estimate can only be amended before formal matches/training")
    if any(path.exists() and any(path.rglob("training_state.pt")) for path in formal_paths[2:]):
        raise ValueError("evaluation estimate can only be amended before formal training")
    preflight_path = root_path / "research/g_v3/results/preflight/preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    counts = evaluation_game_counts(
        seed_count=len(config["seeds"]), stage_count=len(config["stage_nodes"]),
        development_groups=int(config["evaluation"]["milestone_opening_groups_per_opponent"]),
        promotion_groups=int(config["promotion"]["opening_groups"]),
        blind_groups=int(config["evaluation"]["blind_opening_groups_per_matchup_seed"]),
    )
    previous_evaluation_game_estimate = int(config["runtime_estimate"].get("evaluation_games", -1))
    if (int(config["runtime_estimate"].get("evaluation_games", -1)) == counts["total"]
            and config.get("evaluation", {}).get("opponent_aliases", {}).get(
                "G-F2:own_initial") == "G-F2:F2@frozen"):
        return config
    state_rates = [
        float(row["sample_positions"]) / float(row["cpu_seconds"])
        for row in preflight["selfplay_games"] if int(row.get("sample_positions", 0)) > 0
    ]
    update_costs = [
        float(row["cpu_seconds"]) / int(row["updates"])
        for row in preflight["training_updates"] if int(row.get("updates", 0)) > 0
    ]
    evaluation_costs = [float(row["cpu_seconds"]) for row in preflight["evaluation_games"]
                        if row.get("completed") is True]
    estimate = estimate_v3_runtime(
        states_per_cpu_second=float(statistics.median(state_rates)),
        update_cpu_seconds=float(statistics.median(update_costs)),
        evaluation_cpu_seconds_per_game=float(statistics.median(evaluation_costs)),
        states_per_seed=int(config["states_per_seed"]), seed_count=len(config["seeds"]),
        route_count=2, total_evaluation_games=counts["total"],
    )
    estimate["evaluation_games"] = counts["total"]
    estimate["evaluation_games_by_phase"] = counts
    estimate["estimate_basis"] = (
        "development: 6 trained nodes x 2 seeds x (5 G0 + 4 G-F2 distinct matchups), "
        "plus node0 x 2 seeds x 8 baseline matchups, all x 4 groups x 2 modes x 2 colors; "
        "promotion: 6 nodes x 2 routes x 2 seeds x "
        "12 groups x 2 colors; blind: 2 seeds x (5 G0 + 4 G-F2 matchups) x "
        "20 groups x 2 modes x 2 colors. G-F2 initial is the same frozen F2 checkpoint."
    )
    if float(estimate["conservative_cpu_seconds"]) > float(config["hard_cpu_hour_cap"]) * 3600:
        raise ValueError("corrected conservative runtime estimate exceeds the registered CPU cap")
    preflight_old_sha = _sha(preflight_path.read_bytes())
    preflight["runtime_estimate_10k_per_seed_each_route"]["evaluation_games"] = counts["total"]
    preflight["runtime_estimate_10k_per_seed_each_route"]["evaluation_games_by_phase"] = counts
    # Recompute the estimate from the same measured rates through the registered helper.
    preflight["runtime_estimate_10k_per_seed_each_route"] = estimate_v3_runtime(
        states_per_cpu_second=float(statistics.median(state_rates)),
        update_cpu_seconds=float(statistics.median(update_costs)),
        evaluation_cpu_seconds_per_game=float(statistics.median(evaluation_costs)),
        states_per_seed=10_000, seed_count=len(config["seeds"]), route_count=2,
        total_evaluation_games=counts["total"],
    )
    preflight["runtime_estimate_10k_per_seed_each_route"].update(
        {"evaluation_games": counts["total"], "evaluation_games_by_phase": counts,
         "estimate_basis": estimate["estimate_basis"]}
    )
    preflight["evaluation_games_estimate_basis"] = {
        **counts,
        "note": "Before formal training, corrected the schedule count to include node-0 baselines and every promotion while counting G-F2's identical initial/F2 checkpoint once.",
    }
    _atomic_json(preflight_path, preflight)
    config["runtime_estimate"] = estimate
    config["evaluation"]["opponent_aliases"] = {"G-F2:own_initial": "G-F2:F2@frozen"}
    config["preflight_sha256"] = _sha(preflight_path.read_bytes())
    reason = (
        "The initial estimate omitted promotion games and counted G-F2's initial checkpoint separately from its identical frozen F2 weights."
        if previous_evaluation_game_estimate == 3520 else
        "The schedule estimate was extended to include node-0 development baselines; training budgets and evaluation rules remain unchanged."
    )
    config.setdefault("pre_training_amendments", []).append({
        "previous_config_sha256": old_sha,
        "previous_preflight_sha256": preflight_old_sha,
        "reason": reason,
        "formal_training_had_started": False,
        "previous_evaluation_game_estimate": previous_evaluation_game_estimate,
        "corrected_evaluation_game_estimate": counts["total"],
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    })
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    config["config_sha256"] = _sha(_canonical(unsigned))
    _atomic_json(config_path, config)
    return config
