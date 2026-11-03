"""Build and freeze V4.2 search/training registration from immutable history."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .v3_schedule import _opening_digest
from .v4_2_config import config_sha256, validate_v4_2_config
from .v4_2_schedule import build_v4_2_schedule


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(
    root: Path,
    candidate_id: str,
    checkpoint_path: str,
    training_states: int,
    lineage: list[tuple[str, int]],
) -> dict[str, Any]:
    checkpoint = root / checkpoint_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"historical G0 checkpoint missing: {checkpoint_path}")
    segments = []
    for relative, states in lineage:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"historical G0 self-play log missing: {relative}")
        segments.append({"path": relative, "budgeted_states": int(states), "sha256": _sha256(path)})
    return {
        "candidate_id": candidate_id,
        "training_family": "G0",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint),
        "training_states": int(training_states),
        "lineage_segments": segments,
    }


def historical_candidate_registry(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    v3_510901 = "research/g_v3/results/selfplay/G0-seed-510901.jsonl"
    v3_510902 = "research/g_v3/results/selfplay/G0-seed-510902.jsonl"
    candidates = [
        _candidate(root_path, "G0-V3-510901-5K", "research/g_v3/checkpoints/G0/seed-510901/node_5000.pt", 5_000, [(v3_510901, 5_000)]),
        _candidate(root_path, "G0-V3-510901-10K", "research/g_v3/checkpoints/G0/seed-510901/node_10000.pt", 10_000, [(v3_510901, 10_000)]),
        _candidate(root_path, "G0-V3-510901-20K", "research/g_v3/checkpoints/G0/seed-510901/node_20000.pt", 20_000, [(v3_510901, 20_000)]),
        _candidate(root_path, "G0-V3-510902-30K", "research/g_v3/checkpoints/G0/seed-510902/node_30000.pt", 30_000, [(v3_510902, 30_000)]),
    ]
    for seed in ("540401", "540402", "540403"):
        candidates.append(_candidate(
            root_path, f"G0-V4-{seed}-30K", f"research/g_v4/checkpoints/A-G0-s{seed}/node_30000.pt", 30_000,
            [(f"research/g_v4/results/selfplay/A-G0-s{seed}.jsonl", 30_000)],
        ))
    candidates.append(_candidate(
        root_path, "G0-V4-CONT-100K",
        "research/g_v4/checkpoints/B-G0-v3-champion-continue/node_100000.pt", 100_000,
        [("research/g_v4/results/selfplay/B-G0-v3-champion-continue.jsonl", 100_000)],
    ))
    base_lineage = [(v3_510901, 20_000)]
    for arm in ("control_v4", "state_uniform", "winner_balanced", "low_lr"):
        candidates.append(_candidate(
            root_path, f"G0-V4_1-{arm}-10K",
            f"research/g_v4_1/checkpoints/{arm}/node_10000.pt", 30_000,
            [*base_lineage, (f"research/g_v4_1/results/selfplay/{arm}.jsonl", 10_000)],
        ))
    return candidates


def _collect_historical_openings(root: Path) -> list[list[dict[str, Any]]]:
    paths = [root / f"research/{version}/config/openings.json" for version in ("g_v3", "g_v4", "g_v4_1")]
    found: dict[str, list[dict[str, Any]]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            opening = value.get("opening")
            if isinstance(opening, list) and len(opening) == 4:
                found.setdefault(_opening_digest(opening), opening)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"historical G0 opening schedule missing: {path}")
        visit(json.loads(path.read_text(encoding="utf-8")))
    return list(found.values())


def create_v4_2_registration(
    root: str | Path,
    *,
    states_per_arm: int = 10_000,
    include_search_arm: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_path = Path(root).resolve()
    config_dir = root_path / "research/g_v4_2/config"
    config_dir.mkdir(parents=True, exist_ok=True)
    development_groups, blind_groups = 16, 20
    train_groups = 800
    schedule = build_v4_2_schedule(
        seed=742_201, train_groups_per_arm=train_groups,
        development_groups=development_groups, blind_groups=blind_groups,
        excluded_openings=_collect_historical_openings(root_path),
    )
    schedule_path = config_dir / "openings.json"
    schedule_path.write_text(json.dumps(schedule, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    baseline = {"search_variant": "baseline", "opponent_pool": "current", "sampling_mode": "uniform_game"}
    arms: dict[str, dict[str, str]] = {"baseline": dict(baseline)}
    if include_search_arm:
        arms["search_target"] = {**baseline, "search_variant": "selected_diagnostic"}
    arms["history_pool"] = {**baseline, "opponent_pool": "historical_g0"}
    candidates = historical_candidate_registry(root_path)
    historical_audit_path = root_path / "research/g_v4_1/results/diagnostics/historical_g0_checkpoint_audit.json"
    config: dict[str, Any] = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4-2",
        "branch": "research/alphazero-g-v4-2",
        "base_commit": "4dd0c449c7a4afe1a4f68b3e2a234cdd70e186df",
        "board_size": 9,
        "network": {"channels": 32, "residual_blocks": 2, "policy_actions": 81, "value_head": True},
        "compute": {
            "cpu_hour_cap": 5.0, "device": "cpu", "torch_threads": 1,
            "snapshot_intervals_hours": [1, 2, 4, 6, 10, 16, 26],
            "drive_folder_name": "WuziQ G V4.2 Recovery",
            "drive_folder_id": "1SU-tHwFH79oY1KtVXgK3VsTFR3ECk1jR",
        },
        "arms": arms,
        "training": {
            "states_per_arm": int(states_per_arm),
            "maximum_total_new_states": int(states_per_arm) * len(arms),
            "simulations_per_move": 64, "optimizer": "AdamW", "learning_rate": 1e-4,
            "weight_decay": 1e-4, "batch_size": 16, "update_every_states": 16,
            "checkpoint_every_states": 256,
            "replay_capacity_positions": 20_000, "replay_sampling": "uniform_game",
            "batchnorm": "freeze", "root_noise_fraction": 0.25,
            "temperature_moves": 12, "late_temperature": 0.1,
            "policy_target": "PUCT root visits", "optimizer_reset": True,
            "history_opponents": [
                "G0-V3-510901-5K", "G0-V3-510901-10K", "G0-V3-510901-20K",
                "G0-V4-540402-30K", "G0-V4_1-control_v4-10K",
            ],
        },
        "search_variants": {
            "baseline": {"c_puct": 1.5, "prior_smoothing_fraction": 0.0, "leaf_value_scale": 1.0},
            "c_puct_5": {"c_puct": 5.0, "prior_smoothing_fraction": 0.0, "leaf_value_scale": 1.0},
            "c_puct_10": {"c_puct": 10.0, "prior_smoothing_fraction": 0.0, "leaf_value_scale": 1.0},
            "prior_smoothing_10pct": {"c_puct": 1.5, "prior_smoothing_fraction": 0.10, "leaf_value_scale": 1.0},
            "value_neutral": {"c_puct": 1.5, "prior_smoothing_fraction": 0.0, "leaf_value_scale": 0.0},
        },
        "diagnostics": {
            "simulations": [64, 128, 256], "formal_simulations": 64,
            "tactical_seed": 742_202, "tactical_per_category": 16,
            "tactical_split": {"search_development_per_category": 8, "search_validation_per_category": 8},
            "historical_candidate_count": len(candidates),
            "historical_checkpoint_audit_path": "research/g_v4_1/results/diagnostics/historical_g0_checkpoint_audit.json",
            "historical_checkpoint_audit_sha256": _sha256(historical_audit_path),
            "search_selection": {
                "metric": "macro_exact_tactical_accuracy_across_categories_and_top3_history_models",
                "minimum_absolute_gain": 0.05,
                "blocking_accuracy_minimum_ratio_to_baseline": 0.90,
                "validation_minimum_absolute_gain": 0.0,
                "validation_after_selection": True,
                "tie_order": ["c_puct_5", "prior_smoothing_10pct", "value_neutral", "c_puct_10"],
            },
        },
        "evaluation": {
            "historical_groups": 16, "development_groups": development_groups,
            "stage_groups": development_groups,
            "blind_groups": blind_groups, "blind_used_for_selection": False,
            "paired_colors": ["B", "W"], "formal_simulations": 64,
            "opponents": ["A", "B", "G0-V3-510901-20K"],
            "promotion": {"score_rate_at_least": 0.625,
                          "paired_ci95_lower_above": 0.5,
                          "each_color_at_least": 0.5},
        },
        "historical_candidates": candidates,
        "historical_g0_training_only_sources": True,
        "forbidden_training_inputs": ["A", "B", "C", "E", "F", "F1", "F2", "teacher_labels", "handcrafted_tactical_reward"],
        "opening_schedule_path": "research/g_v4_2/config/openings.json",
        "opening_schedule_sha256": schedule["schedule_sha256"],
    }
    config["config_sha256"] = config_sha256(config)
    validate_v4_2_config(config, require_branch=True, root=root_path)
    config_path = config_dir / "experiment.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return config, schedule
