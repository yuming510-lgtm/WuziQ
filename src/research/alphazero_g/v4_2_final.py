"""Development-only candidate selection and once-only V4.2 blind evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .puct import PUCTConfig
from .training import load_g_checkpoint, save_g_checkpoint
from .v3_evaluation import summarize_v3_matches
from .v4_2_config import load_v4_2_config
from .v4_2_experiment import _ACTIVE_CONFIG, _match_record, _model_by_id
from .v4_2_runtime import V4_2Runtime
from .v4_2_storage import atomic_write_json, game_result_path, load_json, sha256_file


CHAMPION_ID = "G0-V3-510901-20K"


def _paired_difference(candidate_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = {(row["opening_group_id"], row["agent_color"]): float(row["score_for_agent"])
                 for row in candidate_rows}
    reference = {(row["opening_group_id"], row["agent_color"]): float(row["score_for_agent"])
                 for row in reference_rows}
    if set(candidate) != set(reference):
        raise ValueError("paired blind score difference requires identical groups and colors")
    groups = sorted({group for group, _ in candidate})
    group_differences = np.asarray([
        np.mean([candidate[(group, color)] - reference[(group, color)] for color in ("B", "W")])
        for group in groups
    ], dtype=np.float64)
    rng = np.random.default_rng(742_229)
    sampled = rng.choice(group_differences, size=(20_000, len(group_differences)), replace=True).mean(axis=1)
    return {
        "paired_opening_groups": len(groups), "mean_score_rate_difference": float(group_differences.mean()),
        "paired_bootstrap_ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
    }


def evaluate_arm_final_head_to_head(
    root: str | Path, config: dict[str, Any], schedule: dict[str, Any],
    active_arms: list[str], runtime: V4_2Runtime,
) -> dict[str, Any]:
    from . import v4_2_experiment
    v4_2_experiment._ACTIVE_CONFIG = config
    output_path = Path(root) / "research/g_v4_2/results/training/arm_internal_summary.json"
    if output_path.exists():
        return load_json(output_path)
    models = {}
    for arm_id in active_arms:
        path = Path(root) / "research/g_v4_2/checkpoints" / arm_id / "node_10000.pt"
        if not path.is_file():
            raise FileNotFoundError(f"active V4.2 arm lacks its 10K checkpoint: {arm_id}")
        models[arm_id] = load_g_checkpoint(path, device="cpu")
    groups = schedule["development_openings"][:8]
    summary: dict[str, Any] = {}
    for index, challenger in enumerate(active_arms):
        for incumbent in active_arms[index + 1:]:
            rows = []
            for group in groups:
                for color in ("B", "W"):
                    match = v4_2_experiment._match_record(
                        Path(root), model=models[challenger], candidate_id=f"{challenger}@10000",
                        opponent_id=f"{incumbent}@10000", opponent_name="MODEL", group=group,
                        color=color, config=PUCTConfig(), opponent_config=PUCTConfig(),
                        opponent_model_override=models[incumbent], namespace="arm_internal",
                        phase="development_arm_head_to_head", runtime=runtime,
                    )
                    rows.append(match)
            summary[f"{challenger}_vs_{incumbent}"] = summarize_v3_matches(rows)
    result = {"groups": 8, "paired_colors": ["B", "W"], "comparisons": summary}
    atomic_write_json(output_path, result)
    return result


def freeze_final_candidate(
    root: str | Path, config: dict[str, Any], active_arms: list[str]
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = root_path / "research/g_v4_2/results/selection/final_candidate.json"
    if path.exists():
        existing = load_json(path)
        checkpoint = root_path / existing["checkpoint_path"]
        if sha256_file(checkpoint) != existing["checkpoint_sha256"]:
            raise ValueError("frozen V4.2 final-candidate checkpoint changed")
        return existing
    rows = []
    for arm_id in active_arms:
        for states in (1_000, 2_000, 5_000, 10_000):
            evaluation_path = root_path / "research/g_v4_2/results/training" / arm_id / f"node_{states}_evaluation.json"
            checkpoint_path = root_path / "research/g_v4_2/checkpoints" / arm_id / f"node_{states}.pt"
            if not evaluation_path.is_file() or not checkpoint_path.is_file():
                continue
            evaluation = load_json(evaluation_path)
            gate = evaluation["challenger_gate_vs_v3_g0_20k"]
            rows.append({
                "candidate_id": f"{arm_id}@{states}", "arm_id": arm_id, "training_states": states,
                "checkpoint_path": str(checkpoint_path.relative_to(root_path)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "promotion_gate_passed_on_development": bool(gate["promote"]),
                "score_vs_v3_champion": float(evaluation["matches"][CHAMPION_ID]["score_rate"]),
                "score_vs_A": float(evaluation["matches"]["A-rule"]["score_rate"]),
                "score_vs_B": float(evaluation["matches"]["B-50ms"]["score_rate"]),
                "tactical_accuracy": float(evaluation["tactical_search_puct64"]["accuracy"]),
            })
    if not rows:
        raise ValueError("V4.2 has no complete development candidate to freeze")
    passed = [row for row in rows if row["promotion_gate_passed_on_development"]]
    if passed:
        selected = max(passed, key=lambda row: (
            row["score_vs_v3_champion"], row["score_vs_A"], row["tactical_accuracy"], -row["training_states"], row["candidate_id"]
        ))
    else:
        selected = max(rows, key=lambda row: (
            (row["score_vs_A"] + row["score_vs_B"] + row["score_vs_v3_champion"]) / 3.0,
            row["tactical_accuracy"], -row["training_states"], row["candidate_id"]
        ))
    frozen_path = root_path / "research/g_v4_2/checkpoints/final_best_dev.pt"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_g_checkpoint(root_path / selected["checkpoint_path"], device="cpu")
    save_g_checkpoint(frozen_path, model, metadata={
        "model_family": "G", "candidate_id": selected["candidate_id"],
        "selection_source": "development_only", "config_sha256": config["config_sha256"],
        "not_a_new_champion_unless_blind_gate_passes": not bool(selected["promotion_gate_passed_on_development"]),
    })
    selected = dict(selected)
    selected.update({
        "checkpoint_path": str(frozen_path.relative_to(root_path)),
        "checkpoint_sha256": sha256_file(frozen_path),
        "development_promotion_gate_passed": bool(selected["promotion_gate_passed_on_development"]),
        "selection_candidates": rows,
        "blind_results_used_for_selection": False,
        "frozen_before_blind_evaluation": True,
    })
    atomic_write_json(path, selected)
    return selected


def run_v4_2_blind(root: str | Path, runtime: V4_2Runtime) -> dict[str, Any]:
    from . import v4_2_experiment

    root_path = Path(root).resolve()
    config = load_v4_2_config(root_path)
    v4_2_experiment._ACTIVE_CONFIG = config
    schedule = load_json(root_path / config["opening_schedule_path"])
    selection = freeze_final_candidate(
        root_path, config, load_json(root_path / "research/g_v4_2/results/training/summary.json")["active_arms"]
    )
    candidate = load_g_checkpoint(root_path / selection["checkpoint_path"], device="cpu")
    champion = _model_by_id(root_path, config, CHAMPION_ID)
    groups = schedule["blind_openings"]
    if len(groups) < 20:
        raise ValueError("V4.2 blind set must contain at least 20 held-out openings")
    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    candidate_id = str(selection["candidate_id"])
    for group in groups:
        for opponent_name, opponent_id in (("A", "A-rule"), ("B", "B-50ms"), ("MODEL", CHAMPION_ID)):
            for color in ("B", "W"):
                record = v4_2_experiment._match_record(
                    root_path, model=candidate, candidate_id=candidate_id,
                    opponent_id=opponent_id, opponent_name=opponent_name, group=group,
                    color=color, config=PUCTConfig(), namespace="blind_puct64",
                    phase="final_blind_puct64", opponent_model_override=champion if opponent_name == "MODEL" else None,
                    runtime=runtime,
                )
                rows_by_key[f"candidate_vs_{opponent_id}"] = rows_by_key.get(f"candidate_vs_{opponent_id}", []) + [record]
        for opponent_name, opponent_id in (("A", "A-rule"), ("B", "B-50ms")):
            for color in ("B", "W"):
                reference = v4_2_experiment._match_record(
                    root_path, model=champion, candidate_id=CHAMPION_ID,
                    opponent_id=opponent_id, opponent_name=opponent_name, group=group,
                    color=color, config=PUCTConfig(), namespace="blind_reference_puct64",
                    phase="blind_reference_v3_champion_puct64", runtime=runtime,
                )
                rows_by_key[f"v3_champion_vs_{opponent_id}"] = rows_by_key.get(f"v3_champion_vs_{opponent_id}", []) + [reference]
        for opponent_name, opponent_id in (("A", "A-rule"), ("B", "B-50ms")):
            for color in ("B", "W"):
                network = v4_2_experiment._match_record(
                    root_path, model=candidate, candidate_id=candidate_id,
                    opponent_id=opponent_id, opponent_name=opponent_name, group=group,
                    color=color, mode="network", config=PUCTConfig(), namespace="blind_network",
                    phase="final_blind_network_only", runtime=runtime,
                )
                rows_by_key[f"candidate_network_vs_{opponent_id}"] = rows_by_key.get(f"candidate_network_vs_{opponent_id}", []) + [network]

    summaries = {name: summarize_v3_matches(rows) for name, rows in rows_by_key.items()}
    paired_gains = {}
    for opponent_id in ("A-rule", "B-50ms"):
        paired_gains[f"candidate_vs_{opponent_id}"] = _paired_difference(
            rows_by_key[f"candidate_vs_{opponent_id}"], rows_by_key[f"v3_champion_vs_{opponent_id}"]
        )
        paired_gains[f"puct64_minus_network_vs_{opponent_id}"] = _paired_difference(
            rows_by_key[f"candidate_vs_{opponent_id}"], rows_by_key[f"candidate_network_vs_{opponent_id}"]
        )
    candidate_vs_champion = summaries[f"candidate_vs_{CHAMPION_ID}"]
    blind_gate = (
        candidate_vs_champion["score_rate"] > 0.5
        and candidate_vs_champion["paired_opening_bootstrap_ci95"] is not None
        and candidate_vs_champion["paired_opening_bootstrap_ci95"][0] > 0.5
    )
    result = {
        "format_version": 1, "config_sha256": config["config_sha256"],
        "opening_schedule_sha256": schedule["schedule_sha256"],
        "frozen_candidate": selection, "blind_groups": len(groups),
        "games": {key: len(value) for key, value in rows_by_key.items()},
        "summaries": summaries, "paired_score_differences": paired_gains,
        "candidate_beats_v3_champion_with_paired_ci_lower_above_50pct": bool(blind_gate),
        "champion_decision": candidate_id if (blind_gate and selection["development_promotion_gate_passed"]) else CHAMPION_ID,
        "champion_promotion_requires_development_gate": bool(selection["development_promotion_gate_passed"]),
        "blind_used_for_selection": False,
    }
    atomic_write_json(root_path / "research/g_v4_2/results/blind/summary.json", result)
    return result
