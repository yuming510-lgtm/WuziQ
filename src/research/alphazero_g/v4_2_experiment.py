"""Pre-registered frozen-checkpoint, search, training and final-blind V4.2 runner."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...gomoku import GomokuBoard
from ..imitation_f.model import PolicyValueNet, encode_board
from .evaluation import play_evaluation_match, validate_match_record
from .puct import PUCTConfig, PUCTSearch
from .selfplay import play_selfplay_game
from .training import load_g_checkpoint
from .v3_evaluation import summarize_v3_matches, v3_value_and_entropy_diagnostics
from .v4_2_config import load_v4_2_config
from .v4_2_diagnostics import (
    build_v4_2_tactical_set, exact_tactical_actions, puzzle_symmetry_key, validate_v4_2_puzzle,
)
from .v4_2_schedule import validate_v4_2_schedule
from .v4_2_storage import atomic_write_json, game_result_path, iter_games, load_json, persist_unique_game
from .v4_2_training import (
    V4SampleReplay, play_history_pool_game, prefix_selfplay_game, validate_history_pool_game,
)


def _stable_id(prefix: str, *parts: Any) -> str:
    joined = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(joined.encode()).hexdigest()[:24]}"


def _stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)


def _config_from_dict(config: dict[str, Any]) -> PUCTConfig:
    return PUCTConfig(
        c_puct=float(config["c_puct"]),
        prior_smoothing_fraction=float(config.get("prior_smoothing_fraction", 0.0)),
        leaf_value_scale=float(config.get("leaf_value_scale", 1.0)),
    )


def _opening_group(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(move) for move in group["opening"]]


_MODEL_CACHE: dict[str, PolicyValueNet] = {}


def _model_by_id(root: Path, config: dict[str, Any], candidate_id: str) -> PolicyValueNet:
    if candidate_id in _MODEL_CACHE:
        return _MODEL_CACHE[candidate_id]
    row = next((item for item in config["historical_candidates"] if item["candidate_id"] == candidate_id), None)
    if row is None:
        raise KeyError(f"unregistered historical G0 model: {candidate_id}")
    path = root / row["checkpoint_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["checkpoint_sha256"]:
        raise ValueError(f"historical G0 checkpoint changed after pre-registration: {candidate_id}")
    model = load_g_checkpoint(path, device="cpu")
    model.eval()
    model.checkpoint_metadata = {**dict(getattr(model, "checkpoint_metadata", {})),
                                 "candidate_id": candidate_id, "checkpoint_sha256": digest}
    _MODEL_CACHE[candidate_id] = model
    return model


def _match_record(
    root: Path,
    *,
    model: PolicyValueNet,
    candidate_id: str,
    opponent_id: str,
    opponent_name: str,
    group: dict[str, Any],
    color: str,
    mode: str = "puct",
    config: PUCTConfig | None = None,
    opponent_config: PUCTConfig | None = None,
    opponent_model_override: PolicyValueNet | None = None,
    namespace: str,
    phase: str,
    runtime: Any | None = None,
) -> dict[str, Any]:
    game_id = _stable_id("g42", namespace, mode, candidate_id, opponent_id, group["opening_group_id"], color,
                         None if config is None else config.__dict__,
                         None if opponent_config is None else opponent_config.__dict__)
    path = game_result_path(root, namespace, game_id)
    if path.exists():
        existing = load_json(path)
        validate_match_record(existing)
        return existing
    if runtime is not None:
        runtime.check(reserve_seconds=1.0)
    opponent_model = opponent_model_override
    if opponent_name == "MODEL" and opponent_model is None:
        opponent_model = _model_by_id(root, _ACTIVE_CONFIG, opponent_id)
    result = play_evaluation_match(
        model, game_id=game_id, opening_group_id=str(group["opening_group_id"]),
        arm_id=candidate_id, mode=mode, opponent=opponent_name,
        opening=_opening_group(group), agent_color=color, seed=_stable_seed(game_id),
        simulations=64, opponent_mode="puct" if opponent_name == "MODEL" else "network",
        opponent_simulations=64, config=config or PUCTConfig(), opponent_config=opponent_config,
        opponent_model=opponent_model, device="cpu",
    )
    result.update({
        "phase": phase, "candidate_checkpoint_id": candidate_id,
        "opponent_checkpoint_id": opponent_id, "opening_kind": group.get("kind", "random4"),
        "search_variant": (config or PUCTConfig()).__dict__,
        "opponent_search_variant": (opponent_config or config or PUCTConfig()).__dict__ if opponent_name == "MODEL" else None,
    })
    validate_match_record(result)
    persist_unique_game(root, namespace, result)
    if runtime is not None:
        runtime.save(phase, last_game_id=game_id, result_namespace=namespace)
    return result


def _tactical_record(model: PolicyValueNet, puzzle: dict[str, Any], config: PUCTConfig,
                     runtime: Any | None = None) -> dict[str, Any]:
    if runtime is not None:
        runtime.check(reserve_seconds=0.2)
    board = GomokuBoard(9)
    expected = "B"
    for move in puzzle["position"]:
        if move["player"] != expected:
            raise ValueError("registered tactical puzzle is not alternating")
        board.place_stone(expected, (int(move["row"]), int(move["col"])))
        if board.check_victory((int(move["row"]), int(move["col"])), expected):
            raise ValueError("tactical puzzle reached a terminal state before its decision")
        expected = "W" if expected == "B" else "B"
    if expected != puzzle["to_move"]:
        raise ValueError("registered puzzle side-to-move changed")
    exact = exact_tactical_actions(puzzle)
    encoded = encode_board(board, puzzle["to_move"])
    legal = encoded[2].reshape(-1).bool()
    model.eval()
    with torch.inference_mode():
        logits, value = model(encoded.unsqueeze(0))
        masked = logits[0].masked_fill(~legal, -1e9)
        probabilities = torch.softmax(masked, dim=-1).cpu().numpy()
    policy_action = int(np.argmax(probabilities))
    search = PUCTSearch(model, config=config, device="cpu")
    result = search.search(board, puzzle["to_move"], 64, add_root_noise=False,
                           temperature=0.0, seed=_stable_seed(puzzle["puzzle_id"], config.__dict__))
    correct = int(result.move[0] * 9 + result.move[1]) in exact
    optimal_probability = float(np.sum(probabilities[exact]))
    visit_probability = float(np.sum(result.visit_policy[exact]))
    return {
        "puzzle_id": puzzle["puzzle_id"], "category": puzzle["category"],
        "position_symmetry_key": puzzle_symmetry_key(puzzle),
        "optimal_actions": exact, "label_source": puzzle["label_source"],
        "network_action": policy_action, "network_correct": policy_action in exact,
        "policy_prior_mass_on_optimal": optimal_probability,
        "network_value": float(value[0].clamp(-1, 1).cpu()),
        "network_policy_entropy": float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))),
        "puct_action": int(result.move[0] * 9 + result.move[1]), "puct_correct": correct,
        "root_prior_mass_on_optimal": float(np.sum(result.root_priors[exact])),
        "root_visit_mass_on_optimal": visit_probability,
        "root_value": float(result.root_value), "simulations": int(result.simulations),
        "tree_nodes": int(result.tree_nodes), "max_depth": int(result.max_depth),
        "network_evaluations": int(result.network_evaluations),
    }


def _tactical_summary(rows: list[dict[str, Any]], *, action_key: str = "puct_correct") -> dict[str, Any]:
    categories = {}
    for category in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "puzzles": len(subset), "correct": sum(bool(row[action_key]) for row in subset),
            "accuracy": float(np.mean([bool(row[action_key]) for row in subset])) if subset else None,
            "mean_optimal_prior_mass": float(np.mean([row["policy_prior_mass_on_optimal"] for row in subset])) if subset else None,
            "mean_root_visit_mass": float(np.mean([row["root_visit_mass_on_optimal"] for row in subset])) if subset else None,
            "mean_root_value": float(np.mean([row["root_value"] for row in subset])) if subset else None,
            "mean_network_value": float(np.mean([row["network_value"] for row in subset])) if subset else None,
            "mean_network_policy_entropy": float(np.mean([row["network_policy_entropy"] for row in subset])) if subset else None,
            "network_correct": sum(bool(row["network_correct"]) for row in subset),
            "network_accuracy": float(np.mean([bool(row["network_correct"]) for row in subset])) if subset else None,
            "mean_depth": float(np.mean([row["max_depth"] for row in subset])) if subset else None,
        }
    return {
        "puzzles": len(rows), "correct": sum(bool(row[action_key]) for row in rows),
        "accuracy": float(np.mean([bool(row[action_key]) for row in rows])) if rows else None,
        "categories": categories,
        "immediate_win_policy_prior_mass": float(np.mean([
            row["policy_prior_mass_on_optimal"] for row in rows
            if row["category"] in {"direct_win", "edge_win"}
        ])) if any(row["category"] in {"direct_win", "edge_win"} for row in rows) else None,
        "immediate_win_root_visit_mass": float(np.mean([
            row["root_visit_mass_on_optimal"] for row in rows
            if row["category"] in {"direct_win", "edge_win"}
        ])) if any(row["category"] in {"direct_win", "edge_win"} for row in rows) else None,
        "immediate_win_network_value_mae_vs_exact_win": float(np.mean([
            abs(float(row["network_value"]) - 1.0) for row in rows
            if row["category"] in {"direct_win", "edge_win"}
        ])) if any(row["category"] in {"direct_win", "edge_win"} for row in rows) else None,
    }


_ACTIVE_CONFIG: dict[str, Any] = {}
_ACTIVE_RUNTIME: Any | None = None


def run_v4_2_diagnostics(root: str | Path, runtime: Any | None = None) -> dict[str, Any]:
    global _ACTIVE_CONFIG, _ACTIVE_RUNTIME
    root_path = Path(root).resolve()
    config = load_v4_2_config(root_path)
    _ACTIVE_CONFIG = config
    _ACTIVE_RUNTIME = runtime
    schedule = load_json(root_path / config["opening_schedule_path"])
    validate_v4_2_schedule(schedule)
    puzzles_path = root_path / "research/g_v4_2/config/tactical_puzzles.json"
    if puzzles_path.exists():
        puzzles = load_json(puzzles_path)
        for puzzle in puzzles:
            validate_v4_2_puzzle(puzzle)
    else:
        puzzles = build_v4_2_tactical_set(
            seed=int(config["diagnostics"]["tactical_seed"]),
            per_category=int(config["diagnostics"]["tactical_per_category"]),
        )
        atomic_write_json(puzzles_path, puzzles)
    keys = [puzzle_symmetry_key(row) for row in puzzles]
    if len(keys) != len(set(keys)) or len(puzzles) != 64:
        raise ValueError("V4.2 tactical suite must contain 64 independent D4-distinct positions")

    baseline_cfg = _config_from_dict(config["search_variants"]["baseline"])
    history_rows = []
    champion_id = "G0-V3-510901-20K"
    champion = _model_by_id(root_path, config, champion_id)
    groups = schedule["development_openings"]
    for candidate in config["historical_candidates"]:
        candidate_id = candidate["candidate_id"]
        model = _model_by_id(root_path, config, candidate_id)
        for opponent_name, opponent_id in (("A", "A-rule"), ("MODEL", champion_id)):
            for group in groups:
                for color in ("B", "W"):
                    record = _match_record(
                        root_path, model=model, candidate_id=candidate_id, opponent_id=opponent_id,
                        opponent_name=opponent_name, group=group, color=color,
                        config=baseline_cfg, namespace="history", phase="historical_puct64",
                        runtime=runtime,
                    )
                    record = dict(record)
                    if opponent_name == "MODEL":
                        record["opponent"] = champion_id
                    history_rows.append(record)
    history_summary = {}
    for candidate in config["historical_candidates"]:
        candidate_id = candidate["candidate_id"]
        history_summary[candidate_id] = {}
        for opponent_id in ("A-rule", champion_id):
            subset = [row for row in history_rows if row["candidate_checkpoint_id"] == candidate_id
                      and row["opponent_checkpoint_id"] == opponent_id]
            history_summary[candidate_id][opponent_id] = summarize_v3_matches(subset)
    history_summary_path = root_path / "research/g_v4_2/results/diagnostics/history_match_summary.json"
    atomic_write_json(history_summary_path, history_summary)

    # Full 64-position report for each historical model; model selection does not use it.
    historical_tactics: dict[str, list[dict[str, Any]]] = {}
    for candidate in config["historical_candidates"]:
        candidate_id = candidate["candidate_id"]
        target = root_path / "research/g_v4_2/results/diagnostics/historical_tactics" / f"{candidate_id}.json"
        model = _model_by_id(root_path, config, candidate_id)
        rows = []
        row_dir = root_path / "research/g_v4_2/results/diagnostics/historical_tactic_rows" / candidate_id
        for puzzle in puzzles:
            row_path = row_dir / f"{puzzle['puzzle_id']}.json"
            if row_path.exists():
                row = load_json(row_path)
            else:
                row = _tactical_record(model, puzzle, baseline_cfg, runtime)
                atomic_write_json(row_path, row)
                if runtime is not None:
                    runtime.save("historical_tactics", current_candidate=candidate_id,
                                 current_puzzle=puzzle["puzzle_id"])
            rows.append(row)
        atomic_write_json(target, rows)
        if len(rows) != len(puzzles) or {row["puzzle_id"] for row in rows} != {row["puzzle_id"] for row in puzzles}:
            raise ValueError(f"historical tactical evaluation incomplete for {candidate_id}")
        historical_tactics[candidate_id] = rows

    # Pre-registered top-three are chosen only from paired development games.
    ranked = []
    for candidate_id, summary in history_summary.items():
        a_rate = summary["A-rule"]["score_rate"]
        champion_rate = summary[champion_id]["score_rate"]
        ranked.append((0.5 * (a_rate + champion_rate), candidate_id))
    top3 = [candidate_id for _, candidate_id in sorted(ranked, key=lambda item: (-item[0], item[1]))[:3]]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for puzzle in puzzles:
        by_category.setdefault(puzzle["category"], []).append(puzzle)
    per_split = int(config["diagnostics"]["tactical_split"]["search_development_per_category"])
    dev_puzzles = [puzzle for category in sorted(by_category) for puzzle in by_category[category][:per_split]]
    validation_puzzles = [puzzle for category in sorted(by_category) for puzzle in by_category[category][per_split:]]

    variant_names = [name for name in config["search_variants"] if name != "baseline"]
    dev_rows: dict[str, list[dict[str, Any]]] = {"baseline": []}
    for candidate_id in top3:
        base_rows = [row for row in historical_tactics[candidate_id]
                     if row["puzzle_id"] in {p["puzzle_id"] for p in dev_puzzles}]
        dev_rows["baseline"].extend(base_rows)
        model = _model_by_id(root_path, config, candidate_id)
        for variant in variant_names:
            target = root_path / "research/g_v4_2/results/diagnostics/search_dev" / f"{candidate_id}-{variant}.json"
            variant_cfg = _config_from_dict(config["search_variants"][variant])
            rows = []
            row_dir = root_path / "research/g_v4_2/results/diagnostics/search_dev_rows" / candidate_id / variant
            for puzzle in dev_puzzles:
                row_path = row_dir / f"{puzzle['puzzle_id']}.json"
                if row_path.exists():
                    row = load_json(row_path)
                else:
                    row = _tactical_record(model, puzzle, variant_cfg, runtime)
                    atomic_write_json(row_path, row)
                    if runtime is not None:
                        runtime.save("search_development_tactics", current_candidate=candidate_id,
                                     current_variant=variant, current_puzzle=puzzle["puzzle_id"])
                rows.append(row)
            dev_rows.setdefault(variant, []).extend(rows)

    metric_rows = {
        name: _tactical_summary(rows)
        for name, rows in dev_rows.items()
    }
    baseline_acc = metric_rows["baseline"]["accuracy"]
    baseline_block = float(np.mean([
        metric_rows["baseline"]["categories"][category]["accuracy"]
        for category in ("unique_block", "false_threat")
    ]))
    selection = config["diagnostics"]["search_selection"]
    candidates = []
    for variant in variant_names:
        metric = metric_rows[variant]
        blocking = float(np.mean([
            metric["categories"][category]["accuracy"]
            for category in ("unique_block", "false_threat")
        ]))
        candidates.append({
            "variant": variant, "development_macro_accuracy": metric["accuracy"],
            "development_gain": metric["accuracy"] - baseline_acc,
            "development_blocking_accuracy": blocking,
            "blocking_ratio_to_baseline": blocking / baseline_block if baseline_block > 0 else None,
        })
    tie_order = list(selection["tie_order"])
    candidates.sort(key=lambda row: (-row["development_macro_accuracy"], tie_order.index(row["variant"])))
    best_dev = candidates[0] if candidates else None
    proposed = "baseline"
    if best_dev and best_dev["development_gain"] >= float(selection["minimum_absolute_gain"]):
        ratio = best_dev["blocking_ratio_to_baseline"]
        if ratio is not None and ratio >= float(selection["blocking_accuracy_minimum_ratio_to_baseline"]):
            proposed = best_dev["variant"]

    validation_rows = {"baseline": [], proposed: []}
    val_ids = {row["puzzle_id"] for row in validation_puzzles}
    for candidate_id in top3:
        base_rows = [row for row in historical_tactics[candidate_id] if row["puzzle_id"] in val_ids]
        validation_rows["baseline"].extend(base_rows)
        if proposed != "baseline":
            target = root_path / "research/g_v4_2/results/diagnostics/search_validation" / f"{candidate_id}-{proposed}.json"
            model = _model_by_id(root_path, config, candidate_id)
            cfg = _config_from_dict(config["search_variants"][proposed])
            rows = []
            row_dir = root_path / "research/g_v4_2/results/diagnostics/search_validation_rows" / candidate_id / proposed
            for puzzle in validation_puzzles:
                row_path = row_dir / f"{puzzle['puzzle_id']}.json"
                if row_path.exists():
                    row = load_json(row_path)
                else:
                    row = _tactical_record(model, puzzle, cfg, runtime)
                    atomic_write_json(row_path, row)
                    if runtime is not None:
                        runtime.save("search_validation_tactics", current_candidate=candidate_id,
                                     current_variant=proposed, current_puzzle=puzzle["puzzle_id"])
                rows.append(row)
            validation_rows[proposed].extend(rows)
    validation_metrics = {name: _tactical_summary(rows) for name, rows in validation_rows.items()}
    validation_passed = proposed == "baseline"
    validation_gain = 0.0
    validation_blocking_ratio = 1.0
    if proposed != "baseline":
        base_metric, variant_metric = validation_metrics["baseline"], validation_metrics[proposed]
        validation_gain = variant_metric["accuracy"] - base_metric["accuracy"]
        base_block = float(np.mean([base_metric["categories"][name]["accuracy"]
                                    for name in ("unique_block", "false_threat")]))
        variant_block = float(np.mean([variant_metric["categories"][name]["accuracy"]
                                       for name in ("unique_block", "false_threat")]))
        validation_blocking_ratio = variant_block / base_block if base_block > 0 else None
        validation_passed = validation_gain >= float(selection.get("validation_minimum_absolute_gain", 0.0)) and (
            validation_blocking_ratio is not None
            and validation_blocking_ratio >= float(selection["blocking_accuracy_minimum_ratio_to_baseline"])
        )
    selected = proposed if validation_passed else "baseline"

    # Compare each search variant against the same frozen model running baseline PUCT-64.
    match_groups = schedule["development_openings"][:8]
    search_match_summary: dict[str, Any] = {}
    for candidate_id in top3:
        model = _model_by_id(root_path, config, candidate_id)
        search_match_summary[candidate_id] = {}
        for variant in variant_names:
            variant_cfg = _config_from_dict(config["search_variants"][variant])
            rows = []
            for group in match_groups:
                for color in ("B", "W"):
                    match = _match_record(
                        root_path, model=model, candidate_id=f"{candidate_id}|{variant}",
                        opponent_id=candidate_id, opponent_name="MODEL", group=group, color=color,
                        config=variant_cfg, opponent_config=baseline_cfg,
                        namespace="frozen_search_matches", phase="frozen_search_head_to_head",
                        runtime=runtime,
                    )
                    match["candidate_checkpoint_id"] = candidate_id
                    match["search_variant_name"] = variant
                    rows.append(match)
            search_match_summary[candidate_id][variant] = summarize_v3_matches(rows)

    # 128/256 are diagnostic only; the frozen G0 champion and top-three use the same positions.
    expanded_search_rows = []
    for candidate_id in top3:
        model = _model_by_id(root_path, config, candidate_id)
        for puzzle in dev_puzzles:
            board = GomokuBoard(9)
            expected = "B"
            for move in puzzle["position"]:
                board.place_stone(expected, (int(move["row"]), int(move["col"])))
                expected = "W" if expected == "B" else "B"
            for simulations in (64, 128, 256):
                row_path = (root_path / "research/g_v4_2/results/diagnostics/expanded_simulation_rows"
                            / candidate_id / puzzle["puzzle_id"] / f"{simulations}.json")
                if row_path.exists():
                    row = load_json(row_path)
                else:
                    if runtime is not None:
                        runtime.check(reserve_seconds=0.2)
                    cfg = _config_from_dict(config["search_variants"][selected])
                    result = PUCTSearch(model, cfg).search(
                        board, puzzle["to_move"], simulations, add_root_noise=False,
                        temperature=0.0, seed=_stable_seed(candidate_id, puzzle["puzzle_id"], simulations),
                    )
                    row = {
                    "candidate_id": candidate_id, "puzzle_id": puzzle["puzzle_id"],
                    "category": puzzle["category"], "simulations": simulations,
                    "action": int(result.move[0] * 9 + result.move[1]),
                    "correct": int(result.move[0] * 9 + result.move[1]) in exact_tactical_actions(puzzle),
                    "tree_nodes": result.tree_nodes, "max_depth": result.max_depth,
                    "root_value": result.root_value, "visit_entropy": result.visit_entropy,
                    "elapsed_seconds": result.elapsed_seconds,
                    }
                    atomic_write_json(row_path, row)
                    if runtime is not None:
                        runtime.save("expanded_search_diagnostics", current_candidate=candidate_id,
                                     current_puzzle=puzzle["puzzle_id"], current_simulations=simulations)
                expanded_search_rows.append(row)
    diagnostics = {
        "format_version": 1, "experiment_id": config["experiment_id"],
        "config_sha256": config["config_sha256"], "opening_schedule_sha256": schedule["schedule_sha256"],
        "historical_candidate_count": len(config["historical_candidates"]),
        "historical_match_summary": history_summary,
        "historical_tactical_summary": {candidate_id: _tactical_summary(rows) for candidate_id, rows in historical_tactics.items()},
        "top3_history_models_selected_on_development_games": top3,
        "tactical_split": {"development_ids": sorted(row["puzzle_id"] for row in dev_puzzles),
                           "validation_ids": sorted(row["puzzle_id"] for row in validation_puzzles)},
        "search_development_metrics": metric_rows,
        "search_development_candidates": candidates,
        "search_validation_metrics": validation_metrics,
        "search_selection": {
            "proposed": proposed, "selected": selected, "validation_passed": validation_passed,
            "validation_macro_gain": validation_gain,
            "validation_blocking_ratio_to_baseline": validation_blocking_ratio,
            "training_search_arm_active": selected != "baseline",
            "reason": "pre-registered exact tactical development and independent validation gates",
        },
        "frozen_search_match_summary": search_match_summary,
        "expanded_simulation_diagnostics": expanded_search_rows,
    }
    atomic_write_json(root_path / "research/g_v4_2/results/diagnostics/diagnostics.json", diagnostics)
    return diagnostics
