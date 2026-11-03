"""Checkpointed, exact-budget three-arm V4.2 G0 training and stage evaluation."""
from __future__ import annotations

import copy
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..imitation_f.model import PolicyValueNet
from .puct import PUCTConfig
from .selfplay import play_selfplay_game
from .training import (
    GTrainer, load_g_checkpoint, load_training_state, save_g_checkpoint, save_training_state,
)
from .v3_evaluation import promotion_decision, summarize_v3_matches
from .v4_2_config import load_v4_2_config
from .v4_2_experiment import (
    _ACTIVE_CONFIG, _MODEL_CACHE, _config_from_dict, _match_record, _model_by_id,
    _stable_id, _stable_seed, _tactical_record, _tactical_summary,
)
from .v4_2_runtime import BudgetExhausted, V4_2Runtime
from .v4_2_storage import atomic_write_json, game_result_path, iter_games, load_json, persist_unique_game
from .v4_2_training import V4SampleReplay, play_history_pool_game, prefix_selfplay_game, validate_history_pool_game


_STAGE_STATES = (1_000, 2_000, 5_000, 10_000)
_SEED = 742_300
_CHAMPION_ID = "G0-V3-510901-20K"


def _initial_arm(root: Path, config: dict[str, Any], arm_id: str):
    row = next(item for item in config["historical_candidates"] if item["candidate_id"] == _CHAMPION_ID)
    checkpoint = root / row["checkpoint_path"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
        raise ValueError("V4.2 training start checkpoint hash differs from registered V3 G0@20K")
    model = load_g_checkpoint(checkpoint, device="cpu")
    model.checkpoint_metadata = {"model_family": "G", "arm_id": arm_id,
                                 "initialization": _CHAMPION_ID,
                                 "initial_checkpoint_sha256": row["checkpoint_sha256"]}
    trainer = GTrainer(
        model, seed=_SEED, learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        expected_updates=math.ceil(int(config["training"]["states_per_arm"])
                                   / int(config["training"]["update_every_states"])),
        device="cpu", bn_mode=str(config["training"]["batchnorm"]),
    )
    replay = V4SampleReplay(int(config["training"]["replay_capacity_positions"]), seed=_SEED)
    return trainer, replay


def _persist_arm_progress(root: Path, arm_id: str, progress: dict[str, Any]) -> None:
    atomic_write_json(root / "research/g_v4_2/results/training" / arm_id / "progress.json", progress)


def _combine_update_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    total_updates = sum(int(row.get("updates", 0)) for row in rows)
    result: dict[str, Any] = {"updates": total_updates, "update_count": int(rows[-1].get("update_count", 0))}
    for key in ("policy_loss", "value_loss", "total_loss"):
        result[key] = float(sum(float(row.get(key, 0.0)) * int(row.get("updates", 0)) for row in rows)
                            / max(1, total_updates))
    for key in ("cpu_seconds", "wall_seconds"):
        result[key] = float(sum(float(row.get(key, 0.0)) for row in rows))
    result["learning_rate"] = float(rows[-1].get("learning_rate", 0.0))
    return result


def _validate_baseline_game(game: dict[str, Any], state_count: int) -> dict[str, Any]:
    clipped = prefix_selfplay_game(game, state_count)
    from .replay import _validate_game
    _validate_game(clipped)
    return clipped


def _evaluate_stage(
    root: Path,
    config: dict[str, Any],
    schedule: dict[str, Any],
    *,
    model: PolicyValueNet,
    arm_id: str,
    states: int,
    diagnostics: dict[str, Any],
    runtime: V4_2Runtime,
    optimizer_metrics: dict[str, Any] | None = None,
    selfplay_cpu_seconds: float = 0.0,
) -> dict[str, Any]:
    match_groups = schedule["development_openings"][:int(config["evaluation"]["stage_groups"])]
    rows = []
    for opponent_name, opponent_id in (("A", "A-rule"), ("B", "B-50ms"), ("MODEL", _CHAMPION_ID)):
        for group in match_groups:
            for color in ("B", "W"):
                runtime.check(reserve_seconds=1.0)
                record = _match_record(
                    root, model=model, candidate_id=f"{arm_id}@{states}", opponent_id=opponent_id,
                    opponent_name=opponent_name, group=group, color=color,
                    config=PUCTConfig(), namespace="training_eval", phase=f"development_{arm_id}_{states}",
                    runtime=runtime,
                )
                rows.append(record)
                runtime.save("training_stage_evaluation", current_arm=arm_id,
                             current_state_node=states, last_game_id=record["game_id"])
    by_opponent = {}
    for opponent_id in ("A-rule", "B-50ms", _CHAMPION_ID):
        subset = [row for row in rows if row["opponent_checkpoint_id"] == opponent_id]
        by_opponent[opponent_id] = summarize_v3_matches(subset)

    puzzle_ids = set(diagnostics["tactical_split"]["development_ids"])
    puzzle_rows = load_json(root / "research/g_v4_2/config/tactical_puzzles.json")
    by_id = {item["puzzle_id"]: item for item in puzzle_rows}
    selected_puzzles = [by_id[puzzle_id] for puzzle_id in sorted(puzzle_ids)]
    tactical = [_tactical_record(model, puzzle, PUCTConfig(), runtime) for puzzle in selected_puzzles]
    tactical_summary = _tactical_summary(tactical)
    history_matches = [row for row in rows if row["opponent_checkpoint_id"] == _CHAMPION_ID]
    promotion = promotion_decision(
        history_matches, candidate_id=f"{arm_id}@{states}", incumbent_id=_CHAMPION_ID,
        min_score_rate=float(config["evaluation"]["promotion"]["score_rate_at_least"]),
        lower_ci_minimum=float(config["evaluation"]["promotion"]["paired_ci95_lower_above"]),
        min_color_score_rate=float(config["evaluation"]["promotion"]["each_color_at_least"]),
        required_groups=len(match_groups), seed=742_310 + states,
    )
    summary = {
        "arm_id": arm_id, "training_states": states,
        "matches": by_opponent, "challenger_gate_vs_v3_g0_20k": promotion,
        "tactical_search_puct64": tactical_summary,
        "tactical_records": tactical,
        "optimizer": dict(optimizer_metrics or model.checkpoint_metadata.get("latest_optimizer_metrics", {})),
        "selfplay_cpu_seconds_since_previous_node": float(selfplay_cpu_seconds),
        "evaluation_cpu_seconds": float(sum(float(value["total_cpu_seconds"]) for value in by_opponent.values())),
        "completed_selfplay_games": int(
            load_json(root / "research/g_v4_2/results/training" / arm_id / "progress.json").get("games", 0)
        ),
    }
    atomic_write_json(root / "research/g_v4_2/results/training" / arm_id / f"node_{states}_evaluation.json", summary)
    return summary


def train_v4_2_arms(root: str | Path, runtime: V4_2Runtime) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_2_config(root_path)
    schedule = load_json(root_path / config["opening_schedule_path"])
    diagnostics = load_json(root_path / "research/g_v4_2/results/diagnostics/diagnostics.json")
    from . import v4_2_experiment
    v4_2_experiment._ACTIVE_CONFIG = config
    selected_search = str(diagnostics["search_selection"]["selected"])
    active_arms = ["baseline", "history_pool"]
    if selected_search != "baseline":
        active_arms.insert(1, "search_target")
    search_configs = {name: _config_from_dict(item) for name, item in config["search_variants"].items()}
    search_configs["selected_diagnostic"] = search_configs[selected_search]
    history_ids = list(config["training"]["history_opponents"])
    history_models = {candidate_id: _model_by_id(root_path, config, candidate_id) for candidate_id in history_ids}
    opening_groups = schedule["train_openings"]
    all_stage_summaries: dict[str, list[dict[str, Any]]] = {}
    total_states_by_arm: dict[str, int] = {}

    for arm_id in active_arms:
        runtime.check(reserve_seconds=2.0)
        arm_dir = root_path / "research/g_v4_2/checkpoints" / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        state_path = arm_dir / "training_state.pt"
        trainer, replay = _initial_arm(root_path, config, arm_id)
        if state_path.exists():
            state_progress = load_training_state(state_path, trainer, replay, restore_global_rng=True)
            progress = dict(state_progress)
            if progress.get("optimizer_metrics"):
                trainer.model.checkpoint_metadata["latest_optimizer_metrics"] = progress["optimizer_metrics"]
        else:
            progress = {"arm_id": arm_id, "training_states": 0, "next_game_index": 0,
                        "last_checkpoint_states": 0, "games": 0, "last_stage": 0,
                        "selfplay_cpu_seconds": 0.0, "last_stage_selfplay_cpu_seconds": 0.0,
                        "stage_update_metrics": []}
            save_g_checkpoint(arm_dir / "node_0.pt", trainer.model, metadata={
                "arm_id": arm_id, "training_states": 0, "base_candidate": _CHAMPION_ID,
                "search_variant": (selected_search if arm_id == "search_target" else "baseline"),
            })
            save_training_state(state_path, trainer, replay, progress=progress)
        model = trainer.model
        model.eval()
        stage_summaries: list[dict[str, Any]] = []
        persisted_stage_path = root_path / "research/g_v4_2/results/training" / arm_id
        persisted_stage_path.mkdir(parents=True, exist_ok=True)
        stage_eval_files = sorted(persisted_stage_path.glob("node_*_evaluation.json"))
        stage_summaries.extend(load_json(path) for path in stage_eval_files)

        for milestone in _STAGE_STATES:
            if int(progress.get("training_states", 0)) >= milestone:
                evaluation_path = persisted_stage_path / f"node_{milestone}_evaluation.json"
                if not evaluation_path.exists():
                    checkpoint_path = arm_dir / f"node_{milestone}.pt"
                    if not checkpoint_path.exists():
                        raise ValueError(f"missing committed V4.2 checkpoint at node {milestone}")
                    stage_model = load_g_checkpoint(checkpoint_path, device="cpu")
                    summary = _evaluate_stage(root_path, config, schedule, model=stage_model,
                                              arm_id=arm_id, states=milestone, diagnostics=diagnostics,
                                              runtime=runtime,
                                              optimizer_metrics=dict(progress.get("optimizer_metrics", {})),
                                              selfplay_cpu_seconds=(float(progress.get("selfplay_cpu_seconds", 0.0))
                                                  - float(progress.get("last_stage_selfplay_cpu_seconds", 0.0))))
                    stage_summaries.append(summary)
                progress["stage_update_metrics"] = []
                progress["last_stage_selfplay_cpu_seconds"] = float(progress.get("selfplay_cpu_seconds", 0.0))
                save_training_state(state_path, trainer, replay, progress=progress)
                continue
            while int(progress["training_states"]) < milestone:
                chunk_size = int(config["training"]["checkpoint_every_states"])
                current_states = int(progress["training_states"])
                collection_target = min(milestone, (current_states // chunk_size + 1) * chunk_size)
                while int(progress["training_states"]) < collection_target:
                    runtime.check(reserve_seconds=2.0)
                    game_index = int(progress["next_game_index"])
                    group = opening_groups[game_index % len(opening_groups)]
                    game_id = _stable_id("g42-selfplay", arm_id, game_index, group["opening_group_id"],
                                         config["config_sha256"])
                    namespace = f"selfplay/{arm_id}"
                    path = game_result_path(root_path, namespace, game_id)
                    if path.exists():
                        game = load_json(path)
                    elif arm_id == "history_pool":
                        opponent_id = history_ids[game_index % len(history_ids)]
                        learner_color = "B" if game_index % 2 == 0 else "W"
                        game = play_history_pool_game(
                            model, history_models[opponent_id], game_id=game_id,
                            opening_group_id=str(group["opening_group_id"]), arm_id=arm_id,
                            opening=[dict(move) for move in group["opening"]],
                            learner_color=learner_color,
                            simulations=int(config["training"]["simulations_per_move"]),
                            seed=_stable_seed("train", game_index), config=search_configs["baseline"],
                        )
                    else:
                        variant_name = "selected_diagnostic" if arm_id == "search_target" else "baseline"
                        game = play_selfplay_game(
                            model, group["opening"], arm_id=arm_id, game_id=game_id,
                            opening_group_id=str(group["opening_group_id"]),
                            simulations=int(config["training"]["simulations_per_move"]),
                            seed=_stable_seed("train", game_index), config=search_configs[variant_name],
                        )
                    if not game.get("completed") or not game.get("moves"):
                        raise ValueError("V4.2 refuses incomplete self-play episodes")
                    remaining = collection_target - int(progress["training_states"])
                    rows = list(game.get("experience_rows", []))
                    take = min(remaining, len(rows))
                    if take < 1:
                        raise ValueError("completed game yielded no learner training states")
                    if arm_id == "history_pool":
                        validate_history_pool_game(game, str(game["learner_color"]))
                        selected_rows = rows[:take]
                    else:
                        clipped_game = _validate_baseline_game(game, take)
                        selected_rows = list(clipped_game["experience_rows"])
                    game["budgeted_training_state_count"] = take
                    game["training_state_ids"] = [row["sample_id"] for row in selected_rows]
                    persist_unique_game(root_path, namespace, game)
                    replay.add_game(game, selected_rows)
                    progress["training_states"] = int(progress["training_states"]) + take
                    progress["next_game_index"] = game_index + 1
                    progress["games"] = int(progress.get("games", 0)) + 1
                    progress["last_completed_game_id"] = game_id
                    progress["selfplay_cpu_seconds"] = float(progress.get("selfplay_cpu_seconds", 0.0)) + float(
                        game.get("stats", {}).get("cpu_seconds", 0.0)
                    )
                    _persist_arm_progress(root_path, arm_id, progress)
                    runtime.save("training_selfplay_collection", current_arm=arm_id,
                                 current_state_node=int(progress["training_states"]), last_game_id=game_id)

                target_updates = int(progress["training_states"]) // int(config["training"]["update_every_states"])
                delta = target_updates - int(trainer.update_count)
                update_metrics = trainer.train_updates(
                    replay, updates=delta, batch_size=int(config["training"]["batch_size"]),
                    sampling_mode=str(config["training"]["replay_sampling"]),
                )
                update_history = list(progress.get("stage_update_metrics", []))
                update_history.append(update_metrics)
                progress["stage_update_metrics"] = update_history
                aggregate_metrics = _combine_update_metrics(update_history)
                model.checkpoint_metadata["latest_optimizer_metrics"] = aggregate_metrics
                progress.update({"update_count": trainer.update_count,
                                 "optimizer_metrics": aggregate_metrics,
                                 "last_checkpoint_states": int(progress["training_states"])})
                save_training_state(state_path, trainer, replay, progress=progress)

            checkpoint_path = arm_dir / f"node_{milestone}.pt"
            update_metrics = dict(progress.get("optimizer_metrics", {}))
            save_g_checkpoint(checkpoint_path, model, metadata={
                "arm_id": arm_id, "training_states": milestone, "update_count": trainer.update_count,
                "base_candidate": _CHAMPION_ID, "optimizer_reset": True,
                "learning_rate": config["training"]["learning_rate"],
                "batchnorm": config["training"]["batchnorm"],
                "search_variant": (selected_search if arm_id == "search_target" else "baseline"),
                "opponent_pool": (history_ids if arm_id == "history_pool" else ["current"]),
                "checkpoint_sha256_base": config["historical_candidates"][2]["checkpoint_sha256"],
                "policy_loss": float(update_metrics["policy_loss"]),
                "value_loss": float(update_metrics["value_loss"]),
            })
            _persist_arm_progress(root_path, arm_id, progress)
            runtime.save("training_checkpoint", current_arm=arm_id, current_state_node=milestone,
                         checkpoint_path=str(checkpoint_path.relative_to(root_path)),
                         checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
            summary = _evaluate_stage(root_path, config, schedule, model=model,
                                     arm_id=arm_id, states=milestone, diagnostics=diagnostics,
                                     runtime=runtime,
                                     optimizer_metrics=update_metrics,
                                     selfplay_cpu_seconds=(float(progress.get("selfplay_cpu_seconds", 0.0))
                                         - float(progress.get("last_stage_selfplay_cpu_seconds", 0.0))))
            stage_summaries.append(summary)
            all_stage_summaries[arm_id] = stage_summaries
            total_states_by_arm[arm_id] = int(progress["training_states"])
            atomic_write_json(root_path / "research/g_v4_2/results/training/stage_summary.json", {
                "stage_summaries": all_stage_summaries, "states_by_arm": total_states_by_arm,
            })
            progress["stage_update_metrics"] = []
            progress["last_stage_selfplay_cpu_seconds"] = float(progress.get("selfplay_cpu_seconds", 0.0))
            save_training_state(state_path, trainer, replay, progress=progress)

        all_stage_summaries[arm_id] = stage_summaries
        total_states_by_arm[arm_id] = int(progress["training_states"])

    training_summary = {
        "format_version": 1, "active_arms": active_arms,
        "skipped_search_arm_reason": None if "search_target" in active_arms else (
            "frozen-weight search configuration did not pass the pre-registered development/validation gates"
        ),
        "states_by_arm": total_states_by_arm,
        "new_training_states_total": int(sum(total_states_by_arm.values())),
        "stage_summaries": all_stage_summaries,
        "optimizer_reset_from_v3_champion": True,
        "starting_candidate_id": _CHAMPION_ID,
        "training_inputs_excluded": config["forbidden_training_inputs"],
    }
    atomic_write_json(root_path / "research/g_v4_2/results/training/summary.json", training_summary)
    return training_summary
