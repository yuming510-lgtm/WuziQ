"""Explicit BatchNorm running-statistic controls used by G V3 probes."""
from __future__ import annotations

from typing import Any, Literal
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import append_evaluation_match, load_evaluation_log, play_evaluation_match
from .replay import ReplayBuffer, append_complete_game, load_game_log
from .selfplay import play_selfplay_game
from .v3_schedule import validate_v3_schedule

BNBufferMap = dict[str, dict[str, torch.Tensor]]
_BUFFER_NAMES = ("running_mean", "running_var", "num_batches_tracked")


def capture_bn_buffers(model: nn.Module) -> BNBufferMap:
    """Copy every BatchNorm running buffer by stable module name."""
    result: BNBufferMap = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            buffers = {}
            for field in _BUFFER_NAMES:
                value = getattr(module, field, None)
                if value is not None:
                    buffers[field] = value.detach().clone()
            result[name] = buffers
    return result


def restore_bn_buffers(model: nn.Module, buffers: BNBufferMap) -> None:
    """Restore only running statistics; affine parameters and all other weights are untouched."""
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    if set(modules) != set(buffers):
        raise ValueError("BatchNorm layer names differ from the saved buffer map")
    with torch.no_grad():
        for name, module in modules.items():
            expected = {field for field in _BUFFER_NAMES if getattr(module, field, None) is not None}
            values = buffers[name]
            if set(values) != expected:
                raise ValueError(f"BatchNorm buffer names differ for layer {name!r}")
            for field, value in values.items():
                target = getattr(module, field)
                if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                    raise ValueError(f"BatchNorm buffer shape differs for layer {name!r}: {field}")
                target.copy_(value.to(device=target.device, dtype=target.dtype))


def set_bn_training_mode(model: nn.Module, mode: Literal["update", "freeze"]) -> None:
    """Train the model and either update or freeze BN running statistics.

    In freeze mode the BatchNorm affine parameters remain trainable. Only each
    layer's behavior for running mean/variance is switched to evaluation mode.
    """
    if mode not in {"update", "freeze"}:
        raise ValueError("bn mode must be 'update' or 'freeze'")
    model.train()
    if mode == "freeze":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _budgeted_prefix(game: dict[str, Any], count: int) -> dict[str, Any]:
    copy_game = copy.deepcopy(game)
    total = len(copy_game["experience_rows"])
    copy_game["experience_rows"] = copy_game["experience_rows"][:count]
    copy_game["budgeted_environment_steps"] = count
    copy_game["completion_overrun_steps"] = total - count
    copy_game["replay_prefix"] = True
    return copy_game


def _buffer_drift(before: BNBufferMap, after: BNBufferMap) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name in before:
        row: dict[str, float | int] = {}
        for field in before[name]:
            left = before[name][field].detach().cpu().to(torch.float64)
            right = after[name][field].detach().cpu().to(torch.float64)
            delta = right - left
            row[f"{field}_l2"] = float(torch.linalg.vector_norm(delta).item())
            row[f"{field}_max_abs"] = float(delta.abs().max().item()) if delta.numel() else 0.0
        result[name] = row
    return result


def run_bn_probe(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Compare dynamic/frozen BN and restore-before-evaluation on matched data.

    Each route generates its own 256-state self-play prefix. All three BN modes
    for that route then start from the identical initial weights and train on
    that same route-local experience. Restore is an attribution control only.
    """
    from .training import GTrainer, initialize_arm, save_g_checkpoint

    root_path = Path(root).resolve()
    schedule_path = root_path / "research/g_v3/config/openings.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    seeds = [int(seed) for seed in config.get("seeds", schedule["seeds"])]
    if seeds != [int(seed) for seed in schedule["seeds"]]:
        raise ValueError("BN probe seeds differ from the registered opening schedule")
    settings = config.get("bn_probe_selection", {})
    states_per_route = int(settings.get("states_per_route", 256))
    updates = int(settings.get("updates", 16))
    opening_count = int(settings.get("opening_groups", 8))
    tie_margin = float(settings.get("tie_score_rate_margin", 0.125))
    if states_per_route < 1 or updates < 1 or opening_count < 1:
        raise ValueError("BN probe counts must be positive")
    if len(schedule["development_openings"]) < opening_count:
        raise ValueError("development opening pool is smaller than the BN probe quota")

    torch.set_num_threads(1)
    output_dir = root_path / "research/g_v3/results/bn_probe"
    selfplay_log = output_dir / "selfplay.jsonl"
    evaluation_log = output_dir / "evaluation.jsonl"
    saved_games, game_ids = load_game_log(selfplay_log)
    saved_by_id = {str(game["game_id"]): game for game in saved_games}
    saved_matches, match_ids = load_evaluation_log(evaluation_log)
    match_by_id = {str(match["game_id"]): match for match in saved_matches}
    route_results: dict[str, Any] = {}
    candidate_models: dict[tuple[str, str], torch.nn.Module] = {}
    route_initials: dict[str, torch.nn.Module] = {}
    started = time.perf_counter()
    cpu_started = time.process_time()

    for arm in ("G0", "G-F2"):
        seed = seeds[0]
        initial_model = initialize_arm(arm, seed=seed, device="cpu")
        initial_model.eval()
        route_initials[arm] = initial_model
        train_groups = schedule["train_openings_by_seed"][str(seed)]
        replay = ReplayBuffer(states_per_route * 2, seed=seed + (0 if arm == "G0" else 17))
        completed_rows = 0
        next_game_index = 0
        while completed_rows < states_per_route:
            group = train_groups[next_game_index % len(train_groups)]
            game_id = f"g-v3-bn-{arm}-seed-{seed}-game-{next_game_index:05d}"
            game = saved_by_id.get(game_id)
            if game is None:
                game_model = initialize_arm(arm, seed=seed, device="cpu")
                game = play_selfplay_game(
                    game_model, group["opening"], arm_id=arm, game_id=game_id,
                    opening_group_id=group["opening_group_id"], simulations=64,
                    seed=seed + 41_000 + next_game_index * 13 + (0 if arm == "G0" else 1),
                )
                append_complete_game(selfplay_log, game, game_ids)
                saved_by_id[game_id] = game
            remaining = states_per_route - completed_rows
            accepted = min(remaining, len(game["experience_rows"]))
            replay.add_complete_game(_budgeted_prefix(game, accepted))
            completed_rows += accepted
            next_game_index += 1

        initial_buffers = capture_bn_buffers(initial_model)
        metrics_by_mode: dict[str, Any] = {}
        for mode in ("update", "freeze", "restore_before_evaluation"):
            model = copy.deepcopy(initial_model)
            trainer_mode = "update" if mode == "restore_before_evaluation" else mode
            trainer = GTrainer(
                model, seed=seed + 51_000 + (0 if arm == "G0" else 100) + len(metrics_by_mode),
                learning_rate=float(config.get("learning_rate", 1e-4)),
                weight_decay=float(config.get("weight_decay", 1e-4)),
                expected_updates=updates, device="cpu", bn_mode=trainer_mode,
            )
            training_metrics = trainer.train_updates(replay, updates=updates, batch_size=16)
            trained_buffers = capture_bn_buffers(model)
            if mode == "restore_before_evaluation":
                restore_bn_buffers(model, initial_buffers)
                evaluation_buffers = capture_bn_buffers(model)
            else:
                evaluation_buffers = trained_buffers
            model.eval()
            drift = _buffer_drift(initial_buffers, trained_buffers)
            checkpoint = output_dir / "checkpoints" / f"{arm}-seed-{seed}-{mode}.pt"
            checkpoint_sha = save_g_checkpoint(checkpoint, model, metadata={
                "experiment": "g_v3_bn_probe", "arm": arm, "seed": seed,
                "bn_mode": mode, "updates": updates,
            })
            candidate_models[(arm, mode)] = model
            scores: list[float] = []
            color_scores = {"B": [], "W": []}
            for group_index, group in enumerate(schedule["development_openings"][:opening_count]):
                for color in ("B", "W"):
                    game_id = f"g-v3-bn-eval-{arm}-{mode}-{group_index:02d}-{color}"
                    result = match_by_id.get(game_id)
                    if result is None:
                        result = play_evaluation_match(
                            model,
                            game_id=game_id,
                            opening_group_id=group["opening_group_id"],
                            arm_id=f"{arm}-{mode}",
                            mode="puct",
                            opponent="MODEL",
                            opponent_model=initial_model,
                            opponent_mode="puct",
                            opening=group["opening"],
                            agent_color=color,
                            seed=seed + 61_000 + group_index * 2 + (color == "W"),
                            simulations=64,
                            opponent_simulations=64,
                            device="cpu",
                        )
                        result["bn_probe_mode"] = mode
                        append_evaluation_match(evaluation_log, result, match_ids)
                        saved_matches.append(result)
                        match_by_id[game_id] = result
                    score = float(result["score_for_agent"])
                    scores.append(score)
                    color_scores[color].append(score)
            metrics_by_mode[mode] = {
                "training": training_metrics,
                "bn_running_stat_drift": drift,
                "evaluation_bn_buffers_equal_initial": all(
                    torch.equal(evaluation_buffers[name][field], initial_buffers[name][field])
                    for name in initial_buffers for field in initial_buffers[name]
                ),
                "checkpoint": checkpoint.relative_to(root_path).as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "puct_score_rate_vs_own_initial": float(np.mean(scores)),
                "black_score_rate": float(np.mean(color_scores["B"])),
                "white_score_rate": float(np.mean(color_scores["W"])),
                "paired_games": len(scores),
            }
        route_results[arm] = {
            "seed": seed,
            "budgeted_states": completed_rows,
            "selfplay_game_count": next_game_index,
            "experience_games": replay.game_count,
            "experience_positions": replay.position_count,
            "modes": metrics_by_mode,
        }

    mode_scores = {
        mode: float(np.mean([
            route_results[arm]["modes"][mode]["puct_score_rate_vs_own_initial"]
            for arm in ("G0", "G-F2")
        ]))
        for mode in ("update", "freeze")
    }
    difference = abs(mode_scores["update"] - mode_scores["freeze"])
    selected = "freeze" if difference < tie_margin else max(mode_scores, key=mode_scores.get)
    result = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v3-bn-probe",
        "formal_training": False,
        "selection_rule": {
            "combined_score_rates": mode_scores,
            "absolute_difference": difference,
            "tie_margin": tie_margin,
            "tie_default": "freeze",
            "selected_common_mode": selected,
            "restore_before_evaluation_is_selection_eligible": False,
        },
        "routes": route_results,
        "selfplay_log": selfplay_log.relative_to(root_path).as_posix(),
        "evaluation_log": evaluation_log.relative_to(root_path).as_posix(),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
    }
    _atomic_json(output_dir / "probe.json", result)
    return result
