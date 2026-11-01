#!/usr/bin/env python3
"""Run pre-registered development nodes and one final blind evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from src.research.alphazero_g.evaluation import (
    append_evaluation_match,
    load_evaluation_log,
    load_frozen_opponent,
    play_evaluation_match,
    summarize_matches,
)
from src.research.alphazero_g.training import load_g_checkpoint


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _seed_from(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _checkpoint_path(root: Path, arm: str, checkpoint_id: str) -> Path:
    suffix = "initial" if checkpoint_id == "initial" else checkpoint_id
    return root / "research/g_v1/checkpoints" / arm / f"{suffix}.pt"


def _run_node(
    root: Path,
    config: dict[str, Any],
    arm: str,
    checkpoint_id: str,
    opening_groups: list[dict[str, Any]],
    log_path: Path,
    existing_ids: set[str],
    opponent_models: dict[str, torch.nn.Module],
    *,
    phase: str,
    cpu_used_before: float,
    cpu_cap_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    checkpoint = _checkpoint_path(root, arm, checkpoint_id)
    model = load_g_checkpoint(checkpoint, device="cpu")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    matches, _ = load_evaluation_log(log_path)
    already_used = cpu_used_before + sum(float(match.get("runner_cpu_seconds", 0.0)) for match in matches)
    if already_used >= cpu_cap_seconds:
        raise RuntimeError("registered G V1 CPU-hour cap is exhausted before evaluation")
    new_count = 0
    per_game_simulations = int(config["evaluation"]["puct_simulations"])
    for mode in config["evaluation"]["modes"]:
        for opponent in config["evaluation"]["opponents"]:
            frozen = opponent_models.get(opponent)
            for group in opening_groups:
                for color in ("B", "W"):
                    game_id = f"{phase}-{arm}-{checkpoint_id}-{mode}-{opponent}-{group['opening_group_id']}-{color}"
                    if game_id in existing_ids:
                        continue
                    if already_used >= cpu_cap_seconds:
                        raise RuntimeError("registered G V1 CPU-hour cap is exhausted")
                    start_cpu = time.process_time()
                    match = play_evaluation_match(
                        model,
                        game_id=game_id,
                        opening_group_id=group["opening_group_id"],
                        arm_id=arm,
                        mode=mode,
                        opponent=opponent,
                        opening=group["opening"],
                        agent_color=color,
                        seed=_seed_from(game_id),
                        simulations=per_game_simulations,
                        opponent_model=frozen,
                        device="cpu",
                    )
                    match["checkpoint_id"] = checkpoint_id
                    match["checkpoint_sha256"] = checkpoint_sha
                    match["phase"] = phase
                    match["registered_puct_simulations"] = per_game_simulations if mode == "puct" else 0
                    match["runner_cpu_seconds"] = time.process_time() - start_cpu
                    if not append_evaluation_match(log_path, match, existing_ids):
                        raise ValueError(f"evaluation game id was already persisted: {game_id}")
                    matches.append(match)
                    already_used += float(match["runner_cpu_seconds"])
                    new_count += 1
    return matches, new_count


def _summarize_development(matches: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for match in matches:
        key = (match["arm_id"], match["checkpoint_id"], match["mode"], match["opponent"])
        groups.setdefault(key, []).append(match)
    for key, group_matches in sorted(groups.items()):
        summaries.append({
            "arm_id": key[0], "checkpoint_id": key[1], "mode": key[2], "opponent": key[3],
            **summarize_matches(group_matches),
            "mean_agent_seconds": sum(match["stats"]["agent_seconds"] for match in group_matches) / len(group_matches),
            "mean_agent_simulations": sum(match["stats"]["agent_simulations"] for match in group_matches) / len(group_matches),
        })
    return {"summaries": summaries}


def _select_checkpoints(root: Path, config: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    selected = {}
    for arm in ("G0", "G-F2"):
        candidates = []
        for checkpoint_id in config["checkpoint_nodes"]:
            puct_rows = [
                row for row in matches
                if row["arm_id"] == arm and row["checkpoint_id"] == checkpoint_id and row["mode"] == "puct"
            ]
            opponent_rates = []
            for opponent in config["evaluation"]["opponents"]:
                rows = [row for row in puct_rows if row["opponent"] == opponent]
                if not rows:
                    raise ValueError(f"development selection lacks {arm}/{checkpoint_id}/puct/{opponent}")
                opponent_rates.append(summarize_matches(rows)["score_rate"])
            checkpoint = _checkpoint_path(root, arm, checkpoint_id)
            candidates.append({
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "mean_puct_score_over_opponents": sum(opponent_rates) / len(opponent_rates),
                "opponent_score_rates": dict(zip(config["evaluation"]["opponents"], opponent_rates, strict=True)),
            })
        # Checkpoint nodes are ordered initial -> stage-1 -> stage-2. Python's
        # stable max keeps the earlier node when scores tie exactly.
        best = max(candidates, key=lambda row: row["mean_puct_score_over_opponents"])
        selected[arm] = {**best, "candidates": candidates}
    return {
        "format_version": 1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "schedule_sha256": config["opening_schedule_sha256"],
        "selection_rule": config["development_selection_rule"],
        "arms": selected,
    }


def run_evaluations(root: str | Path, config_path: str | Path, phase: str = "all") -> dict[str, Any]:
    if phase not in {"development", "blind", "all"}:
        raise ValueError("phase must be development, blind, or all")
    root_path = Path(root).resolve()
    config = json.loads(Path(config_path).read_text())
    schedule = config["opening_schedule"]
    output_dir = root_path / "research/g_v1/results/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_threads = int(config.get("torch_threads", config.get("resources", {}).get("torch_threads", 1)))
    torch.set_num_threads(max(1, torch_threads))
    opponent_models = {name: load_frozen_opponent(name, device="cpu") for name in ("F1", "F2")}
    training_summary_path = root_path / "research/g_v1/results/training/summary.json"
    if not training_summary_path.exists():
        raise FileNotFoundError("complete dual-arm training summary is required before evaluation")
    training_summary = json.loads(training_summary_path.read_text())
    if training_summary.get("completed") is not True or training_summary.get("training_budget_equal") is not True:
        raise ValueError("registered dual-arm training quotas did not complete equally")
    cpu_cap_seconds = float(config["hard_cpu_hour_cap"]) * 3600
    training_cpu_seconds = float(training_summary["total_active_cpu_seconds"])
    progress = {"phase": phase, "new_matches": 0}
    dev_path = output_dir / "development.jsonl"
    dev_matches, dev_ids = load_evaluation_log(dev_path)
    blind_existing_path = output_dir / "blind.jsonl"
    blind_existing_cpu = sum(
        float(match.get("runner_cpu_seconds", 0.0)) for match in load_evaluation_log(blind_existing_path)[0]
    )
    if phase in {"development", "all"}:
        for arm in ("G0", "G-F2"):
            for checkpoint_id in config["checkpoint_nodes"]:
                _, count = _run_node(
                    root_path, config, arm, checkpoint_id, schedule["development_openings"],
                    dev_path, dev_ids, opponent_models, phase="development",
                    cpu_used_before=training_cpu_seconds + blind_existing_cpu,
                    cpu_cap_seconds=cpu_cap_seconds,
                )
                progress["new_matches"] += count
        dev_matches, dev_ids = load_evaluation_log(dev_path)
        expected_dev = int(config["evaluation"]["development_games_total"])
        if len(dev_matches) != expected_dev:
            raise ValueError(f"development schedule has {len(dev_matches)} records; expected {expected_dev}")
        dev_summary = _summarize_development(dev_matches)
        _atomic_json(output_dir / "development_summary.json", dev_summary)
        selection = _select_checkpoints(root_path, config, dev_matches)
        selection_path = output_dir / "selection.json"
        if selection_path.exists():
            existing = json.loads(selection_path.read_text())
            if existing != selection:
                raise ValueError("development selection changed after it was frozen")
        else:
            _atomic_json(selection_path, selection)
        progress["selection"] = selection
    selection_path = output_dir / "selection.json"
    if phase in {"blind", "all"}:
        if not selection_path.exists():
            raise ValueError("blind evaluation cannot start before a frozen development selection")
        selection = json.loads(selection_path.read_text())
        selection_sha = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
        blind_path = output_dir / "blind.jsonl"
        blind_matches, blind_ids = load_evaluation_log(blind_path)
        blind_state_path = output_dir / "blind_run.json"
        if blind_state_path.exists():
            state = json.loads(blind_state_path.read_text())
            if state.get("selection_sha256") != selection_sha or state.get("schedule_sha256") != config["opening_schedule_sha256"]:
                raise ValueError("existing blind run belongs to a different selection or opening schedule")
        else:
            _atomic_json(blind_state_path, {
                "selection_sha256": selection_sha,
                "schedule_sha256": config["opening_schedule_sha256"],
                "blind_run_count": 1,
                "opened_before_first_blind_game": True,
            })
        for arm in ("G0", "G-F2"):
            checkpoint_id = selection["arms"][arm]["checkpoint_id"]
            _, count = _run_node(
                root_path, config, arm, checkpoint_id, schedule["blind_openings"],
                blind_path, blind_ids, opponent_models, phase="blind",
                cpu_used_before=training_cpu_seconds + sum(
                    float(match.get("runner_cpu_seconds", 0.0)) for match in load_evaluation_log(dev_path)[0]
                ),
                cpu_cap_seconds=cpu_cap_seconds,
            )
            progress["new_matches"] += count
        blind_matches, _ = load_evaluation_log(blind_path)
        expected_blind = int(config["evaluation"]["blind_games_per_selected_checkpoint"])
        if len(blind_matches) != expected_blind:
            raise ValueError(f"blind schedule has {len(blind_matches)} records; expected {expected_blind}")
        _atomic_json(output_dir / "blind_summary.json", _summarize_development(blind_matches))
        progress["blind_games"] = len(blind_matches)
        progress["blind_selection_sha256"] = selection_sha
    progress["completed"] = (
        phase == "development" and len(dev_matches) == int(config["evaluation"]["development_games_total"])
    ) or (
        phase == "blind" and (output_dir / "blind_summary.json").exists()
    ) or (
        phase == "all" and (output_dir / "blind_summary.json").exists()
        and len(load_evaluation_log(dev_path)[0]) == int(config["evaluation"]["development_games_total"])
    )
    _atomic_json(output_dir / "progress.json", progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("research/g_v1/config/experiment.json"))
    parser.add_argument("--phase", choices=("development", "blind", "all"), default="all")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else args.root / args.config
    result = run_evaluations(args.root, config, phase=args.phase)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
