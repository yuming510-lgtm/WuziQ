"""Paired V6.2 stage and blind evaluation with isolated artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .v5_evaluation import _puzzle_board
from .v5_reference import ReferencePolicyValueNet, instrumented_player
from .v6_2_training import ARTIFACT, ROOT, total_cpu_used_seconds
from .v6_evaluation import run_match_set


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_reference_model(path: str | Path) -> ReferencePolicyValueNet:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise ValueError(f"checkpoint has no model_state: {path}")
    model = ReferencePolicyValueNet(9, seed=1)
    model.load_policy_param(state)
    model.network.eval()
    return model


def evaluate_tactics(
    model: ReferencePolicyValueNet,
    puzzles: list[dict[str, Any]],
    *,
    simulations: int,
    seed: int,
    cpu_deadline: float | None = None,
) -> dict[str, Any]:
    """Report naked policy, target probability mass, PUCT, and available value labels."""
    rows = []
    for index, puzzle in enumerate(puzzles):
        if cpu_deadline is not None and time.process_time() >= float(cpu_deadline):
            break
        normalized = {
            **puzzle,
            "board_size": int(puzzle.get("board_size", 9)),
            "n_in_row": int(puzzle.get("n_in_row", 5)),
            "moves": puzzle.get("moves", puzzle.get("position", [])),
            "target_actions": puzzle.get("target_actions", puzzle.get("optimal_actions", [])),
        }
        board = _puzzle_board(normalized)
        priors, value = model.policy_value_fn(board)
        prior_map = {int(action): float(prob) for action, prob in priors}
        targets = set(map(int, normalized["target_actions"]))
        network_action = max(prior_map, key=lambda action: (prior_map[action], -action))
        np.random.seed((int(seed) + index) & 0xFFFFFFFF)
        player = instrumented_player(model.policy_value_fn, c_puct=5.0,
                                     n_playout=int(simulations) + 1, is_selfplay=False)
        puct_action, visit_distribution = player.get_action(board, temp=1e-3, return_prob=1)
        exact_value = normalized.get("solved_value_to_move")
        row = {
            "puzzle_id": normalized["puzzle_id"],
            "category": normalized["category"],
            "network_action": int(network_action),
            "network_correct": int(network_action) in targets,
            "network_target_probability_mass": float(sum(prior_map.get(action, 0.0) for action in targets)),
            "network_value_to_move": float(value),
            "solved_value_to_move": exact_value,
            "absolute_value_error": (None if exact_value is None
                                      else abs(float(value) - float(exact_value))),
            "puct_action": int(puct_action),
            "puct_correct": int(puct_action) in targets,
            "puct_target_visit_mass": float(sum(float(visit_distribution[action]) for action in targets)),
            "tree_depth": int(player.last_search_stats.get("tree_depth", 0)),
            "tree_nodes": int(player.last_search_stats.get("tree_nodes", 0)),
            "root_action_visits": int(player.last_search_stats.get("root_visits", 0)),
            "search_cpu_seconds": float(player.last_search_stats.get("search_cpu_seconds", 0.0)),
        }
        rows.append(row)
    categories = sorted({row["category"] for row in rows})
    labeled = [row for row in rows if row["absolute_value_error"] is not None]
    return {
        "simulations": int(simulations),
        "puzzles": len(rows),
        "complete": len(rows) == len(puzzles),
        "network_top1_accuracy": float(np.mean([row["network_correct"] for row in rows])) if rows else None,
        "network_target_probability_mass": float(np.mean([row["network_target_probability_mass"] for row in rows])) if rows else None,
        "puct_accuracy": float(np.mean([row["puct_correct"] for row in rows])) if rows else None,
        "puct_target_visit_mass": float(np.mean([row["puct_target_visit_mass"] for row in rows])) if rows else None,
        "value_labeled_puzzles": len(labeled),
        "mean_absolute_value_error": float(np.mean([row["absolute_value_error"] for row in labeled])) if labeled else None,
        "mean_tree_depth": float(np.mean([row["tree_depth"] for row in rows])) if rows else None,
        "mean_tree_nodes": float(np.mean([row["tree_nodes"] for row in rows])) if rows else None,
        "mean_search_cpu_seconds": float(np.mean([row["search_cpu_seconds"] for row in rows])) if rows else None,
        "by_category": {
            category: {
                "puzzles": sum(row["category"] == category for row in rows),
                "network_top1_accuracy": float(np.mean([row["network_correct"] for row in rows
                                                         if row["category"] == category])),
                "puct_accuracy": float(np.mean([row["puct_correct"] for row in rows
                                                 if row["category"] == category])),
                "network_target_probability_mass": float(np.mean([
                    row["network_target_probability_mass"] for row in rows
                    if row["category"] == category])),
            } for category in categories
        },
        "rows": rows,
    }


def run_stage(root: str | Path = ROOT, *, game_count: int) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    cpu_cap = float(json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
                    ["compute"]["cpu_hard_cap_total_new_hours"]) * 3600.0
    used_cpu = total_cpu_used_seconds(root)
    if used_cpu >= cpu_cap:
        raise RuntimeError("V6.2 cumulative new-CPU hard cap has been reached")
    cpu_deadline = time.process_time() + (cpu_cap - used_cpu)
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    model_path = artifact / f"checkpoints/reference400_empty/node_games_{int(game_count)}.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    candidate = load_reference_model(model_path)
    baseline_path = artifact / "checkpoints/reference400_empty/node_20000.pt"
    baseline = load_reference_model(baseline_path)
    stage_path = artifact / f"results/development/stage_games_{int(game_count)}.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.is_file() else {
        "game_count": int(game_count),
        "candidate_checkpoint_sha256": _sha(model_path),
        "opening_schedule_sha256": _sha(artifact / "config/opening_schedule.json"),
        "comparisons": {"64": {}, "400": {}},
        "tactics": {"64": {}, "400": {}},
        "complete": False,
    }
    if stage["candidate_checkpoint_sha256"] != _sha(model_path):
        raise ValueError("existing V6.2 stage belongs to a different checkpoint")
    openings = schedule["development"]
    pair_count = int(registration["evaluation"]["development_pairs_per_stage"])
    for budget in registration["evaluation"]["budgets_root_action_visits"]:
        budget_key = str(budget)
        for opponent in ("G20K", "A", "B-50ms"):
            if opponent in stage["comparisons"][budget_key] and stage["comparisons"][budget_key][opponent].get("complete"):
                continue
            pairs = pair_count
            if int(game_count) == 2_000 and opponent == "G20K" and int(budget) == 400:
                pairs = 80
            kwargs: dict[str, Any] = {
                "root": root,
                "phase": f"development-games-{int(game_count)}",
                "candidate_name": f"ref-v6-2-g{int(game_count)}",
                "opponent_name": ("ref-v6-2-G20K" if opponent == "G20K" else opponent),
                "openings": openings,
                "pairs": int(pairs),
                "candidate_reference_model": candidate,
                "simulations": int(budget),
                "board_size": 9,
                "n_in_row": 5,
                "artifact_dir": ARTIFACT,
                "cpu_deadline": cpu_deadline,
            }
            if opponent == "G20K":
                kwargs["opponent_reference_model"] = baseline
            summary = run_match_set(**kwargs)
            stage["comparisons"][budget_key][opponent] = summary
            _atomic_json(stage_path, stage)

        for label, path in (
            ("historical_v6_64", artifact / "config/tactics_historical_v6_64.json"),
            ("new_independent_128", artifact / "config/tactics_new_128.json"),
        ):
            result_path = artifact / f"results/tactics/games_{int(game_count)}_{label}_puct{int(budget)}.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
            if result and result.get("checkpoint_sha256") != _sha(model_path):
                raise ValueError("V6.2 tactical result belongs to another checkpoint")
            if not result or not result.get("complete"):
                puzzles = json.loads(path.read_text(encoding="utf-8"))
                result = evaluate_tactics(candidate, puzzles, simulations=int(budget),
                                          seed=861_300 + int(game_count) + int(budget),
                                          cpu_deadline=cpu_deadline)
                result.update({
                    "checkpoint_sha256": _sha(model_path),
                    "puzzle_set": label,
                    "puzzle_set_sha256": _sha(path),
                    "game_count": int(game_count),
                    "budget_role": "evaluation only; never a self-play training target",
                })
                _atomic_json(result_path, result)
            stage["tactics"][budget_key][label] = {
                "path": str(result_path.relative_to(root)),
                "puzzles": result["puzzles"],
                "complete": result["complete"],
                "network_top1_accuracy": result["network_top1_accuracy"],
                "network_target_probability_mass": result["network_target_probability_mass"],
                "puct_accuracy": result["puct_accuracy"],
                "puct_target_visit_mass": result["puct_target_visit_mass"],
                "mean_absolute_value_error": result["mean_absolute_value_error"],
            }
            _atomic_json(stage_path, stage)
    stage["complete"] = all(
        stage["comparisons"][budget].get(opponent, {}).get("complete", False)
        and stage["comparisons"][budget][opponent]["completed_pairs"] == (
            80 if int(game_count) == 2_000 and budget == "400" and opponent == "G20K" else pair_count
        )
        for budget in ("64", "400") for opponent in ("G20K", "A", "B-50ms")
    ) and all(set(stage["tactics"][budget]) == {"historical_v6_64", "new_independent_128"}
              and all(stage["tactics"][budget][name].get("complete") for name in stage["tactics"][budget])
              for budget in ("64", "400"))
    stage["stage_checkpoint_counters"] = {
        "games": int(game_count),
        "states": int(torch.load(model_path, map_location="cpu", weights_only=False)["environment_states"]),
    }
    _atomic_json(stage_path, stage)
    return stage


def select_final_candidate(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    gate_path = artifact / "results/gate_2000.json"
    if not gate_path.is_file():
        raise FileNotFoundError("V6.2 2,000-game advancement gate has not been computed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    eligible_games = [2_000]
    if gate["passed"]:
        eligible_games.extend(game for game in (2_500, 3_000)
                              if (artifact / f"results/development/stage_games_{game}.json").is_file())
    rows = []
    for game_count in eligible_games:
        path = artifact / f"results/development/stage_games_{game_count}.json"
        if not path.is_file():
            continue
        stage = json.loads(path.read_text(encoding="utf-8"))
        if not stage.get("comparisons", {}).get("400", {}).get("G20K", {}).get("complete"):
            continue
        match = stage["comparisons"]["400"]["G20K"]
        rows.append({
            "game_count": game_count,
            "score_rate": match["score_rate"],
            "paired_ci95": match["paired_opening_bootstrap_ci95"],
            "checkpoint_path": f"research/g_v6_2/checkpoints/reference400_empty/node_games_{game_count}.pt",
            "checkpoint_sha256": _sha(artifact / f"checkpoints/reference400_empty/node_games_{game_count}.pt"),
        })
    if not rows:
        raise ValueError("no development checkpoint has a completed primary PUCT-400 comparison")
    selected = sorted(rows, key=lambda row: (-float(row["score_rate"]), int(row["game_count"])))[0]
    result = {
        "frozen_before_blind": True,
        "blind_results_used_for_selection": False,
        "selection_rule": registration["evaluation"]["blind_selection"],
        "development_candidates": rows,
        "selected": selected,
    }
    _atomic_json(artifact / "results/selection.json", result)
    return result


def run_final_blind(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    cpu_cap = float(registration["compute"]["cpu_hard_cap_total_new_hours"]) * 3600.0
    used_cpu = total_cpu_used_seconds(root)
    if used_cpu >= cpu_cap:
        raise RuntimeError("V6.2 cumulative new-CPU hard cap has been reached")
    cpu_deadline = time.process_time() + (cpu_cap - used_cpu)
    selection_path = artifact / "results/selection.json"
    if not selection_path.is_file():
        select_final_candidate(root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("frozen_before_blind") or selection.get("blind_results_used_for_selection"):
        raise ValueError("V6.2 blind selection lock is invalid")
    selected = selection["selected"]
    candidate = load_reference_model(root / selected["checkpoint_path"])
    baseline = load_reference_model(artifact / "checkpoints/reference400_empty/node_20000.pt")
    openings = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))["blind"]
    summaries = {}
    for budget in (64, 400):
        summaries[str(budget)] = {}
        for opponent in ("G20K", "A", "B-50ms"):
            kwargs: dict[str, Any] = {
                "root": root,
                "phase": "blind-final-v6-2",
                "candidate_name": f"ref-v6-2-final-g{selected['game_count']}",
                "opponent_name": ("ref-v6-2-G20K" if opponent == "G20K" else opponent),
                "openings": openings,
                "pairs": 40,
                "candidate_reference_model": candidate,
                "simulations": budget,
                "board_size": 9,
                "n_in_row": 5,
                "artifact_dir": ARTIFACT,
                "cpu_deadline": cpu_deadline,
            }
            if opponent == "G20K":
                kwargs["opponent_reference_model"] = baseline
            summaries[str(budget)][opponent] = run_match_set(**kwargs)
    result = {
        "complete": all(summary.get("complete") and summary.get("paired_openings") == 40
                         for budget in summaries.values() for summary in budget.values()),
        "selection": selection,
        "blind_seed": 861_202,
        "opening_ids": [opening["opening_group_id"] for opening in openings],
        "matches": summaries,
    }
    _atomic_json(artifact / "results/final_blind.json", result)
    return result
