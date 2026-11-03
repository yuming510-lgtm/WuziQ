"""Auditable aggregation for G V4.1 development/blind logs."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import load_evaluation_log, summarize_matches
from .v4_1_config import ARM_IDS


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float] | None:
    """Two-sided 95% Wilson interval for a binary outcome rate."""
    if isinstance(successes, bool) or isinstance(trials, bool) or trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials")
    if trials == 0:
        return None
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _model_mode_metrics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    selected_moves = []
    seconds: list[float] = []
    for match in matches:
        candidate_moves = [move for move in match["moves"] if move.get("actor") == "agent"]
        seconds.append(float(match["stats"].get("agent_seconds", 0.0)) / max(len(candidate_moves), 1))
        winner = match.get("winner_color")
        for move in candidate_moves:
            target = 0.0 if winner is None else 1.0 if move.get("player") == winner else -1.0
            prediction = move.get("root_value") if match.get("registered_mode") == "puct" else move.get("value")
            if prediction is not None and math.isfinite(float(prediction)):
                selected_moves.append({
                    "prediction": float(prediction), "target": target,
                    "policy_entropy": move.get("search_entropy", move.get("policy_entropy")),
                    "search_depth": int(move.get("search_depth", 0)),
                    "nodes": int(move.get("nodes", 0)),
                    "simulations": int(move.get("simulations", 0)),
                })
    predictions = np.asarray([row["prediction"] for row in selected_moves], dtype=np.float64)
    targets = np.asarray([row["target"] for row in selected_moves], dtype=np.float64)
    residual = predictions - targets if len(predictions) else np.asarray([], dtype=np.float64)
    entropy = [float(row["policy_entropy"]) for row in selected_moves if row["policy_entropy"] is not None]
    puct = matches[0].get("registered_mode") == "puct" if matches else False
    return {
        "games": len(matches),
        "candidate_moves": sum(sum(move.get("actor") == "agent" for move in match["moves"])
                                for match in matches),
        "mean_candidate_decision_seconds": float(np.mean(seconds)) if seconds else None,
        "mean_value_prediction": float(predictions.mean()) if len(predictions) else None,
        "mean_terminal_target": float(targets.mean()) if len(targets) else None,
        "value_mae": float(np.abs(residual).mean()) if len(residual) else None,
        "value_mse": float(np.square(residual).mean()) if len(residual) else None,
        "mean_policy_or_search_entropy": float(np.mean(entropy)) if entropy else None,
        "mean_search_depth": float(np.mean([row["search_depth"] for row in selected_moves])) if puct and selected_moves else 0.0,
        "mean_nodes_per_move": float(np.mean([row["nodes"] for row in selected_moves])) if puct and selected_moves else 0.0,
        "mean_simulations_per_move": float(np.mean([row["simulations"] for row in selected_moves])) if puct and selected_moves else 0.0,
        "value_samples": len(selected_moves),
    }


def summarize_phase(path: str | Path, *, phase: str) -> dict[str, Any]:
    matches, ids = load_evaluation_log(path)
    required_groups = 12 if phase == "development" else 20
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        key = (str(match.get("registered_candidate")),
               str(match.get("registered_opponent")),
               str(match.get("registered_mode")))
        grouped[key].append(match)
    if phase == "development":
        expected_keys = expected_development_comparisons()
    elif phase == "blind":
        finalist_values = {str(row.get("blind_finalist")) for row in matches}
        if len(finalist_values) != 1:
            raise ValueError("blind matches do not share exactly one frozen finalist")
        finalist = next(iter(finalist_values))
        expected_keys = {("G0-V3-20K", opponent, mode)
                         for opponent in ("A", "B") for mode in ("network", "puct")}
        expected_keys.update((finalist, opponent, mode)
                             for opponent in ("A", "B") for mode in ("network", "puct"))
        expected_keys.add((finalist, "G0-V3-20K", "puct"))
    else:
        raise ValueError("phase must be development or blind")
    if set(grouped) != expected_keys:
        raise ValueError("evaluation match log is missing or has unexpected registered comparisons")
    comparisons = []
    for (candidate, opponent, mode), rows in sorted(grouped.items()):
        if len(rows) != required_group_games(phase):
            raise ValueError(f"paired comparison {candidate}/{opponent}/{mode} is incomplete")
        colors_by_group: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            colors_by_group[str(row["opening_group_id"])].add(str(row["agent_color"]))
        if len(colors_by_group) != required_groups or any(colors != {"B", "W"} for colors in colors_by_group.values()):
            raise ValueError(f"paired comparison {candidate}/{opponent}/{mode} lacks complete color swaps")
        summary = summarize_matches(rows)
        summary["win_rate_wilson_ci95"] = wilson_interval(int(summary["wins"]), int(summary["games"]))
        summary["nonloss_rate_wilson_ci95"] = wilson_interval(
            int(summary["wins"] + summary["draws"]), int(summary["games"])
        )
        comparisons.append({
            "candidate": candidate, "opponent": opponent, "mode": mode,
            "summary": summary, "compute": _model_mode_metrics(rows),
        })
    expected = 816 if phase == "development" else 360
    return {
        "phase": phase,
        "games": len(matches), "unique_game_ids": len(ids),
        "registered_expected_games": expected,
        "complete": len(matches) == expected,
        "paired_group_count": required_groups,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "all_moves": sum(len(match["moves"]) for match in matches),
        "cpu_seconds": sum(float(match["stats"].get("total_cpu_seconds", 0.0)) for match in matches),
        "winner_counts": {
            "black": sum(match.get("winner_color") == "B" for match in matches),
            "white": sum(match.get("winner_color") == "W" for match in matches),
            "draw": sum(match.get("winner_color") is None for match in matches),
        },
    }


def write_phase_summary(path: str | Path, output: str | Path, *, phase: str) -> dict[str, Any]:
    summary = summarize_phase(path, phase=phase)
    if not summary["complete"]:
        raise ValueError(f"{phase} match log is incomplete ({summary['games']}/{summary['registered_expected_games']})")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(destination)
    return summary


def expected_development_comparisons() -> set[tuple[str, str, str]]:
    expected = {("G0-V3-20K", opponent, mode) for opponent in ("A", "B") for mode in ("network", "puct")}
    for candidate in ARM_IDS:
        expected.update((candidate, opponent, mode)
                        for opponent in ("G0-V3-20K", "A", "B")
                        for mode in ("network", "puct"))
    for index, candidate in enumerate(ARM_IDS):
        expected.update((candidate, opponent, "puct") for opponent in ARM_IDS[index + 1:])
    return expected


def required_group_games(phase: str) -> int:
    groups = 12 if phase == "development" else 20
    return groups * 2
