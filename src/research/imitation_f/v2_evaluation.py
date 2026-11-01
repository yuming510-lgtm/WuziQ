"""Paired, resumable F V2 model-vs-A evaluation and uncertainty estimates."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..runner import derive_seed
from .dataset import validate_game_record
from .evaluation import play_match
from .model import load_f_checkpoint
from .v2_data import verify_network_actions


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid match JSON at {path}:{line_number}") from exc
    return rows


def _clean(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    if not result:
        raise ValueError("candidate id must contain a letter or digit")
    return result


def _score(record: dict[str, Any], model_name: str) -> float:
    winner = record.get("winner_model")
    return 0.5 if winner is None else (1.0 if winner == model_name else 0.0)


def _score_summary(records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    counts = Counter()
    points = 0.0
    for row in records:
        winner = row.get("winner_model")
        key = "draws" if winner is None else ("wins" if winner == model_name else "losses")
        counts[key] += 1
        points += _score(row, model_name)
    games = len(records)
    return {
        "games": games, "wins": counts["wins"], "draws": counts["draws"], "losses": counts["losses"],
        "points": points, "score_rate": points / games if games else 0.0,
    }


def _bootstrap_ci(values: list[float], rng: np.random.Generator, samples: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_opening_statistics(
    records: list[dict[str, Any]], model_name: str, bootstrap_seed: int, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    """Summarize paired color-swapped groups, bootstrapping whole openings."""
    if bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if not records:
        raise ValueError("paired match records cannot be empty")
    seen = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        game_id = str(row["game_id"])
        if game_id in seen:
            raise ValueError(f"duplicate match game id: {game_id}")
        seen.add(game_id)
        grouped[str(row["group_id"])].append(row)
    by_group = []
    black_scores, white_scores = [], []
    for group_id, games in sorted(grouped.items()):
        if len(games) != 2:
            raise ValueError(f"opening group {group_id} must contain exactly two color-swapped games")
        colors: dict[str, dict[str, Any]] = {}
        for game in games:
            if model_name not in (game.get("black_model"), game.get("white_model")):
                raise ValueError("challenger must play one side in every game")
            color = "black" if game["black_model"] == model_name else "white"
            if color in colors:
                raise ValueError(f"opening group {group_id} does not swap challenger colors")
            colors[color] = game
        if set(colors) != {"black", "white"}:
            raise ValueError(f"opening group {group_id} does not cover both colors")
        color_scores = {color: _score(game, model_name) for color, game in colors.items()}
        black_scores.append(color_scores["black"])
        white_scores.append(color_scores["white"])
        group_score = (color_scores["black"] + color_scores["white"]) / 2
        by_group.append({
            "group_id": group_id, "score_rate": group_score,
            "black_score": color_scores["black"], "white_score": color_scores["white"],
            "game_ids": [colors["black"]["game_id"], colors["white"]["game_id"]],
        })
    rng = np.random.default_rng(int(bootstrap_seed))
    group_scores = [float(row["score_rate"]) for row in by_group]
    summary = _score_summary(records, model_name)
    summary.update({
        "opening_groups": len(by_group),
        "score_rate_ci95": _bootstrap_ci(group_scores, rng, bootstrap_samples),
        "bootstrap_method": "percentile bootstrap resampling paired opening groups",
        "bootstrap_samples": int(bootstrap_samples), "bootstrap_seed": int(bootstrap_seed),
        "by_color": {
            "black": {**_score_summary([grouped[entry["group_id"]][0] if grouped[entry["group_id"]][0]["black_model"] == model_name else grouped[entry["group_id"]][1] for entry in by_group], model_name),
                      "score_rate_ci95": _bootstrap_ci(black_scores, rng, bootstrap_samples)},
            "white": {**_score_summary([grouped[entry["group_id"]][0] if grouped[entry["group_id"]][0]["white_model"] == model_name else grouped[entry["group_id"]][1] for entry in by_group], model_name),
                      "score_rate_ci95": _bootstrap_ci(white_scores, rng, bootstrap_samples)},
        },
        "opening_group_scores": by_group,
    })
    return summary


def run_opening_matches(
    model,
    challenger_model: str,
    candidate_id: str,
    opponent: str,
    openings: list[dict[str, Any]],
    output_path: str | Path,
    stage: str,
    checkpoint_sha256: str,
    seed: int,
    opponent_budget_seconds: float = 0.05,
) -> dict[str, Any]:
    """Run/resume paired opening games and atomically persist every completed game."""
    if challenger_model not in ("F1", "F2") or opponent != "A":
        raise ValueError("F V2 schedules support only F1/F2 against A")
    if stage not in ("development", "blind") or not openings or opponent_budget_seconds < 0:
        raise ValueError("invalid F V2 match schedule")
    candidate = _clean(candidate_id)
    path = Path(output_path)
    progress_path = path.with_suffix(path.suffix + ".progress.json")
    schedule_payload = {
        "stage": stage, "candidate_id": candidate, "challenger_model": challenger_model,
        "opponent": opponent, "checkpoint_sha256": checkpoint_sha256, "seed": int(seed),
        "opponent_budget_seconds": float(opponent_budget_seconds),
        "openings": openings,
    }
    fingerprint = hashlib.sha256(
        json.dumps(schedule_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    existing = {}
    for row in _read_jsonl(path):
        validate_game_record(row)
        game_id = str(row["game_id"])
        if game_id in existing:
            raise ValueError(f"duplicate persisted match id: {game_id}")
        if row.get("candidate_id") != candidate or row.get("challenger_model") != challenger_model:
            raise ValueError("persisted match belongs to another candidate")
        existing[game_id] = row
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    if progress and progress.get("schedule_fingerprint") != fingerprint:
        raise ValueError("persisted match progress does not match the frozen schedule")

    expected: dict[str, tuple[dict[str, Any], str]] = {}
    for opening in openings:
        for color in ("B", "W"):
            game_id = f"f-v2-{stage}-{candidate}-vs-a-s{int(opening['opening_seed'])}-f{color.lower()}"
            expected[game_id] = (opening, color)
    if set(existing) - set(expected):
        raise ValueError("persisted match results include games outside the frozen schedule")
    _atomic_json(progress_path, {
        "schema_version": 1, "schedule_fingerprint": fingerprint,
        "candidate_id": candidate, "stage": stage, "scheduled_games": len(expected),
        "completed_games": len(existing), "completed_game_ids": sorted(existing),
        "remaining_games": len(expected) - len(existing),
    })
    for game_id, (opening, color) in expected.items():
        if game_id not in existing:
            game_seed = derive_seed(int(seed), "f-v2-paired-match", stage, int(opening["opening_seed"]), color)
            record = play_match(
                model, opponent, opening["opening"], color, game_seed,
                opponent_budget_seconds=opponent_budget_seconds,
            )
            mapping = {"F": challenger_model, opponent: opponent}
            for key in ("black_model", "white_model", "winner_model"):
                if record.get(key) in mapping:
                    record[key] = mapping[record[key]]
            for move in record["moves"]:
                if move.get("model") == "F":
                    move["model"] = challenger_model
            record.update({
                "game_id": game_id, "group_id": str(opening["opening_group_id"]),
                "experiment_id": "wuziq-imitation-learning-f-v2", "stage": stage,
                "candidate_id": candidate, "challenger_model": challenger_model,
                "checkpoint_sha256": checkpoint_sha256,
                "pair": [challenger_model, opponent], "opening_seed": int(opening["opening_seed"]),
                "opening": [dict(move) for move in opening["opening"]],
                "color_swap": color == "W", "game_seed": game_seed,
                "budget_seconds_by_model": {challenger_model: None, opponent: opponent_budget_seconds},
            })
            validate_game_record(record)
            existing[game_id] = record
            _atomic_jsonl(path, [existing[key] for key in sorted(existing)])
        _atomic_json(progress_path, {
            "schema_version": 1, "schedule_fingerprint": fingerprint,
            "candidate_id": candidate, "stage": stage, "scheduled_games": len(expected),
            "completed_games": len(existing), "completed_game_ids": sorted(existing),
            "remaining_games": len(expected) - len(existing), "last_completed_game_id": game_id,
        })
    records = [existing[key] for key in sorted(existing)]
    action_audit = verify_network_actions(model, records)
    summary = _score_summary(records, challenger_model)
    return {**summary, "candidate_id": candidate, "stage": stage, "action_audit": action_audit,
            "records_path": str(path), "checkpoint_sha256": checkpoint_sha256}


def paired_difference_ci(
    left: list[dict[str, Any]], right: list[dict[str, Any]], left_model: str, right_model: str,
    bootstrap_seed: int, bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap the paired per-opening F2 minus F1 score-rate difference."""
    left_groups = {str(row["group_id"]): row for row in paired_opening_statistics(
        left, left_model, bootstrap_seed, bootstrap_samples
    )["opening_group_scores"]}
    right_groups = {str(row["group_id"]): row for row in paired_opening_statistics(
        right, right_model, bootstrap_seed + 1, bootstrap_samples
    )["opening_group_scores"]}
    if set(left_groups) != set(right_groups):
        raise ValueError("paired model comparisons must use identical opening groups")
    differences = [right_groups[group]["score_rate"] - left_groups[group]["score_rate"] for group in sorted(left_groups)]
    rng = np.random.default_rng(int(bootstrap_seed) + 2)
    return {
        "opening_groups": len(differences), "mean_score_rate_difference": float(np.mean(differences)),
        "f2_minus_f1_ci95": _bootstrap_ci(differences, rng, bootstrap_samples),
        "method": "paired percentile bootstrap resampling opening groups",
    }


def load_checkpoint_spec(path: str | Path, device: str = "cpu") -> tuple[Any, str]:
    checkpoint = Path(path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return load_f_checkpoint(checkpoint, device=device), digest
