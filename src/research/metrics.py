"""Statistical summaries computed exclusively from saved game records."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import NormalDist
from typing import Any, Iterable


_Z_95 = NormalDist().inv_cdf(0.975)


def wilson_interval(wins: int, losses: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval for the decisive-game win probability."""

    if wins < 0 or losses < 0 or not 0 < confidence < 1:
        raise ValueError("wins/losses must be non-negative and confidence must be in (0, 1)")
    total = wins + losses
    if total == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = wins / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _paired_block_interval(
    block_values: dict[str, list[float]], samples: int, seed: int
) -> list[float]:
    blocks = [sum(values) / len(values) for _, values in sorted(block_values.items()) if values]
    if not blocks:
        return [0.0, 1.0]
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(blocks[rng.randrange(len(blocks))] for _ in range(len(blocks))) / len(blocks)
        )
    estimates.sort()
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _opening_cluster_id(record: dict[str, Any]) -> str:
    """Cluster repeated games by the actual starting position, not its seed index."""

    opening = record.get("opening")
    if opening is not None:
        payload = [record.get("board_size"), opening]
        import json

        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(record.get("pair_id", record["game_id"]))


def paired_score_interval(
    records: Iterable[dict[str, Any]],
    model: str,
    bootstrap_samples: int = 5_000,
    seed: int = 0,
) -> list[float]:
    """Bootstrap a 95% score-rate interval by distinct opening-position cluster."""

    blocks: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if model not in record.get("matchup", ()):
            continue
        winner = record.get("winner_model")
        score = 1.0 if winner == model else 0.5 if winner is None else 0.0
        block_id = _opening_cluster_id(record)
        blocks[block_id].append(score)
    return _paired_block_interval(blocks, bootstrap_samples, seed)


def paired_pure_win_interval(
    records: Iterable[dict[str, Any]],
    model: str,
    bootstrap_samples: int = 5_000,
    seed: int = 0,
) -> list[float]:
    """Bootstrap the unconditional win fraction by distinct opening layout."""

    blocks: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if model not in record.get("matchup", ()):
            continue
        blocks[_opening_cluster_id(record)].append(
            1.0 if record.get("winner_model") == model else 0.0
        )
    return _paired_block_interval(blocks, bootstrap_samples, seed)


def _model_move_stats(records: Iterable[dict[str, Any]], model: str) -> dict[str, float | int]:
    moves = [
        move
        for record in records
        for move in record.get("moves", [])
        if move.get("model") == model
    ]
    count = len(moves)
    return {
        "moves": count,
        "average_seconds_per_move": (
            sum(float(move.get("elapsed_seconds", 0.0)) for move in moves) / count
            if count
            else 0.0
        ),
        "average_nodes_per_move": (
            sum(int(move.get("nodes", 0)) for move in moves) / count if count else 0.0
        ),
        "average_simulations_per_move": (
            sum(int(move.get("simulations", 0)) for move in moves) / count if count else 0.0
        ),
        "average_completed_depth": (
            sum(int(move.get("completed_depth", 0)) for move in moves) / count if count else 0.0
        ),
        "timed_out_moves": sum(bool(move.get("timed_out", False)) for move in moves),
        "illegal_moves": sum(bool(move.get("illegal", False)) for move in moves),
    }


def _outcome_counts(records: list[dict[str, Any]], model: str) -> dict[str, int]:
    relevant = [record for record in records if model in record.get("matchup", ())]
    wins = sum(record.get("winner_model") == model for record in relevant)
    draws = sum(record.get("winner_model") is None for record in relevant)
    losses = len(relevant) - wins - draws
    return {"games": len(relevant), "wins": wins, "losses": losses, "draws": draws}


def _model_result_stats(
    records: list[dict[str, Any]], model: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    counts = _outcome_counts(records, model)
    games = counts["games"]
    decisive = counts["wins"] + counts["losses"]
    score_points = counts["wins"] + 0.5 * counts["draws"]
    return {
        **counts,
        "win_rate": counts["wins"] / games if games else 0.0,
        "pure_win_rate": counts["wins"] / games if games else 0.0,
        "pure_win_rate_ci95": paired_pure_win_interval(
            records, model, bootstrap_samples=bootstrap_samples, seed=derive_summary_seed(seed, "pure", model)
        ),
        "pure_win_rate_wilson_ci95": list(
            wilson_interval(counts["wins"], counts["losses"] + counts["draws"])
        ),
        "decisive_win_rate": counts["wins"] / decisive if decisive else 0.0,
        "decisive_win_rate_ci95": list(wilson_interval(counts["wins"], counts["losses"])),
        "score_rate": score_points / games if games else 0.0,
        "score_rate_ci95": paired_score_interval(
            records, model, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        **_model_move_stats(records, model),
    }


def _pairwise_records(records: list[dict[str, Any]], pair: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if tuple(sorted(record.get("matchup", ()))) == pair
    ]


def _color_summary(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    black_wins = sum(record.get("winner_color") == "B" for record in records)
    white_wins = sum(record.get("winner_color") == "W" for record in records)
    draws = sum(record.get("winner_color") is None for record in records)
    blocks: dict[str, list[float]] = defaultdict(list)
    for record in records:
        winner_color = record.get("winner_color")
        score = 1.0 if winner_color == "B" else 0.5 if winner_color is None else 0.0
        blocks[_opening_cluster_id(record)].append(score)
    interval = _paired_block_interval(blocks, samples, seed)
    games = len(records)
    return {
        "games": games,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "black_score_rate": (black_wins + 0.5 * draws) / games if games else 0.0,
        "black_score_rate_ci95": interval,
    }


def aggregate_results(
    records: Iterable[dict[str, Any]],
    models: Iterable[str] = ("A", "B", "C", "D"),
    bootstrap_samples: int = 5_000,
    seed: int = 5102026,
) -> dict[str, Any]:
    """Build complete pair matrix, paired score intervals, and overall ranking."""

    games = list(records)
    model_names = tuple(sorted(str(model).upper() for model in models))
    win_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for row_model in model_names:
        win_matrix[row_model] = {}
        for column_model in model_names:
            pair_games = _pairwise_records(games, tuple(sorted((row_model, column_model))))
            if row_model == column_model:
                win_matrix[row_model][column_model] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "score_rate": 0.0,
                }
                continue
            wins = sum(record.get("winner_model") == row_model for record in pair_games)
            losses = sum(record.get("winner_model") == column_model for record in pair_games)
            draws = sum(record.get("winner_model") is None for record in pair_games)
            count = len(pair_games)
            win_matrix[row_model][column_model] = {
                "games": count,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score_rate": (wins + 0.5 * draws) / count if count else 0.0,
            }

    pairwise: dict[str, dict[str, Any]] = {}
    pair_index = 0
    for first_index, first in enumerate(model_names):
        for second in model_names[first_index + 1 :]:
            pair = (first, second)
            pair_games = _pairwise_records(games, pair)
            first_stats = _model_result_stats(
                pair_games, first, bootstrap_samples, derive_summary_seed(seed, pair_index, first)
            )
            second_stats = _model_result_stats(
                pair_games, second, bootstrap_samples, derive_summary_seed(seed, pair_index, second)
            )
            first_interval = first_stats["score_rate_ci95"]
            second_interval = second_stats["score_rate_ci95"]
            pairwise[f"{first}-{second}"] = {
                "models": {first: first_stats, second: second_stats},
                "games": len(pair_games),
                "wins_first": first_stats["wins"],
                "wins_second": second_stats["wins"],
                "draws": first_stats["draws"],
                "score_intervals_overlap": not (
                    first_interval[1] < second_interval[0]
                    or second_interval[1] < first_interval[0]
                ),
            }
            pair_index += 1

    ranking = [
        {
            "model": model,
            **_model_result_stats(
                games, model, bootstrap_samples, derive_summary_seed(seed, "overall", model)
            ),
        }
        for model in model_names
    ]
    ranking.sort(key=lambda item: (-item["score_rate"], item["model"]))
    return {
        "game_count": len(games),
        "models": list(model_names),
        "win_matrix": win_matrix,
        "pairwise": pairwise,
        "ranking": ranking,
        "color_advantage": _color_summary(games, bootstrap_samples, seed),
    }


def derive_summary_seed(base_seed: int, *parts: object) -> int:
    """Make bootstrap sampling repeatable across interpreter processes."""

    import hashlib
    import json

    content = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(content.encode()).digest()[:8], "big")
