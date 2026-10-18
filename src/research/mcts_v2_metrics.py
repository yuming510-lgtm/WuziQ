"""Statistics for the paired V2 MCTS tournaments."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .metrics import aggregate_results, wilson_interval


def _metric_record(record: dict[str, Any]) -> dict[str, Any]:
    """Use the runner's paired opening ID as the bootstrap cluster.

    Color-swapped V2 openings reverse the B/W labels. The coordinates are the
    same paired position, so clustering on the serialized player labels would
    incorrectly split the paired games into separate bootstrap blocks.
    """

    normalized = dict(record)
    normalized["opening"] = None
    return normalized


def _compute_usage(
    records: list[dict[str, Any]], model: str, score_points: float
) -> dict[str, Any]:
    moves = [
        move
        for record in records
        for move in record.get("moves", ())
        if move.get("model") == model
    ]
    seconds = sum(float(move.get("elapsed_seconds", 0.0)) for move in moves)
    simulations = sum(int(move.get("simulations", 0)) for move in moves)
    return {
        "model_moves": len(moves),
        "model_thinking_seconds_total": seconds,
        "model_simulations_total": simulations,
        "model_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
        "model_rollout_plies_total": sum(int(move.get("rollout_plies", 0)) for move in moves),
        "model_terminal_simulations_total": sum(
            int(move.get("terminal_simulations", 0)) for move in moves
        ),
        "model_truncated_simulations_total": sum(
            int(move.get("truncated_simulations", 0)) for move in moves
        ),
        "model_timed_out_moves": sum(bool(move.get("timed_out", False)) for move in moves),
        "model_illegal_moves": sum(bool(move.get("illegal", False)) for move in moves),
        "model_score_points": score_points,
        "score_points_per_model_thinking_second": (
            score_points / seconds if seconds else None
        ),
        "simulations_per_model_thinking_second": (
            simulations / seconds if seconds else None
        ),
    }


def _paired_block_wilson(
    records: Iterable[dict[str, Any]], model: str, *, score_rate: bool
) -> list[float]:
    """Wilson interval over paired opening blocks, including boundary cases.

    Percentile bootstrap intervals collapse to [0, 0] when every observed
    block is a loss. Treat each color-swapped opening pair as one effective
    Bernoulli block and use its average win or score fraction as a fractional
    success count, so zero- and all-win samples retain non-degenerate limits.
    """

    blocks: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if model not in record.get("matchup", ()):
            continue
        winner = record.get("winner_model")
        if score_rate:
            value = 1.0 if winner == model else 0.5 if winner is None else 0.0
        else:
            value = 1.0 if winner == model else 0.0
        block_id = str(record.get("pair_id", record["game_id"]))
        blocks[block_id].append(value)
    if not blocks:
        return [0.0, 1.0]
    block_means = [sum(values) / len(values) for values in blocks.values() if values]
    successes = sum(block_means)
    return list(wilson_interval(successes, len(block_means) - successes))


def summarize_phases(
    records: Iterable[dict[str, Any]],
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 20_261_018,
) -> dict[str, Any]:
    """Summarize win matrices, ranks, compute use, and paired 95% intervals.

    Each configured phase is summarized independently. Fair phases therefore
    produce a five-model ranking at one board size and one shared budget;
    extreme phases produce the D2-versus-C non-equal-budget comparison.
    """

    games = list(records)
    identifiers = [str(record["game_id"]) for record in games]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate game_id in V2 result records")

    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in games:
        phase = str(record.get("phase_name", ""))
        if not phase:
            raise ValueError(f"Game {record['game_id']} is missing phase_name")
        by_phase[phase].append(record)

    phase_summaries: dict[str, Any] = {}
    for phase_name in sorted(by_phase):
        phase_games = by_phase[phase_name]
        first = phase_games[0]
        track = str(first["track"])
        board_size = int(first["board_size"])
        budget_map = {
            str(model): float(seconds)
            for model, seconds in first["budget_seconds_by_model"].items()
        }
        first_budget_values = sorted(budget_map.values())
        budgets_by_model: dict[str, set[float]] = defaultdict(set)
        for record in phase_games:
            record_budget_values = sorted(
                float(seconds)
                for seconds in record["budget_seconds_by_model"].values()
            )
            for model, seconds in record["budget_seconds_by_model"].items():
                budgets_by_model[str(model)].add(float(seconds))
            if (
                str(record["track"]) != track
                or int(record["board_size"]) != board_size
                or record_budget_values != first_budget_values
            ):
                raise ValueError(f"Inconsistent phase metadata: {phase_name}")

        models = sorted(
            {
                str(model)
                for record in phase_games
                for model in record.get("matchup", ())
            }
        )
        for model in models:
            if len(budgets_by_model[model]) != 1:
                raise ValueError(f"Inconsistent per-model budget in phase: {phase_name}/{model}")
        budget_map = {
            model: next(iter(budgets_by_model[model]))
            for model in models
        }
        normalized_games = [_metric_record(record) for record in phase_games]
        stats = aggregate_results(
            normalized_games,
            models=models,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for row in stats["ranking"]:
            points = float(row["wins"]) + 0.5 * float(row["draws"])
            model = str(row["model"])
            row.update(_compute_usage(phase_games, model, points))
            row["pure_win_rate_paired_wilson_ci95"] = _paired_block_wilson(
                phase_games, model, score_rate=False
            )
            row["score_rate_paired_wilson_ci95"] = _paired_block_wilson(
                phase_games, model, score_rate=True
            )

        for pair_name, pair in stats["pairwise"].items():
            first_model, second_model = pair_name.split("-")
            pair_games = [
                record
                for record in phase_games
                if {first_model, second_model}
                == {str(value) for value in record.get("matchup", ())}
            ]
            for model, row in pair["models"].items():
                points = float(row["wins"]) + 0.5 * float(row["draws"])
                row.update(_compute_usage(pair_games, str(model), points))
                row["pure_win_rate_paired_wilson_ci95"] = _paired_block_wilson(
                    pair_games, str(model), score_rate=False
                )
                row["score_rate_paired_wilson_ci95"] = _paired_block_wilson(
                    pair_games, str(model), score_rate=True
                )
        pair_counts: dict[str, int] = {}
        for model_index, first_model in enumerate(models):
            for second_model in models[model_index + 1 :]:
                matching = [
                    record
                    for record in phase_games
                    if {first_model, second_model}
                    == {str(value) for value in record.get("matchup", ())}
                ]
                pair_counts[f"{first_model}-{second_model}"] = len(
                    {str(record.get("pair_id", record["game_id"])) for record in matching}
                )

        phase_summaries[phase_name] = {
            "phase_name": phase_name,
            "track": track,
            "board_size": board_size,
            "budget_seconds_by_model": budget_map,
            "game_count": len(phase_games),
            "paired_opening_count_by_matchup": pair_counts,
            "interval_method": (
                f"{bootstrap_samples}-sample percentile bootstrap clustered by pair_id; "
                "paired-block Wilson is also reported for boundary-stable limits"
            ),
            **stats,
        }

    return {
        "game_count": len(games),
        "phases": phase_summaries,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }
