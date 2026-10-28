"""Replay, budget, paired-opening, and standings checks for V6.2 games."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..v6.validation import validate_game_record


_MODELS = ("C", "D", "E5")
_MOVE_COUNTERS = (
    "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations",
    "rollout_plies", "terminal_simulations", "truncated_simulations",
    "root_candidates", "root_min_visits", "reused_nodes",
)


def validate_v6_2_game_record(record: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Validate one completed game using V6's board replay plus strict V6.2 timing fields."""
    replay = validate_game_record(record, spec, verify_proofs=True)
    for index, move in enumerate(record["moves"]):
        if move.get("opening"):
            continue
        model = str(move.get("model"))
        budget = float(spec["budget_seconds_by_model"][model])
        elapsed = float(move["elapsed_seconds"])
        if elapsed > budget and not bool(move.get("timed_out")):
            raise ValueError(
                f"V6.2 game {spec['game_id']} move {index} exceeded its budget without timeout"
            )
        for counter in _MOVE_COUNTERS:
            value = move.get(counter, 0)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"V6.2 game {spec['game_id']} has invalid {counter} at move {index}"
                )
    if record.get("source_type") != "new" or record.get("source_game_id") is not None:
        raise ValueError("V6.2 standings may contain new games only")
    return replay


def _new_standing() -> dict[str, Any]:
    return {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0.0,
        "black_games": 0,
        "black_wins": 0,
        "white_games": 0,
        "white_wins": 0,
        "thinking_seconds": 0.0,
        "moves": 0,
        "nodes": 0,
        "cutoffs": 0,
        "simulations": 0,
        "max_search_depth": 0,
        "max_tree_depth": 0,
        "timeouts": 0,
        "over_budget_moves": 0,
    }


def validate_v6_2_records(
    records: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate exact-once records and recompute pair results and champion points."""
    specs = {str(game["game_id"]): game for game in schedule}
    if len(specs) != len(schedule):
        raise ValueError("V6.2 schedule contains duplicate game IDs")
    seen: set[str] = set()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validation_results: dict[str, dict[str, Any]] = {}
    for record in records:
        game_id = str(record.get("game_id", ""))
        if game_id in seen:
            raise ValueError(f"Duplicate V6.2 game ID: {game_id}")
        seen.add(game_id)
        spec = specs.get(game_id)
        if spec is None:
            raise ValueError(f"V6.2 results contain an unscheduled game: {game_id}")
        validation_results[game_id] = validate_v6_2_game_record(record, spec)
        by_group[str(spec["group_id"])].append(record)
    if any(len(rows) > 2 for rows in by_group.values()):
        raise ValueError("A V6.2 matchup contains more than two games")
    if require_complete and seen != set(specs):
        missing = sorted(set(specs) - seen)
        raise ValueError(f"V6.2 championship is incomplete; missing {missing}")

    scheduled_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in schedule:
        scheduled_groups[str(spec["group_id"])].append(spec)
    for group_id, specs_for_group in scheduled_groups.items():
        if len(specs_for_group) != 2:
            raise ValueError(f"V6.2 matchup {group_id} is not scheduled for two games")
        first, second = specs_for_group
        if (first["opening"] != second["opening"]
                or first["black_model"] != second["white_model"]
                or first["white_model"] != second["black_model"]
                or {bool(first["color_swap"]), bool(second["color_swap"])} != {False, True}):
            raise ValueError(f"V6.2 matchup {group_id} lacks its exact color swap")
        rows = by_group.get(group_id, [])
        if len(rows) == 2 and {bool(row["color_swap"]) for row in rows} != {False, True}:
            raise ValueError(f"V6.2 played matchup {group_id} does not swap colors")
        if require_complete and len(rows) != 2:
            raise ValueError(f"V6.2 matchup {group_id} is missing a paired game")

    standings = {model: _new_standing() for model in _MODELS}
    pair_rows: list[dict[str, Any]] = []
    for group_id in sorted(scheduled_groups):
        specs_for_group = scheduled_groups[group_id]
        pair = tuple(specs_for_group[0]["pair"])
        games = sorted(by_group.get(group_id, []), key=lambda row: bool(row["color_swap"]))
        pair_result = {"pair": list(pair), "group_id": group_id, "games": len(games),
                       "wins": {model: 0 for model in pair}, "draws": 0,
                       "losses": {model: 0 for model in pair}, "points": {model: 0.0 for model in pair}}
        for game in games:
            winner = game.get("winner_model")
            if winner is None:
                for model in pair:
                    standings[model]["draws"] += 1
                    standings[model]["points"] += 0.5
                    pair_result["draws"] += 1
                    pair_result["points"][model] += 0.5
            else:
                loser = pair[1] if winner == pair[0] else pair[0]
                standings[winner]["wins"] += 1
                standings[winner]["points"] += 1.0
                standings[loser]["losses"] += 1
                pair_result["wins"][winner] += 1
                pair_result["points"][winner] += 1.0
                pair_result["losses"][loser] += 1

            for model in pair:
                standings[model]["games"] += 1
            for color, model in (("black", game["black_model"]), ("white", game["white_model"])):
                row = standings[model]
                row[f"{color}_games"] += 1
                if winner == model:
                    row[f"{color}_wins"] += 1
            for move in game["moves"]:
                model = move.get("model")
                if model not in standings:
                    continue
                row = standings[model]
                if move.get("opening"):
                    continue
                row["moves"] += 1
                row["thinking_seconds"] += float(move["elapsed_seconds"])
                row["nodes"] += int(move.get("nodes", 0))
                row["cutoffs"] += int(move.get("cutoffs", 0))
                row["simulations"] += int(move.get("simulations", 0))
                row["max_search_depth"] = max(row["max_search_depth"], int(move.get("completed_depth", 0)))
                row["max_tree_depth"] = max(row["max_tree_depth"], int(move.get("tree_depth", 0)))
                row["timeouts"] += int(bool(move.get("timed_out")))
                row["over_budget_moves"] += int(float(move["elapsed_seconds"]) > 200.0)
        for model, row in pair_result["points"].items():
            if not math.isclose(row, pair_result["wins"][model] + pair_result["draws"] / 2):
                raise ValueError(f"V6.2 pair points are inconsistent for {model}")
        pair_rows.append(pair_result)

    for row in standings.values():
        if not math.isclose(row["points"], row["wins"] + row["draws"] / 2):
            raise ValueError("V6.2 total points do not equal wins + draws/2")
    max_points = max((row["points"] for row in standings.values()), default=0.0)
    champions = sorted(model for model, row in standings.items() if row["points"] == max_points) if records else []
    return {
        "valid": True,
        "games": len(records),
        "groups": sum(len(rows) == 2 for rows in by_group.values()),
        "partial_groups": sum(len(rows) == 1 for rows in by_group.values()),
        "replayed_moves": sum(int(value["move_count"]) for value in validation_results.values()),
        "unique_ids": len(seen),
        "standings": standings,
        "pair_results": pair_rows,
        "champions": champions,
    }
