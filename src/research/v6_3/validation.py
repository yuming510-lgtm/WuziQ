"""Replay and score validation for the V6.3 paired tiebreak."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..v6.validation import validate_game_record


MODELS = ("C", "E5")
COUNTERS = (
    "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations",
    "rollout_plies", "terminal_simulations", "truncated_simulations",
    "root_candidates", "root_min_visits", "reused_nodes",
)


def validate_v6_3_game_record(record: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    replay = validate_game_record(record, spec, verify_proofs=True)
    if record.get("source_type") != "new" or record.get("source_game_id") is not None:
        raise ValueError("V6.3 contains only newly played games")
    for index, move in enumerate(record["moves"]):
        if move.get("opening"):
            continue
        model = str(move.get("model"))
        budget = float(spec["budget_seconds_by_model"][model])
        elapsed = float(move["elapsed_seconds"])
        if elapsed > budget and not bool(move.get("timed_out")):
            raise ValueError(f"V6.3 {spec['game_id']} over-budget move {index} is not flagged")
        for counter in COUNTERS:
            value = move.get(counter, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"V6.3 {spec['game_id']} has invalid {counter} at move {index}")
    return replay


def _empty_standing() -> dict[str, Any]:
    return {
        "games": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
        "black_games": 0, "black_wins": 0, "white_games": 0, "white_wins": 0,
        "decisions": 0, "thinking_seconds": 0.0, "nodes": 0, "cutoffs": 0,
        "simulations": 0, "timeouts": 0, "over_budget_moves": 0,
        "max_search_depth": 0, "max_tree_depth": 0,
    }


def validate_v6_3_records(
    records: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    specs = {str(spec["game_id"]): spec for spec in schedule}
    if len(specs) != 6 or len(schedule) != 6:
        raise ValueError("V6.3 potential schedule must contain exactly six unique IDs")
    seen: set[str] = set()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replayed = {}
    for record in records:
        game_id = str(record.get("game_id", ""))
        if game_id in seen:
            raise ValueError(f"Duplicate V6.3 game ID: {game_id}")
        spec = specs.get(game_id)
        if spec is None:
            raise ValueError(f"V6.3 result is not in the preregistered schedule: {game_id}")
        seen.add(game_id)
        by_group[str(spec["group_id"])].append(record)
        by_stage[str(spec["stage"])].append(record)
        replayed[game_id] = validate_v6_3_game_record(record, spec)

    scheduled_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in schedule:
        scheduled_groups[str(spec["group_id"])].append(spec)
    for group_id, pair_specs in scheduled_groups.items():
        if len(pair_specs) != 2:
            raise ValueError(f"V6.3 group {group_id} is not exactly a paired opening")
        first, second = pair_specs
        if (first["opening"] != second["opening"]
                or first["black_model"] != second["white_model"]
                or first["white_model"] != second["black_model"]
                or {bool(first["color_swap"]), bool(second["color_swap"])} != {False, True}):
            raise ValueError(f"V6.3 group {group_id} does not exchange colors on one opening")
        played = by_group.get(group_id, [])
        if len(played) > 2:
            raise ValueError(f"V6.3 group {group_id} has duplicate or excess games")
        if len(played) == 2 and {bool(row["color_swap"]) for row in played} != {False, True}:
            raise ValueError(f"V6.3 group {group_id} results do not contain both colors")

    initial_specs = [spec for spec in schedule if spec["stage"] == "initial"]
    conditional_specs = [spec for spec in schedule if spec["stage"] == "conditional_tiebreak"]
    if len(initial_specs) != 4 or len(conditional_specs) != 2:
        raise ValueError("V6.3 schedule must comprise four initial and two conditional games")
    initial_records = by_stage.get("initial", [])
    conditional_records = by_stage.get("conditional_tiebreak", [])
    if len(initial_records) > 4 or len(conditional_records) > 2:
        raise ValueError("V6.3 stage has excess games")

    initial_points = {model: 0.0 for model in MODELS}
    for game in initial_records:
        winner = game.get("winner_model")
        if winner is None:
            for model in MODELS:
                initial_points[model] += 0.5
        elif winner in MODELS:
            initial_points[winner] += 1.0
        else:
            raise ValueError(f"V6.3 game {game['game_id']} names an unknown winner")
    initial_complete = len(initial_records) == 4
    tied_after_initial = initial_complete and math.isclose(initial_points["C"], initial_points["E5"])
    if conditional_records and not tied_after_initial:
        raise ValueError("V6.3 conditional tiebreak games exist without a first-stage tie")
    if len(conditional_records) == 1 and require_complete:
        raise ValueError("V6.3 tiebreak is incomplete")

    standings = {model: _empty_standing() for model in MODELS}
    pair_result = {
        "pair": list(MODELS), "games": len(records),
        "wins": {model: 0 for model in MODELS}, "draws": 0,
        "losses": {model: 0 for model in MODELS},
        "points": {model: 0.0 for model in MODELS},
    }
    black_wins = {model: 0 for model in MODELS}
    white_wins = {model: 0 for model in MODELS}
    for game in records:
        winner = game.get("winner_model")
        if winner is None:
            pair_result["draws"] += 1
            for model in MODELS:
                standings[model]["draws"] += 1
                standings[model]["points"] += 0.5
                pair_result["points"][model] += 0.5
        else:
            if winner not in MODELS:
                raise ValueError(f"Unknown winner in {game['game_id']}")
            loser = "E5" if winner == "C" else "C"
            standings[winner]["wins"] += 1
            standings[winner]["points"] += 1.0
            standings[loser]["losses"] += 1
            pair_result["wins"][winner] += 1
            pair_result["losses"][loser] += 1
            pair_result["points"][winner] += 1.0
            winning_color = str(game["winner_color"])
            (black_wins if winning_color == "B" else white_wins)[winner] += 1
        for model in MODELS:
            standings[model]["games"] += 1
        for color, model in (("black", game["black_model"]), ("white", game["white_model"])):
            row = standings[model]
            row[f"{color}_games"] += 1
            if winner == model:
                row[f"{color}_wins"] += 1
        for move in game["moves"]:
            model = move.get("model")
            if model not in standings or move.get("opening"):
                continue
            row = standings[model]
            row["decisions"] += 1
            row["thinking_seconds"] += float(move["elapsed_seconds"])
            row["nodes"] += int(move.get("nodes", 0))
            row["cutoffs"] += int(move.get("cutoffs", 0))
            row["simulations"] += int(move.get("simulations", 0))
            row["timeouts"] += int(bool(move.get("timed_out")))
            row["over_budget_moves"] += int(float(move["elapsed_seconds"]) > 200.0)
            row["max_search_depth"] = max(row["max_search_depth"], int(move.get("completed_depth", 0)))
            row["max_tree_depth"] = max(row["max_tree_depth"], int(move.get("tree_depth", 0)))

    for model, row in standings.items():
        if not math.isclose(row["points"], row["wins"] + row["draws"] / 2):
            raise ValueError(f"V6.3 points mismatch for {model}")
        if not math.isclose(pair_result["points"][model], pair_result["wins"][model] + pair_result["draws"] / 2):
            raise ValueError(f"V6.3 pair score mismatch for {model}")

    complete = initial_complete and (not tied_after_initial or len(conditional_records) == 2)
    if require_complete and not complete:
        if initial_complete and tied_after_initial:
            raise ValueError("V6.3 has a tied first stage but missing the required tiebreak")
        raise ValueError("V6.3 initial four-game stage is incomplete")
    if not initial_complete:
        tiebreak_status = "pending_initial_stage"
    elif not tied_after_initial:
        tiebreak_status = "skipped_not_tied"
    elif len(conditional_records) < 2:
        tiebreak_status = "required"
    elif math.isclose(standings["C"]["points"], standings["E5"]["points"]):
        tiebreak_status = "played_still_tied"
    else:
        tiebreak_status = "played_decided"
    final_points = {model: standings[model]["points"] for model in MODELS}
    leaders = sorted(model for model in MODELS if final_points[model] == max(final_points.values())) if records else []
    return {
        "valid": True,
        "complete": complete,
        "games": len(records),
        "unique_ids": len(seen),
        "replayed_moves": sum(int(value["move_count"]) for value in replayed.values()),
        "initial_games": len(initial_records),
        "conditional_games": len(conditional_records),
        "initial_points": initial_points,
        "initial_tied": bool(tied_after_initial),
        "tiebreak_status": tiebreak_status,
        "standings": standings,
        "pair_result": pair_result,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "leaders": leaders,
        "replays": replayed,
    }

