"""Replay-validated outcome, advancement and resource analysis for V5.1."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .abcd_ladder_v5 import _project_root, load_config as load_v5_config
from .abcd_ladder_v5_analysis import validate_records as validate_v5_records
from .abcd_ladder_v5_1 import build_schedule, engine_config, load_config
from .abcd_ladder_v5_1_reuse import derive_state
from .abcd_ladder_v5_1_runner import _rekey_v5_record, _validate_v51_record
from .checkpoint import load_game_records, write_json_atomic

_FAMILIES = ("A", "B", "C", "D")
_BUDGETS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")
_FAMILY_PAIRS = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
_Z95 = 1.959963984540054


def model_result_points(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    wins = sum(row.get("winner_model") == model for row in rows)
    draws = sum(row.get("winner_model") is None for row in rows)
    losses = sum(row.get("winner_model") not in (model, None) for row in rows)
    games = len(rows)
    points = wins + draws / 2.0
    return {
        "wins": int(wins), "draws": int(draws), "losses": int(losses), "games": games,
        "points": points, "score_rate": points / games if games else None,
    }


def wilson_fractional(values: list[float]) -> list[float] | None:
    """Descriptive Wilson interval for group-level fractions in [0, 1]."""
    if not values:
        return None
    if any(not 0 <= float(value) <= 1 for value in values):
        raise ValueError("Wilson interval inputs must be fractions in [0, 1]")
    n = float(len(values))
    p = sum(float(value) for value in values) / n
    z2 = _Z95**2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = _Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pair_key(pair: list[str] | tuple[str, str]) -> str:
    return "-".join(pair)


def _empty_count() -> dict[str, Any]:
    return {"games": 0, "wins_first": 0, "draws": 0, "wins_second": 0,
            "first_points": 0.0, "second_points": 0.0, "group_fractions_first": [],
            "group_fractions_second": [], "groups": 0}


def _accumulate_pair(count: dict[str, Any], rows: list[dict[str, Any]], first: str, second: str) -> None:
    first_result = model_result_points(rows, first)
    second_result = model_result_points(rows, second)
    count["games"] += len(rows)
    count["groups"] += 1
    count["wins_first"] += first_result["wins"]
    count["draws"] += first_result["draws"]
    count["wins_second"] += second_result["wins"]
    count["first_points"] += first_result["points"]
    count["second_points"] += second_result["points"]
    count["group_fractions_first"].append(first_result["points"] / 2.0)
    count["group_fractions_second"].append(second_result["points"] / 2.0)


def _finish_pair(key: str, count: dict[str, Any]) -> dict[str, Any]:
    games = int(count["games"])
    first_points = float(count["first_points"])
    second_points = float(count["second_points"])
    return {
        "pair": key,
        "groups": int(count["groups"]),
        "games": games,
        "first_wins": int(count["wins_first"]),
        "draws": int(count["draws"]),
        "second_wins": int(count["wins_second"]),
        "first_points": first_points if games else None,
        "second_points": second_points if games else None,
        "first_score_rate": first_points / games if games else None,
        "second_score_rate": second_points / games if games else None,
        "first_ci95_paired_groups": wilson_fractional(count["group_fractions_first"]),
        "second_ci95_paired_groups": wilson_fractional(count["group_fractions_second"]),
    }


def _source_rows() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = _project_root()
    config_path = root / "research/v5/results/config.json"
    records_path = root / "research/v5/results/9x9_games.jsonl"
    v5_config = load_v5_config(config_path)
    rows = load_game_records(records_path)
    validation = validate_v5_records(rows, v5_config, require_complete=True)
    return {str(row["game_id"]): row for row in rows}, {
        "experiment_id": v5_config["experiment_id"],
        "commit": rows[0]["implementation_commit"] if rows else None,
        "config_sha256": _sha(config_path.read_bytes()),
        "validation": validation,
    }


def validate_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    schedule: list[dict[str, Any]] | None = None,
    require_complete: bool = False,
    reuse_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay every full game, enforce exact IDs/pairs and verify any V5 reuse."""
    frozen = load_config(config)
    groups = schedule if schedule is not None else build_schedule(frozen)
    if len(groups) != 168:
        raise ValueError("V5.1 validation requires the complete frozen 168-group schedule")
    game_specs = {str(game["game_id"]): game for group in groups for game in group["games"]}
    if len(game_specs) != 336:
        raise ValueError("V5.1 schedule contains duplicate game IDs")
    engine = engine_config(frozen)
    seen: set[str] = set()
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows, source_info = _source_rows()
    manifest_by_game: dict[str, dict[str, Any]] = {}
    if reuse_manifest is not None:
        for group in reuse_manifest.get("groups", []):
            for source_map in group.get("games", []):
                manifest_by_game[str(source_map["game_id"])] = source_map

    for row in records:
        game_id = str(row.get("game_id", ""))
        if not game_id:
            raise ValueError("V5.1 game record is missing a game ID")
        if game_id in seen:
            raise ValueError(f"Duplicate V5.1 game ID: {game_id}")
        seen.add(game_id)
        spec = game_specs.get(game_id)
        if spec is None:
            raise ValueError(f"Unscheduled V5.1 game: {game_id}")
        _validate_v51_record(row, spec, engine)
        rows_by_group[str(spec["group_id"])].append(row)
        if row.get("source_type") == "reused_v5":
            source_id = str(row.get("source_game_id", ""))
            source = source_rows.get(source_id)
            if source is None:
                raise ValueError(f"V5.1 reused game has missing V5 source {source_id}")
            mapped = _rekey_v5_record(source, spec)
            if mapped != row:
                raise ValueError(f"V5.1 reused game differs from its exact V5 source: {game_id}")
            if reuse_manifest is not None and manifest_by_game.get(game_id, {}).get("source_game_id") != source_id:
                raise ValueError(f"V5.1 reused game is absent from its frozen provenance manifest: {game_id}")
        elif row.get("source_type") != "new":
            raise ValueError(f"Unknown V5.1 game source type in {game_id}")
    if any(len(rows) > 2 for rows in rows_by_group.values()):
        raise ValueError("V5.1 group contains more than two games")
    for group_id, rows in rows_by_group.items():
        expected = {str(game["game_id"]) for game in next(g for g in groups if g["group_id"] == group_id)["games"]}
        if not {str(row["game_id"]) for row in rows}.issubset(expected):
            raise ValueError(f"V5.1 group {group_id} contains a non-scheduled game")

    state = derive_state(records, frozen, groups)
    if require_complete and not state["complete"]:
        raise ValueError(f"V5.1 route graph has not completed at {state['pending_group_id']}")
    if require_complete and any(len(rows) != 2 for rows in rows_by_group.values()):
        raise ValueError("V5.1 has an incomplete color-swapped pair")
    actual_reused_games = sum(row.get("source_type") == "reused_v5" for row in records)
    actual_reused_groups = sum(
        len(rows) == 2 and all(row.get("source_type") == "reused_v5" for row in rows)
        for rows in rows_by_group.values()
    )
    if actual_reused_games != sum(row.get("source_type") == "reused_v5" for row in records):
        raise ValueError("V5.1 reused game count mismatch")
    if reuse_manifest is not None:
        if reuse_manifest.get("source_experiment_id") != source_info["experiment_id"]:
            raise ValueError("V5.1 reuse manifest references a different V5 experiment")
        if reuse_manifest.get("source_config_sha256") != source_info["config_sha256"]:
            raise ValueError("V5.1 reuse manifest pins a different V5 config")
        if int(reuse_manifest.get("reused_games", 0)) != 2 * int(reuse_manifest.get("reused_groups", 0)):
            raise ValueError("V5.1 reuse manifest has inconsistent group and game counts")
    return {
        "valid": True,
        "complete": bool(state["complete"]),
        "games": len(records),
        "groups": sum(len(rows) == 2 for rows in rows_by_group.values()),
        "partial_groups": sum(len(rows) == 1 for rows in rows_by_group.values()),
        "unique_game_ids": len(seen),
        "reused_games": actual_reused_games,
        "reused_groups": actual_reused_groups,
        "eligible_v5_reuse_groups": int(reuse_manifest.get("reused_groups", 0)) if reuse_manifest else None,
        "route_state": state,
        "v5_source_validation": source_info,
    }


def audit_v5_report_scores(
    v5_results_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Recompute V5 matrix points from raw games and compare to its committed table."""
    results = Path(v5_results_dir)
    config = load_v5_config(results / "config.json")
    records = load_game_records(results / "9x9_games.jsonl")
    source_validation = validate_v5_records(records, config, require_complete=True)
    text = Path(report_path).read_text(encoding="utf-8")
    marker = "### 完整胜负矩阵"
    if marker not in text:
        raise ValueError("V5 report does not contain the complete win-loss matrix heading")
    section = text.split(marker, 1)[1].split("\n### ", 1)[0]
    table: dict[str, dict[str, str]] = {}
    for line in section.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] not in _FAMILIES or len(cells) < 5:
            continue
        table[cells[0]] = dict(zip(_FAMILIES, cells[1:5], strict=True))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(row["pair"])].append(row)
    rows: list[dict[str, Any]] = []
    for first, second in _FAMILY_PAIRS:
        pair_rows = grouped.get((first, second), [])
        points = model_result_points(pair_rows, first)
        cell = table.get(first, {}).get(second, "")
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)/([0-9]+)分", cell)
        if pair_rows and match is None:
            raise ValueError(f"Could not parse V5 reported score for {first}-{second}")
        reported = float(match.group(1)) if match else None
        rows.append({
            "pair": f"{first}-{second}",
            "games": points["games"], "wins": points["wins"], "draws": points["draws"],
            "losses": points["losses"],
            "expected_first_points": points["points"],
            "reported_first_points": reported,
            "first_points_match": reported == points["points"] if reported is not None else None,
            "first_score_rate": points["score_rate"],
        })
    mismatches = [row["pair"] for row in rows if row["first_points_match"] is False]
    return {
        "raw_games_valid": bool(source_validation["valid"] and source_validation["complete"]),
        "raw_games": len(records),
        "source_report": str(Path(report_path)),
        "formula": "wins + draws / 2",
        "mismatched_report_pairs": mismatches,
        "pairs": rows,
    }


def _score_for_group(rows: list[dict[str, Any]], model: str) -> float:
    return float(model_result_points(rows, model)["points"])


def summarize_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    schedule: list[dict[str, Any]] | None = None,
    reuse_manifest: dict[str, Any] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    frozen = load_config(config)
    groups = schedule if schedule is not None else build_schedule(frozen)
    validation = validate_records(records, frozen, schedule=groups, require_complete=require_complete,
                                  reuse_manifest=reuse_manifest)
    state = validation["route_state"]
    game_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    game_by_id = {str(row["game_id"]): row for row in records}
    for row in records:
        game_rows_by_group[str(row["group_id"])].append(row)
    groups_by_id = {str(group["group_id"]): group for group in groups}

    model_counts = {family: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0,
                             "black_games": 0, "black_points": 0.0, "white_games": 0, "white_points": 0.0}
                    for family in _FAMILIES}
    version_counts: dict[tuple[str, str], dict[str, Any]] = {
        (family, budget): {"model": family, "budget_label": budget, "games": 0, "wins": 0,
                           "draws": 0, "losses": 0, "points": 0.0, "decisions": 0,
                           "thinking_seconds": 0.0, "score_per_thinking_second": None}
        for family, budget in [("A", "natural"), *((model, label) for model in ("B", "C", "D") for label in _BUDGETS)]
    }
    side_stats = {family: {"black_games": 0, "black_wins": 0, "black_points": 0.0,
                           "white_games": 0, "white_wins": 0, "white_points": 0.0}
                  for family in _FAMILIES}
    version_pair_counts: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(_empty_count)
    game_summaries: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: str(item["game_id"])):
        first, second = row["pair"]
        winner = row.get("winner_model")
        versions = row["model_versions"]
        for family in (first, second):
            result = model_result_points([row], family)
            entry = model_counts[family]
            for key in ("wins", "draws", "losses"):
                entry[key] += result[key]
            entry["games"] += 1
            entry["points"] += result["points"]
            side = "black" if row["black_model"] == family else "white"
            entry[f"{side}_games"] += 1
            entry[f"{side}_points"] += result["points"]
            side_stats[family][f"{side}_games"] += 1
            side_stats[family][f"{side}_points"] += result["points"]
            if family == "A":
                label = "natural"
            else:
                label = next(rung["label"] for rung in frozen["budget_ladder"]
                             if float(rung["seconds"]) == float(row["budget_seconds_by_model"][family]))
            version = version_counts[(family, label)]
            version["games"] += 1
            version["wins"] += result["wins"]
            version["draws"] += result["draws"]
            version["losses"] += result["losses"]
            version["points"] += result["points"]
        black_winner = row.get("winner_color") == "B"
        white_winner = row.get("winner_color") == "W"
        if black_winner:
            side_stats[row["black_model"]]["black_wins"] += 1
        if white_winner:
            side_stats[row["white_model"]]["white_wins"] += 1
        group_id = str(row["group_id"])
        game_summaries.append({
            "game_id": row["game_id"], "group_id": group_id, "pair": _pair_key(row["pair"]),
            "first_model": first, "second_model": second,
            "first_version": versions[first], "second_version": versions[second],
            "opening_seed": int(row["opening_seed"]), "color_swap": bool(row["color_swap"]),
            "black_model": row["black_model"], "white_model": row["white_model"],
            "winner_model": winner or "draw", "winner_color": row.get("winner_color"),
            "outcome": row["outcome"], "termination": row["termination"],
            "move_count": int(row["move_count"]), "thinking_seconds": float(row["thinking_seconds_total"]),
            "timeouts": int(row["timeout_count"]), "nodes": int(row["search_nodes_total"]),
            "simulations": int(row["simulations_total"]), "max_tree_depth": int(row["tree_depth_max"]),
            "source_type": row["source_type"],
            "source_game_id": row.get("source_game_id"),
        })

    group_summaries: list[dict[str, Any]] = []
    route_level_group_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group_id, rows in sorted(game_rows_by_group.items()):
        group = groups_by_id[group_id]
        if len(rows) != 2:
            continue
        first, second = group["pair"]
        route_state = state["groups"].get(group_id, {})
        active_route_ids = list(route_state.get("active_route_ids", []))
        inactive_refs = sorted(set(group["route_ids"]) - set(active_route_ids))
        first_points = model_result_points(rows, first)["points"]
        second_points = model_result_points(rows, second)["points"]
        exact_key = (first, second, group["model_versions"][first], group["model_versions"][second])
        _accumulate_pair(version_pair_counts[exact_key], rows, first, second)
        route_scores: dict[str, float] = {}
        for ref in group["route_refs"]:
            route_id, level = str(ref["route_id"]), int(ref["level_index"])
            if route_id not in active_route_ids:
                continue
            challenger = route_id.split(">", 1)[0].split("@", 1)[0]
            route_scores[route_id] = _score_for_group(rows, challenger)
            route_level_group_rows[(route_id, level)].extend(rows)
        outcome_counts = Counter(row.get("winner_model") or "draw" for row in rows)
        group_summaries.append({
            "group_id": group_id, "pair": _pair_key(group["pair"]),
            "model_versions": dict(group["model_versions"]),
            "budget_seconds_by_model": dict(group["budget_seconds_by_model"]),
            "route_ids": list(group["route_ids"]), "counted_route_ids": active_route_ids,
            "shared_result_after_failure_route_ids": inactive_refs,
            "opening_seed": int(group["opening_seed"]), "games": 2,
            "wins_first": sum(row.get("winner_model") == first for row in rows),
            "draws": outcome_counts.get("draw", 0), "wins_second": sum(row.get("winner_model") == second for row in rows),
            "first_points": first_points, "second_points": second_points,
            "first_score_rate": first_points / 2.0, "second_score_rate": second_points / 2.0,
            "black_wins": sum(row.get("winner_color") == "B" for row in rows),
            "white_wins": sum(row.get("winner_color") == "W" for row in rows),
            "route_scores": route_scores,
            "source_type": "reused_v5" if all(row.get("source_type") == "reused_v5" for row in rows) else "new",
        })

    family_pair_counts = {_pair_key(pair): _empty_count() for pair in _FAMILY_PAIRS}
    for group_row in group_summaries:
        key = str(group_row["pair"])
        count = family_pair_counts[key]
        count["groups"] += 1
        count["games"] += 2
        count["wins_first"] += int(group_row["wins_first"])
        count["draws"] += int(group_row["draws"])
        count["wins_second"] += int(group_row["wins_second"])
        count["first_points"] += float(group_row["first_points"])
        count["second_points"] += float(group_row["second_points"])
        count["group_fractions_first"].append(float(group_row["first_points"]) / 2.0)
        count["group_fractions_second"].append(float(group_row["second_points"]) / 2.0)
    family_pair_rows = [_finish_pair(key, family_pair_counts[key]) for key in family_pair_counts]
    version_pair_rows: list[dict[str, Any]] = []
    schedule_status: dict[str, dict[str, Any]] = state["groups"]
    for group in groups:
        first, second = group["pair"]
        key = (first, second, group["model_versions"][first], group["model_versions"][second])
        count = version_pair_counts.get(key, _empty_count())
        row = _finish_pair(f"{first}@{key[2]}-{second}@{key[3]}", count)
        row.update({
            "group_id": group["group_id"], "model_versions": dict(group["model_versions"]),
            "route_ids": list(group["route_ids"]), "route_refs": deepcopy_list(group["route_refs"]),
            "opening_seed": int(group["opening_seed"]),
            "status": schedule_status.get(group["group_id"], {}).get("status", "not_reached"),
        })
        if row["games"]:
            # The group has exactly two results; expose absolute points as well as rates.
            actual = next(item for item in group_summaries if item["group_id"] == group["group_id"])
            row["first_points"] = actual["first_points"]
            row["second_points"] = actual["second_points"]
        version_pair_rows.append(row)

    route_rows: list[dict[str, Any]] = []
    for route_id, route in state["routes"].items():
        for level in route["levels"]:
            group_id = str(level["group_id"])
            rows = route_level_group_rows.get((route_id, int(level.get("level_index", -1))), [])
            # Each scheduled route rung corresponds to exactly one paired group.
            counts = model_result_points(rows, str(route["challenger_family"])) if rows else None
            route_rows.append({
                "route_id": route_id,
                "challenger_family": route["challenger_family"],
                "challenger_version": route["challenger_version"],
                "opponent_family": route["opponent_family"],
                "route_status": route["status"],
                "highest_passed_budget": route["highest_passed_budget"],
                "failed_budget": route["failed_budget"],
                "level_index": level.get("level_index"),
                "group_id": group_id,
                "opponent_budget_label": level["opponent_budget_label"],
                "status": level["status"],
                "reason": level.get("reason"),
                "games": counts["games"] if counts else None,
                "wins": counts["wins"] if counts else None,
                "draws": counts["draws"] if counts else None,
                "losses": counts["losses"] if counts else None,
                "points": counts["points"] if counts else None,
                "score_rate": counts["score_rate"] if counts else None,
                "ci95_paired_groups": wilson_fractional([counts["points"] / 2.0]) if counts else None,
            })

    usage_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for family in ("A", "B", "C", "D"):
        labels = ("natural",) if family == "A" else _BUDGETS
        for label in labels:
            usage_acc[(family, label)] = {
                "model": family, "budget_label": label, "budget_seconds": 0.0 if label == "natural"
                else float(next(row["seconds"] for row in frozen["budget_ladder"] if row["label"] == label)),
                "decisions": 0, "elapsed_seconds": 0.0, "nodes": 0, "cutoffs": 0,
                "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0,
                "truncated_simulations": 0, "depth_sum": 0, "max_depth": 0,
                "timeouts": 0, "reused_nodes": 0, "move_elapsed": [],
            }
    for record in records:
        for move in record["moves"]:
            family = move.get("model")
            if family is None:
                continue
            budget = record["budget_seconds_by_model"][family]
            label = "natural" if budget is None else next(
                rung["label"] for rung in frozen["budget_ladder"] if float(rung["seconds"]) == float(budget)
            )
            row = usage_acc[(str(family), label)]
            row["decisions"] += 1
            row["elapsed_seconds"] += float(move["elapsed_seconds"])
            row["move_elapsed"].append(float(move["elapsed_seconds"]))
            for target, field in (("nodes", "nodes"), ("cutoffs", "cutoffs"),
                                  ("simulations", "simulations"), ("rollout_plies", "rollout_plies"),
                                  ("terminal_simulations", "terminal_simulations"),
                                  ("truncated_simulations", "truncated_simulations"),
                                  ("reused_nodes", "reused_nodes")):
                row[target] += int(move[field])
            row["depth_sum"] += int(move["completed_depth"])
            row["max_depth"] = max(row["max_depth"], int(move["tree_depth"]))
            row["timeouts"] += int(bool(move["timed_out"]))
    model_usage: list[dict[str, Any]] = []
    for key, row in usage_acc.items():
        decisions = int(row["decisions"])
        seconds = float(row["elapsed_seconds"])
        row.pop("move_elapsed")
        row.update({
            "mean_seconds_per_decision": seconds / decisions if decisions else None,
            "mean_completed_depth": row["depth_sum"] / decisions if decisions else None,
            "mean_nodes_per_decision": row["nodes"] / decisions if decisions else None,
            "mean_cutoffs_per_decision": row["cutoffs"] / decisions if decisions else None,
            "cutoffs_per_node": row["cutoffs"] / row["nodes"] if row["nodes"] else None,
            "mean_simulations_per_decision": row["simulations"] / decisions if decisions else None,
            "simulations_per_second": row["simulations"] / seconds if seconds else None,
            "nodes_per_second": row["nodes"] / seconds if seconds else None,
            "timeout_rate": row["timeouts"] / decisions if decisions else None,
            "score_per_thinking_second": version_counts.get(key, {}).get("points", 0.0) / seconds if seconds else None,
        })
        model_usage.append(row)

    model_summary: dict[str, dict[str, Any]] = {}
    for family, counts in model_counts.items():
        games = int(counts["games"])
        points = float(counts["points"])
        model_summary[family] = {
            **counts,
            "score_rate": points / games if games else None,
            "ci95_paired_groups": wilson_fractional([
                (group["first_points"] if group["pair"].split("-")[0] == family else group["second_points"]) / 2.0
                for group in group_summaries if family in group["pair"].split("-")
            ]),
        }
    version_summaries: list[dict[str, Any]] = []
    for key, entry in version_counts.items():
        games = int(entry["games"])
        points = float(entry["points"])
        usage = usage_acc[key]
        seconds = float(usage["elapsed_seconds"])
        version_summaries.append({
            **entry,
            "score_rate": points / games if games else None,
            "mean_seconds_per_decision": seconds / usage["decisions"] if usage["decisions"] else None,
            "mean_completed_depth": usage["depth_sum"] / usage["decisions"] if usage["decisions"] else None,
            "max_completed_depth": usage["max_depth"],
            "total_nodes": usage["nodes"], "total_cutoffs": usage["cutoffs"],
            "total_simulations": usage["simulations"], "mean_simulations_per_decision": (
                usage["simulations"] / usage["decisions"] if usage["decisions"] else None
            ),
            "simulations_per_second": usage["simulations"] / seconds if seconds else None,
            "score_per_thinking_second": points / seconds if seconds else None,
        })

    total_seconds = sum(float(row["thinking_seconds_total"]) for row in records)
    nodes_total = sum(int(row["search_nodes_total"]) for row in records)
    simulations_total = sum(int(row["simulations_total"]) for row in records)
    timeout_total = sum(int(row["timeout_count"]) for row in records)
    return {
        "experiment_id": frozen["experiment_id"],
        "board_size": 9,
        "maximum_routes": 66, "maximum_groups": 168, "maximum_games": 336,
        "actual_games": len(records), "actual_groups": len(group_summaries),
        "partial_groups": validation["partial_groups"],
        "complete": validation["complete"],
        "route_state": state,
        "validation": {key: value for key, value in validation.items() if key != "route_state"},
        "models": model_summary,
        "versions": version_summaries,
        "pairwise": family_pair_rows,
        "version_pairwise": version_pair_rows,
        "groups": group_summaries,
        "schedule_groups": [
            {"group_id": group["group_id"], "pair": _pair_key(group["pair"]),
             "model_versions": dict(group["model_versions"]), "opening_seed": int(group["opening_seed"]),
             "route_ids": list(group["route_ids"]),
             "status": state["groups"].get(group["group_id"], {}).get("status", "not_reached"),
             "games": len(game_rows_by_group.get(str(group["group_id"]), []))}
            for group in groups
        ],
        "games": game_summaries,
        "routes": [row for row in route_rows],
        "model_usage": model_usage,
        "side_effects": side_stats,
        "actual_metrics": {
            "ai_thinking_seconds": total_seconds,
            "search_nodes_total": nodes_total,
            "mcts_simulations_total": simulations_total,
            "timeout_moves": timeout_total,
            "actual_reused_games": validation["reused_games"],
            "actual_reused_groups": validation["reused_groups"],
            "eligible_v5_reuse_groups": validation["eligible_v5_reuse_groups"],
            "unique_game_ids": len(game_by_id),
            "terminations": dict(sorted(Counter(row["termination"] for row in records).items())),
        },
        "history_v5_score_audit": audit_v5_report_scores(
            _project_root() / "research/v5/results", _project_root() / "research/v5/REPORT.md"
        ),
    }


def deepcopy_list(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in value]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(results_dir: str | Path, output_dir: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(results_dir)
    output = Path(output_dir)
    config = load_config(root / "config.json")
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    reuse = json.loads((root / "reuse_manifest.json").read_text(encoding="utf-8"))
    records = load_game_records(root / "9x9_games.jsonl")
    summary = summarize_records(records, config, schedule=schedule, reuse_manifest=reuse,
                                require_complete=require_complete)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "summary.json", summary)
    _write_csv(output / "family_pairwise.csv", summary["pairwise"])
    _write_csv(output / "version_pairwise.csv", summary["version_pairwise"])
    _write_csv(output / "group_summary.csv", summary["groups"])
    _write_csv(output / "schedule_group_status.csv", summary["schedule_groups"])
    _write_csv(output / "game_summary.csv", summary["games"])
    _write_csv(output / "route_ladder.csv", summary["routes"])
    _write_csv(output / "version_summary.csv", summary["versions"])
    _write_csv(output / "model_usage.csv", summary["model_usage"])
    _write_csv(output / "color_effect.csv", [dict(model=model, **stats) for model, stats in summary["side_effects"].items()])
    _write_csv(output / "v5_score_correction.csv", summary["history_v5_score_audit"]["pairs"])
    return summary
