"""V5 reuse audit and independent V5.1 route advancement."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .abcd_ladder_v5 import _project_root, build_schedule as build_v5_schedule, load_config as load_v5_config
from .abcd_ladder_v5_1 import build_routes, build_schedule, load_config
from .abcd_ladder_v5_analysis import validate_records as validate_v5_records
from .checkpoint import load_game_records


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _winner_score(winner: str | None, model: str) -> float:
    return 0.5 if winner is None else 1.0 if winner == model else 0.0


def _v5_group_key(group: dict[str, Any]) -> tuple[Any, ...]:
    opening = group.get("opening")
    budget = group.get("budget_seconds_by_model")
    if budget is None:
        games = group.get("games", [])
        if not games:
            raise ValueError("Cannot verify budget for a group without game specifications")
        budget = games[0]["budget_seconds_by_model"]
        opening = games[0]["opening"]
        if any(game["budget_seconds_by_model"] != budget for game in games):
            raise ValueError("Color-swapped games disagree on budget configuration")
        if any(game["opening"] != opening for game in games):
            raise ValueError("Color-swapped games disagree on opening configuration")
    return (
        tuple(group["pair"]),
        _json_key(budget),
        int(group["opening_seed"]),
        _json_key(opening),
    )


def _v51_group_key(group: dict[str, Any]) -> tuple[Any, ...]:
    return _v5_group_key(group)


def prepare_v5_reuse(
    schedule: list[dict[str, Any]],
    v5_results_dir: str | Path,
    v5_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pin every exact V5 matchup to its V5 seed; never select on its outcome."""
    target_config = load_config(v5_config)
    results = Path(v5_results_dir)
    source_config_path = results / "config.json"
    source_games_path = results / "9x9_games.jsonl"
    if not source_config_path.is_file() or not source_games_path.is_file():
        raise FileNotFoundError("V5 reuse requires committed V5 config and official game records")
    source_config_raw = source_config_path.read_bytes()
    source_config = load_v5_config(source_config_path)
    if _sha256((_project_root() / target_config["engine_config_path"]).read_bytes()) != target_config["engine_config_sha256"]:
        raise ValueError("V5.1 V5 engine configuration pin does not match")
    if _json_key(source_config) != _json_key(load_v5_config(_project_root() / target_config["engine_config_path"])):
        raise ValueError("V5 source config differs from the V5 engine config pinned by V5.1")
    if source_config["engine_sha256"] != target_config["engine_sha256"]:
        raise ValueError("V5 reuse engine hashes differ from V5.1")

    source_records = load_game_records(source_games_path)
    validation = validate_v5_records(source_records, source_config, require_complete=True)
    old_schedule = build_v5_schedule(source_config)
    old_groups = {group["group_id"]: group for group in old_schedule}
    old_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_records:
        old_rows_by_group[str(row["group_id"])].append(row)

    new_schedule = deepcopy(schedule)
    candidates: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for group in new_schedule:
        candidates[_v51_group_key(group)].append(group)
    used_new_ids: set[str] = set()
    manifest_groups: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    for old_group_id, old_rows in sorted(old_rows_by_group.items()):
        if len(old_rows) != 2:
            rejections.append({"source_group_id": old_group_id, "reason": "source_pair_incomplete"})
            continue
        old_group = old_groups[old_group_id]
        matches = candidates.get(_v5_group_key(old_group), [])
        if len(matches) != 1:
            rejections.append({"source_group_id": old_group_id, "reason": "no_unique_exact_v5_1_group_match"})
            continue
        new_group = matches[0]
        new_id = str(new_group["group_id"])
        if new_id in used_new_ids:
            raise ValueError(f"More than one V5 group maps to V5.1 group {new_id}")
        old_by_swap = {bool(row["color_swap"]): row for row in old_rows}
        new_by_swap = {bool(row["color_swap"]): row for row in new_group["games"]}
        if set(old_by_swap) != {False, True} or set(new_by_swap) != {False, True}:
            rejections.append({"source_group_id": old_group_id, "reason": "color_swap_pair_mismatch"})
            continue
        source_map: list[dict[str, Any]] = []
        for color_swap in (False, True):
            source = old_by_swap[color_swap]
            target = new_by_swap[color_swap]
            if (source["black_model"], source["white_model"]) != (target["black_model"], target["white_model"]):
                source_map = []
                break
            if source["budget_seconds_by_model"] != target["budget_seconds_by_model"]:
                source_map = []
                break
            if source["opening_seed"] != target["opening_seed"] or source["opening"] != target["opening"]:
                source_map = []
                break
            # The original deterministic seed is carried into V5.1 before the
            # schedule is frozen, making this exact seeded matchup reusable.
            target["game_seed"] = int(source["game_seed"])
            target["source_type"] = "reused_v5"
            target["source_game_id"] = str(source["game_id"])
            target["source_group_id"] = old_group_id
            target["source_game_seed"] = int(source["game_seed"])
            target["source_implementation_commit"] = str(source["implementation_commit"])
            source_map.append({
                "game_id": str(target["game_id"]),
                "source_game_id": str(source["game_id"]),
                "game_seed": int(source["game_seed"]),
                "color_swap": color_swap,
                "source_implementation_commit": str(source["implementation_commit"]),
            })
        if len(source_map) != 2:
            rejections.append({"source_group_id": old_group_id, "reason": "game_level_condition_mismatch"})
            continue
        new_group["reuse_source"] = {"experiment_id": source_config["experiment_id"], "group_id": old_group_id, "games": source_map}
        new_group["source_type"] = "reused_v5"
        used_new_ids.add(new_id)
        manifest_groups.append({
            "group_id": new_id,
            "source_group_id": old_group_id,
            "pair": list(old_group["pair"]),
            "model_versions": dict(new_group["model_versions"]),
            "opening_seed": int(new_group["opening_seed"]),
            "games": source_map,
        })
    manifest = {
        "source_experiment_id": source_config["experiment_id"],
        "source_commit": source_records[0]["implementation_commit"] if source_records else "unknown",
        "source_config_sha256": _sha256(source_config_raw),
        "engine_sha256": dict(source_config["engine_sha256"]),
        "reused_groups": len(manifest_groups),
        "reused_games": sum(len(group["games"]) for group in manifest_groups),
        "groups": manifest_groups,
        "rejections": rejections,
    }
    manifest["source_validation"] = {
        "valid": bool(validation["valid"]),
        "complete": bool(validation["complete"]),
        "games": int(validation["games"]),
        "groups": int(validation["groups"]),
    }
    return new_schedule, manifest


def derive_state(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    schedule: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute advancement independently for every fixed-version challenger route."""
    frozen = load_config(config)
    groups = deepcopy(schedule if schedule is not None else build_schedule(frozen))
    route_specs = build_routes(frozen)
    by_game: dict[str, dict[str, Any]] = {}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_game_ids: set[str] = set()
    for group in groups:
        for game in group["games"]:
            by_game[str(game["game_id"])] = game
    for record in records:
        game_id = str(record.get("game_id", ""))
        if game_id in seen_game_ids:
            raise ValueError(f"Duplicate V5.1 game ID: {game_id}")
        spec = by_game.get(game_id)
        if spec is None:
            raise ValueError(f"Unscheduled V5.1 game ID: {game_id}")
        if record.get("group_id") != spec["group_id"]:
            raise ValueError(f"V5.1 game {game_id} has mismatched group ID")
        if record.get("winner_model") not in (*spec["pair"], None):
            raise ValueError(f"V5.1 game {game_id} has an impossible winner model")
        seen_game_ids.add(game_id)
        by_group[str(spec["group_id"])].append(record)
    for group_id, rows in by_group.items():
        if len(rows) > 2:
            raise ValueError(f"V5.1 group {group_id} contains more than two games")

    runtime = {
        route_id: {"active": True, "next_level": 0, "highest_passed": None, "failed_level": None}
        for route_id in route_specs
    }
    evaluated: dict[tuple[str, int], dict[str, Any]] = {}
    group_states: dict[str, dict[str, Any]] = {}
    pending_group_id: str | None = None
    complete = True
    processed_groups = 0
    for schedule_index, group in enumerate(groups):
        group_id = str(group["group_id"])
        rows = by_group.get(group_id, [])
        active_refs = [
            ref for ref in group["route_refs"]
            if runtime[ref["route_id"]]["active"]
            and runtime[ref["route_id"]]["next_level"] == int(ref["level_index"])
        ]
        if rows and not active_refs:
            raise ValueError(f"V5.1 group {group_id} has results but no referenced route is active")
        if not rows and not active_refs:
            group_states[group_id] = {"status": "skipped_no_active_route", "games": 0}
            continue
        if not rows or len(rows) == 1:
            pending_group_id = group_id
            complete = False
            group_states[group_id] = {
                "status": "incomplete_pair" if rows else "pending",
                "games": len(rows),
                "active_route_ids": [ref["route_id"] for ref in active_refs],
            }
            for later in groups[schedule_index + 1:]:
                if by_group.get(str(later["group_id"])):
                    raise ValueError("V5.1 results exist after an unfinished active group")
                group_states[str(later["group_id"])] = {
                    "status": "not_reached", "games": 0,
                    "active_route_ids": [
                        str(ref["route_id"]) for ref in later["route_refs"]
                        if runtime[str(ref["route_id"])]["active"]
                    ],
                }
            break
        by_id = {str(row["game_id"]): row for row in rows}
        expected = {str(game["game_id"]) for game in group["games"]}
        if len(by_id) != 2 or set(by_id) != expected:
            raise ValueError(f"V5.1 group {group_id} is missing or mismatches its color-swapped pair")
        processed_groups += 1
        group_states[group_id] = {"status": "played", "games": 2, "active_route_ids": [ref["route_id"] for ref in active_refs]}
        for ref in active_refs:
            route_id = str(ref["route_id"])
            level_index = int(ref["level_index"])
            challenger = route_specs[route_id]["challenger_family"]
            wins = draws = losses = 0
            for game in group["games"]:
                winner = by_id[str(game["game_id"])].get("winner_model")
                wins += int(winner == challenger)
                draws += int(winner is None)
                losses += int(winner is not None and winner != challenger)
            points = wins + draws / 2.0
            passed = points >= 1.0
            evaluated[(route_id, level_index)] = {
                "status": "passed" if passed else "failed",
                "score": points,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "games": 2,
                "group_id": group_id,
            }
            state = runtime[route_id]
            state["next_level"] += 1
            if passed:
                state["highest_passed"] = level_index
            else:
                state["active"] = False
                state["failed_level"] = level_index

    route_states: dict[str, dict[str, Any]] = {}
    for route_id, spec in route_specs.items():
        state = runtime[route_id]
        levels: list[dict[str, Any]] = []
        for level in spec["levels"]:
            index = int(level["level_index"])
            group_id = str(level["group_id"])
            group_records = by_group.get(group_id, [])
            if (route_id, index) in evaluated:
                level_state = dict(evaluated[(route_id, index)])
                level_state["opponent_budget_label"] = level["opponent_budget_label"]
            elif state["failed_level"] is not None and index > int(state["failed_level"]):
                level_state = {
                    "status": "shared_result_after_failure" if group_records else "not_run_after_failure",
                    "score": None,
                    "games": len(group_records),
                    "group_id": group_id,
                    "opponent_budget_label": level["opponent_budget_label"],
                    "reason": f"route_failed_at_level_{state['failed_level']}",
                }
            elif state["active"] and index == int(state["next_level"]):
                level_state = {
                    "status": "incomplete_pair" if group_records else "pending",
                    "score": None,
                    "games": len(group_records),
                    "group_id": group_id,
                    "opponent_budget_label": level["opponent_budget_label"],
                }
            else:
                level_state = {
                    "status": "not_reached",
                    "score": None,
                    "games": len(group_records),
                    "group_id": group_id,
                    "opponent_budget_label": level["opponent_budget_label"],
                }
            levels.append(level_state)
        if state["failed_level"] is not None:
            status = "failed"
        elif state["failed_level"] is None and state["next_level"] == len(spec["levels"]):
            status = "completed"
        else:
            status = "active"
        failed_index = state["failed_level"]
        route_states[route_id] = {
            **spec,
            "status": status,
            "highest_passed_budget": (
                spec["levels"][int(state["highest_passed"])]["opponent_budget_label"]
                if state["highest_passed"] is not None else None
            ),
            "failed_budget": (
                spec["levels"][int(failed_index)]["opponent_budget_label"]
                if failed_index is not None else None
            ),
            "levels": levels,
        }
    if complete and any(state["status"] == "active" for state in route_states.values()):
        complete = False
    return {
        "routes": route_states,
        "groups": group_states,
        "pending_group_id": pending_group_id,
        "completed_groups": processed_groups,
        "completed_games": sum(len(rows) for rows in by_group.values()),
        "complete": complete,
    }
