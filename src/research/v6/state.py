"""Deterministic route-local advancement with empirical/inferred separation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import BUDGET_LABELS


def _winner_score(winner: str | None, challenger: str) -> tuple[int, int, int]:
    if winner is None:
        return 0, 1, 0
    if winner == challenger:
        return 1, 0, 0
    return 0, 0, 1


def derive_state(
    records: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    route_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild all route states from full pairs; inferred levels have no score."""
    game_specs = {str(game["game_id"]): game for group in schedule for game in group["games"]}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        game_id = str(record.get("game_id", ""))
        if game_id in seen:
            raise ValueError(f"Duplicate V6 game ID: {game_id}")
        seen.add(game_id)
        spec = game_specs.get(game_id)
        if spec is None:
            raise ValueError(f"Unscheduled V6 game ID: {game_id}")
        if record.get("group_id") != spec["group_id"]:
            raise ValueError(f"V6 game {game_id} has a mismatched group ID")
        if record.get("winner_model") not in (*spec["pair"], None):
            raise ValueError(f"V6 game {game_id} has an impossible winner")
        by_group[str(spec["group_id"])].append(record)
    for group_id, rows in by_group.items():
        if len(rows) > 2:
            raise ValueError(f"V6 group {group_id} has more than two game records")

    runtime = {
        route_id: {"active": True, "next_level": 0, "highest_passed": None, "failed_level": None}
        for route_id in route_specs
    }
    evaluated: dict[tuple[str, int], dict[str, Any]] = {}
    group_states: dict[str, dict[str, Any]] = {}
    pending_group_ids: list[str] = []
    inferred_groups = 0
    completed_groups = 0

    def inference_source(group: dict[str, Any], route_id: str) -> dict[str, Any] | None:
        other_refs = [ref for ref in group["route_refs"] if str(ref["route_id"]) != route_id]
        if len(other_refs) != 1:
            return None
        reverse_id = str(other_refs[0]["route_id"])
        reverse = runtime.get(reverse_id)
        if reverse is None or reverse["failed_level"] is None:
            return None
        challenger_spec = route_specs[route_id]
        challenger = challenger_spec["challenger_family"]
        if challenger == "A":
            return None
        budget_index = BUDGET_LABELS.index(challenger_spec["challenger_version"])
        if budget_index <= int(reverse["failed_level"]):
            return None
        reverse_spec = route_specs[reverse_id]
        source_level = int(reverse["failed_level"])
        return {
            "source_route_id": reverse_id,
            "source_group_id": reverse_spec["levels"][source_level]["group_id"],
            "source_failed_level": source_level,
            "rule": "higher-opponent-budget-assumed-no_weaker_after-lower-budget-loss",
        }

    def mark_inferred(route_id: str, level_index: int, source: dict[str, Any]) -> None:
        route = runtime[route_id]
        evaluated[(route_id, level_index)] = {
            "status": "inferred_pass",
            "score": None,
            "score_rate": None,
            "wins": None,
            "draws": None,
            "losses": None,
            "games": 0,
            "group_id": route_specs[route_id]["levels"][level_index]["group_id"],
            "inference": source,
        }
        route["next_level"] += 1
        route["highest_passed"] = level_index

    for group in schedule:
        group_id = str(group["group_id"])
        rows = by_group.get(group_id, [])
        active_refs = [
            ref for ref in group["route_refs"]
            if runtime[str(ref["route_id"])]["active"]
            and runtime[str(ref["route_id"])]["next_level"] == int(ref["level_index"])
        ]
        complete_pair = len(rows) == 2
        if complete_pair:
            expected = {str(game["game_id"]) for game in group["games"]}
            if {str(row["game_id"]) for row in rows} != expected:
                raise ValueError(f"V6 group {group_id} has an incomplete/mismatched color pair")
            completed_groups += 1
            group_states[group_id] = {
                "status": "played" if active_refs else "played_after_route_stop",
                "games": 2,
                "active_route_ids": [str(ref["route_id"]) for ref in active_refs],
                "source_type": group.get("source_type", "new"),
            }
            by_id = {str(row["game_id"]): row for row in rows}
            for ref in active_refs:
                route_id = str(ref["route_id"])
                level_index = int(ref["level_index"])
                challenger = route_specs[route_id]["challenger_family"]
                wins = draws = losses = 0
                for game in group["games"]:
                    w, d, l = _winner_score(
                        by_id[str(game["game_id"])].get("winner_model"), challenger
                    )
                    wins += w
                    draws += d
                    losses += l
                points = wins + draws / 2.0
                passed = points >= 1.0
                evaluated[(route_id, level_index)] = {
                    "status": "passed" if passed else "failed",
                    "score": points,
                    "score_rate": points / 2.0,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "games": 2,
                    "group_id": group_id,
                    "source_type": group.get("source_type", "new"),
                }
                route = runtime[route_id]
                route["next_level"] += 1
                if passed:
                    route["highest_passed"] = level_index
                else:
                    route["active"] = False
                    route["failed_level"] = level_index
            continue
        if len(rows) > 0 and not active_refs:
            group_states[group_id] = {
                "status": "partial_observation_after_route_stop",
                "games": len(rows),
                "active_route_ids": [],
                "excluded_game_ids": [str(row["game_id"]) for row in rows],
            }
            continue
        if len(rows) == 1 and active_refs:
            sources = {str(ref["route_id"]): inference_source(group, str(ref["route_id"])) for ref in active_refs}
            if all(sources.values()) and group.get("source_type") != "reused_v5_1":
                for ref in active_refs:
                    route_id = str(ref["route_id"])
                    mark_inferred(route_id, int(ref["level_index"]), sources[route_id])
                inferred_groups += 1
                group_states[group_id] = {
                    "status": "inferred_dominated_partial_observation",
                    "games": 1,
                    "active_route_ids": [str(ref["route_id"]) for ref in active_refs],
                    "inferences": sources,
                    "excluded_game_ids": [str(rows[0]["game_id"])],
                }
            else:
                pending_group_ids.append(group_id)
                group_states[group_id] = {
                    "status": "incomplete_pair",
                    "games": 1,
                    "active_route_ids": [str(ref["route_id"]) for ref in active_refs],
                }
            continue
        if rows:
            raise ValueError(f"V6 group {group_id} contains an unhandled partial pair")
        if not active_refs:
            group_states[group_id] = {
                "status": "skipped_no_active_route",
                "games": 0,
                "active_route_ids": [],
            }
            continue
        sources = {str(ref["route_id"]): inference_source(group, str(ref["route_id"])) for ref in active_refs}
        if all(sources.values()) and group.get("source_type") != "reused_v5_1":
            for ref in active_refs:
                route_id = str(ref["route_id"])
                mark_inferred(route_id, int(ref["level_index"]), sources[route_id])
            inferred_groups += 1
            group_states[group_id] = {
                "status": "inferred_dominated",
                "games": 0,
                "active_route_ids": [str(ref["route_id"]) for ref in active_refs],
                "inferences": sources,
            }
        else:
            pending_group_ids.append(group_id)
            group_states[group_id] = {
                "status": "pending",
                "games": 0,
                "active_route_ids": [str(ref["route_id"]) for ref in active_refs],
            }

    route_states: dict[str, dict[str, Any]] = {}
    for route_id, spec in route_specs.items():
        state = runtime[route_id]
        levels: list[dict[str, Any]] = []
        for level in spec["levels"]:
            index = int(level["level_index"])
            group_id = str(level["group_id"])
            group_rows = by_group.get(group_id, [])
            if (route_id, index) in evaluated:
                value = dict(evaluated[(route_id, index)])
                value.update({
                    "level_index": index,
                    "opponent_budget_label": level["opponent_budget_label"],
                })
            elif state["failed_level"] is not None and index > int(state["failed_level"]):
                group_status = group_states[group_id]["status"]
                if len(group_rows) == 2:
                    status = "shared_result_after_failure"
                elif len(group_rows) == 1:
                    status = "partial_observation_after_failure"
                else:
                    status = "inferred_fail"
                value = {
                    "status": status,
                    "score": None,
                    "score_rate": None,
                    "wins": None,
                    "draws": None,
                    "losses": None,
                    "games": len(group_rows) if status == "partial_observation_after_failure" else 0,
                    "group_id": group_id,
                    "level_index": index,
                    "opponent_budget_label": level["opponent_budget_label"],
                    "reason": f"route_failed_at_level_{state['failed_level']}",
                    "inference": (
                        "higher opponent budget assumed no weaker"
                        if status == "inferred_fail" else None
                    ),
                }
            elif state["active"] and index == int(state["next_level"]):
                value = {
                    "status": "incomplete_pair" if group_rows else "pending",
                    "score": None,
                    "score_rate": None,
                    "wins": None,
                    "draws": None,
                    "losses": None,
                    "games": len(group_rows),
                    "group_id": group_id,
                    "level_index": index,
                    "opponent_budget_label": level["opponent_budget_label"],
                }
            else:
                value = {
                    "status": "not_reached",
                    "score": None,
                    "score_rate": None,
                    "wins": None,
                    "draws": None,
                    "losses": None,
                    "games": len(group_rows),
                    "group_id": group_id,
                    "level_index": index,
                    "opponent_budget_label": level["opponent_budget_label"],
                }
            levels.append(value)
        if state["failed_level"] is not None:
            status = "failed"
        elif state["next_level"] == len(spec["levels"]):
            status = "completed"
        else:
            status = "active"
        empirical_passes = [
            index for (candidate, index), result in evaluated.items()
            if candidate == route_id and result["status"] == "passed"
        ]
        route_states[route_id] = {
            **spec,
            "status": status,
            "highest_passed_level": state["highest_passed"],
            "highest_passed_budget": (
                spec["levels"][int(state["highest_passed"])]["opponent_budget_label"]
                if state["highest_passed"] is not None else None
            ),
            "highest_actual_passed_budget": (
                spec["levels"][max(empirical_passes)]["opponent_budget_label"]
                if empirical_passes else None
            ),
            "failed_level": state["failed_level"],
            "failed_budget": (
                spec["levels"][int(state["failed_level"])]["opponent_budget_label"]
                if state["failed_level"] is not None else None
            ),
            "inferred_passed_levels": sum(
                item["status"] == "inferred_pass"
                for (candidate, _index), item in evaluated.items()
                if candidate == route_id
            ),
            "levels": levels,
        }
    return {
        "routes": route_states,
        "groups": group_states,
        "pending_group_ids": pending_group_ids,
        "pending_group_id": pending_group_ids[0] if pending_group_ids else None,
        "completed_groups": completed_groups,
        "inferred_groups": inferred_groups,
        "completed_games": len(records),
        "complete": not pending_group_ids and all(
            route["status"] != "active" for route in route_states.values()
        ),
    }
