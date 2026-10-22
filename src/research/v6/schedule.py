"""Independent fixed-version challenge routes and unique paired-game schedule."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .config import BUDGET_LABELS, BUDGET_SECONDS, FAMILIES, SEARCH_FAMILIES, load_config
from ..abcd_ladder_v5 import derive_seed
from ..abcd_ladder_v5_1 import _opening_index as _v51_opening_index


def _route_id(challenger: str, version: str, opponent: str) -> str:
    return f"{challenger}>{opponent}" if challenger == "A" else f"{challenger}@{version}>{opponent}"


def _opening_index(
    config: dict[str, Any],
    pair: tuple[str, str],
    first_index: int,
    second_index: int,
    pair_index: int,
) -> int:
    legacy_config = dict(config)
    legacy_config["base_seed"] = int(config.get("v5_1_base_seed", 20261021))
    if pair[0] == "A" and pair[1] in {"B", "C", "D"}:
        return _v51_opening_index(legacy_config, pair, 0, second_index)
    if pair[0] in {"B", "C"} and pair[1] in {"C", "D"} and pair[0] < pair[1]:
        # Exact V5.1 opening coordinates/seeds for the three legacy search pairs.
        return _v51_opening_index(legacy_config, pair, first_index, second_index)
    count = len(config["opening_seeds"])
    base = pair_index * 3
    if pair[0] == "A":
        return (pair_index * 5 + second_index) % count
    if first_index == 0 and second_index == 0:
        return base % count
    if second_index == 0:
        return (base + first_index) % count
    if first_index == 0:
        return (base + second_index + 5) % count
    return derive_seed(
        int(config["base_seed"]), "v6-opening", pair, first_index, second_index
    ) % count


def _opening(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return [
        {"player": str(player), "row": int(row), "col": int(col)}
        for player, row, col in config["opening_catalog"][str(seed)]
    ]


def _append_route_level(
    routes: dict[str, dict[str, Any]],
    route_id: str,
    challenger: str,
    challenger_version: str,
    opponent: str,
    level_index: int,
    group_id: str,
    opponent_label: str,
) -> None:
    route = routes.setdefault(route_id, {
        "route_id": route_id,
        "challenger_family": challenger,
        "challenger_version": challenger_version,
        "challenger_budget_label": challenger_version,
        "opponent_family": opponent,
        "levels": [],
    })
    route["levels"].append({
        "level_index": level_index,
        "group_id": group_id,
        "opponent_budget_label": opponent_label,
        "status": "pending",
    })


def _make_group(
    config: dict[str, Any],
    routes: dict[str, dict[str, Any]],
    pair: tuple[str, str],
    versions: dict[str, str],
    budget_indexes: dict[str, int | None],
    route_levels: list[tuple[str, str, str, int, str]],
    opening_index: int,
    pair_index: int,
) -> dict[str, Any]:
    first, second = pair
    opening_seed = int(config["opening_seeds"][opening_index])
    opening = _opening(config, opening_seed)
    group_id = (
        f"v6-9x9-{first.lower()}-{versions[first]}-{second.lower()}-{versions[second]}"
        f"-o{opening_seed}"
    )
    route_refs: list[dict[str, Any]] = []
    for route_id, challenger, challenger_version, level, opponent_label in route_levels:
        _append_route_level(
            routes, route_id, challenger, challenger_version,
            second if challenger == first else first,
            level, group_id, opponent_label,
        )
        route_refs.append({"route_id": route_id, "level_index": level})
    budgets = {
        family: None if family == "A" else BUDGET_SECONDS[int(budget_indexes[family])]
        for family in pair
    }
    games = []
    for color_swap in (False, True):
        black, white = pair if not color_swap else (pair[1], pair[0])
        game_id = f"{group_id}-s{int(color_swap)}"
        game_seed = derive_seed(int(config["base_seed"]), "v6-game", game_id)
        games.append({
            "experiment_id": config["experiment_id"],
            "game_id": game_id,
            "group_id": group_id,
            "board_size": 9,
            "pair": list(pair),
            "model_versions": dict(versions),
            "budget_seconds_by_model": dict(budgets),
            "opening_index": opening_index,
            "opening_seed": opening_seed,
            "opening": opening,
            "color_swap": color_swap,
            "black_model": black,
            "white_model": white,
            "game_seed": game_seed,
            "route_refs": [dict(ref) for ref in route_refs],
            "source_type": "new",
            "source_game_id": None,
        })
    return {
        "experiment_id": config["experiment_id"],
        "group_id": group_id,
        "pair": list(pair),
        "model_versions": dict(versions),
        "budget_seconds_by_model": dict(budgets),
        "route_ids": [ref["route_id"] for ref in route_refs],
        "route_refs": route_refs,
        "opening_index": opening_index,
        "opening_seed": opening_seed,
        "opening": opening,
        "shared_result": True,
        "source_type": "new",
        "games": games,
    }


def build_schedule(config_source: dict[str, Any] | str | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config = load_config(config_source)
    schedule: list[dict[str, Any]] = []
    routes: dict[str, dict[str, Any]] = {}
    family_pairs = list(combinations(FAMILIES, 2))
    for pair_index, pair in enumerate(family_pairs):
        first, second = pair
        if first == "A":
            opponent = second
            route_a = _route_id("A", "natural", opponent)
            for opponent_index, label in enumerate(BUDGET_LABELS):
                versions = {"A": "natural", opponent: label}
                refs = [
                    (route_a, "A", "natural", opponent_index, label),
                    (_route_id(opponent, label, "A"), opponent, label, 0, "natural"),
                ]
                schedule.append(_make_group(
                    config, routes, pair, versions,
                    {"A": None, opponent: opponent_index},
                    refs,
                    _opening_index(config, pair, 0, opponent_index, pair_index),
                    pair_index,
                ))
        else:
            for first_index, first_label in enumerate(BUDGET_LABELS):
                for second_index, second_label in enumerate(BUDGET_LABELS):
                    versions = {first: first_label, second: second_label}
                    refs = [
                        (
                            _route_id(first, first_label, second),
                            first, first_label, second_index, second_label,
                        ),
                        (
                            _route_id(second, second_label, first),
                            second, second_label, first_index, first_label,
                        ),
                    ]
                    schedule.append(_make_group(
                        config, routes, pair, versions,
                        {first: first_index, second: second_index},
                        refs,
                        _opening_index(config, pair, first_index, second_index, pair_index),
                        pair_index,
                    ))
    validate_schedule(schedule, routes)
    return schedule, routes


def validate_schedule(
    schedule: list[dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> dict[str, int]:
    group_ids = [str(group["group_id"]) for group in schedule]
    game_ids = [str(game["game_id"]) for group in schedule for game in group["games"]]
    pair_keys = [
        (tuple(group["pair"]), tuple(group["model_versions"][family] for family in group["pair"]))
        for group in schedule
    ]
    if len(group_ids) != len(set(group_ids)) or len(game_ids) != len(set(game_ids)):
        raise ValueError("V6 schedule contains duplicate group or game IDs")
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("V6 schedule repeats a model-version pairing")
    if any(len(group["games"]) != 2 for group in schedule):
        raise ValueError("Every V6 group must contain exactly two games")
    positions = {group_id: index for index, group_id in enumerate(group_ids)}
    for group in schedule:
        if len(group["pair"]) != 2 or group["pair"][0] == group["pair"][1]:
            raise ValueError("Same-family matches are not part of V6")
        games = group["games"]
        if games[0]["opening"] != games[1]["opening"] or games[0]["opening_seed"] != games[1]["opening_seed"]:
            raise ValueError(f"V6 pair does not share an opening: {group['group_id']}")
        if (games[0]["black_model"], games[0]["white_model"]) != (
            games[1]["white_model"], games[1]["black_model"]
        ):
            raise ValueError(f"V6 pair does not exchange colors: {group['group_id']}")
        if games[0]["game_seed"] == games[1]["game_seed"]:
            raise ValueError(f"V6 paired game seeds must be distinct: {group['group_id']}")
    for route_id, route in routes.items():
        levels = route["levels"]
        if [int(level["level_index"]) for level in levels] != list(range(len(levels))):
            raise ValueError(f"V6 route levels are not contiguous: {route_id}")
        indices = [positions[str(level["group_id"])] for level in levels]
        if indices != sorted(indices):
            raise ValueError(f"V6 schedule is not topological for route {route_id}")
    expected_groups = 56 + 28 * 49
    if (len(schedule), len(routes), len(game_ids)) != (expected_groups, 456, 2856):
        raise ValueError("V6 full schedule counts do not match the frozen maximum")
    return {
        "routes": len(routes),
        "groups": len(schedule),
        "games": len(game_ids),
        "families": len(FAMILIES),
        "same_family_groups": sum(g["pair"][0] == g["pair"][1] for g in schedule),
        "unique_game_ids": len(set(game_ids)),
    }
