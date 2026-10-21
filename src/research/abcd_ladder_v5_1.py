"""Frozen schedule model for independent ABCD budget-version ladders (V5.1)."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from ..gomoku import GomokuBoard
from .abcd_ladder_v5 import _project_root, derive_seed, load_config as load_v5_engine_config

_FAMILIES = ("A", "B", "C", "D")
_SEARCH_FAMILIES = ("B", "C", "D")
_FAMILY_PAIRS = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
_BUDGET_LABELS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")
_BUDGET_SECONDS = (0.05, 0.2, 1.0, 5.0, 10.0, 20.0, 50.0)
_REQUIRED_ENGINE_PINS = {"src/ai.py", "src/gomoku.py", "src/research/mcts_d2.py"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root() -> Path:
    return _project_root()


def _engine_config(config: dict[str, Any]) -> dict[str, Any]:
    path = _root() / str(config["engine_config_path"])
    return load_v5_engine_config(path)


def load_config(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load V5.1 metadata and verify it still points at the frozen V5 engines."""
    if isinstance(source, dict):
        config = deepcopy(source)
    else:
        config = json.loads(Path(source).read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported V5.1 config schema")
    if config.get("experiment_id") != "wuziq-abcd-independent-ladder-v5-1-2026-10-21":
        raise ValueError("Unexpected V5.1 experiment ID")
    if int(config.get("board_size", 0)) != 9 or tuple(config.get("families", ())) != _FAMILIES:
        raise ValueError("V5.1 is frozen to A/B/C/D on a 9x9 board")
    ladder = config.get("budget_ladder", ())
    if tuple(item.get("label") for item in ladder) != _BUDGET_LABELS:
        raise ValueError("V5.1 must use the seven frozen budget rungs")
    if tuple(float(item.get("seconds", -1)) for item in ladder) != _BUDGET_SECONDS:
        raise ValueError("V5.1 budget seconds do not match labels")
    if int(config.get("opening_plies", 0)) != 4:
        raise ValueError("V5.1 openings are frozen to four alternating plies")
    if (int(config.get("maximum_routes", 0)), int(config.get("maximum_groups", 0)),
            int(config.get("maximum_games", 0))) != (66, 168, 336):
        raise ValueError("V5.1 maximum schedule must be 66 routes, 168 groups and 336 games")

    engine_path = _root() / str(config.get("engine_config_path", ""))
    if not engine_path.is_file() or _sha256(engine_path.read_bytes()) != config.get("engine_config_sha256"):
        raise ValueError("Pinned V5 engine configuration changed")
    engine_config = load_v5_engine_config(engine_path)
    if set(config.get("engine_sha256", {})) != _REQUIRED_ENGINE_PINS:
        raise ValueError("V5.1 must pin exactly the three verified decision engine files")
    if config["engine_sha256"] != engine_config["engine_sha256"]:
        raise ValueError("V5.1 engine pins differ from the frozen V5 configuration")
    if config.get("opening_seeds") != engine_config["opening_seeds"]:
        raise ValueError("V5.1 opening seeds differ from V5")
    if config.get("opening_catalog") != engine_config["opening_catalog"]:
        raise ValueError("V5.1 opening catalog differs from V5")
    seeds = tuple(int(seed) for seed in config["opening_seeds"])
    if len(seeds) != 12 or len(set(seeds)) != 12:
        raise ValueError("V5.1 requires twelve unique frozen opening seeds")
    for seed in seeds:
        stones = config["opening_catalog"].get(str(seed))
        if not isinstance(stones, list) or len(stones) != 4:
            raise ValueError(f"V5.1 opening {seed} must contain exactly four plies")
        board = GomokuBoard(9)
        for index, stone in enumerate(stones):
            if len(stone) != 3 or str(stone[0]) != ("B" if index % 2 == 0 else "W"):
                raise ValueError(f"V5.1 opening {seed} must alternate B-W-B-W")
            coord = int(stone[1]), int(stone[2])
            board.place_stone(str(stone[0]), coord)
            if board.check_victory(coord, str(stone[0])):
                raise ValueError(f"V5.1 opening {seed} is terminal")
    return config


def engine_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the unchanged V5 engine settings used by `run_game` for actual decisions."""
    return _engine_config(load_config(config))


def _budget_label(family: str, index: int) -> str:
    return "natural" if family == "A" else _BUDGET_LABELS[index]


def _route_id(challenger: str, challenger_version: str, opponent: str) -> str:
    if challenger == "A":
        return f"A>{opponent}"
    return f"{challenger}@{challenger_version}>{opponent}"


def _opening_index(
    config: dict[str, Any], pair: tuple[str, str], first_index: int, second_index: int
) -> int:
    pair_index = _FAMILY_PAIRS.index(pair)
    count = len(config["opening_seeds"])
    if pair[0] == "A":
        return (pair_index * 3 + second_index) % count
    # Keep every V5 baseline/axis game's exact opening for eligible reuse.
    base = pair_index * 3
    if first_index == 0 and second_index == 0:
        return base % count
    if second_index == 0:
        return (base + first_index) % count
    if first_index == 0:
        return (base + second_index + 5) % count
    return derive_seed(
        int(config["base_seed"]), "v5.1-opening", pair, first_index, second_index
    ) % count


def _opening(config: dict[str, Any], seed: int) -> list[tuple[str, tuple[int, int]]]:
    return [
        (str(player), (int(row), int(col)))
        for player, row, col in config["opening_catalog"][str(seed)]
    ]


def _route(
    routes: dict[str, dict[str, Any]],
    route_id: str,
    challenger: str,
    challenger_version: str,
    opponent: str,
) -> None:
    routes.setdefault(route_id, {
        "route_id": route_id,
        "challenger_family": challenger,
        "challenger_version": challenger_version,
        "challenger_budget_label": _budget_label(challenger, _BUDGET_LABELS.index(challenger_version))
        if challenger != "A" else "natural",
        "opponent_family": opponent,
        "levels": [],
    })


def _append_group(
    config: dict[str, Any],
    schedule: list[dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    pair: tuple[str, str],
    model_versions: dict[str, str],
    budget_indexes: dict[str, int | None],
    route_levels: list[tuple[str, int, str]],
    opening_index: int,
) -> None:
    first, second = pair
    seed = int(config["opening_seeds"][opening_index])
    first_label = model_versions[first]
    second_label = model_versions[second]
    group_id = (
        f"v5-1-9x9-{first.lower()}-{first_label}-{second.lower()}-{second_label}-o{seed}"
    )
    if any(group["group_id"] == group_id for group in schedule):
        raise ValueError(f"Duplicate V5.1 group ID: {group_id}")
    opening = [
        {"player": player, "row": coord[0], "col": coord[1]}
        for player, coord in _opening(config, seed)
    ]
    route_refs: list[dict[str, Any]] = []
    for route_id, level_index, opponent_label in route_levels:
        route_refs.append({"route_id": route_id, "level_index": level_index})
        challenger = route_id.split(">", 1)[0].split("@", 1)[0]
        opponent = route_id.split(">", 1)[1]
        version = "natural" if challenger == "A" else route_id.split("@", 1)[1].split(">", 1)[0]
        _route(routes, route_id, challenger, version, opponent)
        routes[route_id]["levels"].append({
            "level_index": level_index,
            "group_id": group_id,
            "opponent_budget_label": opponent_label,
        })

    budgets = {
        family: None if family == "A" else _BUDGET_SECONDS[int(budget_indexes[family])]
        for family in pair
    }
    route_ids = [ref[0] for ref in route_levels]
    games: list[dict[str, Any]] = []
    for color_swap in (False, True):
        black, white = pair if not color_swap else (pair[1], pair[0])
        game_id = f"{group_id}-s{int(color_swap)}"
        games.append({
            "experiment_id": config["experiment_id"],
            "group_id": group_id,
            "game_id": game_id,
            "board_size": 9,
            "pair": list(pair),
            "matchup": list(pair),
            "route_ids": route_ids,
            "route_refs": [dict(route_id=route_id, level_index=index) for route_id, index, _ in route_levels],
            "model_versions": dict(model_versions),
            "budget_seconds_by_model": dict(budgets),
            "opening_index": opening_index,
            "opening_seed": seed,
            "color_swap": color_swap,
            "black_model": black,
            "white_model": white,
            "pair_id": group_id,
            "game_seed": derive_seed(int(config["base_seed"]), "v5.1-game", game_id),
            "opening": opening,
            "challenger_model": None,
            "challenger_budget_label": None,
            "opponent_budget_label": None,
            "source_type": "new",
        })
    schedule.append({
        "experiment_id": config["experiment_id"],
        "group_id": group_id,
        "pair": list(pair),
        "model_versions": dict(model_versions),
        "budget_seconds_by_model": dict(budgets),
        "route_ids": route_ids,
        "route_refs": [dict(route_id=route_id, level_index=index) for route_id, index, _ in route_levels],
        "opening_index": opening_index,
        "opening_seed": seed,
        "opening": opening,
        "shared_result": True,
        "games": games,
    })


def _schedule_and_routes(config_source: str | Path | dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config = load_config(config_source)
    schedule: list[dict[str, Any]] = []
    routes: dict[str, dict[str, Any]] = {}

    # A has one natural-time version and challenges each opponent version in order.
    for pair in _FAMILY_PAIRS[:3]:
        _, opponent = pair
        for opponent_index, opponent_budget in enumerate(_BUDGET_LABELS):
            a_route = _route_id("A", "natural", opponent)
            opponent_route = _route_id(opponent, opponent_budget, "A")
            _append_group(
                config, schedule, routes, pair,
                {"A": "natural", opponent: opponent_budget},
                {"A": None, opponent: opponent_index},
                [(a_route, opponent_index, opponent_budget), (opponent_route, 0, "natural")],
                _opening_index(config, pair, 0, opponent_index),
            )

    # Each fixed version is a challenger independently. Row-major order advances
    # both directional ladders monotonically through the 7x7 version grid.
    for pair in _FAMILY_PAIRS[3:]:
        first, second = pair
        for first_index, first_budget in enumerate(_BUDGET_LABELS):
            for second_index, second_budget in enumerate(_BUDGET_LABELS):
                first_route = _route_id(first, first_budget, second)
                second_route = _route_id(second, second_budget, first)
                _append_group(
                    config, schedule, routes, pair,
                    {first: first_budget, second: second_budget},
                    {first: first_index, second: second_index},
                    [(first_route, second_index, second_budget), (second_route, first_index, first_budget)],
                    _opening_index(config, pair, first_index, second_index),
                )

    expected_game_count = 2 * len(schedule)
    ids = [game["game_id"] for group in schedule for game in group["games"]]
    if len(schedule) != 168 or len(routes) != 66 or expected_game_count != 336:
        raise ValueError("V5.1 maximum schedule must be 66 independent routes, 168 groups and 336 games")
    if len(ids) != len(set(ids)) or len({group["group_id"] for group in schedule}) != len(schedule):
        raise ValueError("V5.1 schedule has duplicate group or game IDs")
    positions = {group["group_id"]: index for index, group in enumerate(schedule)}
    for route in routes.values():
        levels = route["levels"]
        if [row["level_index"] for row in levels] != list(range(len(levels))):
            raise ValueError(f"V5.1 route levels are not contiguous: {route['route_id']}")
        group_positions = [positions[row["group_id"]] for row in levels]
        if group_positions != sorted(group_positions):
            raise ValueError(f"V5.1 route has a cycle or non-topological schedule: {route['route_id']}")
    return schedule, routes


def build_schedule(config: str | Path | dict[str, Any]) -> list[dict[str, Any]]:
    return _schedule_and_routes(config)[0]


def build_routes(config: str | Path | dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _schedule_and_routes(config)[1]
