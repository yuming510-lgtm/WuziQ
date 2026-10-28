"""Deterministic six-game V6.2 round-robin schedule."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed without Python's process-randomized hash."""
    payload = json.dumps(
        [int(base_seed), *parts], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Create exactly two common-opening, color-swapped games for each fixed matchup."""
    opening = [dict(move) for move in config["opening"]]
    time_limit = float(config["time_limit_seconds"])
    games: list[dict[str, Any]] = []
    for first, second in config["pairs"]:
        pair = (str(first), str(second))
        pair_slug = "-".join(model.lower() for model in pair)
        group_id = f"v6.2-9x9-{pair_slug}-o{config['opening_id']}"
        for color_swap in (False, True):
            black_model, white_model = pair
            if color_swap:
                black_model, white_model = white_model, black_model
            game_id = f"{group_id}-s{int(color_swap)}"
            games.append(
                {
                    "game_id": game_id,
                    "group_id": group_id,
                    "experiment_id": config["experiment_id"],
                    "board_size": 9,
                    "pair": list(pair),
                    "model_versions": {model: "200s" for model in pair},
                    "budget_seconds_by_model": {model: time_limit for model in pair},
                    "opening_index": 0,
                    "opening_seed": int(config["opening_id"]),
                    "opening": [dict(move) for move in opening],
                    "color_swap": color_swap,
                    "black_model": black_model,
                    "white_model": white_model,
                    "game_seed": derive_seed(int(config["base_seed"]), "game", game_id),
                    "route_refs": [],
                    "source_type": "new",
                    "source_game_id": None,
                }
            )
    if len(games) != 6 or len({game["game_id"] for game in games}) != 6:
        raise ValueError("V6.2 schedule must contain six unique games")
    return games
