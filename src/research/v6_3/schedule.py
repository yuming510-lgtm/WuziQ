"""Deterministic four-game opening round plus two conditional tiebreak games."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


def derive_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps([int(base_seed), *parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    opening_ids = [*config["initial_opening_ids"], str(config["tiebreak_opening_id"])]
    schedule: list[dict[str, Any]] = []
    for index, opening_id in enumerate(opening_ids):
        stage = "initial" if index < 2 else "conditional_tiebreak"
        group_id = f"v6.3-9x9-c-e5-o{opening_id}"
        opening = [dict(move) for move in config["openings"][opening_id]]
        for color_swap in (False, True):
            black_model, white_model = ("C", "E5")
            if color_swap:
                black_model, white_model = white_model, black_model
            game_id = f"{group_id}-s{int(color_swap)}"
            schedule.append({
                "game_id": game_id,
                "group_id": group_id,
                "experiment_id": config["experiment_id"],
                "stage": stage,
                "board_size": 9,
                "pair": ["C", "E5"],
                "model_versions": dict(config["model_versions"]),
                "budget_seconds_by_model": dict(config["budget_seconds_by_model"]),
                "opening_index": index,
                "opening_seed": int(opening_id),
                "opening": opening,
                "color_swap": color_swap,
                "black_model": black_model,
                "white_model": white_model,
                "game_seed": derive_seed(int(config["base_seed"]), "game", game_id),
                "route_refs": [],
                "source_type": "new",
                "source_game_id": None,
            })
    if len(schedule) != 6 or len({row["game_id"] for row in schedule}) != 6:
        raise ValueError("V6.3 must preregister exactly six unique potential game IDs")
    return schedule

