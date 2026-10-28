from __future__ import annotations

import json
from pathlib import Path

from src.research.v6_3.config import load_v6_3_config
from src.research.v6_3.schedule import build_schedule


def test_preregistered_openings_are_fresh_legal_and_color_paired() -> None:
    config = load_v6_3_config()
    schedule = build_schedule(config)

    assert len(schedule) == 6
    assert len({row["game_id"] for row in schedule}) == 6
    assert [row["stage"] for row in schedule] == [
        "initial", "initial", "initial", "initial",
        "conditional_tiebreak", "conditional_tiebreak",
    ]
    assert len({tuple((m["player"], m["row"], m["col"]) for m in item)
                for item in config["openings"].values()}) == 3

    for group_index in range(3):
        first, second = schedule[group_index * 2:group_index * 2 + 2]
        assert first["opening"] == second["opening"]
        assert first["black_model"] == second["white_model"]
        assert first["white_model"] == second["black_model"]
        assert first["color_swap"] is False
        assert second["color_swap"] is True

    v6 = json.loads(Path("configs/v6/experiment.json").read_text())
    old = {
        tuple((stone[0], int(stone[1]), int(stone[2])) for stone in opening)
        for opening in v6["opening_catalog"].values()
    }
    old.add((('B', 4, 4), ('W', 3, 5), ('B', 5, 3), ('W', 2, 6)))
    assert all(
        tuple((m["player"], m["row"], m["col"]) for m in opening) not in old
        for opening in config["openings"].values()
    )

