from __future__ import annotations

import importlib
import importlib.util
import json

import pytest


def _api():
    name = "src.research.imitation_f.v2_evaluation"
    if importlib.util.find_spec(name) is None:
        pytest.fail("F V2 paired schedule and bootstrap evaluation API is not implemented")
    return importlib.import_module(name)


def _game(group: str, color: str, winner: str | None) -> dict:
    return {
        "game_id": f"{group}-{color}", "group_id": group,
        "pair": ["F2", "A"], "black_model": "F2" if color == "B" else "A",
        "white_model": "A" if color == "B" else "F2",
        "winner_model": winner, "winner_color": None if winner is None else (color if winner == "F2" else ("W" if color == "B" else "B")),
        "opening_seed": 1, "opening": [], "moves": [],
    }


def test_opening_level_bootstrap_uses_paired_color_scores_and_counts_draw_half_points() -> None:
    api = _api()
    records = []
    for index, outcomes in enumerate((("F2", "A"), (None, None), ("F2", "F2"), ("A", "F2"))):
        group = f"g{index}"
        records.extend((_game(group, "B", outcomes[0]), _game(group, "W", outcomes[1])))

    summary = api.paired_opening_statistics(records, "F2", bootstrap_seed=732601, bootstrap_samples=2000)
    assert summary["games"] == 8
    assert summary["wins"] == 4
    assert summary["draws"] == 2
    assert summary["losses"] == 2
    assert summary["points"] == pytest.approx(5.0)
    assert summary["score_rate"] == pytest.approx(0.625)
    assert summary["opening_groups"] == 4
    assert summary["score_rate_ci95"][0] <= summary["score_rate"] <= summary["score_rate_ci95"][1]
    assert summary["by_color"]["black"]["games"] == 4
    assert summary["by_color"]["white"]["games"] == 4


def test_match_runner_is_unique_paired_and_idempotent(tmp_path) -> None:
    api = _api()
    from src.research.imitation_f.generation import build_evaluation_openings
    from src.research.imitation_f.model import PolicyValueNet

    openings = build_evaluation_openings(732602, 1)
    model = PolicyValueNet(channels=8, blocks=1, seed=732603)
    path = tmp_path / "matches.jsonl"
    first = api.run_opening_matches(
        model, "F2", "test-candidate", "A", openings, path, "development",
        "checkpoint-sha", 732604, 0.0,
    )
    before = path.read_bytes()
    second = api.run_opening_matches(
        model, "F2", "test-candidate", "A", openings, path, "development",
        "checkpoint-sha", 732604, 0.0,
    )
    rows = [json.loads(line) for line in before.splitlines()]
    assert first["games"] == second["games"] == 2
    assert len(rows) == len({row["game_id"] for row in rows}) == 2
    assert {row["black_model"] for row in rows} == {"F2", "A"}
    assert {row["color_swap"] for row in rows} == {False, True}
