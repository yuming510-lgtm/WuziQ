from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.research.alphazero_g.v5_evaluation import generate_opening_pool
from src.research.alphazero_g.v5_reference import ReferencePolicyValueNet


def _evaluation_module():
    name = "src.research.alphazero_g.v6_evaluation"
    if importlib.util.find_spec(name) is None:
        pytest.fail("V6 paired, namespaced evaluator is not implemented")
    from src.research.alphazero_g import v6_evaluation

    return v6_evaluation


def test_v6_paired_games_are_atomic_resumable_and_namespaced(tmp_path: Path, monkeypatch) -> None:
    torch.set_num_threads(1)
    evaluator = _evaluation_module()
    openings = generate_opening_pool(size=6, n_in_row=4, groups=1, plies=2, seed=6116)
    openings[0]["opening_group_id"] = "v6-eval-smoke-0"
    model = ReferencePolicyValueNet(6, seed=6117)
    args = {
        "root": tmp_path,
        "phase": "smoke",
        "candidate_name": "ref-v6-smoke",
        "opponent_name": "ref-v6-smoke-opponent",
        "openings": openings,
        "pairs": 1,
        "candidate_reference_model": model,
        "opponent_reference_model": model,
        "simulations": 2,
        "board_size": 6,
        "n_in_row": 4,
    }
    summary = evaluator.run_match_set(**args)

    assert summary["games"] == 2
    assert summary["paired_openings"] == 1
    assert summary["simulations_per_reference_turn"] == 2
    assert {row["candidate_color"] for row in summary["game_rows"]} == {"B", "W"}
    paths = list((tmp_path / "research/g_v6/results/matches").rglob("v6m-*.json"))
    assert len(paths) == 2
    assert not (tmp_path / "research/g_v5/results/matches").exists()

    def unexpected_replay(**_kwargs):
        raise AssertionError("a complete unique game was counted or computed twice")

    monkeypatch.setattr(evaluator, "play_match", unexpected_replay)
    resumed = evaluator.run_match_set(**args)
    assert resumed["wins"] == summary["wins"]
    assert resumed["draws"] == summary["draws"]
    assert resumed["losses"] == summary["losses"]
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["simulations_per_reference_turn"] == 2
        assert len(row["moves"]) == row["move_count"]


def test_v6_budget_comparison_reuses_exact_registered_opening_identities() -> None:
    registration_name = "src.research.alphazero_g.v6_registration"
    if importlib.util.find_spec(registration_name) is None:
        pytest.fail("V6 registration must exist before budget pairing can be audited")
    from src.research.alphazero_g.v6_registration import build_opening_schedule

    schedule = build_opening_schedule()
    opening_ids_64 = {row["opening_group_id"] for row in schedule["development"]}
    opening_ids_400 = {row["opening_group_id"] for row in schedule["development"]}
    assert opening_ids_64 == opening_ids_400
    assert len(opening_ids_64) == 8
    assert not opening_ids_64.intersection(row["opening_group_id"] for row in schedule["blind"])
