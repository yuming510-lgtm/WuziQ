from __future__ import annotations

import copy

import pytest
import torch

from src.research.alphazero_g.evaluation import (
    _network_move,
    append_evaluation_match,
    load_evaluation_log,
    play_evaluation_match,
    summarize_matches,
    validate_match_record,
)
from src.research.alphazero_g.experiment import build_g_schedule, register_g_budget, validate_g_schedule
from src.research.alphazero_g.puct import PUCTConfig


class FixedPolicyValue(torch.nn.Module):
    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros((planes.shape[0], 81), dtype=planes.dtype, device=planes.device)
        logits[:, 40] = 100.0
        values = torch.zeros((planes.shape[0],), dtype=planes.dtype, device=planes.device)
        return logits, values


def _opening() -> list[dict]:
    return [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 3, "col": 5},
        {"player": "W", "row": 5, "col": 3},
    ]


def test_schedule_is_deterministic_legal_disjoint_and_excludes_history() -> None:
    historical = _opening()
    first = build_g_schedule(5301, 3, 2, 2, ["old-group-1"], [historical])
    second = build_g_schedule(5301, 3, 2, 2, ["old-group-1"], [historical])
    assert first == second
    assert len(first["train_openings"]) == 3
    assert len(first["development_openings"]) == 2
    assert len(first["blind_openings"]) == 2
    validate_g_schedule(first, excluded_openings=[historical])
    all_ids = {
        group["opening_group_id"]
        for section in ("train_openings", "development_openings", "blind_openings")
        for group in first[section]
    }
    assert "old-group-1" not in all_ids
    corrupted = copy.deepcopy(first)
    corrupted["blind_openings"][0]["opening"] = historical
    with pytest.raises(ValueError, match="overlaps"):
        validate_g_schedule(corrupted, excluded_openings=[historical])


def test_network_policy_masks_occupied_actions() -> None:
    from src.gomoku import GomokuBoard

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    move, _ = _network_move(FixedPolicyValue(), board, "W", torch.device("cpu"))
    assert move != (4, 4)


def test_paired_match_is_complete_legal_and_records_mode_and_search_counts(tmp_path) -> None:
    model = FixedPolicyValue()
    game = play_evaluation_match(
        model,
        game_id="g-eval-test-01", opening_group_id="test-open-01",
        arm_id="G0", mode="puct", opponent="F1",
        opening=_opening(), agent_color="B", seed=5302, simulations=1,
        config=PUCTConfig(), opponent_model=FixedPolicyValue(),
    )
    assert game["completed"] is True
    assert game["mode"] == "puct"
    assert game["termination"] in {"five_in_row", "full_board"}
    assert game["move_count"] == len(game["moves"])
    validate_match_record(game)
    assert game["stats"]["agent_simulations"] == sum(
        move.get("simulations", 0) for move in game["moves"] if move.get("actor") == "agent"
    )
    assert game["stats"]["agent_network_evaluations"] == sum(
        move.get("network_evaluations", 0) for move in game["moves"] if move.get("actor") == "agent"
    )
    board = [[None] * 9 for _ in range(9)]
    for move in game["moves"]:
        row, col = move["row"], move["col"]
        assert 0 <= row < 9 and 0 <= col < 9 and board[row][col] is None
        board[row][col] = move["player"]
    summary = summarize_matches([game])
    assert summary["games"] == 1
    assert summary["points"] == game["score_for_agent"]
    log = tmp_path / "matches.jsonl"
    seen: set[str] = set()
    assert append_evaluation_match(log, game, seen) is True
    assert append_evaluation_match(log, game, seen) is False
    with log.open("ab") as handle:
        handle.write(b'{"partial":')
    matches, restored_ids = load_evaluation_log(log)
    assert len(matches) == 1
    assert restored_ids == {"g-eval-test-01"}
    assert log.read_bytes().endswith(b"\n")


def test_budget_registration_requires_complete_measured_preflight_and_finite_cap() -> None:
    schedule = build_g_schedule(5303, 2, 2, 2)
    preflight = {
        "simulation_benchmarks": [
            {"simulations": sims, "position_id": position, "mean_seconds": sims / 160.0}
            for sims in (16, 32, 64) for position in ("opening", "middle", "tactical")
        ],
        "selfplay_games": [
            {"arm_id": arm, "completed": True, "sample_positions": 40, "wall_seconds": 2.0}
            for arm in ("G0", "G-F2") for _ in range(2)
        ],
        "training_updates": [
            {"arm_id": arm, "completed": True, "updates": 2, "wall_seconds": 1.0}
            for arm in ("G0", "G-F2")
        ],
    }
    parameters = {
        "simulations_per_move": 16,
        "environment_steps_per_arm": 128,
        "updates_per_arm": 16,
        "batch_size": 16,
        "replay_capacity": 512,
        "stage_environment_steps": [64, 64],
        "stage_updates": [8, 8],
        "cpu_hour_cap": 10,
    }
    registered = register_g_budget(preflight, schedule, parameters)
    assert registered["registered"] is True
    assert registered["arms"]["G0"] == registered["arms"]["G-F2"]
    assert registered["evaluation"]["blind_runs"] == 1
    assert registered["evaluation"]["blind_games_per_selected_checkpoint"] == 64

    with pytest.raises(ValueError, match="missing measured simulation_benchmarks"):
        register_g_budget({**preflight, "simulation_benchmarks": []}, schedule, parameters)
    with pytest.raises(ValueError, match="hard CPU-hour cap"):
        register_g_budget(preflight, schedule, {**parameters, "cpu_hour_cap": 1e-9})
