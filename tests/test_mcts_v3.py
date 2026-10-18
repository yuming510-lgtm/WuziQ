"""V3 high-compute challenge registration and D3 routing tests."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import SearchConfig, SearchResult, choose_move
from src.gomoku import GomokuBoard


def _runner():
    return importlib.import_module("src.research.mcts_v2")


def _v3_config() -> dict:
    phases = []
    for budget in (10.0, 20.0, 50.0):
        for opponent in ("A", "B", "C"):
            phases.append(
                {
                    "name": f"d3-{int(budget)}s-vs-{opponent.lower()}",
                    "track": "high-compute",
                    "board_size": 9,
                    "games_per_matchup": 6,
                    "matchups": [["D3", opponent]],
                    "budget_seconds_by_model": {"D3": budget, opponent: 0.05},
                }
            )
    return {
        "experiment_id": "wuziq-mcts-high-compute-v3-test",
        "base_seed": 20261018,
        "opening_seeds": [731091, 731092, 731093],
        "opening_plies": 4,
        "search": {
            "max_depth": 4,
            "time_limit_seconds": 0.05,
            "mcts_simulations": 1_000_000,
            "mcts_rollout_max_plies": 24,
            "candidate_radius": 2,
            "uct_constant": 2**0.5,
            "mcts_d2_tactical_root": True,
            "mcts_d2_fair_root": True,
            "mcts_d2_tactical_rollout": True,
            "mcts_d2_node_reuse": True,
        },
        "phases": phases,
    }


def test_d3_alias_uses_the_same_pure_mcts_policy_as_d2() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    settings = SearchConfig(time_limit_seconds=10, mcts_simulations=24)

    d2 = choose_move(board, "B", "D2", settings, seed=9231)
    d3 = choose_move(board, "B", "D3", settings, seed=9231)

    assert d3.move == d2.move
    assert d3.root_value == d2.root_value
    assert d3.nodes == d2.nodes
    assert d3.simulations == d2.simulations == 24


def test_v3_registered_schedule_has_54_games_three_pairs_and_shared_openings() -> None:
    runner = _runner()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/mcts_high_compute_v3.json").read_text())

    schedule = runner.build_schedule(config)

    assert len(schedule) == 54
    assert len({spec["game_id"] for spec in schedule}) == 54
    phase_names = {spec["phase_name"] for spec in schedule}
    assert len(phase_names) == 9
    for phase_name in phase_names:
        phase_games = [spec for spec in schedule if spec["phase_name"] == phase_name]
        assert len(phase_games) == 6
        assert sum(bool(spec["color_swap"]) for spec in phase_games) == 3
        assert {spec["budget_seconds_by_model"]["D3"] for spec in phase_games} in (
            {10.0},
            {20.0},
            {50.0},
        )
        opponent = next(model for model in phase_games[0]["matchup"] if model != "D3")
        assert all(spec["budget_seconds_by_model"][opponent] == 0.05 for spec in phase_games)
        for index in range(3):
            pair = [spec for spec in phase_games if spec["opening_index"] == index]
            assert len(pair) == 2
            assert {spec["opening_seed"] for spec in pair} == {config["opening_seeds"][index]}
            assert {spec["black_model"] for spec in pair} == {"D3", opponent}
            assert pair[0]["color_swap"] != pair[1]["color_swap"]

    opening_signatures = {
        tuple(runner.paired_opening(9, seed)) for seed in config["opening_seeds"]
    }
    assert len(opening_signatures) == 3
    assert {
        spec["opening_seed"] for spec in schedule
    } == set(config["opening_seeds"])
    for seed in config["opening_seeds"]:
        actual = runner.paired_opening(9, seed)
        expected = [
            (player, (row, col))
            for player, row, col in config["opening_catalog"][str(seed)]
        ]
        assert actual == expected


def test_v3_schedule_rejects_wrong_opponent_budget() -> None:
    runner = _runner()
    config = _v3_config()
    config["phases"][0]["budget_seconds_by_model"]["A"] = 0.1

    try:
        runner.build_schedule(config)
    except ValueError as error:
        assert "0.05" in str(error)
    else:
        raise AssertionError("high-compute schedule accepted an opponent budget other than 50 ms")


def test_v3_game_runner_uses_persistent_mcts_agent_for_d3(monkeypatch) -> None:
    runner = _runner()
    config = _v3_config()
    spec = runner.build_schedule(config)[0]
    calls = []

    def fake_mcts(agent, board, player, settings, seed):
        calls.append((player, seed))
        move = next(
            (row, col)
            for row in range(board.size)
            for col in range(board.size)
            if board.get((row, col)) is None
        )
        return SearchResult(move, 0, 0, 2, 0, 1, 0.001, False)

    def fake_other_model(board, player, model, settings, seed):
        move = next(
            (row, col)
            for row in range(board.size)
            for col in range(board.size)
            if board.get((row, col)) is None
        )
        return SearchResult(move, 0, 0, 0, 0, 0, 0.001, False)

    monkeypatch.setattr(runner.MCTSAgent, "choose_move", fake_mcts)
    monkeypatch.setattr(runner, "choose_move", fake_other_model)

    record = runner.run_game(spec, config)

    assert calls
    assert any(move["model"] == "D3" and move["simulations"] == 1 for move in record["moves"])
