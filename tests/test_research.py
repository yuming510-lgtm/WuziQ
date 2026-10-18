"""Durability, pairing, and resumability checks for research runs."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _runner():
    assert importlib.util.find_spec("src.research.runner") is not None
    return importlib.import_module("src.research.runner")


def _checkpoint():
    assert importlib.util.find_spec("src.research.checkpoint") is not None
    return importlib.import_module("src.research.checkpoint")


def _metrics():
    assert (Path(__file__).resolve().parents[1] / "src/research/metrics.py").is_file()
    return importlib.import_module("src.research.metrics")


def _efficiency():
    assert (Path(__file__).resolve().parents[1] / "src/research/efficiency.py").is_file()
    return importlib.import_module("src.research.efficiency")


def _validation():
    assert importlib.util.find_spec("src.research.validation") is not None
    return importlib.import_module("src.research.validation")


def _config(models=("A", "B"), board_size=5, games_per_matchup=2):
    return {
        "experiment_id": "test-run",
        "base_seed": 510,
        "models": list(models),
        "search": {
            "max_depth": 1,
            "time_limit_seconds": 0.01,
            "mcts_simulations": 8,
            "candidate_radius": 2,
            "uct_constant": 1.4142135623730951,
        },
        "opening_policy": "center-plus-seeded-neighbor",
        "phases": [
            {"name": "test", "board_size": board_size, "games_per_matchup": games_per_matchup}
        ],
    }


def test_research_runner_package_available() -> None:
    assert (Path(__file__).resolve().parents[1] / "src/research/runner.py").is_file()


def test_main_schedule_has_all_six_pairs_and_balanced_colors() -> None:
    runner = _runner()
    config = _config(models=("A", "B", "C", "D"), board_size=9, games_per_matchup=200)
    games = runner.build_schedule(config)
    assert len(games) == 1_200
    assert len({game["game_id"] for game in games}) == 1_200
    pairs = {tuple(game["matchup"]) for game in games}
    assert pairs == {("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")}
    for pair in pairs:
        paired = [game for game in games if tuple(game["matchup"]) == pair]
        assert len(paired) == 200
        assert sum(game["black_model"] == pair[0] for game in paired) == 100
        assert sum(game["black_model"] == pair[1] for game in paired) == 100


def test_paired_openings_are_stable_and_swap_models() -> None:
    runner = _runner()
    config = _config(games_per_matchup=4)
    first = runner.build_schedule(config)
    second = runner.build_schedule(config)
    assert first == second
    for opening_index in range(2):
        pair = [
            game
            for game in first
            if game["opening_index"] == opening_index
        ]
        assert len(pair) == 2
        assert pair[0]["opening_seed"] == pair[1]["opening_seed"]
        assert pair[0]["black_model"] == pair[1]["white_model"]
        assert pair[0]["white_model"] == pair[1]["black_model"]
        assert runner.paired_opening(5, pair[0]["opening_seed"]) == runner.paired_opening(
            5, pair[1]["opening_seed"]
        )


def test_schedule_rejects_unpaired_game_counts() -> None:
    runner = _runner()
    config = _config(games_per_matchup=3)
    with pytest.raises(ValueError, match="even number"):
        runner.build_schedule(config)


def test_jsonl_recovery_truncates_only_an_incomplete_final_record(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    game_file = tmp_path / "games.jsonl"
    complete = {"game_id": "game-1", "result": "draw"}
    game_file.write_text(json.dumps(complete) + "\n{" + '"game_id":"broken"', encoding="utf-8")

    records = checkpoint.load_game_records(game_file)
    assert [record["game_id"] for record in records] == ["game-1"]
    assert game_file.read_text(encoding="utf-8") == json.dumps(complete) + "\n"


def test_jsonl_recovery_rejects_duplicate_completed_ids(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    game_file = tmp_path / "games.jsonl"
    duplicate = {"game_id": "game-1", "result": "draw"}
    game_file.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate game_id"):
        checkpoint.load_game_records(game_file)


def test_snapshot_round_trip_verifies_and_restores_only_recovery_files(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    config = tmp_path / "config.json"
    progress = tmp_path / "progress.json"
    games = tmp_path / "9x9_games.jsonl"
    config.write_text('{"base_seed":510}\n', encoding="utf-8")
    progress.write_text('{"next_schedule_index":1}\n', encoding="utf-8")
    games.write_text('{"game_id":"smoke-1","result":"draw"}\n', encoding="utf-8")
    snapshot = tmp_path / "checkpoint.zip"
    checkpoint.create_snapshot(config, progress, [games], snapshot)

    restored = tmp_path / "restored"
    recovered = checkpoint.restore_snapshot(snapshot, restored)
    assert recovered == ["smoke-1"]
    assert (restored / "config.json").read_bytes() == config.read_bytes()
    assert (restored / "progress.json").read_bytes() == progress.read_bytes()
    assert (restored / "9x9_games.jsonl").read_bytes() == games.read_bytes()
    assert sorted(path.name for path in restored.iterdir()) == [
        "9x9_games.jsonl",
        "config.json",
        "progress.json",
    ]


def test_tournament_resume_never_counts_a_game_twice(tmp_path: Path) -> None:
    runner = _runner()
    config = _config(games_per_matchup=2)
    first = runner.run_tournament(config, tmp_path, max_games=1)
    second = runner.run_tournament(config, tmp_path)
    third = runner.run_tournament(config, tmp_path)
    records = runner.load_all_games(tmp_path)

    assert len(first) == 1
    assert len(second) == 1
    assert third == []
    assert len(records) == 2
    assert len({record["game_id"] for record in records}) == 2
    assert all(record["termination"] in {"five_in_row", "full_board", "illegal_move_forfeit"} for record in records)


def test_result_validation_replays_complete_tournament_and_rejects_illegal_move(tmp_path: Path) -> None:
    runner = _runner()
    config = _config(games_per_matchup=2)
    runner.run_tournament(config, tmp_path)
    records = runner.load_all_games(tmp_path)

    validation = _validation()
    report = validation.validate_tournament(records, config)
    assert report["game_count"] == 2
    assert report["unique_game_ids"] == 2
    assert report["illegal_moves"] == 0

    corrupted = json.loads(json.dumps(records))
    next(move for move in corrupted[0]["moves"] if not move.get("opening"))["row"] = 99
    with pytest.raises(ValueError, match="illegal move"):
        validation.validate_tournament(corrupted, config)


def test_metrics_module_available() -> None:
    assert (Path(__file__).resolve().parents[1] / "src/research/metrics.py").is_file()


def test_efficiency_module_available() -> None:
    assert (Path(__file__).resolve().parents[1] / "src/research/efficiency.py").is_file()


def test_wilson_interval_has_expected_bounds() -> None:
    lower, upper = _metrics().wilson_interval(75, 25)
    assert 0.65 < lower < 0.67
    assert 0.82 < upper < 0.84


def test_result_aggregation_builds_complete_matrix_and_paired_scores() -> None:
    metrics = _metrics()
    games = [
        {"game_id": "ab0", "pair_id": "ab-open-0", "matchup": ["A", "B"], "black_model": "A", "white_model": "B", "winner_model": "A", "winner_color": "B", "outcome": "black_win", "thinking_seconds_total": 1.0, "moves": [{"model": "A", "elapsed_seconds": 0.1, "nodes": 10, "simulations": 0, "completed_depth": 1, "timed_out": False}]},
        {"game_id": "ab1", "pair_id": "ab-open-0", "matchup": ["A", "B"], "black_model": "B", "white_model": "A", "winner_model": "B", "winner_color": "B", "outcome": "black_win", "thinking_seconds_total": 1.0, "moves": [{"model": "B", "elapsed_seconds": 0.2, "nodes": 20, "simulations": 0, "completed_depth": 1, "timed_out": False}]},
        {"game_id": "ac0", "pair_id": "ac-open-0", "matchup": ["A", "C"], "black_model": "A", "white_model": "C", "winner_model": None, "winner_color": None, "outcome": "draw", "thinking_seconds_total": 2.0, "moves": [{"model": "C", "elapsed_seconds": 0.3, "nodes": 30, "simulations": 2, "completed_depth": 1, "timed_out": True}]},
        {"game_id": "ad0", "pair_id": "ad-open-0", "matchup": ["A", "D"], "black_model": "D", "white_model": "A", "winner_model": "A", "winner_color": "W", "outcome": "white_win", "thinking_seconds_total": 3.0, "moves": [{"model": "A", "elapsed_seconds": 0.4, "nodes": 40, "simulations": 0, "completed_depth": 2, "timed_out": False}]},
        {"game_id": "bc0", "pair_id": "bc-open-0", "matchup": ["B", "C"], "black_model": "B", "white_model": "C", "winner_model": "C", "winner_color": "W", "outcome": "white_win", "thinking_seconds_total": 4.0, "moves": []},
        {"game_id": "bd0", "pair_id": "bd-open-0", "matchup": ["B", "D"], "black_model": "B", "white_model": "D", "winner_model": None, "winner_color": None, "outcome": "draw", "thinking_seconds_total": 5.0, "moves": []},
        {"game_id": "cd0", "pair_id": "cd-open-0", "matchup": ["C", "D"], "black_model": "C", "white_model": "D", "winner_model": "D", "winner_color": "W", "outcome": "white_win", "thinking_seconds_total": 6.0, "moves": []},
    ]

    summary = metrics.aggregate_results(games, bootstrap_samples=200)
    assert set(summary["win_matrix"]) == {"A", "B", "C", "D"}
    assert set(summary["win_matrix"]["A"]) == {"A", "B", "C", "D"}
    assert summary["win_matrix"]["A"]["B"]["wins"] == 1
    assert summary["win_matrix"]["B"]["A"]["wins"] == 1
    assert summary["pairwise"]["A-B"]["draws"] == 0
    assert summary["pairwise"]["A-B"]["models"]["A"]["score_rate"] == 0.5
    assert summary["pairwise"]["A-B"]["models"]["B"]["score_rate"] == 0.5
    assert summary["pairwise"]["A-C"]["models"]["A"]["score_rate"] == 0.5
    assert summary["pairwise"]["A-C"]["models"]["C"]["score_rate"] == 0.5
    assert len(summary["ranking"]) == 4
    assert summary["ranking"][0]["score_rate_ci95"][0] <= summary["ranking"][0]["score_rate_ci95"][1]
    assert summary["color_advantage"]["games"] == len(games)


def test_pure_win_rate_counts_draws_as_non_wins_and_exposes_decisive_rate() -> None:
    records = [
        {"game_id": "ab-win", "pair_id": "open-win", "matchup": ["A", "B"], "winner_model": "A"},
        {"game_id": "ab-draw", "pair_id": "open-draw", "matchup": ["A", "B"], "winner_model": None},
        {"game_id": "ab-loss", "pair_id": "open-loss", "matchup": ["A", "B"], "winner_model": "B"},
    ]
    stats = _metrics().aggregate_results(records, bootstrap_samples=50)["pairwise"]["A-B"]["models"]["A"]
    assert stats["pure_win_rate"] == pytest.approx(1 / 3)
    assert stats["decisive_win_rate"] == pytest.approx(1 / 2)
    assert stats["score_rate"] == pytest.approx(1 / 2)


def test_fixed_depth_efficiency_uses_same_positions_and_values() -> None:
    efficiency = _efficiency()
    positions = efficiency.generate_position_suite(count=4, board_size=9, seed=888)
    assert positions == efficiency.generate_position_suite(count=4, board_size=9, seed=888)
    results = efficiency.run_fixed_depth_comparison(
        positions, depth=2, time_limit_seconds=10, seed=888
    )
    assert len(results) == 4
    for result in results:
        assert result["minimax"]["completed_depth"] == 2
        assert result["alpha_beta"]["completed_depth"] == 2
        assert result["minimax"]["root_value"] == result["alpha_beta"]["root_value"]
        assert result["alpha_beta"]["move"] in result["minimax"]["optimal_moves"]
        assert set(map(tuple, result["alpha_beta"]["optimal_moves"])) <= set(
            map(tuple, result["minimax"]["optimal_moves"])
        )
        assert 0 <= result["prune_ratio"] <= 1
