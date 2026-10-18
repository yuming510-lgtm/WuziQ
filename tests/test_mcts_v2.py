"""Correctness checks for the V2 MCTS research variants."""
from __future__ import annotations

import sys
from pathlib import Path
import importlib
import subprocess
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import SearchConfig, choose_move, generate_candidates
from src.gomoku import GomokuBoard
from src.research.mcts_d2 import MCTSAgent, _Node, _uct_child
from src.ai import SearchResult


def test_d1_matches_original_uct_when_the_rollout_horizon_is_already_full() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    config = SearchConfig(
        time_limit_seconds=10,
        mcts_simulations=16,
        mcts_rollout_max_plies=24,
    )

    baseline = choose_move(board, "B", "D", config, seed=731)
    high_budget = choose_move(board, "B", "D1", config, seed=731)

    assert high_budget.move == baseline.move
    assert high_budget.root_value == baseline.root_value
    assert high_budget.nodes == baseline.nodes
    assert high_budget.simulations == baseline.simulations


def test_d1_reports_completed_terminal_rollouts_separately() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    result = choose_move(
        board,
        "B",
        "D1",
        SearchConfig(time_limit_seconds=10, mcts_simulations=8),
        seed=219,
    )

    assert result.simulations == 8
    assert result.terminal_simulations == result.simulations
    assert result.truncated_simulations == 0
    assert result.rollout_plies >= result.simulations


def test_d2_takes_an_immediate_win_without_minimax_search() -> None:
    board = GomokuBoard(5)
    for col in range(4):
        board.place_stone("B", (2, col))
    original_history = list(board.history)

    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(time_limit_seconds=0.05, mcts_simulations=2),
        seed=1,
    )

    assert result.move == (2, 4)
    assert board.history == original_history


def test_d2_blocks_a_unique_immediate_opponent_win() -> None:
    board = GomokuBoard(5)
    for col in range(4):
        board.place_stone("W", (2, col))

    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(time_limit_seconds=0.05, mcts_simulations=2),
        seed=2,
    )

    assert result.move == (2, 4)


def test_d2_covers_each_root_candidate_before_revisiting_one() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    candidates = generate_candidates(board, "B", "W", radius=1)
    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(
            time_limit_seconds=10,
            mcts_simulations=len(candidates),
            candidate_radius=1,
        ),
        seed=33,
    )

    assert result.root_candidates == len(candidates)
    assert result.root_min_visits >= 1


def test_d2_terminal_rollout_counts_are_explicit_and_board_is_restored() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    initial_history = list(board.history)
    initial_grid = [row[:] for row in board.grid]
    result = choose_move(
        board,
        "B",
        "D2",
        SearchConfig(time_limit_seconds=10, mcts_simulations=8),
        seed=219,
    )

    assert result.simulations == 8
    assert result.terminal_simulations == result.simulations
    assert result.truncated_simulations == 0
    assert result.rollout_plies >= result.simulations
    assert board.history == initial_history
    assert board.grid == initial_grid


def test_d2_is_deterministic_for_a_fixed_simulation_budget_and_seed() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    config = SearchConfig(time_limit_seconds=10, mcts_simulations=20)

    first = choose_move(board, "B", "D2", config, seed=999)
    second = choose_move(board, "B", "D2", config, seed=999)

    assert first.move == second.move
    assert first.simulations == second.simulations == 20


def test_d2_reuses_matching_subtree_after_its_move_and_opponent_reply() -> None:
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))
    config = SearchConfig(time_limit_seconds=10, mcts_simulations=40)
    agent = MCTSAgent()

    first = agent.choose_move(board, "B", config, seed=51)
    board.place_stone("B", first.move)
    selected = next(child for child in agent._root.children if child.move == first.move)
    assert selected.children
    opponent_reply = selected.children[0].move
    assert opponent_reply is not None
    board.place_stone("W", opponent_reply)

    second = agent.choose_move(board, "B", config, seed=52)

    assert board.get(second.move) is None
    assert second.reused_nodes >= 1


def test_d2_zero_budget_returns_a_legal_fallback() -> None:
    board = GomokuBoard(7)
    board.place_stone("B", (3, 3))
    result = choose_move(
        board,
        "W",
        "D2",
        SearchConfig(time_limit_seconds=0, mcts_simulations=2),
        seed=4,
    )

    assert result.timed_out
    assert board.get(result.move) is None


def test_d2_uct_selects_unvisited_nodes_without_dividing_by_zero() -> None:
    parent = _Node(None, "W", visits=1)
    first = _Node((0, 0), "B", parent=parent)
    second = _Node((0, 1), "B", parent=parent)
    parent.children = [first, second]

    assert _uct_child(parent, 1.4142135623730951) is first


def test_d2_fast_candidates_preserve_the_shared_radius_candidate_set() -> None:
    module = importlib.import_module("src.research.mcts_d2")
    board = GomokuBoard(7)
    for player, coord in [
        ("B", (3, 3)),
        ("W", (3, 4)),
        ("B", (4, 3)),
        ("W", (2, 4)),
    ]:
        board.place_stone(player, coord)

    fast = module._d2_candidates(board, "B", "W", radius=2)
    shared = generate_candidates(board, "B", "W", radius=2)

    assert set(fast) == set(shared)
    assert len(fast) == len(set(fast))


def test_d2_inline_win_check_matches_board_rules_for_every_empty_cell() -> None:
    module = importlib.import_module("src.research.mcts_d2")
    board = GomokuBoard(9)
    for player, coord in [
        ("B", (4, 1)),
        ("B", (4, 2)),
        ("B", (4, 3)),
        ("B", (4, 4)),
        ("W", (2, 2)),
        ("W", (3, 3)),
        ("W", (5, 5)),
    ]:
        board.place_stone(player, coord)

    for row in range(board.size):
        for col in range(board.size):
            if board.get((row, col)) is not None:
                continue
            for player in ("B", "W"):
                assert module._would_win_fast(board, (row, col), player) == (
                    board.potential_line_length((row, col), player) >= 5
                )


def test_d2_combined_tactical_scan_matches_individual_win_and_block_checks() -> None:
    module = importlib.import_module("src.research.mcts_d2")
    board = GomokuBoard(9)
    for col in range(1, 5):
        board.place_stone("B", (4, col))
    for row in range(2, 6):
        board.place_stone("W", (row, 7))
    candidates = module._d2_candidates(board, "B", "W", radius=2)

    wins, blocks = module._immediate_tactics(board, candidates, "B")
    expected_wins = [coord for coord in candidates if module._would_win_fast(board, coord, "B")]
    expected_blocks = [
        coord
        for coord in candidates
        if coord not in expected_wins and module._would_win_fast(board, coord, "W")
    ]

    assert set(wins) == set(expected_wins)
    assert set(blocks) == set(expected_blocks)


def _v2_runner():
    return importlib.import_module("src.research.mcts_v2")


def _small_runner_config(games_per_matchup: int = 2) -> dict:
    return {
        "experiment_id": "wuziq-mcts-v2-test",
        "base_seed": 20261018,
        "opening_plies": 4,
        "search": {"max_depth": 1, "time_limit_seconds": 0, "mcts_simulations": 2},
        "phases": [
            {
                "name": "fair-5x5-50ms",
                "track": "fair",
                "board_size": 5,
                "games_per_matchup": games_per_matchup,
                "matchups": [["A", "D1"]],
                "budget_seconds_by_model": {"default": 0.05},
            }
        ],
    }


def test_v2_opening_family_is_diverse_and_color_swap_preserves_coordinates() -> None:
    runner = _v2_runner()
    openings = [
        tuple((row, col) for _, (row, col) in runner.paired_opening(9, seed))
        for seed in range(100)
    ]
    first = runner.paired_opening(9, 510)
    swapped = runner.paired_opening(9, 510, color_swap=True)

    assert len(set(openings)) >= 90
    assert [coord for _, coord in first] == [coord for _, coord in swapped]
    assert [player for player, _ in swapped] == [
        "W" if player == "B" else "B" for player, _ in first
    ]


def test_v2_schedule_ids_are_unique_and_fair_budgets_match() -> None:
    runner = _v2_runner()
    config = _small_runner_config()
    config["phases"].append(
        {
            "name": "fair-5x5-200ms",
            "track": "fair",
            "board_size": 5,
            "games_per_matchup": 2,
            "matchups": [["A", "D1"]],
            "budget_seconds_by_model": {"default": 0.2},
        }
    )
    schedule = runner.build_schedule(config)
    ids = [spec["game_id"] for spec in schedule]

    assert len(ids) == len(set(ids))
    assert len({spec["opening_index"] for spec in schedule}) >= 1
    fair = [spec for spec in schedule if spec["track"] == "fair"]
    assert all(
        len(set(spec["budget_seconds_by_model"].values())) == 1 for spec in fair
    )


def test_v2_extreme_schedule_keeps_c_at_v1_budget() -> None:
    runner = _v2_runner()
    config = _small_runner_config()
    config["phases"] = [
        {
            "name": "extreme-5x5-d2-1s-vs-c-50ms",
            "track": "extreme",
            "board_size": 5,
            "games_per_matchup": 2,
            "matchups": [["D2", "C"]],
            "budget_seconds_by_model": {"D2": 1.0, "C": 0.05},
        }
    ]

    schedule = runner.build_schedule(config)

    assert len(schedule) == 2
    assert all(spec["budget_seconds_by_model"]["C"] == 0.05 for spec in schedule)
    assert all(spec["budget_seconds_by_model"]["D2"] == 1.0 for spec in schedule)


def test_v2_runner_records_legal_timeouts_without_forfeiting(monkeypatch) -> None:
    runner = _v2_runner()
    config = _small_runner_config()
    spec = runner.build_schedule(config)[0]

    def timed_out_legal_move(board, player, model, settings, seed):
        move = next(
            (row, col)
            for row in range(board.size)
            for col in range(board.size)
            if board.get((row, col)) is None
        )
        return SearchResult(
            move, 0, 0, 0, 0, 0, settings.time_limit_seconds + 0.01, True, (move,)
        )

    monkeypatch.setattr(runner, "choose_move", timed_out_legal_move)
    record = runner.run_game(spec, config)

    assert record["timeout_count"] > 0
    assert record["termination"] != "illegal_move_forfeit"


def test_v2_runner_forfeits_on_the_first_illegal_move(monkeypatch) -> None:
    runner = _v2_runner()
    config = _small_runner_config()
    spec = runner.build_schedule(config)[0]

    def illegal_move(board, player, model, settings, seed):
        return SearchResult((board.size, board.size), 0, 0, 0, 0, 0, 0.001, False)

    monkeypatch.setattr(runner, "choose_move", illegal_move)
    record = runner.run_game(spec, config)

    assert record["termination"] == "illegal_move_forfeit"
    assert record["winner_model"] in spec["matchup"]
    assert record["moves"][-1]["illegal"] is True


def test_v2_runner_resume_fills_remaining_ids_without_duplicates(tmp_path: Path) -> None:
    runner = _v2_runner()
    config = _small_runner_config()
    first = runner.run_experiment(config, tmp_path, max_games=1)
    second = runner.run_experiment(config, tmp_path)
    third = runner.run_experiment(config, tmp_path)

    assert first["games_completed_this_run"] == 1
    assert second["games_completed_this_run"] == 1
    assert third["games_completed_this_run"] == 0
    assert second["games_completed_total"] == 2
    assert third["games_completed_total"] == 2


def test_v2_run_records_the_source_commit_for_recovery(tmp_path: Path) -> None:
    runner = _v2_runner()
    config = _small_runner_config()
    root = Path(__file__).resolve().parents[1]
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    runner.run_experiment(config, tmp_path, max_games=0)
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))

    assert saved["code_commit"] == expected_commit
    assert progress["code_commit"] == expected_commit


def test_v2_cli_exposes_reproducible_config_and_output_options() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_mcts_v2.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--max-games" in completed.stdout


def test_fixed_simulation_benchmark_records_actual_completed_count() -> None:
    benchmark = importlib.import_module("src.research.mcts_v2_benchmark")
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))

    records = benchmark.benchmark_position(
        board,
        "B",
        models=("D1", "D2"),
        target_simulations=(3,),
        max_seconds=10,
        base_seed=871,
    )

    assert {item["requested_simulations"] for item in records} == {3}
    assert {item["actual_simulations"] for item in records} == {3}
    assert all(item["elapsed_seconds"] > 0 for item in records)


def test_d2_ablation_variants_cover_each_engine_optimization() -> None:
    ablation = importlib.import_module("src.research.mcts_v2_ablation")
    variants = ablation.ablation_variants()

    assert set(variants) == {
        "full",
        "no_root_tactics",
        "no_fair_root",
        "no_tactical_rollout",
        "no_node_reuse",
    }
    assert variants["no_root_tactics"]["mcts_d2_tactical_root"] is False
    assert variants["no_fair_root"]["mcts_d2_fair_root"] is False
    assert variants["no_tactical_rollout"]["mcts_d2_tactical_rollout"] is False
    assert variants["no_node_reuse"]["mcts_d2_node_reuse"] is False


def test_ablation_runner_records_d1_d2_features_and_reuse_probe() -> None:
    ablation = importlib.import_module("src.research.mcts_v2_ablation")
    board = GomokuBoard(5)
    board.place_stone("B", (2, 2))
    board.place_stone("W", (2, 3))

    result = ablation.run_ablation(
        [{"position_id": "reuse-test", "board": board, "player": "B"}],
        budgets_seconds=(10,),
        simulations=8,
        base_seed=92,
    )

    assert any(row["model"] == "D1" for row in result["rows"])
    assert any(row["variant"] == "full" for row in result["rows"])
    assert any(row["variant"] == "no_tactical_rollout" for row in result["rows"])


def test_d2_ablation_tactical_cases_accept_any_valid_win_or_unique_block() -> None:
    benchmark = importlib.import_module("src.research.mcts_v2_benchmark")
    ablation = importlib.import_module("src.research.mcts_v2_ablation")
    positions = [
        position
        for position in benchmark.representative_positions(81)
        if "forced-" in position["position_id"] and position["board"].size == 9
    ]
    result = ablation.run_ablation(
        positions, budgets_seconds=(10,), simulations=2, base_seed=81
    )
    full_rows = [
        row for row in result["rows"] if row["model"] == "D2" and row["variant"] == "full"
    ]

    assert len(full_rows) == 2
    assert all(row["expected_move_found"] is True for row in full_rows)


def test_v2_snapshot_local_drill_verifies_restore_and_keeps_other_files(
    tmp_path: Path,
) -> None:
    snapshot = importlib.import_module("src.research.mcts_v2_snapshot")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")

    result = snapshot.run_local_restore_drill(tmp_path)

    assert result["restore_verified"] is True
    assert result["checksum_corruption_rejected"] is True
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_formal_v2_config_pre_registers_both_mcts_models_and_all_opponents() -> None:
    runner = _v2_runner()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mcts_potential_v2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schedule = runner.build_schedule(config)
    fair_9x9 = [spec for spec in schedule if spec["board_size"] == 9 and spec["track"] == "fair"]
    pairs = {tuple(spec["matchup"]) for spec in fair_9x9}

    assert len(schedule) == 2960
    assert pairs == {
        ("A", "D1"), ("B", "D1"), ("C", "D1"),
        ("A", "D2"), ("B", "D2"), ("C", "D2"),
    }
