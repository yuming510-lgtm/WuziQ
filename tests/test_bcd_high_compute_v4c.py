from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ai import SearchConfig, SearchResult, choose_move
from src.research import bcd_high_compute_v4c as v4c
from src.research.bcd_high_compute_v4c_analysis import validate_game_record, validate_records
from src.research.bcd_high_compute_v4c_snapshot import create_snapshot, restore_snapshot


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "bcd_high_compute_v4c.json"


def _config() -> dict:
    return v4c.load_config(CONFIG_PATH)


def _winning_record(spec: dict, config: dict) -> dict:
    """Create a short legal black win for runner and replay tests."""
    moves = []
    for index, opening in enumerate(spec["opening"]):
        moves.append({
            "ply": index, "player": opening["player"], "model": None,
            "row": opening["row"], "col": opening["col"], "elapsed_seconds": 0.0,
            "nodes": 0, "completed_depth": 0, "tree_depth": 0, "cutoffs": 0,
            "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0,
            "truncated_simulations": 0, "root_candidates": 0, "root_min_visits": 0,
            "reused_nodes": 0, "timed_out": False, "opening": True,
        })
    sequence = [("B", (8, 0)), ("W", (0, 0)), ("B", (8, 1)), ("W", (1, 0)),
                ("B", (8, 2)), ("W", (2, 0)), ("B", (8, 3)), ("W", (3, 0)), ("B", (8, 4))]
    for player, (row, col) in sequence:
        model = spec["black_model"] if player == "B" else spec["white_model"]
        moves.append({
            "ply": len(moves), "player": player, "model": model, "row": row, "col": col,
            "elapsed_seconds": 0.001, "nodes": 1, "completed_depth": 1, "tree_depth": 1,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0,
            "truncated_simulations": 0, "root_candidates": 1, "root_min_visits": 0,
            "reused_nodes": 0, "timed_out": False, "opening": False,
        })
    return {
        **spec, "matchup": list(spec["matchup"]), "source_type": "new",
        "experiment_id": config["experiment_id"], "implementation_commit": "test-commit",
        "moves": moves, "move_count": len(moves), "winner_color": "B",
        "winner_model": spec["black_model"], "outcome": "b_win", "termination": "five_in_row",
        "timeout_count": 0, "search_nodes_total": 9, "simulations_total": 0,
        "rollout_plies_total": 0, "terminal_simulations_total": 0,
        "truncated_simulations_total": 0, "tree_depth_max": 1,
        "thinking_seconds_total": 0.009,
    }


def test_schedule_has_18_unique_new_games_and_three_paired_openings() -> None:
    config = _config()
    schedule = v4c.build_schedule(config)

    assert len(schedule) == 18
    assert len({game["game_id"] for game in schedule}) == 18
    assert {"-".join(row["matchup"]) for row in schedule} == {"B-C", "B-D", "C-D"}
    assert all(row["source_type"] == "new" for row in schedule)
    for matchup in ("B-C", "B-D", "C-D"):
        games = [game for game in schedule if "-".join(game["matchup"]) == matchup]
        assert len(games) == 6
        assert {game["opening_seed"] for game in games} == {731091, 731092, 731093}
        assert {game["black_model"] for game in games} == set(matchup.split("-"))
        for seed in (731091, 731092, 731093):
            pair = [game for game in games if game["opening_seed"] == seed]
            assert len(pair) == 2
            assert {game["color_swap"] for game in pair} == {False, True}
            assert pair[0]["opening"] == pair[1]["opening"]
    assert not any(game["game_id"].startswith("v4b-") for game in schedule)


def test_b_and_c_share_all_search_settings_except_registered_budget() -> None:
    config = _config()
    b = v4c.settings_for_model(config, "B", occupied=12)
    c = v4c.settings_for_model(config, "C", occupied=12)
    assert replace(b, time_limit_seconds=c.time_limit_seconds) == c
    assert b.time_limit_seconds == pytest.approx(1.0)
    assert c.time_limit_seconds == pytest.approx(0.2)
    assert b.max_depth == c.max_depth == 69
    assert b.candidate_radius == c.candidate_radius == 2


def test_d_uses_v4b_mcts_settings_with_only_budget_increased() -> None:
    config = _config()
    d = v4c.settings_for_model(config, "D", occupied=12)
    assert d.time_limit_seconds == pytest.approx(50.0)
    assert d.candidate_radius == 2
    assert d.mcts_d2_tactical_root and d.mcts_d2_fair_root
    assert d.mcts_d2_tactical_rollout and d.mcts_d2_node_reuse
    assert d.mcts_simulations == 1_000_000
    assert d.mcts_rollout_max_plies == 24


def test_same_depth_b_and_c_return_equal_root_score_and_only_c_prunes() -> None:
    config = _config()
    board = v4c.GomokuBoard(9)
    for player, (row, col) in v4c._opening(config, 731091):
        board.place_stone(player, (row, col))
    common = SearchConfig(max_depth=2, time_limit_seconds=10.0, candidate_radius=2)
    b = choose_move(board, "B", "B", common, seed=731091)
    c = choose_move(board, "B", "C", common, seed=731091)
    assert b.completed_depth == c.completed_depth == 2
    assert b.root_value == c.root_value
    assert b.cutoffs == 0
    assert c.cutoffs > 0


def test_seed_derivation_is_stable_and_v4c_scoped() -> None:
    a = v4c.derive_seed(20261020, "v4c-game", "v4c-9x9-b-c-o731091-s0")
    assert a == v4c.derive_seed(20261020, "v4c-game", "v4c-9x9-b-c-o731091-s0")
    assert a != v4c.derive_seed(20261020, "v4b-game", "v4c-9x9-b-c-o731091-s0")


def test_game_replay_accepts_legal_win_and_rejects_tampered_outcome(monkeypatch) -> None:
    config = _config()
    spec = v4c.build_schedule(config)[0]
    black = iter(((8, 0), (8, 1), (8, 2), (8, 3), (8, 4)))
    white = iter(((0, 0), (1, 0), (2, 0), (3, 0)))

    def fake_choose(board, player, model, settings, seed):
        return SearchResult(next(black if player == "B" else white), 0, 1, 1, 0, 0, 0.001, False)

    monkeypatch.setattr(v4c, "choose_move", fake_choose)
    record = v4c.run_game(spec, config, implementation_commit="test-commit")
    assert record["termination"] == "five_in_row"
    assert validate_game_record(record, config, spec)["winner_model"] == record["black_model"]
    record["winner_model"] = record["white_model"]
    with pytest.raises(ValueError, match="outcome|winner"):
        validate_game_record(record, config, spec)


def test_resume_skips_completed_ids_and_deduplicates(tmp_path, monkeypatch) -> None:
    config = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(v4c, "run_game", lambda spec, frozen, implementation_commit: (calls.append(spec["game_id"]) or _winning_record(spec, frozen)))

    first = v4c.run_experiment(config_path, tmp_path / "results", max_games=1, implementation_commit="test-commit")
    second = v4c.run_experiment(config_path, tmp_path / "results", max_games=1, implementation_commit="test-commit")
    rows = [json.loads(line) for line in (tmp_path / "results" / "9x9_games.jsonl").read_text().splitlines()]
    assert first["completed_games"] == 1
    assert second["completed_games"] == 2
    assert len(calls) == 2
    assert len(rows) == len({row["game_id"] for row in rows}) == 2


def test_result_records_must_be_complete_and_color_paired() -> None:
    config = _config()
    rows = [_winning_record(spec, config) for spec in v4c.build_schedule(config)]
    result = validate_records(rows, config, require_complete=True)
    assert result["complete"] is True
    assert result["validated_games"] == 18
    assert result["new_games"] == 18
    with pytest.raises(ValueError, match="duplicate"):
        validate_records(rows + [rows[0]], config, require_complete=False)


def test_v4c_snapshot_round_trip_verifies_hash_and_ids(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    config = _config()
    (results / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (results / "progress.json").write_text(json.dumps({
        "schema_version": 1, "experiment_id": config["experiment_id"],
        "completed_games": 0, "completed_ids": [], "implementation_commit": "test-commit",
    }), encoding="utf-8")
    (results / "9x9_games.jsonl").write_text("", encoding="utf-8")
    archive = tmp_path / "v4c-recovery.zip"
    manifest = create_snapshot(results, archive, code_version="test-commit")
    restored = tmp_path / "empty-restore"
    result = restore_snapshot(archive, restored)
    assert result["game_ids"] == []
    assert result["completed_games"] == 0
    assert result["code_version"] == "test-commit"
    assert "9x9_games.jsonl" in manifest["files"]
    assert {p.name for p in restored.iterdir()} == {"config.json", "progress.json", "9x9_games.jsonl", "code_version.txt"}


def test_snapshot_restore_rejects_modified_archive(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    config = _config()
    (results / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (results / "progress.json").write_text(json.dumps({"experiment_id": config["experiment_id"], "completed_games": 0, "completed_ids": [], "implementation_commit": "test-commit"}), encoding="utf-8")
    (results / "9x9_games.jsonl").write_text("", encoding="utf-8")
    archive = tmp_path / "v4c-recovery.zip"
    create_snapshot(results, archive, code_version="test-commit")
    damaged = bytearray(archive.read_bytes())
    damaged[-20] ^= 0xFF
    broken = tmp_path / "broken.zip"
    broken.write_bytes(damaged)
    with pytest.raises((ValueError, Exception)):
        restore_snapshot(broken, tmp_path / "broken-restore")
