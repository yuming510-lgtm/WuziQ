from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from src.ai import SearchResult
from src.research import abcd_final_v4b as v4b
from src.research.abcd_final_v4b_analysis import summarize_records, validate_game_record, validate_records
from src.research import abcd_final_v4b_preflight as preflight
from src.research.abcd_final_v4b_report import build_report
from src.research.abcd_final_v4b_snapshot import create_snapshot, restore_snapshot


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "abcd_final_v4b.json"


def _config() -> dict:
    return v4b.load_config(CONFIG_PATH)


def test_schedule_is_36_games_with_fixed_openings_and_balanced_colors() -> None:
    config = _config()
    schedule = v4b.build_schedule(config)

    assert len(schedule) == 36
    assert len({game["game_id"] for game in schedule}) == 36
    for matchup in ("A-B", "A-C", "B-C", "A-D", "B-D", "C-D"):
        games = [game for game in schedule if "-".join(game["matchup"]) == matchup]
        assert len(games) == 6
        assert {game["opening_seed"] for game in games} == {731091, 731092, 731093}
        for seed in (731091, 731092, 731093):
            pair = [game for game in games if game["opening_seed"] == seed]
            assert len(pair) == 2
            assert {game["color_swap"] for game in pair} == {False, True}
            assert pair[0]["opening"] == pair[1]["opening"]
    assert [game["game_id"] for game in schedule] == [
        game["game_id"] for game in v4b.build_schedule(config)
    ]


def test_b_and_c_use_same_v4_search_settings_and_deadline() -> None:
    config = _config()
    b = v4b.settings_for_model(config, "B", occupied=12)
    c = v4b.settings_for_model(config, "C", occupied=12)

    assert b == c
    assert b.time_limit_seconds == pytest.approx(0.2)
    assert b.max_depth == 69
    assert b.candidate_radius == 2


def test_v3_d20_vs_a_records_are_reused_only_after_full_provenance_check() -> None:
    config = _config()
    records, audit = v4b.load_v3_reuse(config, ROOT)

    assert audit["verified"] is True
    assert audit["source_code_commit"] == "ff6aa837cf2d38268485fab51548c0e8327eb56d"
    assert audit["reused_games"] == 6
    assert len(records) == 6
    assert all(record["experiment_id"] == config["experiment_id"] for record in records)
    assert all(record["source_experiment_id"] == "wuziq-mcts-high-compute-v3" for record in records)
    assert {record["matchup"][1] for record in records} == {"D"}
    assert all(record["source_type"] == "reused_v3" for record in records)
    assert all(record["budget_seconds_by_model"]["D"] == 20.0 for record in records)
    assert validate_records(records, config, require_complete=False)["reused_v3_games"] == 6


def test_run_game_keeps_one_persistent_d3_agent_and_records_real_mcts_metrics(monkeypatch) -> None:
    config = _config()
    spec = next(
        game
        for game in v4b.build_schedule(config)
        if game["matchup"] == ("A", "D")
        and game["opening_seed"] == 731091
        and game["color_swap"] is True
    )
    constructed: list[object] = []

    class FakeMCTSAgent:
        def __init__(self) -> None:
            self.moves = iter(((4, 5), (4, 6), (4, 7), (4, 8)))
            constructed.append(self)

        def choose_move(self, board, player, search_config, seed):
            assert player == "B"
            assert search_config.time_limit_seconds == pytest.approx(20.0)
            move = next(self.moves)
            return SearchResult(move, 0, 17, 3, 0, 11, 19.99, True)

    white_moves = iter(((0, 0), (1, 0), (2, 0)))

    def fake_choose_move(board, player, model, search_config, seed):
        assert model == "A"
        move = next(white_moves)
        return SearchResult(move, 0, 0, 0, 0, 0, 0.0001, False)

    monkeypatch.setattr(v4b, "MCTSAgent", FakeMCTSAgent)
    monkeypatch.setattr(v4b, "choose_move", fake_choose_move)
    record = v4b.run_game(spec, config)

    assert len(constructed) == 1
    assert record["winner_model"] == "D"
    assert record["termination"] == "five_in_row"
    d_moves = [move for move in record["moves"] if move.get("model") == "D"]
    assert len(d_moves) == 4
    assert all(move["simulations"] == 11 for move in d_moves)
    assert all(move["tree_depth"] == 3 for move in d_moves)
    assert all(move["timed_out"] for move in d_moves)


def test_illegal_move_is_recorded_as_a_forfeit_without_crashing(monkeypatch) -> None:
    config = _config()
    spec = next(
        game
        for game in v4b.build_schedule(config)
        if game["matchup"] == ("A", "D") and game["opening_seed"] == 731091 and game["color_swap"]
    )

    class IllegalMCTSAgent:
        def choose_move(self, board, player, search_config, seed):
            return SearchResult("invalid", 0, 9, 1, 0, 3, 20.1, True)  # type: ignore[arg-type]

    monkeypatch.setattr(v4b, "MCTSAgent", IllegalMCTSAgent)
    record = v4b.run_game(spec, config)

    assert record["termination"] == "illegal_move_forfeit"
    assert record["winner_model"] == "A"
    assert record["moves"][-1]["illegal"] is True
    assert "row" not in record["moves"][-1]
    validate_game_record(record, config, spec)
    record["moves"][-1]["attempted_move"] = [0, 0]
    with pytest.raises(ValueError, match="attempted_move"):
        validate_game_record(record, config, spec)


def test_experiment_resume_skips_completed_unique_ids(tmp_path, monkeypatch) -> None:
    config = _config()
    config["reuse_v3"]["enabled"] = False
    calls: list[str] = []

    def fake_run_game(spec, frozen):
        calls.append(spec["game_id"])
        moves = []
        for index, opening_move in enumerate(spec["opening"]):
            moves.append(v4b._opening_log(opening_move["player"], (opening_move["row"], opening_move["col"]), index))
        main_line = (((4, 5), (0, 0)), ((4, 6), (1, 0)), ((4, 7), (2, 0)), ((4, 8), None))
        for black_coord, white_coord in main_line:
            moves.append({"ply": len(moves), "player": "B", "model": spec["black_model"], "row": black_coord[0], "col": black_coord[1], "elapsed_seconds": 0.001, "nodes": 1, "completed_depth": 1, "tree_depth": 1, "cutoffs": 0, "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0, "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False})
            if white_coord is not None:
                moves.append({"ply": len(moves), "player": "W", "model": spec["white_model"], "row": white_coord[0], "col": white_coord[1], "elapsed_seconds": 0.0, "nodes": 0, "completed_depth": 0, "tree_depth": 0, "cutoffs": 0, "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0, "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False})
        return {
            **spec,
            "matchup": list(spec["matchup"]),
            "source_type": "new",
            "moves": moves,
            "move_count": len(moves),
            "winner_color": "B",
            "winner_model": spec["black_model"],
            "outcome": "b_win",
            "termination": "five_in_row",
            "timeout_count": 0,
            "search_nodes_total": 4,
            "simulations_total": 0,
            "rollout_plies_total": 0,
            "terminal_simulations_total": 0,
            "truncated_simulations_total": 0,
            "tree_depth_max": 1,
            "thinking_seconds_total": 0.004,
        }

    monkeypatch.setattr(v4b, "run_game", fake_run_game)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    first = v4b.run_experiment(config_path, tmp_path / "results", max_games=1)
    second = v4b.run_experiment(config_path, tmp_path / "results", max_games=1)

    assert first["new_games"] == 1
    assert second["new_games"] == 2
    assert second["games_completed_this_run"] == 1
    assert len(calls) == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "9x9_games.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len({row["game_id"] for row in rows}) == 2


def test_v4b_snapshot_round_trip_checks_hash_and_restores_only_recovery_state(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for name, value in (
        ("config.json", {"experiment_id": "snapshot-drill"}),
        ("progress.json", {"completed_ids": ["game-1"], "completed_games": 1}),
        ("reuse_audit.json", {"verified": False, "reused_games": 0}),
    ):
        (results / name).write_text(json.dumps(value), encoding="utf-8")
    (results / "9x9_games.jsonl").write_text(
        json.dumps({"game_id": "game-1", "moves": []}) + "\n", encoding="utf-8"
    )

    archive = tmp_path / "v4b-drill.zip"
    manifest = create_snapshot(results, archive, code_version="implementation-commit")
    restored = tmp_path / "restored"
    result = restore_snapshot(archive, restored)

    assert "9x9_games.jsonl" in manifest["files"]
    assert result["game_ids"] == ["game-1"]
    assert (restored / "code_version.txt").read_text().strip() == "implementation-commit"


def test_full_match_analysis_has_all_pairings_and_descriptive_paired_intervals() -> None:
    config = _config()
    records = []
    for spec in v4b.build_schedule(config):
        moves = [
            v4b._opening_log(opening["player"], (opening["row"], opening["col"]), index)
            for index, opening in enumerate(spec["opening"])
        ]
        for index, (black_coord, white_coord) in enumerate(
            (((8, 0), (0, 0)), ((8, 1), (1, 0)), ((8, 2), (2, 0)), ((8, 3), (3, 0)), ((8, 4), None))
        ):
            for player, coordinate in (("B", black_coord), ("W", white_coord)):
                if coordinate is None:
                    continue
                model = spec["black_model"] if player == "B" else spec["white_model"]
                moves.append({"ply": len(moves), "player": player, "model": model, "row": coordinate[0], "col": coordinate[1], "elapsed_seconds": 0.001, "nodes": 1, "completed_depth": 1, "tree_depth": 1, "cutoffs": 0, "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0, "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False})
        records.append({**spec, "matchup": list(spec["matchup"]), "source_type": "new", "moves": moves, "move_count": len(moves), "winner_color": "B", "winner_model": spec["black_model"], "outcome": "b_win", "termination": "five_in_row", "timeout_count": 0, "search_nodes_total": 9, "simulations_total": 0, "rollout_plies_total": 0, "terminal_simulations_total": 0, "truncated_simulations_total": 0, "tree_depth_max": 1, "thinking_seconds_total": 0.009})

    summary = summarize_records(records, config)

    assert summary["game_count"] == 36
    assert list(summary["matchups"]) == ["A-B", "A-C", "B-C", "A-D", "B-D", "C-D"]
    assert all(summary["matchups"][pair]["games"] == 6 for pair in summary["matchups"])
    assert all(len(summary["matchups"][pair]["paired_wilson_95_ci"]["A" if pair.startswith("A-") else pair[0]]) == 2 for pair in summary["matchups"])
    assert "non-comparable" in summary["usage"]["A"]["node_metric_semantics"]


def test_report_generation_includes_historical_v1_to_v4_context(tmp_path) -> None:
    config = _config()
    results = tmp_path / "results"
    results.mkdir()
    (results / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Reuse the deterministic, legal all-pair sample generated by the analysis test.
    records = []
    for spec in v4b.build_schedule(config):
        moves = [v4b._opening_log(opening["player"], (opening["row"], opening["col"]), index) for index, opening in enumerate(spec["opening"])]
        line = (((8, 0), (0, 0)), ((8, 1), (1, 0)), ((8, 2), (2, 0)), ((8, 3), (3, 0)), ((8, 4), None))
        for black_coord, white_coord in line:
            for player, coord in (("B", black_coord), ("W", white_coord)):
                if coord is None:
                    continue
                model = spec["black_model"] if player == "B" else spec["white_model"]
                moves.append({"ply": len(moves), "player": player, "model": model, "row": coord[0], "col": coord[1], "elapsed_seconds": 0.001, "nodes": 1, "completed_depth": 1, "tree_depth": 1, "cutoffs": 0, "simulations": 0, "rollout_plies": 0, "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0, "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False})
        records.append({**spec, "matchup": list(spec["matchup"]), "source_type": "new", "moves": moves, "move_count": len(moves), "winner_color": "B", "winner_model": spec["black_model"], "outcome": "b_win", "termination": "five_in_row", "timeout_count": 0, "search_nodes_total": 9, "simulations_total": 0, "rollout_plies_total": 0, "terminal_simulations_total": 0, "truncated_simulations_total": 0, "tree_depth_max": 1, "thinking_seconds_total": 0.009})
    (results / "9x9_games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (results / "reuse_audit.json").write_text(json.dumps({"verified": False, "reused_games": 0}), encoding="utf-8")
    (results / "preflight.json").write_text(json.dumps({"estimate": {"estimated_formal_run_minutes": 1.0, "estimated_formal_run_range_minutes": [1.0, 1.25], "D20_fullboard_theoretical_ceiling_seconds_for_12_new_D_games": 10}, "resources": {}, "benchmarks": [{"model": "D", "simulations": 1000, "elapsed_seconds_wall": 20}], "preflight_elapsed_seconds": 20, "storage_estimate": {"estimated_final_recovery_bundle_bytes": 1000}}), encoding="utf-8")
    (results / "execution.json").write_text(json.dumps({"wall_seconds_this_invocation": 12, "wall_seconds_accumulated": 34}), encoding="utf-8")

    output = build_report(results, ROOT)

    assert output["validation"]["validated_games"] == 36
    report = (results / "report_zh.md").read_text(encoding="utf-8")
    assert "V1–V4 历史结果" in report
    assert "A–B" in report and "B–D" in report and "C–D" in report
    assert "34.0 秒" in report
    assert "Drive 恢复演练未验证" in report
    assert "复用 V3 **0 局**" in report
    assert (tmp_path / "figures" / "matchup_score_rates.svg").is_file()


def test_preflight_persists_estimate_and_commit_metadata(tmp_path, monkeypatch) -> None:
    config = _config()

    class FakeMCTSAgent:
        def choose_move(self, board, player, search_config, seed):
            return SearchResult((4, 3), 0, 12, 2, 0, 5, 0.001, False)

    def fake_choose_move(board, player, model, search_config, seed):
        return SearchResult((4, 3), 0, 12, 2, 0, 0, 0.001, False)

    monkeypatch.setattr(preflight, "MCTSAgent", FakeMCTSAgent)
    monkeypatch.setattr(preflight, "choose_move", fake_choose_move)
    monkeypatch.setattr(preflight, "_historical_calibration", lambda root: {"B": {"D3_total_seconds": 10, "opponent_moves_total": 10}, "C": {"D3_total_seconds": 20, "opponent_moves_total": 20}})
    monkeypatch.setattr(preflight, "_resource_snapshot", lambda directory: {"workspace_disk_free_bytes": 123456})

    output = tmp_path / "preflight.json"
    result = preflight.run_preflight(config, output, project_root=ROOT)

    assert output.is_file()
    assert result["official_games_started"] == 0
    assert re.fullmatch(r"[0-9a-f]{40}", result["base_code_commit"])
    assert len(result["benchmarks"]) == 4


def test_historical_game_loader_never_truncates_a_corrupt_tail(tmp_path) -> None:
    path = tmp_path / "historic.jsonl"
    original = b'{"game_id":"good"}\nnot-json\n'
    path.write_bytes(original)

    with pytest.raises(ValueError, match="Invalid historical JSONL"):
        v4b.read_historical_games(path)

    assert path.read_bytes() == original
