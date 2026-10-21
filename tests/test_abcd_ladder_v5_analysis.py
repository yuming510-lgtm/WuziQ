from __future__ import annotations

from pathlib import Path

import pytest

from src.research.abcd_ladder_v5 import build_schedule, load_config
from src.research.abcd_ladder_v5_analysis import _write_csv, summarize_records, validate_game_record, wilson_fractional
from src.research.abcd_ladder_v5_report import build_report


CONFIG = Path(__file__).parents[1] / "configs" / "abcd_ladder_v5.json"


def _valid_black_win(spec: dict) -> dict:
    moves = []
    for index, stone in enumerate(spec["opening"]):
        moves.append({
            "ply": index, "player": stone["player"], "model": None,
            "row": stone["row"], "col": stone["col"], "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 0, "tree_depth": 0,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
            "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": True,
        })
    continuation = [("B", 4, 0), ("W", 0, 0), ("B", 4, 1), ("W", 0, 1),
                    ("B", 4, 2), ("W", 0, 2), ("B", 4, 3)]
    elapsed_total = 0.0
    for player, row, col in continuation:
        model = spec["black_model"] if player == "B" else spec["white_model"]
        elapsed = 0.001
        elapsed_total += elapsed
        moves.append({
            "ply": len(moves), "player": player, "model": model,
            "row": row, "col": col, "elapsed_seconds": elapsed,
            "root_value": 0, "nodes": 0, "completed_depth": 0, "tree_depth": 0,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
            "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False,
        })
    return {
        **spec,
        "implementation_commit": "test-commit",
        "moves": moves,
        "move_count": len(moves),
        "winner_color": "B",
        "winner_model": spec["black_model"],
        "outcome": "b_win",
        "termination": "five_in_row",
        "timeout_count": 0,
        "search_nodes_total": 0,
        "simulations_total": 0,
        "tree_depth_max": 0,
        "thinking_seconds_total": elapsed_total,
    }


def test_valid_game_replays_and_reconstructs_actual_winner():
    config = load_config(CONFIG)
    first_group = build_schedule(config)[0]
    record = _valid_black_win(first_group["games"][0])

    result = validate_game_record(record, config)

    assert result["winner_model"] == "A"
    assert result["move_count"] == 11
    assert result["termination"] == "five_in_row"


def test_replay_rejects_repeated_move_even_if_winner_metadata_looks_valid():
    config = load_config(CONFIG)
    record = _valid_black_win(build_schedule(config)[0]["games"][0])
    record["moves"][-1]["row"] = 4
    record["moves"][-1]["col"] = 4

    with pytest.raises(ValueError, match="repeats or misplaces"):
        validate_game_record(record, config)


def test_incomplete_color_pair_is_not_counted_as_a_group_or_promotion():
    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    record = _valid_black_win(group["games"][0])

    summary = summarize_records([record], config)

    assert summary["actual_games"] == 1
    assert summary["actual_groups"] == 0
    assert summary["route_state"]["routes"]["A>B"]["levels"][0]["status"] == "incomplete_pair"
    assert summary["route_state"]["routes"]["A>B"]["highest_passed_budget"] is None


def test_fractional_score_interval_is_bounded_and_requires_data():
    assert wilson_fractional([]) is None
    interval = wilson_fractional([2.0, 0.0])
    assert interval is not None
    assert 0.0 <= interval[0] < 0.5 < interval[1] <= 1.0


def test_group_intervals_use_fractional_group_scores_and_report_renders(tmp_path):
    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    records = [_valid_black_win(spec) for spec in group["games"]]
    summary = summarize_records(records, config)
    pair = summary["pairwise"][0]

    assert summary["actual_metrics"]["unique_game_ids"] == 2
    assert summary["actual_metrics"]["terminations"] == {"five_in_row": 2}
    assert pair["first_score_rate"] == 0.5
    assert pair["second_score_rate"] == 0.5
    assert pair["first_group_score_ci95"][0] < 0.5 < pair["first_group_score_ci95"][1]
    report_path = build_report(summary, tmp_path / "results", tmp_path / "v5")
    assert report_path.is_file()
    assert (tmp_path / "v5" / "figures" / "ladder_gradient.svg").is_file()
    assert "未运行" in report_path.read_text(encoding="utf-8")
    assert not report_path.read_bytes().endswith(b"\n\n")


def test_csv_writer_uses_diff_clean_line_endings(tmp_path):
    path = tmp_path / "rows.csv"

    _write_csv(path, [{"value": "one"}, {"value": "two"}])

    assert path.read_bytes() == b"\xef\xbb\xbfvalue\none\ntwo\n"


def test_report_includes_run_window_drive_verification_and_discarded_pilot(tmp_path):
    import json

    config = load_config(CONFIG)
    group = build_schedule(config)[0]
    summary = summarize_records([_valid_black_win(spec) for spec in group["games"]], config)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "progress.json").write_text(json.dumps({
        "first_started_at_utc": "2026-10-20T23:18:36Z",
        "completed_at_utc": "2026-10-21T00:20:33Z",
        "elapsed_seconds_total": 42.0,
        "elapsed_wall_seconds_total": 75.0,
        "completed_games": 2,
        "completed_groups": 1,
    }), encoding="utf-8")
    (results_dir / "preflight.json").write_text(json.dumps({
        "preflight_elapsed_seconds": 120.0,
        "estimate": {"estimated_full_schedule_wall_minutes": 787.548},
    }), encoding="utf-8")
    (results_dir / "drive_recovery.json").write_text(json.dumps({
        "status": "已验证 Drive 上传、下载、SHA-256 与空目录恢复",
        "latest_snapshot": {"drive_file_id": "snapshot-123", "sha256": "abc123", "completed_games": 2},
        "files_remaining": 1,
    }), encoding="utf-8")
    (results_dir / "preliminary_attempt_audit.json").write_text(json.dumps({
        "completed_games": 10,
        "counted_in_official_results": False,
        "reason": "review fix",
    }), encoding="utf-8")

    report_path = build_report(summary, results_dir, tmp_path / "v5")
    report = report_path.read_text(encoding="utf-8")

    assert "2026-10-20T23:18:36Z" in report and "2026-10-21T00:20:33Z" in report
    assert "Drive 上传、下载、SHA-256 与空目录恢复" in report
    assert "snapshot-123" in report
    assert "10 局前测已作废且未计入正式成绩" in report
    assert "预检按理论 60 组估算约 13.13 小时" in report
    assert "路线 `A>B` 截至本报告已通过至 50ms，后续档尚未完成。" in report
    assert "本轮非等预算混合样本的描述性得分率排序为：A 50.0%、B 50.0%、C —、D —。" in report
    assert "C 98.3%" not in report
    assert "路线 `C>B` 通过全部预算档" not in report
