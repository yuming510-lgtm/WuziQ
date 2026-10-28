from __future__ import annotations

import json
import subprocess
from pathlib import Path
import shutil

import pytest

from src.gomoku import GomokuBoard
from src.research.checkpoint import load_game_records
from src.research.v6_3.config import load_v6_3_config
from src.research.v6_3.analysis import summarize_v6_3
from src.research.v6_3.runner import run_championship
from src.research.v6_3.schedule import build_schedule
from src.research.v6_3.validation import validate_v6_3_records
from src.research.v6_3.validation import validate_v6_3_game_record


def _fake_game(spec, _engine_config, _v6_config, implementation_commit, winner_rule):
    board = GomokuBoard(9)
    moves = []

    def add(player, coord, model, opening):
        board.place_stone(player, coord)
        moves.append({
            "ply": len(moves), "player": player, "model": model,
            "row": coord[0], "col": coord[1], "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 0,
            "tree_depth": 0, "cutoffs": 0, "simulations": 0,
            "rollout_plies": 0, "terminal_simulations": 0,
            "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False,
            "opening": opening, "proof_status": "not_applicable",
            "proof_kind": None, "proof_certificate": None, "stats": {},
        })

    for move in spec["opening"]:
        add(move["player"], (move["row"], move["col"]), None, True)

    winner = winner_rule(spec)
    if winner == "black":
        target_color = "B"
    elif winner == "white":
        target_color = "W"
    else:
        target_color = "B" if spec["black_model"] == winner else "W"
    target_model = spec["black_model"] if target_color == "B" else spec["white_model"]
    target_index = other_index = 0
    other_color = "W" if target_color == "B" else "B"
    other_model = spec["black_model"] if other_color == "B" else spec["white_model"]
    other_coords = [(8, col) for col in (0, 2, 4, 6, 8)]
    current = "B"
    winner_color = winner_model = None
    while True:
        if current == target_color:
            coord = (0, target_index)
            target_index += 1
        else:
            coord = other_coords[other_index]
            other_index += 1
        model = target_model if current == target_color else other_model
        add(current, coord, model, False)
        if board.check_victory(coord, current):
            winner_color, winner_model = current, model
            break
        current = "W" if current == "B" else "B"

    return {
        **spec, "implementation_commit": implementation_commit,
        "moves": moves, "move_count": len(moves),
        "winner_color": winner_color, "winner_model": winner_model,
        "outcome": f"{winner_color.lower()}_win", "termination": "five_in_row",
        "timeout_count": 0,
        "search_nodes_total": 0, "simulations_total": 0,
        "tree_depth_max": 0, "search_depth_max": 0, "cutoffs_total": 0,
        "thinking_seconds_total": 0.0, "proof_count": 0,
        "proof_verified_count": 0, "source_type": "new", "source_game_id": None,
    }


def _repo_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_first_four_wins_without_tie_skips_conditional_games(tmp_path: Path) -> None:
    config = load_v6_3_config()
    calls = []

    def game(spec, engine_config, v6_config, commit):
        calls.append(spec["game_id"])
        return _fake_game(spec, engine_config, v6_config, commit, lambda _spec: "C")

    result = run_championship(config, tmp_path, game_runner=game, implementation_commit=_repo_commit())
    records = load_game_records(tmp_path / "9x9_games.jsonl")
    schedule = json.loads((tmp_path / "schedule.json").read_text())
    checked = validate_v6_3_records(records, schedule, require_complete=True)

    assert len(calls) == 4
    assert result["complete"] is True
    assert result["tiebreak_status"] == "skipped_not_tied"
    assert len(records) == 4
    assert checked["standings"]["C"]["points"] == 4.0
    summary = summarize_v6_3(config, tmp_path)
    assert summary["new_games"] == 4
    assert summary["combined_with_v6_2_direct"]["C"]["games"] == 6
    assert summary["combined_with_v6_2_direct"]["C"]["points"] == 5.0
    from scripts.build_v6_3_report import _report, _svg_report, csv_dicts

    report_root = tmp_path / "research" / "v6_3"
    report_root.mkdir(parents=True)
    (report_root / "results").mkdir()
    shutil.copyfile(tmp_path / "config.json", report_root / "results" / "config.json")
    summary["_game_rows"] = [
        {**row, "winner_model": row["winner_model"] or None, "color_swap": row["color_swap"] == "True"}
        for row in csv_dicts(tmp_path / "games.csv")
    ]
    summary["_opening_rows"] = list(csv_dicts(tmp_path / "opening_summary.csv"))
    _svg_report(summary, report_root / "figures")
    _report(summary, {}, records, report_root / "REPORT.md")
    assert "探索性冠军" in (report_root / "REPORT.md").read_text(encoding="utf-8")


def test_tied_first_four_run_only_the_two_preregistered_tiebreak_games(tmp_path: Path) -> None:
    config = load_v6_3_config()
    calls = []

    def game(spec, engine_config, v6_config, commit):
        calls.append(spec["game_id"])
        return _fake_game(spec, engine_config, v6_config, commit, lambda _spec: "black")

    result = run_championship(config, tmp_path, game_runner=game, implementation_commit=_repo_commit())
    schedule = json.loads((tmp_path / "schedule.json").read_text())
    records = load_game_records(tmp_path / "9x9_games.jsonl")
    checked = validate_v6_3_records(records, schedule, require_complete=True)

    assert len(calls) == 6
    assert result["complete"] is True
    assert result["tiebreak_status"] == "played_still_tied"
    assert len(records) == 6
    assert checked["standings"]["C"]["points"] == 3.0
    assert checked["standings"]["E5"]["points"] == 3.0
    assert [spec["game_id"] for spec in schedule[-2:]] == calls[-2:]


def test_resume_keeps_completed_games_and_never_counts_an_interrupted_game(tmp_path: Path) -> None:
    config = load_v6_3_config()
    attempted = []

    def interrupted(spec, engine_config, v6_config, commit):
        attempted.append(spec["game_id"])
        if len(attempted) == 2:
            raise RuntimeError("simulated interruption")
        return _fake_game(spec, engine_config, v6_config, commit, lambda _spec: "C")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_championship(config, tmp_path, game_runner=interrupted, implementation_commit=_repo_commit())
    assert len(load_game_records(tmp_path / "9x9_games.jsonl")) == 1

    def finish(spec, engine_config, v6_config, commit):
        return _fake_game(spec, engine_config, v6_config, commit, lambda _spec: "C")

    resumed = run_championship(config, tmp_path, game_runner=finish, implementation_commit=_repo_commit())
    records = load_game_records(tmp_path / "9x9_games.jsonl")
    assert resumed["complete"] is True
    assert len(records) == 4
    assert len({row["game_id"] for row in records}) == 4
    assert attempted == ["v6.3-9x9-c-e5-o731301-s0", "v6.3-9x9-c-e5-o731301-s1"]


def test_completed_resume_is_idempotent(tmp_path: Path) -> None:
    config = load_v6_3_config()

    def game(spec, engine_config, v6_config, commit):
        return _fake_game(spec, engine_config, v6_config, commit, lambda _spec: "C")

    first = run_championship(config, tmp_path, game_runner=game, implementation_commit=_repo_commit())
    second = run_championship(config, tmp_path, game_runner=game, implementation_commit=_repo_commit())
    assert first["games_completed_this_run"] == 4
    assert second["games_completed_this_run"] == 0
    assert len(load_game_records(tmp_path / "9x9_games.jsonl")) == 4


def test_single_game_checkpoint_can_be_snapshotted_and_resumed(tmp_path: Path) -> None:
    config = load_v6_3_config()
    calls = []

    def game(spec, engine_config, v6_config, commit):
        calls.append(spec["game_id"])
        return _fake_game(spec, engine_config, v6_config, commit, lambda _spec: "C")

    first = run_championship(
        config, tmp_path, game_runner=game, implementation_commit=_repo_commit(), max_games_this_run=1
    )
    assert first["complete"] is False
    assert len(load_game_records(tmp_path / "9x9_games.jsonl")) == 1
    second = run_championship(
        config, tmp_path, game_runner=game, implementation_commit=_repo_commit(), max_games_this_run=1
    )
    assert second["complete"] is False
    assert len(load_game_records(tmp_path / "9x9_games.jsonl")) == 2
    assert calls == ["v6.3-9x9-c-e5-o731301-s0", "v6.3-9x9-c-e5-o731301-s1"]


def test_over_budget_decision_must_be_flagged(tmp_path: Path) -> None:
    config = load_v6_3_config()
    spec = build_schedule(config)[0]
    record = _fake_game(spec, {}, {}, _repo_commit(), lambda _spec: "C")
    decision = next(move for move in record["moves"] if not move["opening"])
    decision["elapsed_seconds"] = 200.01
    decision["timed_out"] = False
    record["thinking_seconds_total"] = 200.01
    with pytest.raises(ValueError, match="over-budget"):
        validate_v6_3_game_record(record, spec)

