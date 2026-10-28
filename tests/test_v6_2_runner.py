from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gomoku import GomokuBoard
from src.research.checkpoint import load_game_records
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.runner import run_championship
from src.research.v6_2.schedule import build_schedule
from src.research.v6_2.validation import validate_v6_2_records


def _fake_game(spec, _engine_config, _v6_config, implementation_commit):
    board = GomokuBoard(9)
    moves = []
    for index, opening in enumerate(spec["opening"]):
        coord = int(opening["row"]), int(opening["col"])
        board.place_stone(opening["player"], coord)
        moves.append({
            "ply": index, "player": opening["player"], "model": None,
            "row": coord[0], "col": coord[1], "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 0,
            "tree_depth": 0, "cutoffs": 0, "simulations": 0,
            "rollout_plies": 0, "terminal_simulations": 0,
            "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False,
            "opening": True, "proof_status": "not_applicable", "proof_kind": None,
            "proof_certificate": None, "stats": {},
        })
    winner_color = winner_model = None
    termination = "full_board"
    current = "B"
    count = {"B": 0, "W": 0}
    while not board.is_full():
        model = spec["black_model"] if current == "B" else spec["white_model"]
        coord = (0, count[current]) if current == "B" else (8, count[current])
        count[current] += 1
        board.place_stone(current, coord)
        moves.append({
            "ply": len(moves), "player": current, "model": model,
            "row": coord[0], "col": coord[1], "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 0,
            "tree_depth": 0, "cutoffs": 0, "simulations": 0,
            "rollout_plies": 0, "terminal_simulations": 0,
            "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False,
            "opening": False, "proof_status": "not_applicable", "proof_kind": None,
            "proof_certificate": None, "stats": {},
        })
        if board.check_victory(coord, current):
            winner_color, winner_model, termination = current, model, "five_in_row"
            break
        current = "W" if current == "B" else "B"
    return {
        **spec,
        "implementation_commit": implementation_commit,
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "draw" if winner_color is None else f"{winner_color.lower()}_win",
        "termination": termination,
        "timeout_count": sum(bool(move["timed_out"]) for move in moves),
        "search_nodes_total": sum(move["nodes"] for move in moves),
        "simulations_total": sum(move["simulations"] for move in moves),
        "tree_depth_max": max((move["tree_depth"] for move in moves), default=0),
        "search_depth_max": max((move["completed_depth"] for move in moves), default=0),
        "cutoffs_total": sum(move["cutoffs"] for move in moves),
        "thinking_seconds_total": sum(move["elapsed_seconds"] for move in moves),
        "proof_count": 0,
        "proof_verified_count": 0,
        "source_type": "new",
        "source_game_id": None,
    }


def test_v6_2_runner_persists_six_games_and_rerun_does_not_duplicate(tmp_path: Path) -> None:
    config = load_v6_2_config()
    first = run_championship(
        config, tmp_path, game_runner=_fake_game, implementation_commit="test-commit"
    )
    records = load_game_records(tmp_path / "9x9_games.jsonl")
    schedule = json.loads((tmp_path / "schedule.json").read_text())
    replay = validate_v6_2_records(records, schedule, require_complete=True)

    assert first["games_completed_this_run"] == 6
    assert first["complete"] is True
    assert replay["games"] == 6
    assert len({row["game_id"] for row in records}) == 6
    assert all(row["termination"] == "five_in_row" for row in records)

    second = run_championship(
        config, tmp_path, game_runner=_fake_game, implementation_commit="test-commit"
    )
    assert second["games_completed_this_run"] == 0
    assert len(load_game_records(tmp_path / "9x9_games.jsonl")) == 6


def test_interrupted_game_is_not_counted_and_resume_starts_it_again(tmp_path: Path) -> None:
    config = load_v6_2_config()
    attempted: list[str] = []

    def interrupted(spec, *args):
        attempted.append(spec["game_id"])
        raise RuntimeError("simulated process interruption before game completion")

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        run_championship(
            config, tmp_path, game_runner=interrupted, implementation_commit="test-commit"
        )
    assert not load_game_records(tmp_path / "9x9_games.jsonl")

    resumed = run_championship(
        config, tmp_path, game_runner=_fake_game, implementation_commit="test-commit"
    )
    records = load_game_records(tmp_path / "9x9_games.jsonl")
    assert attempted == ["v6.2-9x9-c-d-o731201-s0"]
    assert resumed["games_completed_this_run"] == 6
    assert len(records) == 6
    assert records[0]["game_id"] == attempted[0]


def test_initialize_only_freezes_resume_state_without_starting_a_game(tmp_path: Path) -> None:
    config = load_v6_2_config()

    result = run_championship(
        config, tmp_path,
        game_runner=lambda *args: pytest.fail("initialization must not run a game"),
        implementation_commit="test-commit",
        initialize_only=True,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert result["complete"] is False
    assert result["games"] == 0
    assert progress["completed_games"] == 0
    assert progress["current_game_id"] is None
    assert not (tmp_path / "9x9_games.jsonl").exists()


def test_validation_recomputes_round_robin_points_and_joint_champion() -> None:
    config = load_v6_2_config()
    schedule = build_schedule(config)
    records = [_fake_game(spec, {}, {}, "test-commit") for spec in schedule]

    result = validate_v6_2_records(records, schedule, require_complete=True)

    assert result["games"] == 6
    assert result["groups"] == 3
    assert result["standings"]["C"]["wins"] == 2
    assert result["standings"]["C"]["losses"] == 2
    assert result["standings"]["C"]["points"] == 2.0
    assert result["champions"] == ["C", "D", "E5"]
    for model, row in result["standings"].items():
        assert row["points"] == row["wins"] + row["draws"] / 2


def test_validation_rejects_a_game_using_wrong_budget_or_duplicate_id() -> None:
    config = load_v6_2_config()
    schedule = build_schedule(config)
    records = [_fake_game(spec, {}, {}, "test-commit") for spec in schedule]
    records[0]["moves"][4]["elapsed_seconds"] = 201.0
    records[0]["moves"][4]["timed_out"] = False
    records[0]["thinking_seconds_total"] = 201.0

    with pytest.raises(ValueError, match="budget|deadline|timeout"):
        validate_v6_2_records(records, schedule, require_complete=False)

    records[0] = _fake_game(schedule[0], {}, {}, "test-commit")
    records[1]["game_id"] = records[0]["game_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_v6_2_records(records, schedule, require_complete=False)
