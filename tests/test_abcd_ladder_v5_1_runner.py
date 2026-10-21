from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gomoku import GomokuBoard
from src.research.abcd_ladder_v5_1 import load_config
from src.research.abcd_ladder_v5_1_runner import run_experiment


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "abcd_ladder_v5_1.json"


def _complete_legal_record(spec: dict, config: dict) -> dict:
    """Small deterministic black win used to exercise runner durability."""
    board = GomokuBoard(9)
    moves = []
    for index, stone in enumerate(spec["opening"]):
        board.place_stone(stone["player"], (stone["row"], stone["col"]))
        moves.append({
            "ply": index, "player": stone["player"], "model": None,
            "row": stone["row"], "col": stone["col"], "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 0, "tree_depth": 0,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
            "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": True,
        })
    occupied_black = {coord for player, coord in board.history if player == "B"}
    occupied_white = {coord for player, coord in board.history if player == "W"}
    target = None
    for row in range(9):
        for start in range(5):
            line = {(row, col) for col in range(start, start + 5)}
            if not (line & occupied_white) and line - occupied_black:
                target = line
                break
        if target is not None:
            break
    assert target is not None
    current = "B"
    while True:
        if current == "B":
            coord = next((cell for cell in sorted(target) if board.get(cell) is None), None)
            assert coord is not None
        else:
            coord = None
            for row in range(9):
                for col in range(9):
                    candidate = (row, col)
                    if candidate in target or board.get(candidate) is not None:
                        continue
                    board.place_stone("W", candidate)
                    if not board.check_victory(candidate, "W"):
                        coord = candidate
                        break
                    board.history.pop()
                    board.grid[row][col] = None
                if coord is not None:
                    break
            assert coord is not None
        if current == "B":
            board.place_stone("B", coord)
        row, col = coord
        moves.append({
            "ply": len(moves), "player": current,
            "model": spec["black_model"] if current == "B" else spec["white_model"],
            "row": row, "col": col, "elapsed_seconds": 0.0,
            "root_value": 0, "nodes": 0, "completed_depth": 1, "tree_depth": 1,
            "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
            "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 1,
            "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": False,
        })
        if board.check_victory(coord, current):
            winner_color, winner_model = current, spec["black_model"] if current == "B" else spec["white_model"]
            break
        current = "W" if current == "B" else "B"
    node_total = sum(move["nodes"] for move in moves)
    simulation_total = sum(move["simulations"] for move in moves)
    elapsed_total = sum(move["elapsed_seconds"] for move in moves)
    return {
        **spec, "implementation_commit": "unit-test", "moves": moves, "move_count": len(moves),
        "winner_color": winner_color, "winner_model": winner_model,
        "outcome": f"{winner_color.lower()}_win", "termination": "five_in_row", "timeout_count": 0,
        "search_nodes_total": node_total, "simulations_total": simulation_total,
        "tree_depth_max": max(move["tree_depth"] for move in moves), "thinking_seconds_total": elapsed_total,
    }


def test_one_game_partial_pair_resumes_without_advancing_route(tmp_path):
    config = load_config(CONFIG)
    first = run_experiment(config, tmp_path, reuse_v5=False, max_games=1, implementation_commit="test",
                           game_runner=lambda spec, _engine, _commit: _complete_legal_record(spec, config))
    assert first["completed_games"] == 1
    assert first["completed_groups"] == 0
    assert first["pending_group_id"] is not None
    assert first["routes"]["A>B"]["levels"][0]["status"] == "incomplete_pair"

    second = run_experiment(config, tmp_path, reuse_v5=False, max_games=1, implementation_commit="test",
                            game_runner=lambda spec, _engine, _commit: _complete_legal_record(spec, config))
    assert second["completed_games"] == 2
    assert second["completed_groups"] == 1
    assert second["pending_group_id"] != first["pending_group_id"]
    assert len({json.loads(line)["game_id"] for line in (tmp_path / "9x9_games.jsonl").read_text().splitlines()}) == 2


def test_duplicate_game_ids_in_durable_jsonl_are_rejected(tmp_path):
    config = load_config(CONFIG)
    runner = lambda spec, _engine, _commit: _complete_legal_record(spec, config)
    run_experiment(config, tmp_path, reuse_v5=False, max_games=1, implementation_commit="test", game_runner=runner)
    path = tmp_path / "9x9_games.jsonl"
    path.write_bytes(path.read_bytes() + path.read_bytes())
    with pytest.raises(ValueError, match="Duplicate game_id"):
        run_experiment(config, tmp_path, reuse_v5=False, implementation_commit="test", game_runner=runner)


def test_skipped_groups_are_recorded_without_inventing_results(tmp_path):
    config = load_config(CONFIG)
    runner = lambda spec, _engine, _commit: _complete_legal_record(spec, config)
    progress = run_experiment(config, tmp_path, reuse_v5=False, max_groups=3,
                              implementation_commit="test", game_runner=runner)
    assert progress["completed_groups"] == 3
    assert sum(group["status"] == "played" for group in progress["groups"].values()) == 3
    assert all(group["status"] in {"played", "pending", "not_reached", "skipped_no_active_route"}
               for group in progress["groups"].values())
    assert len(json.loads((tmp_path / "progress.json").read_text())["completed_ids"]) == 6


def test_resume_rejects_changed_schedule_or_implementation_commit(tmp_path):
    config = load_config(CONFIG)
    runner = lambda spec, _engine, _commit: _complete_legal_record(spec, config)
    run_experiment(config, tmp_path, reuse_v5=False, max_games=1, implementation_commit="test", game_runner=runner)
    with pytest.raises(ValueError, match="implementation commit"):
        run_experiment(config, tmp_path, reuse_v5=False, max_games=1, implementation_commit="other", game_runner=runner)


def test_exact_v5_group_is_materialized_only_as_the_active_v51_group(tmp_path):
    config = load_config(CONFIG)
    progress = run_experiment(config, tmp_path, reuse_v5=True, max_games=2,
                              implementation_commit="reuse-test")
    rows = [json.loads(line) for line in (tmp_path / "9x9_games.jsonl").read_text().splitlines()]

    assert progress["completed_games"] == 2
    assert progress["completed_groups"] == 1
    assert {row["source_type"] for row in rows} == {"reused_v5"}
    assert all(row["source_game_id"] and row["source_game_seed"] == row["game_seed"] for row in rows)
    assert len({row["source_game_id"] for row in rows}) == 2
    assert all(row["implementation_commit"] != "reuse-test" for row in rows)
