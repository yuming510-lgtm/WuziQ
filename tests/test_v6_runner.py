from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gomoku import GomokuBoard
from src.research.v6.config import load_config
from src.research.v6.runner import run_experiment
from src.research.v6.snapshot import create_v6_snapshot, restore_v6_snapshot
from src.research.v6.validation import validate_records


def _fake_game(spec, _engine_config, _v6_config, implementation_commit):
    board = GomokuBoard(9)
    moves = []
    current = "B"
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
    while not board.is_full():
        model = spec["black_model"] if current == "B" else spec["white_model"]
        coord = next(
            (row, col)
            for row in range(9)
            for col in range(9)
            if board.get((row, col)) is None
        )
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
        "timeout_count": 0,
        "search_nodes_total": 0,
        "simulations_total": 0,
        "tree_depth_max": 0,
        "search_depth_max": 0,
        "cutoffs_total": 0,
        "thinking_seconds_total": 0.0,
        "proof_count": 0,
        "proof_verified_count": 0,
        "source_type": "new",
        "source_game_id": None,
    }


def test_v6_runner_persists_paired_games_and_resumes_without_duplicate_ids(tmp_path: Path) -> None:
    config = load_config()
    first = run_experiment(
        config, tmp_path, reuse_v5_1=False, max_groups=1,
        implementation_commit="test-commit", game_runner=_fake_game,
    )
    assert first["games_completed_this_run"] == 2
    assert first["completed_groups"] == 1
    rows = __import__("src.research.checkpoint", fromlist=["load_game_records"]).load_game_records(
        tmp_path / "9x9_games.jsonl"
    )
    assert len(rows) == 2

    second = run_experiment(
        config, tmp_path, reuse_v5_1=False, max_groups=1,
        implementation_commit="test-commit", game_runner=_fake_game,
    )
    rows = __import__("src.research.checkpoint", fromlist=["load_game_records"]).load_game_records(
        tmp_path / "9x9_games.jsonl"
    )
    assert second["games_completed_this_run"] == 2
    assert len(rows) == 4
    assert len({row["game_id"] for row in rows}) == 4
    schedule = __import__("json").loads((tmp_path / "schedule.json").read_text())
    assert validate_records(rows, schedule, require_complete_groups=True, verify_proofs=False)["games"] == 4


def test_v6_snapshot_roundtrips_to_an_empty_directory_with_hashes(tmp_path: Path) -> None:
    output = tmp_path / "result"
    run_experiment(
        load_config(), output, reuse_v5_1=False, max_groups=1,
        implementation_commit="test-commit", game_runner=_fake_game,
    )
    archive = tmp_path / "v6-snapshot.zip"
    created = create_v6_snapshot(output, archive)
    restored = tmp_path / "empty-restore"

    result = restore_v6_snapshot(archive, restored)

    assert created["size_bytes"] > 0
    assert result["empty_restore"] is True
    assert result["game_count"] == 2
    assert (restored / "9x9_games.jsonl").read_bytes() == (output / "9x9_games.jsonl").read_bytes()
