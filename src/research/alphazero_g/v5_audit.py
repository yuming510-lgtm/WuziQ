"""Independent replay and artifact-integrity checks for G V5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from third_party.AlphaZero_Gomoku.game import Board
from .v5_evaluation import _IndependentRuleBoard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _board(size: int, n_in_row: int) -> Board:
    board = Board(width=size, height=size, n_in_row=n_in_row)
    board.init_board(start_player=0)
    return board


def _play_move(board: Board, oracle: _IndependentRuleBoard, move: dict[str, Any]) -> tuple[str, bool]:
    color = str(move["player"])
    expected = "B" if board.current_player == 1 else "W"
    if color != expected:
        raise ValueError("move color does not match the side to move")
    row, col = int(move["row"]), int(move["col"])
    action = row * int(board.width) + col
    if action not in board.availables:
        raise ValueError("replay contains an illegal or repeated move")
    board.do_move(action)
    oracle.place_stone(color, (row, col))
    oracle_win = oracle.check_victory((row, col), color)
    engine_end, engine_winner = board.game_end()
    if bool(engine_end) != bool(oracle_win or oracle.is_full()):
        raise ValueError("reference game engine and independent rule oracle disagree")
    if oracle_win:
        if engine_winner != (1 if color == "B" else 2):
            raise ValueError("reference engine winner differs from independent rule oracle")
        return color, True
    if oracle.is_full():
        if engine_winner != -1:
            raise ValueError("full-board result differs from draw")
        return "draw", True
    return "", False


def audit_selfplay_game(game_path: Path) -> dict[str, Any]:
    record = json.loads(game_path.read_text(encoding="utf-8"))
    sample_path = game_path.with_suffix(".npz")
    if not sample_path.is_file():
        raise ValueError(f"selfplay sample archive is missing: {sample_path}")
    size, n_in_row = int(record["board_size"]), int(record["n_in_row"])
    board, oracle = _board(size, n_in_row), _IndependentRuleBoard(size, n_in_row)
    for move in record.get("opening", []):
        winner, ended = _play_move(board, oracle, move)
        if ended:
            raise ValueError("selfplay opening is terminal")
    with np.load(sample_path, allow_pickle=False) as archive:
        states = archive["states"].astype(np.float32)
        policies = archive["policies"].astype(np.float32)
        values = archive["values"].astype(np.float32)
    played = [move for move in record.get("moves", []) if not move.get("opening")]
    if len(played) != len(states) or len(states) != len(policies) or len(states) != len(values):
        raise ValueError("selfplay game and training arrays have different state counts")
    if states.shape != (len(played), 4, size, size) or policies.shape != (len(played), size * size):
        raise ValueError("selfplay arrays have unexpected shapes")
    if not np.isfinite(states).all() or not np.isfinite(policies).all() or not np.isfinite(values).all():
        raise ValueError("selfplay arrays contain non-finite values")
    if np.any(policies < -1e-6) or not np.allclose(policies.sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("visit target is negative or is not normalized")
    terminal = ""
    for index, move in enumerate(played):
        state = board.current_state().astype(np.float32)
        if not np.allclose(states[index], state, atol=1e-3):
            raise ValueError(f"training position does not match replay at sample {index}")
        legal = np.zeros(size * size, dtype=bool)
        legal[np.asarray(board.availables, dtype=np.int64)] = True
        if np.any(policies[index, ~legal] > 2e-3):
            raise ValueError(f"visit target assigns mass to an illegal move at sample {index}")
        if terminal:
            raise ValueError("selfplay record continues after a terminal position")
        winner, ended = _play_move(board, oracle, move)
        if ended:
            terminal = winner
    if not terminal:
        raise ValueError("selfplay record has no terminal move")
    if terminal == "draw":
        expected_values = np.zeros(len(played), dtype=np.float32)
        expected_outcome = "draw"
    else:
        winner_num = 1 if terminal == "B" else 2
        # Training positions are stored in side-to-move perspective.
        players = []
        replay_board = _board(size, n_in_row)
        for move in record.get("opening", []):
            replay_board.do_move(int(move["row"]) * size + int(move["col"]))
        for move in played:
            players.append(int(replay_board.current_player))
            replay_board.do_move(int(move["row"]) * size + int(move["col"]))
        expected_values = np.where(np.asarray(players) == winner_num, 1.0, -1.0).astype(np.float32)
        expected_outcome = f"{terminal}_win"
    if not np.array_equal(values.astype(np.int8), expected_values.astype(np.int8)):
        raise ValueError("winner labels do not match the replayed terminal result")
    if record.get("outcome") != expected_outcome or record.get("winner_color") != (None if terminal == "draw" else terminal):
        raise ValueError("selfplay metadata does not match replayed terminal result")
    if record.get("termination") != ("full_board" if terminal == "draw" else "five_in_row"):
        raise ValueError("selfplay termination metadata is inconsistent")
    if int(record.get("move_count", -1)) != len(record.get("moves", [])):
        raise ValueError("selfplay move count metadata is inconsistent")
    return {"game_id": record["game_id"], "samples": len(played), "moves": len(record["moves"]),
            "winner": terminal, "sample_sha256": _sha(sample_path), "game_sha256": _sha(game_path)}


def audit_evaluation_game(game_path: Path) -> dict[str, Any]:
    record = json.loads(game_path.read_text(encoding="utf-8"))
    size = int(record.get("board_size", 9))
    n_in_row = int(record.get("n_in_row", 5))
    board, oracle = _board(size, n_in_row), _IndependentRuleBoard(size, n_in_row)
    opening = record.get("opening", [])
    moves = record.get("moves", [])
    if len(moves) < len(opening):
        raise ValueError("evaluation move list is shorter than its opening")
    if [{k: move[k] for k in ("player", "row", "col")} for move in moves[:len(opening)]] != opening:
        raise ValueError("evaluation opening metadata differs from its full move list")
    winner = ""
    for move in moves:
        if winner:
            raise ValueError("evaluation record continues after game end")
        winner, _ended = _play_move(board, oracle, move)
        if winner == "draw":
            break
    if not winner:
        raise ValueError("evaluation record is not a complete game")
    winner_color = None if winner == "draw" else winner
    candidate_color = str(record["candidate_color"])
    points = 0.5 if winner == "draw" else 1.0 if winner == candidate_color else 0.0
    if record.get("winner_color") != winner_color or float(record.get("points", -1)) != points:
        raise ValueError("evaluation score metadata differs from replay")
    if int(record.get("move_count", -1)) != len(moves):
        raise ValueError("evaluation move count metadata is inconsistent")
    return {"game_id": record["game_id"], "candidate": record["candidate"],
            "opponent": record["opponent"], "winner": winner_color,
            "points": points, "moves": len(record["moves"]), "game_sha256": _sha(game_path)}


def audit_v5(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    v5 = root / "research/g_v5"
    cfg = json.loads((v5 / "config/experiment.json").read_text(encoding="utf-8"))
    declared = cfg.get("config_sha256")
    canonical = dict(cfg)
    canonical.pop("config_sha256", None)
    actual = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    if declared != actual:
        raise ValueError("registered V5 configuration hash mismatch")
    source = json.loads((v5 / "reference/source_manifest.json").read_text(encoding="utf-8"))
    source_hashes = []
    for item in source["files"]:
        path = root / item["path"]
        if _sha(path) != item["sha256"]:
            raise ValueError(f"vendored reference source hash mismatch: {item['path']}")
        source_hashes.append(item["path"])
    for config_key, config_path in (
        ("opening_schedule_sha256", v5 / "config/opening_schedule.json"),
        ("six_by_six_tactics_sha256", v5 / "config/six_by_six_tactics.json"),
        ("historical_midgame_prefix_pool_sha256", v5 / "config/historical_midgame_prefixes.json"),
        ("source_manifest_sha256", v5 / "reference/source_manifest.json"),
    ):
        if config_key in cfg and _sha(config_path) != cfg[config_key]:
            raise ValueError(f"registered V5 artifact hash mismatch: {config_key}")
    for item in cfg.get("nine_by_nine", {}).get("historical_checkpoints", []):
        path = root / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise ValueError(f"frozen historical G0 checkpoint mismatch: {item['id']}")
    training = []
    seen_training: set[str] = set()
    orphan_samples = []
    for directory in sorted((v5 / "raw/selfplay").glob("*")):
        for sample in directory.glob("*.npz"):
            if not sample.with_suffix(".json").exists():
                orphan_samples.append(str(sample.relative_to(root)))
        for path in sorted(directory.glob("*.json")):
            row = audit_selfplay_game(path)
            if row["game_id"] in seen_training:
                raise ValueError(f"duplicate selfplay game ID: {row['game_id']}")
            seen_training.add(row["game_id"])
            training.append(row)
    if orphan_samples:
        raise ValueError(f"unpaired selfplay sample files remain: {orphan_samples[:5]}")
    evaluation = []
    seen_evaluation: set[str] = set()
    for path in sorted((v5 / "results/matches").rglob("v5m-*.json")):
        row = audit_evaluation_game(path)
        if row["game_id"] in seen_evaluation:
            raise ValueError(f"duplicate evaluation game ID: {row['game_id']}")
        seen_evaluation.add(row["game_id"])
        evaluation.append(row)
    return {
        "format_version": 1, "config_sha256": declared,
        "vendored_reference_files_verified": len(source_hashes),
        "registered_schedule_and_checkpoint_hashes_verified": True,
        "selfplay_games_verified": len(training),
        "selfplay_training_states_verified": sum(row["samples"] for row in training),
        "evaluation_games_verified": len(evaluation),
        "unique_selfplay_game_ids": len(seen_training),
        "unique_evaluation_game_ids": len(seen_evaluation),
        "all_complete_games_legal": True,
    }
