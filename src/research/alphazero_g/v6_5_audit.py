"""Independent replay and checkpoint accounting audit for V6.5 self-play."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import v6_5_training as training


def _has_n(board: np.ndarray, row: int, col: int, player: int, n: int = 5) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= n:
            return True
    return False


def audit_game_record(record: dict[str, Any], sample_path: str | Path | None = None) -> dict[str, Any]:
    size = int(record.get("board_size", 0))
    n_in_row = int(record.get("n_in_row", 0))
    if size != 9 or n_in_row != 5:
        raise ValueError("V6.5 self-play record must use the registered 9x9 connect-five rules")
    if record.get("source") != "self_play" or int(record.get("simulations_configured_per_move", -1)) != 400:
        raise ValueError("V6.5 record is not from the frozen 400-simulation self-play engine")
    if record.get("opening") not in ([], None):
        raise ValueError("V6.5 self-play must preserve the V6.3 empty-board opening distribution")
    moves = record.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("self-play game has no recorded moves")
    board = np.zeros((size, size), dtype=np.int8)
    expected = 1
    terminal = False
    winner_color = None
    for index, move in enumerate(moves):
        if terminal:
            raise ValueError(f"move {index} occurs after the game ended")
        player = 1 if move.get("player") == "B" else 2 if move.get("player") == "W" else 0
        if player != expected:
            raise ValueError(f"move {index} violates alternating black/white turn order")
        row, col = int(move.get("row", -1)), int(move.get("col", -1))
        if not (0 <= row < size and 0 <= col < size) or board[row, col] != 0:
            raise ValueError(f"move {index} is out of bounds or occupies a used point")
        if bool(move.get("opening", False)):
            raise ValueError("training move incorrectly marked as an opening prefix")
        if int(move.get("configured_playouts", -1)) != 400:
            raise ValueError(f"move {index} does not use the frozen 400-playout search")
        board[row, col] = player
        if _has_n(board, row, col, player, n_in_row):
            terminal = True
            winner_color = "B" if player == 1 else "W"
            if index != len(moves) - 1:
                raise ValueError("game contains moves after a five-in-row win")
        elif not np.any(board == 0):
            terminal = True
            if index != len(moves) - 1:
                raise ValueError("game contains moves after a full-board draw")
        expected = 3 - expected
    if not terminal:
        raise ValueError("recorded game ends before a legal terminal state")
    outcome = "draw" if winner_color is None else f"{winner_color}_win"
    termination = "full_board" if winner_color is None else "five_in_row"
    if record.get("winner_color") != winner_color or record.get("outcome") != outcome:
        raise ValueError("record winner/outcome does not match independent board replay")
    if record.get("termination") != termination:
        raise ValueError("record termination reason does not match independent board replay")
    if int(record.get("move_count", -1)) != len(moves):
        raise ValueError("move_count differs from the full move list")
    if int(record.get("training_state_count", -1)) != len(moves):
        raise ValueError("training state count differs from the number of recorded decisions")
    if sample_path is not None:
        sample_path = Path(sample_path)
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        with np.load(sample_path, allow_pickle=False) as samples:
            states, policies, values = samples["states"], samples["policies"], samples["values"]
            if states.shape[0] != len(moves) or policies.shape[0] != len(moves) or values.shape[0] != len(moves):
                raise ValueError("recorded NPZ sample count differs from move/state count")
            if tuple(states.shape[1:]) != (4, size, size) or tuple(policies.shape[1:]) != (size * size,):
                raise ValueError("recorded training arrays have unexpected state or policy dimensions")
            current_player = 1
            for value in values:
                current_color = "B" if current_player == 1 else "W"
                expected_value = 0 if winner_color is None else 1 if current_color == winner_color else -1
                if int(value) != expected_value:
                    raise ValueError("recorded value target has a winner-perspective mismatch")
                current_player = 3 - current_player
    return {
        "game_id": str(record["game_id"]), "move_count": len(moves),
        "winner_color": winner_color, "outcome": outcome,
        "root_visits_values": sorted({int(m.get("root_visits", -1)) for m in moves}),
    }


def audit_experiment(root: str | Path = training.ROOT, *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / training.ARTIFACT
    state_path = artifact / "checkpoints" / training.ARM / "training_state.pt"
    state_validation = training.validate_training_state(state_path, root=root)
    current_games = int(state_validation["game_count"])
    current_states = int(state_validation["environment_states"])
    current_updates = int(state_validation["optimizer_updates"])
    if current_games < training.BASE_GAMES or current_games > training.TARGET_GAMES:
        raise ValueError("checkpoint game count is outside the registered V6.5 range")
    directory = artifact / "raw/selfplay" / training.ARM
    expected_ids = {f"{training.ARM}-game-{index:06d}" for index in range(training.BASE_GAMES, current_games)}
    records: dict[str, dict[str, Any]] = {}
    pending_id = f"{training.ARM}-game-{current_games:06d}"
    pending_record: dict[str, Any] | None = None
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(record.get("game_id", ""))
        if game_id in records:
            raise ValueError(f"duplicate self-play game id: {game_id}")
        if path.stem != game_id:
            raise ValueError(f"self-play filename and game ID disagree: {path.name}")
        index = int(game_id.rsplit("-", 1)[-1])
        if index < training.BASE_GAMES or index > current_games:
            raise ValueError(f"self-play record outside checkpoint range: {game_id}")
        expected_seed = training.SEED + index * 10007
        if int(record.get("seed", -1)) != expected_seed:
            raise ValueError(f"deterministic seed mismatch for {game_id}")
        records[game_id] = record
        if index == current_games:
            pending_record = record
    expected_or_pending_ids = expected_ids | ({pending_id} if pending_record is not None else set())
    if set(records) != expected_or_pending_ids:
        missing = sorted(expected_ids - set(records))
        extras = sorted(set(records) - expected_or_pending_ids)
        raise ValueError(f"self-play IDs do not match the checkpoint boundary; missing={missing[:5]}, extras={extras[:5]}")
    if len(list(directory.glob("*.npz"))) != len(records):
        raise ValueError("self-play JSON and NPZ counts differ")
    replayed = []
    total_states = 0
    total_moves = 0
    colors = {"B": 0, "W": 0, "draw": 0}
    for game_id in sorted(expected_ids):
        record = records[game_id]
        result = audit_game_record(record, directory / f"{game_id}.npz")
        replayed.append(result)
        total_states += int(record["training_state_count"])
        total_moves += int(record["move_count"])
        colors[result["winner_color"] or "draw"] += 1
    pending_valid = False
    if pending_record is not None:
        audit_game_record(pending_record, directory / f"{pending_id}.npz")
        pending_valid = True
    baseline_path = artifact / "checkpoints" / training.ARM / "training_state_games_5000.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    expected_state_delta = current_states - int(baseline["environment_states"])
    if total_states != expected_state_delta:
        raise ValueError(f"training states do not reconcile: game logs={total_states}, checkpoint delta={expected_state_delta}")
    if current_updates != current_games - 1:
        raise ValueError("optimizer update count is inconsistent with the one-update-per-game V5 loop")
    history_path = artifact / "results/training/reference400_empty.jsonl"
    history_ids = set()
    if history_path.is_file():
        history_ids = {str(json.loads(line)["game_id"]) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if history_ids != expected_ids:
        raise ValueError("per-game optimizer history IDs do not match complete self-play records")
    complete = current_games == training.TARGET_GAMES
    if require_complete and not complete:
        raise ValueError(f"V6.5 target is incomplete at G{current_games}; target G{training.TARGET_GAMES}")
    result = {
        "experiment_id": "wuziq-alphazero-g-v6-5-selfplay-scale",
        "source_baseline_games": training.BASE_GAMES,
        "target_games": training.TARGET_GAMES,
        "games_completed": current_games,
        "new_games_replayed": len(replayed),
        "unique_game_ids": len(expected_ids),
        "legal_terminal_games": len(replayed),
        "pending_recovery_game_id": pending_id if pending_record is not None else None,
        "pending_recovery_game_valid": pending_valid,
        "new_training_states": total_states,
        "checkpoint_training_state_delta": expected_state_delta,
        "cumulative_training_states": current_states,
        "optimizer_updates": current_updates,
        "winner_distribution": colors,
        "mean_selfplay_moves": (total_moves / len(records)) if records else 0.0,
        "checkpoint": state_validation,
        "complete": complete,
        "valid": True,
    }
    from .v6_5_training import atomic_json
    atomic_json(artifact / "results/integrity_audit.json", result)
    return result
