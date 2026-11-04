"""Durable self-play loop following the independent reference pipeline."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from third_party.AlphaZero_Gomoku.game import Board
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def augment_reference_data(play_data: list[tuple[np.ndarray, np.ndarray, float]]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Apply the same eight rotations/reflections as upstream TrainPipeline."""
    if not play_data:
        return []
    height, width = play_data[0][0].shape[1:]
    if height != width:
        raise ValueError("the upstream augmentation is registered for square boards")
    extended = []
    for state, policy, value in play_data:
        matrix = policy.reshape(height, width)
        for turns in (1, 2, 3, 4):
            state_rotated = np.rot90(state, turns, axes=(-2, -1))
            policy_rotated = np.rot90(np.flipud(matrix), turns)
            extended.append((state_rotated.copy(), np.flipud(policy_rotated).flatten().copy(), float(value)))
            state_flipped = np.flip(state_rotated, axis=-1)
            policy_flipped = np.fliplr(policy_rotated)
            extended.append((state_flipped.copy(), np.flipud(policy_flipped).flatten().copy(), float(value)))
    return extended


def _opening_identity(opening: list[dict[str, Any]]) -> str:
    raw = json.dumps(opening, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _run_game(
    *,
    size: int,
    n_in_row: int,
    simulations: int,
    model: ReferencePolicyValueNet,
    game_id: str,
    seed: int,
    opening: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    opening = list(opening or [])
    np.random.seed(int(seed) & 0xFFFFFFFF)
    board = make_board(size, n_in_row, 0)
    player = instrumented_player(model.policy_value_fn, c_puct=5.0,
                                 n_playout=simulations, is_selfplay=True)
    player.reset_player()
    expected_player = 1
    for move in opening:
        expected = "B" if expected_player == 1 else "W"
        if move["player"] != expected:
            raise ValueError("training opening must alternate from black")
        action = int(move["row"]) * size + int(move["col"])
        if action not in board.availables:
            raise ValueError("training opening is illegal")
        board.do_move(action)
        if board.game_end()[0]:
            raise ValueError("training opening is terminal")
        expected_player = 3 - expected_player

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    current_players: list[int] = []
    moves = [{"player": str(move["player"]), "row": int(move["row"]), "col": int(move["col"]),
              "opening": True} for move in opening]
    move_metrics: list[dict[str, Any]] = []
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    while True:
        action, action_probs = player.get_action(board, temp=1.0, return_prob=1)
        if int(action) not in board.availables:
            raise AssertionError("upstream MCTS returned an illegal action")
        states.append(board.current_state().astype(np.float32, copy=True))
        policies.append(np.asarray(action_probs, dtype=np.float32).copy())
        current_players.append(int(board.current_player))
        color = "B" if int(board.current_player) == 1 else "W"
        h, w = divmod(int(action), size)
        metrics = dict(player.last_search_stats)
        moves.append({"player": color, "row": h, "col": w, "opening": False,
                      "move_index": len(moves), **metrics})
        move_metrics.append(metrics)
        board.do_move(int(action))
        end, winner = board.game_end()
        if end:
            values = np.zeros(len(current_players), dtype=np.float32)
            if winner != -1:
                players = np.asarray(current_players)
                values[players == int(winner)] = 1.0
                values[players != int(winner)] = -1.0
            player.reset_player()
            winner_color = None if winner == -1 else ("B" if winner == 1 else "W")
            record = {
                "format_version": 1,
                "source": "self_play",
                "game_id": game_id,
                "board_size": int(size),
                "n_in_row": int(n_in_row),
                "simulations_configured_per_move": int(simulations),
                "first_decision_root_action_visits": int(move_metrics[0].get("root_visits", 0)) if move_metrics else 0,
                "seed": int(seed),
                "opening_id": _opening_identity(opening),
                "opening": opening,
                "moves": moves,
                "move_count": len(moves),
                "training_state_count": len(states),
                "winner_color": winner_color,
                "outcome": "draw" if winner == -1 else f"{winner_color}_win",
                "termination": "five_in_row" if winner != -1 else "full_board",
                "runner_cpu_seconds": time.process_time() - started_cpu,
                "runner_wall_seconds": time.perf_counter() - started_wall,
                "mean_search_cpu_seconds": float(np.mean([m.get("search_cpu_seconds", 0.0) for m in move_metrics])) if move_metrics else 0.0,
                "mean_tree_depth": float(np.mean([m.get("tree_depth", 0) for m in move_metrics])) if move_metrics else 0.0,
            }
            return record, np.stack(states), np.stack(policies), values


def _artifact_root(root: Path, artifact_dir: str | Path | None) -> Path:
    """Resolve an optional output namespace; the default remains the V5 layout."""
    if artifact_dir is None:
        return root / "research/g_v5"
    path = Path(artifact_dir)
    return path if path.is_absolute() else root / path


def _save_game(root: Path, arm: str, record: dict[str, Any], states: np.ndarray,
               policies: np.ndarray, values: np.ndarray,
               artifact_dir: str | Path | None = None) -> tuple[Path, Path]:
    base = _artifact_root(root, artifact_dir) / "raw/selfplay" / arm
    game_path = base / f"{record['game_id']}.json"
    sample_path = base / f"{record['game_id']}.npz"
    _atomic_npz(sample_path, states=states.astype(np.float16), policies=policies.astype(np.float16), values=values.astype(np.int8))
    _atomic_json(game_path, record)
    return game_path, sample_path


def _load_game_samples(game_path: Path) -> list[tuple[np.ndarray, np.ndarray, float]]:
    sample_path = game_path.with_suffix(".npz")
    with np.load(sample_path, allow_pickle=False) as data:
        states = data["states"].astype(np.float32)
        policies = data["policies"].astype(np.float32)
        values = data["values"].astype(np.float32)
    return [(states[i], policies[i], float(values[i])) for i in range(len(values))]


def _checkpoint_payload(
    model: ReferencePolicyValueNet,
    *,
    arm: str,
    config_hash: str,
    game_count: int,
    state_count: int,
    update_count: int,
    lr_multiplier: float,
    replay: deque,
    base_seed: int,
    cumulative_cpu_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "arm_id": arm,
        "config_sha256": config_hash,
        "board_size": model.board_size,
        "model_state": {key: value.detach().cpu() for key, value in model.get_policy_param().items()},
        "optimizer_state": model.optimizer.state_dict(),
        "game_count": int(game_count),
        "environment_states": int(state_count),
        "update_count": int(update_count),
        "lr_multiplier": float(lr_multiplier),
        "replay_buffer": list(replay),
        "base_seed": int(base_seed),
        "cumulative_cpu_seconds": float(cumulative_cpu_seconds),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }


def load_training_checkpoint(
    path: Path,
    *,
    model: ReferencePolicyValueNet,
    arm: str,
    config_hash: str,
    buffer_size: int,
) -> tuple[int, int, int, float, deque, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint.get("arm_id") != arm:
        raise ValueError("unsupported or mismatched V5 training checkpoint")
    if checkpoint.get("config_sha256") != config_hash:
        raise ValueError("V5 checkpoint config hash mismatch")
    if int(checkpoint.get("board_size", 0)) != model.board_size:
        raise ValueError("V5 checkpoint board size mismatch")
    model.load_policy_param(checkpoint["model_state"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    replay = deque(checkpoint["replay_buffer"], maxlen=int(buffer_size))
    return (
        int(checkpoint["game_count"]), int(checkpoint["environment_states"]),
        int(checkpoint["update_count"]), float(checkpoint["lr_multiplier"]), replay,
        int(checkpoint["base_seed"]), float(checkpoint.get("cumulative_cpu_seconds", 0.0)),
    )


def _policy_update(
    model: ReferencePolicyValueNet,
    replay: deque,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    kl_target: float,
    lr_multiplier: float,
) -> tuple[float, float, float]:
    batch = random.sample(replay, int(batch_size))
    states = np.asarray([row[0] for row in batch], dtype=np.float32)
    probs = np.asarray([row[1] for row in batch], dtype=np.float32)
    winners = np.asarray([row[2] for row in batch], dtype=np.float32)
    old_probs, old_values = model.policy_value(states)
    kl = 0.0
    loss = entropy = 0.0
    for _ in range(int(epochs)):
        loss, entropy = model.train_step(states, probs, winners, learning_rate * lr_multiplier)
        new_probs, _ = model.policy_value(states)
        kl = float(np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)))
        if kl > kl_target * 4:
            break
    if kl > kl_target * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_target / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5
    return float(loss), float(entropy), float(lr_multiplier)


def train_reference_arm(
    root: str | Path,
    *,
    arm: str,
    board_size: int,
    n_in_row: int,
    simulations: int,
    target_states: int,
    milestones: list[int],
    seed: int,
    config: dict[str, Any],
    opening_provider=None,
    initial_model_state: dict[str, torch.Tensor] | None = None,
    stop_at_cpu_seconds: float | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one independent reference model, checkpointing after each game."""
    root = Path(root)
    config_hash = str(config["config_sha256"])
    artifact_root = _artifact_root(root, artifact_dir)
    training_progress_path = artifact_root / "results/progress.json"
    if artifact_dir is not None:
        # A versioned runner may maintain its own top-level task progress file.
        training_progress_path = artifact_root / "results/training_progress.json"
    arm_dir = artifact_root / "checkpoints" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    current_path = arm_dir / "training_state.pt"
    initial_path = arm_dir / "node_0.pt"
    session_cpu_started = time.process_time()
    model = ReferencePolicyValueNet(board_size, seed=seed)
    if initial_model_state is not None:
        model.load_policy_param(initial_model_state)
    buffer_size = int(config["training"]["buffer_size"])
    if current_path.exists():
        game_count, state_count, update_count, lr_multiplier, replay, saved_seed, cumulative_cpu_seconds = load_training_checkpoint(
            current_path, model=model, arm=arm, config_hash=config_hash, buffer_size=buffer_size
        )
        if saved_seed != int(seed):
            raise ValueError("V5 resume seed differs from the registered seed")
    else:
        random.seed(int(seed))
        np.random.seed(int(seed) & 0xFFFFFFFF)
        torch.manual_seed(int(seed))
        game_count = state_count = update_count = 0
        cumulative_cpu_seconds = 0.0
        lr_multiplier = 1.0
        replay = deque(maxlen=buffer_size)
        if initial_path.exists() and initial_model_state is None:
            saved_initial = torch.load(initial_path, map_location="cpu", weights_only=False)
            if saved_initial.get("config_sha256") != config_hash:
                raise ValueError("V5 initial checkpoint config hash mismatch")
            model.load_policy_param(saved_initial["model_state"])
        elif not initial_path.exists():
            _atomic_torch_save(initial_path, {
                "format_version": 1, "arm_id": arm, "board_size": board_size,
                "n_in_row": n_in_row, "simulations_per_move": simulations,
                "environment_states": 0, "games_completed": 0,
                "model_state": {k: v.detach().cpu() for k, v in model.get_policy_param().items()},
                "config_sha256": config_hash,
            })
        if initial_model_state is not None:
            # A transferred state dict is weights-only; optimizer starts fresh.
            model.optimizer = torch.optim.Adam(model.network.parameters(), weight_decay=1e-4)
    base_cpu_seconds = cumulative_cpu_seconds
    history_path = artifact_root / "results/training" / f"{arm}.jsonl"
    stage_path = artifact_root / "results/training" / f"{arm}_metrics.jsonl"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    seen_milestones: set[int] = set()
    seen_games: set[str] = set()
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line:
                seen_games.add(str(json.loads(line)["game_id"]))
    if stage_path.exists():
        for line in stage_path.read_text(encoding="utf-8").splitlines():
            if line:
                seen_milestones.add(int(json.loads(line)["milestone_states"]))
    loss = entropy = 0.0
    while state_count < int(target_states):
        if stop_at_cpu_seconds is not None and time.process_time() >= float(stop_at_cpu_seconds):
            break
        game_id = f"{arm}-game-{game_count:06d}"
        game_path = artifact_root / "raw/selfplay" / arm / f"{game_id}.json"
        if game_path.exists() and game_path.with_suffix(".npz").exists():
            record = json.loads(game_path.read_text(encoding="utf-8"))
            play_data = _load_game_samples(game_path)
        else:
            game_seed = int(seed + game_count * 10007)
            opening = opening_provider(game_count, game_seed) if opening_provider else []
            record, states, probs, values = _run_game(
                size=board_size, n_in_row=n_in_row, simulations=simulations, model=model,
                game_id=game_id, seed=game_seed, opening=opening,
            )
            _save_game(root, arm, record, states, probs, values, artifact_dir=artifact_root)
            play_data = [(states[i], probs[i], float(values[i])) for i in range(len(values))]
        replay.extend(augment_reference_data(play_data))
        loss = entropy = 0.0
        if len(replay) > int(config["training"]["batch_size"]):
            loss, entropy, lr_multiplier = _policy_update(
                model, replay,
                batch_size=int(config["training"]["batch_size"]),
                epochs=int(config["training"]["epochs"]),
                learning_rate=float(config["training"]["learning_rate"]),
                kl_target=float(config["training"]["kl_target"]),
                lr_multiplier=lr_multiplier,
            )
            update_count += 1
        game_count += 1
        state_count += int(record["training_state_count"])
        if game_id not in seen_games:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"game_id": game_id, "loss": loss, "entropy": entropy,
                                         "lr_multiplier": lr_multiplier, "update_count": update_count},
                                        ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            seen_games.add(game_id)
        total_cpu_seconds = base_cpu_seconds + time.process_time() - session_cpu_started
        payload = _checkpoint_payload(
            model, arm=arm, config_hash=config_hash, game_count=game_count,
            state_count=state_count, update_count=update_count,
            lr_multiplier=lr_multiplier, replay=replay, base_seed=seed,
            cumulative_cpu_seconds=total_cpu_seconds,
        )
        _atomic_torch_save(current_path, payload)
        _atomic_json(training_progress_path, {
            "format_version": 1, "experiment_id": config["experiment_id"],
            "config_sha256": config_hash, "active_arm": arm,
            "games_completed": game_count, "environment_states": state_count,
            "optimizer_updates": update_count, "replay_rows": len(replay),
            "cumulative_cpu_seconds": total_cpu_seconds,
            "last_game_id": game_id, "last_game_cpu_seconds": record["runner_cpu_seconds"],
            "completed": False, "phase": "self_play_training",
        })
        for milestone in milestones:
            if state_count >= milestone and milestone not in seen_milestones:
                checkpoint_path = arm_dir / f"node_{milestone}.pt"
                _atomic_torch_save(checkpoint_path, {
                    "format_version": 1, "arm_id": arm, "board_size": board_size,
                    "n_in_row": n_in_row, "simulations_per_move": simulations,
                    "environment_states": state_count, "games_completed": game_count,
                    "model_state": {k: v.detach().cpu() for k, v in model.get_policy_param().items()},
                    "config_sha256": config_hash,
                })
                with stage_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "arm_id": arm, "milestone_states": int(milestone),
                        "actual_states": state_count, "game_count": game_count,
                        "update_count": update_count, "loss": loss, "entropy": entropy,
                        "lr_multiplier": lr_multiplier,
                        "training_cpu_seconds": float(record["runner_cpu_seconds"]),
                        "cumulative_cpu_seconds": total_cpu_seconds,
                    }, ensure_ascii=False) + "\n")
                    handle.flush(); os.fsync(handle.fileno())
                seen_milestones.add(milestone)
    for milestone in milestones:
        if state_count >= milestone and milestone not in seen_milestones:
            checkpoint_path = arm_dir / f"node_{milestone}.pt"
            _atomic_torch_save(checkpoint_path, {
                "format_version": 1, "arm_id": arm, "board_size": board_size,
                "n_in_row": n_in_row, "simulations_per_move": simulations,
                "environment_states": state_count, "games_completed": game_count,
                "model_state": {k: v.detach().cpu() for k, v in model.get_policy_param().items()},
                "config_sha256": config_hash,
            })
            with stage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "arm_id": arm, "milestone_states": int(milestone),
                    "actual_states": state_count, "game_count": game_count,
                    "update_count": update_count, "loss": loss, "entropy": entropy,
                    "lr_multiplier": lr_multiplier,
                    "cumulative_cpu_seconds": base_cpu_seconds + time.process_time() - session_cpu_started,
                }, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            seen_milestones.add(milestone)
    final_payload = _checkpoint_payload(
        model, arm=arm, config_hash=config_hash, game_count=game_count,
        state_count=state_count, update_count=update_count,
        lr_multiplier=lr_multiplier, replay=replay, base_seed=seed,
        cumulative_cpu_seconds=base_cpu_seconds + time.process_time() - session_cpu_started,
    )
    _atomic_torch_save(current_path, final_payload)
    model_path = arm_dir / "final.model"
    temp_model = arm_dir / "final.model.tmp"
    torch.save({k: v.detach().cpu() for k, v in model.get_policy_param().items()}, temp_model)
    os.replace(temp_model, model_path)
    return {"arm_id": arm, "game_count": game_count, "environment_states": state_count,
            "update_count": update_count, "checkpoint": str(current_path),
            "model": str(model_path), "replay_rows": len(replay),
            "completed": state_count >= int(target_states),
            "cumulative_cpu_seconds": base_cpu_seconds + time.process_time() - session_cpu_started}
