"""Frozen G5000/G8000 strength acceptance and ladder evaluation for V6.6."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any

import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from .v5_evaluation import (
    _IndependentRuleBoard,
    _project_board,
    generate_opening_pool,
    opening_key,
    other,
    paired_bootstrap_ci,
    summarize_paired_games,
)
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_6")
EXPERIMENT_ID = "wuziq-alphazero-g-v6-6-acceptance"
BASE_COMMIT = "ba41fa266ef725815cdef44ebf753aedfaecba3a"
MODEL_SEED = 590502
MODEL_SHA256 = {
    "G5000": "cc383f0667cfe02b8e8ccbaecf51eccc2c4f088d56c836a4e91e016cfae3ee97",
    "G8000": "490b70eabb75de1b076e1eb9d13a9d82d2ccc213f3b769bfe6c98eddcb679272",
}
STATE_SHA256 = {
    "G5000": "0fb9886068a014d000227ae7b275ff9043322a678d7f259e9730d44ac95398d2",
    "G8000": "6b56dee19e66365fd3623a80817e397eb71875861e1f28be8809447daef88a82",
}
GAME_COUNT = {"G5000": 5000, "G8000": 8000}
BOARD_SIZE = 9
N_IN_ROW = 5
PUCT_SIMULATIONS = 400
OPENING_SEED = 660606
OPENING_PLIES = 6
OPENING_GROUPS = 40
DIRECT_PAIRS = 40
BASELINE_PAIRS = 20
C5_PAIRS = 6
BASELINE_OPPONENTS = ("A", "B@50ms", "C@50ms", "C@200ms", "C@1s", "C@5s")
CORE_OPPONENT_PAIRS = {
    "A": 20,
    "B@50ms": 20,
    "C@50ms": 20,
    "C@200ms": 20,
    "C@1s": 20,
    "C@5s": 6,
}
OPPONENT_BUDGET_SECONDS: dict[str, float | None] = {
    "A": None,
    "B@50ms": 0.05,
    "C@50ms": 0.05,
    "C@200ms": 0.2,
    "C@1s": 1.0,
    "C@5s": 5.0,
}
MODELS = ("G5000", "G8000")


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _stable_int(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:4], "big")


def _stable_id(*parts: str) -> str:
    return "v66-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _valid_move_sequence(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= OPENING_PLIES
        and all(isinstance(move, dict) and {"row", "col", "player"} <= set(move) for move in value[:OPENING_PLIES])
    )


def _collect_sequences(value: Any, destination: set[str]) -> None:
    if isinstance(value, dict):
        for key in ("opening", "moves"):
            moves = value.get(key)
            if _valid_move_sequence(moves):
                prefix = moves[:OPENING_PLIES]
                try:
                    destination.add(opening_key(prefix, BOARD_SIZE))
                except (KeyError, TypeError, ValueError):
                    pass
        for child in value.values():
            _collect_sequences(child, destination)
    elif isinstance(value, list):
        for child in value:
            _collect_sequences(child, destination)


def collect_historical_opening_keys(root: str | Path = ROOT) -> tuple[set[str], list[str]]:
    """Read old opening schedules, stored match records, and raw game histories."""
    root = Path(root).resolve()
    research = root / "research"
    paths: set[Path] = set()
    for path in research.rglob("*.json"):
        rel = path.relative_to(research).as_posix()
        in_scope = (
            "/config/" in f"/{rel}"
            or "/results/matches/" in f"/{rel}"
            or "/raw/" in f"/{rel}"
            or "opening" in path.name.lower()
            or path.name in {"final_challenge.json", "games.json"}
        )
        if in_scope and path.stat().st_size <= 8 * 1024 * 1024:
            paths.add(path)
    keys: set[str] = set()
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        _collect_sequences(payload, keys)
    return keys, [str(path.relative_to(root)) for path in sorted(paths)]


def validate_opening_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    openings = schedule.get("openings")
    if not isinstance(openings, list) or len(openings) != OPENING_GROUPS:
        raise ValueError(f"V6.6 opening pool must contain exactly {OPENING_GROUPS} groups")
    keys: set[str] = set()
    ids: set[str] = set()
    for index, opening in enumerate(openings):
        moves = opening.get("moves")
        if not isinstance(moves, list) or len(moves) != OPENING_PLIES:
            raise ValueError(f"opening {index} does not contain {OPENING_PLIES} plies")
        board = make_board(BOARD_SIZE, N_IN_ROW, 0)
        independent = _IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
        expected = "B"
        for move in moves:
            player = str(move["player"])
            row, col = int(move["row"]), int(move["col"])
            if player != expected or not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                raise ValueError(f"opening {index} has invalid color order or coordinates")
            flat = row * BOARD_SIZE + col
            if flat not in board.availables or independent.get((row, col)) is not None:
                raise ValueError(f"opening {index} repeats or illegally occupies a point")
            board.do_move(flat)
            independent.place_stone(player, (row, col))
            if board.game_end()[0] or independent.check_victory((row, col), player):
                raise ValueError(f"opening {index} is terminal")
            expected = other(expected)
        key = opening_key(moves, BOARD_SIZE)
        if key in keys:
            raise ValueError("V6.6 opening pool contains D4-equivalent duplicates")
        keys.add(key)
        opening_id = str(opening.get("opening_group_id", ""))
        if opening_id in ids or not opening_id.startswith("v6-6-"):
            raise ValueError("V6.6 opening IDs must be unique and start with v6-6-")
        ids.add(opening_id)
    return {"opening_groups": len(openings), "unique_d4_openings": len(keys), "legal_nonterminal": True}


def build_opening_schedule(root: str | Path = ROOT, *, seed: int = OPENING_SEED) -> dict[str, Any]:
    historical, scanned_paths = collect_historical_opening_keys(root)
    selected_seed = int(seed)
    while True:
        rows = generate_opening_pool(
            size=BOARD_SIZE, n_in_row=N_IN_ROW, groups=OPENING_GROUPS,
            plies=OPENING_PLIES, seed=selected_seed,
        )
        keys = [opening_key(row["moves"], BOARD_SIZE) for row in rows]
        if len(set(keys)) == OPENING_GROUPS and not (set(keys) & historical):
            break
        selected_seed += 1
        if selected_seed - int(seed) > 1000:
            raise RuntimeError("could not build 40 historical-distinct V6.6 openings")
    for index, row in enumerate(rows):
        row["opening_group_id"] = f"v6-6-blind-{selected_seed}-{index:03d}"
        row["seed"] = selected_seed + index
    schedule = {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "board_size": BOARD_SIZE,
        "n_in_row": N_IN_ROW,
        "opening_seed_requested": int(seed),
        "opening_seed_selected": selected_seed,
        "opening_plies": OPENING_PLIES,
        "openings": rows,
        "validation": {
            **validate_opening_schedule({"openings": rows}),
            "historical_opening_keys_scanned": len(historical),
            "historical_files_scanned": scanned_paths,
            "historical_d4_overlap": 0,
        },
    }
    return schedule


def _plan_task(candidate: str, opponent: str, opening: dict[str, Any], color: str,
               *, series: str, opponent_budget: float | None, series_seed: str) -> dict[str, Any]:
    opening_id = str(opening["opening_group_id"])
    game_id = _stable_id(EXPERIMENT_ID, series, candidate, opponent, opening_id, color)
    seed = _stable_int(EXPERIMENT_ID, series_seed, opening_id, color)
    return {
        "game_id": game_id,
        "series": series,
        "candidate": candidate,
        "opponent": opponent,
        "opening_group_id": opening_id,
        "opening_seed": int(opening["seed"]),
        "opening": opening["moves"],
        "candidate_color": color,
        "seed": seed,
        "candidate_simulations": PUCT_SIMULATIONS,
        "opponent_time_limit_seconds": opponent_budget,
    }


def build_match_plan(openings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the pre-registered 504-game schedule with paired, color-swapped starts."""
    if len(openings) != OPENING_GROUPS:
        raise ValueError(f"V6.6 requires a frozen pool of exactly {OPENING_GROUPS} openings")
    plan: list[dict[str, Any]] = []
    for opening in openings[:DIRECT_PAIRS]:
        for color in ("B", "W"):
            plan.append(_plan_task(
                "G8000", "G5000", opening, color,
                series="G8000_vs_G5000", opponent_budget=None, series_seed="direct-G8000-G5000",
            ))
    for candidate in MODELS:
        for opponent in BASELINE_OPPONENTS:
            pairs = CORE_OPPONENT_PAIRS[opponent]
            for opening in openings[:pairs]:
                for color in ("B", "W"):
                    plan.append(_plan_task(
                        candidate, opponent, opening, color,
                        series=f"{candidate}_vs_{opponent}",
                        opponent_budget=OPPONENT_BUDGET_SECONDS[opponent],
                        # Both checkpoints receive the same random stream for a matching opponent/opening/color.
                        series_seed=f"opponent-{opponent}",
                    ))
    ids = [row["game_id"] for row in plan]
    if len(plan) != 504 or len(ids) != len(set(ids)):
        raise ValueError("V6.6 plan must contain 504 unique game IDs")
    return plan


def load_frozen_model(path: str | Path, name: str, *, expected_sha256: str | None = None) -> ReferencePolicyValueNet:
    if name not in MODELS:
        raise ValueError(f"unknown frozen model name {name!r}")
    path = Path(path)
    digest = sha_file(path)
    expected = expected_sha256 or MODEL_SHA256[name]
    if digest != expected or digest != MODEL_SHA256[name]:
        raise ValueError(f"{name} model SHA-256 mismatch: {digest}")
    node = torch.load(path, map_location="cpu", weights_only=False)
    if int(node.get("game_count", -1)) != GAME_COUNT[name]:
        raise ValueError(f"{name} checkpoint game count mismatch")
    if (int(node.get("board_size", -1)), int(node.get("n_in_row", -1))) != (BOARD_SIZE, N_IN_ROW):
        raise ValueError(f"{name} checkpoint rule metadata mismatch")
    if int(node.get("simulations_per_move", -1)) != PUCT_SIMULATIONS:
        raise ValueError(f"{name} checkpoint search metadata mismatch")
    if not isinstance(node.get("model_state"), dict):
        raise ValueError(f"{name} checkpoint does not contain model weights")
    model = ReferencePolicyValueNet(BOARD_SIZE, seed=MODEL_SEED)
    model.load_policy_param(node["model_state"])
    model.network.eval()
    return model


def _make_agent(name: str, models: dict[str, ReferencePolicyValueNet]) -> dict[str, Any]:
    if name in MODELS:
        model = models[name]
        player = instrumented_player(
            model.policy_value_fn, c_puct=5.0,
            n_playout=PUCT_SIMULATIONS + 1, is_selfplay=False,
        )
        return {"kind": "G", "name": name, "model": model, "player": player}
    if name == "A":
        return {"kind": "A", "name": name, "time_limit": 1e9}
    if name == "B@50ms":
        return {"kind": "B", "name": name, "time_limit": 0.05}
    if name.startswith("C@"):
        try:
            limit = {"C@50ms": 0.05, "C@200ms": 0.2, "C@1s": 1.0, "C@5s": 5.0}[name]
        except KeyError as exc:
            raise ValueError(f"unsupported V6.6 opponent {name!r}") from exc
        return {"kind": "C", "name": name, "time_limit": limit}
    raise ValueError(f"unknown V6.6 agent {name!r}")


def _agent_choice(agent: dict[str, Any], board, color: str, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
    started_cpu = time.process_time()
    if agent["kind"] == "G":
        move, _policy = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(move), int(board.width))
        stats = dict(agent["player"].last_search_stats)
        stats.update({
            "simulations": int(agent["player"].mcts._n_playout - 1),
            "search_depth": int(stats.get("tree_depth", 0)),
            "search_nodes": int(stats.get("tree_nodes", 0)),
            "root_action_visits": int(stats.get("root_visits", 0)),
        })
        return (row, col), {**stats, "cpu_seconds": time.process_time() - started_cpu}
    projected = _project_board(board)
    result = choose_move(
        projected, color, agent["kind"],
        SearchConfig(time_limit_seconds=float(agent["time_limit"])), int(seed),
    )
    return result.move, {
        "simulations": int(result.simulations),
        "root_action_visits": 0,
        "search_depth": int(result.completed_depth),
        "search_nodes": int(result.nodes),
        "cutoffs": int(result.cutoffs),
        "cpu_seconds": time.process_time() - started_cpu,
        "elapsed_seconds": float(result.elapsed_seconds),
        "timed_out": bool(result.timed_out),
        "configured_time_limit_seconds": float(agent["time_limit"]),
    }


def play_match(task: dict[str, Any], models: dict[str, ReferencePolicyValueNet]) -> dict[str, Any]:
    candidate_name, opponent_name = str(task["candidate"]), str(task["opponent"])
    candidate_color = str(task["candidate_color"])
    if candidate_color not in {"B", "W"}:
        raise ValueError("candidate color must be B or W")
    agents = {
        candidate_name: _make_agent(candidate_name, models),
        opponent_name: _make_agent(opponent_name, models),
    }
    board = make_board(BOARD_SIZE, N_IN_ROW, 0)
    independent = _IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
    stored_moves: list[dict[str, Any]] = []
    expected_color = "B"
    for opening_move in task["opening"]:
        color = str(opening_move["player"])
        row, col = int(opening_move["row"]), int(opening_move["col"])
        if color != expected_color:
            raise ValueError("V6.6 opening violates alternating black/white order")
        flat = row * BOARD_SIZE + col
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError("V6.6 opening contains an illegal or repeated move")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        if board.game_end()[0] or independent.check_victory((row, col), color):
            raise ValueError("V6.6 opening is terminal")
        stored_moves.append({"player": color, "row": row, "col": col, "opening": True})
        expected_color = other(expected_color)

    seed = int(task["seed"])
    np.random.seed(seed & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    candidate_cpu = opponent_cpu = 0.0
    winner: str | None = None
    termination = ""
    illegal_attempt: dict[str, Any] | None = None
    while True:
        color = "B" if board.current_player == 1 else "W"
        actor_name = candidate_name if color == candidate_color else opponent_name
        agent = agents[actor_name]
        move_seed = int(rng.integers(0, 2**31 - 1))
        decision_started = time.perf_counter()
        coord, metrics = _agent_choice(agent, board, color, move_seed)
        elapsed_wall = time.perf_counter() - decision_started
        if actor_name == candidate_name:
            candidate_cpu += float(metrics.get("cpu_seconds", 0.0))
        else:
            opponent_cpu += float(metrics.get("cpu_seconds", 0.0))
        row, col = map(int, coord)
        flat = row * BOARD_SIZE + col
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE) or flat not in board.availables or independent.get((row, col)) is not None:
            illegal_attempt = {"actor": actor_name, "player": color, "row": row, "col": col}
            winner, termination = other(color), "illegal_move_forfeit"
            break
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        stored_moves.append({
            "player": color, "row": row, "col": col, "opening": False,
            "actor": actor_name, "elapsed_seconds": elapsed_wall, **metrics,
        })
        engine_end, engine_winner = board.game_end()
        rule_won = independent.check_victory((row, col), color)
        if bool(engine_end) != bool(rule_won or independent.is_full()):
            raise AssertionError("Gomoku engine and independent rule oracle disagree")
        if rule_won:
            winner, termination = color, "five_in_row"
            if engine_winner != board.current_player ^ 3:
                raise AssertionError("engine winner differs from independent replay")
            break
        if independent.is_full():
            winner, termination = None, "full_board_draw"
            break
    points = 0.5 if winner is None else 1.0 if winner == candidate_color else 0.0
    return {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "game_id": str(task["game_id"]),
        "series": str(task["series"]),
        "candidate": candidate_name,
        "opponent": opponent_name,
        "candidate_color": candidate_color,
        "winner_color": winner,
        "points": points,
        "termination": termination,
        "illegal_attempt": illegal_attempt,
        "board_size": BOARD_SIZE,
        "n_in_row": N_IN_ROW,
        "opening_group_id": str(task["opening_group_id"]),
        "opening_seed": int(task["opening_seed"]),
        "opening": task["opening"],
        "seed": seed,
        "candidate_model_sha256": MODEL_SHA256[candidate_name],
        "opponent_model_sha256": MODEL_SHA256.get(opponent_name),
        "candidate_simulations_per_move": PUCT_SIMULATIONS,
        "opponent_time_limit_seconds": task["opponent_time_limit_seconds"],
        "moves": stored_moves,
        "move_count": len(stored_moves),
        "decision_count": len(stored_moves) - len(task["opening"]),
        "candidate_cpu_seconds": candidate_cpu,
        "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }


def _series_dir(root: Path, task: dict[str, Any]) -> Path:
    return root / ARTIFACT / "results/matches" / _slug(str(task["series"]))


def _validate_existing(game: dict[str, Any], task: dict[str, Any]) -> None:
    for field in ("game_id", "series", "candidate", "opponent", "opening_group_id", "candidate_color", "seed"):
        if game.get(field) != task.get(field):
            raise ValueError(f"V6.6 stored game disagrees with frozen plan field {field}")
    if len(game.get("moves", [])) != int(game.get("move_count", -1)) or not game.get("termination"):
        raise ValueError("V6.6 stored game is incomplete")


def run_batch(root: str | Path = ROOT, *, batch_games: int = 10) -> dict[str, Any]:
    """Run the next deterministic batch; each completed game is atomically persisted."""
    if batch_games < 1:
        raise ValueError("batch_games must be positive")
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    schedule_hash_payload = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    if _canonical_hash(schedule_hash_payload) != registration.get("opening_schedule_sha256"):
        raise ValueError("V6.6 opening schedule SHA differs from the frozen registration")
    plan = build_match_plan(schedule["openings"])
    plan_hash = _canonical_hash(plan)
    stored_plan = json.loads((artifact / "config/match_plan.json").read_text(encoding="utf-8"))
    if plan_hash != registration.get("match_plan_sha256") or stored_plan.get("plan_sha256") != plan_hash:
        raise ValueError("V6.6 match plan differs from the frozen registration")
    model_paths = registration["checkpoint_paths"]
    models = {
        name: load_frozen_model(root / model_paths[name], name)
        for name in MODELS
    }
    existing: dict[str, dict[str, Any]] = {}
    for task in plan:
        path = _series_dir(root, task) / f"{task['game_id']}.json"
        if path.is_file():
            try:
                game = json.loads(path.read_text(encoding="utf-8"))
                _validate_existing(game, task)
                existing[task["game_id"]] = game
            except (json.JSONDecodeError, ValueError):
                quarantine = artifact / "results/quarantine"
                quarantine.mkdir(parents=True, exist_ok=True)
                target = quarantine / f"{task['game_id']}.corrupt.json"
                if target.exists():
                    target.unlink()
                os.replace(path, target)
    pending = [task for task in plan if task["game_id"] not in existing]
    completed_now: list[str] = []
    started_cpu = time.process_time()
    for task in pending[:int(batch_games)]:
        game = play_match(task, models)
        path = _series_dir(root, task) / f"{task['game_id']}.json"
        _validate_existing(game, task)
        atomic_json(path, game)
        existing[task["game_id"]] = game
        completed_now.append(task["game_id"])
        completed_ids = set(existing)
        atomic_json(artifact / "results/progress.json", {
            "experiment_id": EXPERIMENT_ID,
            "planned_games": len(plan),
            "completed_games": len(completed_ids),
            "completed_game_ids": sorted(completed_ids),
            "last_completed_game_id": task["game_id"],
            "cpu_seconds_this_batch": time.process_time() - started_cpu,
            "complete": len(completed_ids) == len(plan),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return {
        "completed_before": len(existing) - len(completed_now),
        "completed_now": len(completed_now),
        "completed_total": len(existing),
        "planned_games": len(plan),
        "game_ids": completed_now,
        "cpu_seconds_this_batch": time.process_time() - started_cpu,
        "complete": len(existing) == len(plan),
    }


def finalize_v6_5_progress(progress_path: str | Path, *, expected_games: int = 8000) -> dict[str, Any]:
    """Correct the stale V6.5 completion flag without touching any checkpoint."""
    path = Path(progress_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("games_completed", -1)) != int(expected_games):
        raise ValueError(f"V6.5 progress is not at G{expected_games}")
    if int(payload.get("optimizer_updates", -1)) != int(expected_games) - 1:
        raise ValueError("V6.5 progress optimizer count does not match the completed game count")
    payload["completed"] = True
    payload["phase"] = "training_complete"
    payload["completion_flag_corrected_by"] = EXPERIMENT_ID
    payload["completion_flag_correction"] = "metadata only; checkpoint bytes and training state unchanged"
    atomic_json(path, payload)
    return payload


def prepare_v6_6(root: str | Path, *, restored_root: str | Path,
                 drive_folder_id: str) -> dict[str, Any]:
    """Validate both Drive recovery states and freeze models, schedule, and registration."""
    from .v6_5_training import validate_training_state

    root, restored_root = Path(root).resolve(), Path(restored_root).resolve()
    artifact = root / ARTIFACT
    checks: dict[str, Any] = {}
    model_paths: dict[str, str] = {}
    state_paths = {
        "G5000": restored_root / "G5000/research/g_v6_5/checkpoints/reference400_empty/training_state_games_5000.pt",
        "G8000": restored_root / "G8000/research/g_v6_5/checkpoints/reference400_empty/training_state_games_8000.pt",
    }
    node_paths = {
        "G5000": restored_root / "G5000/research/g_v6_5/checkpoints/reference400_empty/node_games_5000.pt",
        "G8000": restored_root / "G8000/research/g_v6_5/checkpoints/reference400_empty/node_games_8000.pt",
    }
    for name in MODELS:
        state_path, node_path = state_paths[name], node_paths[name]
        if not state_path.is_file() or not node_path.is_file():
            raise FileNotFoundError(f"restored {name} full training state/model missing")
        if sha_file(state_path) != STATE_SHA256[name] or sha_file(node_path) != MODEL_SHA256[name]:
            raise ValueError(f"restored {name} source hash mismatch")
        restored = validate_training_state(state_path, root=root, expected_games=GAME_COUNT[name])
        node = torch.load(node_path, map_location="cpu", weights_only=False)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if not _model_state_equal(node["model_state"], payload["model_state"]):
            raise ValueError(f"{name} node weights differ from full recovery state")
        if (int(node["game_count"]), int(node["environment_states"]), int(node["update_count"])) != (
            GAME_COUNT[name], int(restored["environment_states"]), GAME_COUNT[name] - 1
        ):
            raise ValueError(f"{name} model metadata differs from recovered training state")
        load_frozen_model(node_path, name)
        destination = artifact / "checkpoints" / f"{name}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha_file(destination) != MODEL_SHA256[name]:
            raise ValueError(f"existing V6.6 {name} checkpoint was modified")
        if not destination.exists():
            shutil.copy2(node_path, destination)
        model_paths[name] = str(destination.relative_to(root))
        checks[name] = {
            "model_sha256": sha_file(destination), "source_state_sha256": sha_file(state_path),
            "source_model_sha256": sha_file(node_path), "game_count": GAME_COUNT[name],
            "environment_states": int(restored["environment_states"]),
            "optimizer_updates": int(restored["optimizer_updates"]),
            "optimizer_slots": int(restored["optimizer_slots"]), "replay_rows": int(restored["replay_rows"]),
            "full_state_restored": bool(restored["valid"]), "model_matches_full_state": True,
            "model_loadable": True,
        }
    schedule_path = artifact / "config/opening_schedule.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        validate_opening_schedule(schedule)
    else:
        schedule = build_opening_schedule(root)
        schedule_bytes = json.dumps(schedule, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        schedule["schedule_sha256"] = hashlib.sha256(schedule_bytes).hexdigest()
        atomic_json(schedule_path, schedule)
    schedule_digest = str(schedule.get("schedule_sha256", ""))
    if not schedule_digest:
        raise ValueError("opening schedule missing its frozen hash")
    progress_path = root / "research/g_v6_5/results/training_progress.json"
    progress_before = json.loads(progress_path.read_text(encoding="utf-8"))
    if int(progress_before.get("games_completed", -1)) != 8000:
        raise ValueError("stale V6.5 completion status does not point to G8000")
    progress_after = finalize_v6_5_progress(progress_path)
    plan = build_match_plan(schedule["openings"])
    plan_path = artifact / "config/match_plan.json"
    plan_bytes = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old.get("plan_sha256") != plan_digest:
            raise ValueError("existing V6.6 frozen plan differs from regenerated plan")
    else:
        atomic_json(plan_path, {"plan_sha256": plan_digest, "tasks": plan})
    registration = {
        "format_version": 1, "experiment_id": EXPERIMENT_ID, "base_commit": BASE_COMMIT,
        "branch": "research/alphazero-g-v6-6-acceptance", "board_size": BOARD_SIZE,
        "n_in_row": N_IN_ROW, "puct_simulations": PUCT_SIMULATIONS,
        "models": checks, "checkpoint_paths": model_paths,
        "opening_schedule_sha256": schedule_digest, "match_plan_sha256": plan_digest,
        "planned_games": len(plan), "drive_folder_id": drive_folder_id,
        "source_drive_archives": {
            "G5000": {"file_id": "1rwXJd863iQymbt_-OfuoFxS0Taor1PQQ", "sha256": "e102465b3893c95524c2fcfb4913e23d60cda22d4bd1aa3debc55d98c17379ab"},
            "G8000": {"file_id": "1koTkXhGJyrvk09RC5gwhk0G1DSXyGrSA", "sha256": "0b80829fef0b62c7bef653c5c6a9afa0ac0bbd946c8491dce6297cf091d260e0"},
        },
        "v6_5_progress_before": progress_before,
        "v6_5_progress_after": progress_after,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm_changes": False,
    }
    atomic_json(artifact / "config/registration.json", registration)
    return registration


def _model_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def audit_game(game: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    _validate_existing(game, task)
    board = make_board(BOARD_SIZE, N_IN_ROW, 0)
    oracle = _IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
    moves = game["moves"]
    if len(moves) < len(task["opening"]):
        raise ValueError("game ends before its frozen opening is complete")
    if game.get("candidate_model_sha256") != MODEL_SHA256[task["candidate"]]:
        raise ValueError("candidate checkpoint identity does not match the task")
    if game.get("opponent_model_sha256") != MODEL_SHA256.get(task["opponent"]):
        raise ValueError("opponent checkpoint identity does not match the task")
    if int(game.get("candidate_simulations_per_move", -1)) != PUCT_SIMULATIONS:
        raise ValueError("stored candidate simulation budget differs from registration")
    if game.get("opponent_time_limit_seconds") != task.get("opponent_time_limit_seconds"):
        raise ValueError("stored opponent time limit differs from the frozen task")
    if game.get("opening") != task.get("opening") or int(game.get("opening_seed", -1)) != int(task["opening_seed"]):
        raise ValueError("stored opening or opening seed differs from the pre-registered task")
    expected_color = "B"
    for index, move in enumerate(moves):
        color = str(move["player"])
        row, col = int(move["row"]), int(move["col"])
        if color != expected_color or not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise ValueError(f"illegal color/coordinate at ply {index}")
        if index < len(task["opening"]):
            expected = task["opening"][index]
            if (color, row, col) != (str(expected["player"]), int(expected["row"]), int(expected["col"])):
                raise ValueError("stored opening differs from the pre-registered schedule")
        else:
            expected_actor = task["candidate"] if color == task["candidate_color"] else task["opponent"]
            if move.get("actor") != expected_actor:
                raise ValueError(f"wrong model actor recorded at ply {index}")
            elapsed = float(move.get("elapsed_seconds", -1.0))
            if not np.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"missing/invalid decision time at ply {index}")
            if expected_actor in MODELS:
                if int(move.get("simulations", -1)) != PUCT_SIMULATIONS or int(move.get("root_action_visits", -1)) != PUCT_SIMULATIONS:
                    raise ValueError(f"MCTS did not complete exactly {PUCT_SIMULATIONS} root visits at ply {index}")
            elif int(move.get("search_depth", -1)) < 0 or int(move.get("search_nodes", -1)) < 0:
                raise ValueError(f"missing traditional search counters at ply {index}")
        action = row * BOARD_SIZE + col
        if action not in board.availables or oracle.get((row, col)) is not None:
            raise ValueError(f"occupied move at ply {index}")
        board.do_move(action)
        oracle.place_stone(color, (row, col))
        won = oracle.check_victory((row, col), color)
        ended, engine_winner = board.game_end()
        if bool(ended) != bool(won or oracle.is_full()):
            raise ValueError(f"engine/oracle terminal mismatch at ply {index}")
        if won:
            if index != len(moves) - 1 or game["winner_color"] != color:
                raise ValueError("winner or trailing moves after terminal position")
            if int(engine_winner) != (1 if color == "B" else 2):
                raise ValueError("engine winner differs from replay winner")
        elif oracle.is_full():
            if index != len(moves) - 1 or game["winner_color"] is not None:
                raise ValueError("invalid draw termination")
        expected_color = other(expected_color)
    if game["termination"] == "illegal_move_forfeit":
        illegal = game.get("illegal_attempt")
        if not isinstance(illegal, dict) or illegal.get("actor") not in {task["candidate"], task["opponent"]}:
            raise ValueError("illegal-move forfeit lacks its actor record")
        expected_side = "B" if board.current_player == 1 else "W"
        if illegal.get("player") != expected_side or game.get("winner_color") != other(expected_side):
            raise ValueError("illegal-move forfeit winner is inconsistent with the legal prefix")
        if board.game_end()[0]:
            raise ValueError("illegal-move forfeit was recorded after an already terminal board")
    elif not board.game_end()[0] or game["termination"] not in {"five_in_row", "full_board_draw"}:
        raise ValueError("game record is not terminal")
    expected_points = 0.5 if game["winner_color"] is None else 1.0 if game["winner_color"] == task["candidate_color"] else 0.0
    if float(game["points"]) != expected_points:
        raise ValueError("candidate points are inconsistent with the replayed outcome")
    return {"game_id": game["game_id"], "valid": True, "plies": len(moves), "winner_color": game["winner_color"]}


def audit_all(root: str | Path = ROOT, *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    schedule_hash_payload = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    if _canonical_hash(schedule_hash_payload) != registration.get("opening_schedule_sha256"):
        raise ValueError("V6.6 opening schedule hash mismatch")
    plan = build_match_plan(schedule["openings"])
    if _canonical_hash(plan) != registration.get("match_plan_sha256"):
        raise ValueError("V6.6 match plan hash mismatch")
    audits = []
    for task in plan:
        path = _series_dir(root, task) / f"{task['game_id']}.json"
        if not path.is_file():
            if require_complete:
                raise FileNotFoundError(f"missing V6.6 game {task['game_id']}")
            continue
        game = json.loads(path.read_text(encoding="utf-8"))
        audits.append(audit_game(game, task))
    ids = [row["game_id"] for row in audits]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate V6.6 game IDs")
    if require_complete and len(audits) != 504:
        raise ValueError(f"expected 504 valid results, found {len(audits)}")
    result = {
        "experiment_id": EXPERIMENT_ID, "planned_games": len(plan), "audited_games": len(audits),
        "missing_games": len(plan) - len(audits), "all_games_valid": len(audits) == len(plan),
        "unique_game_ids": len(set(ids)), "opening_schedule_sha256": registration["opening_schedule_sha256"],
        "plan_sha256": registration["match_plan_sha256"], "game_audits": audits,
    }
    atomic_json(artifact / "results/audit.json", result)
    return result


def summarize_v6_6(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    audit_all(root, require_complete=True)
    schedule = json.loads((root / ARTIFACT / "config/opening_schedule.json").read_text(encoding="utf-8"))
    plan = build_match_plan(schedule["openings"])
    games = []
    for task in plan:
        games.append(json.loads((_series_dir(root, task) / f"{task['game_id']}.json").read_text(encoding="utf-8")))
    by_id = {game["game_id"]: game for game in games}
    summaries = []
    for candidate in MODELS:
        for opponent in BASELINE_OPPONENTS:
            tasks = [task for task in plan if task["candidate"] == candidate and task["opponent"] == opponent]
            selected = [by_id[task["game_id"]] for task in tasks]
            result = summarize_paired_games(selected, seed=_stable_int(EXPERIMENT_ID, candidate, opponent))
            colors = {}
            for color in ("B", "W"):
                color_games = [game for game in selected if game["candidate_color"] == color]
                score = float(np.mean([game["points"] for game in color_games])) if color_games else float("nan")
                colors[color] = {"games": len(color_games), "points": sum(game["points"] for game in color_games), "score_rate": score}
            summaries.append({
                "candidate": candidate, "opponent": opponent, "games": len(selected),
                **result, "candidate_black": colors["B"], "candidate_white": colors["W"],
                "candidate_cpu_seconds": float(np.mean([game["candidate_cpu_seconds"] for game in selected])),
                "opponent_cpu_seconds": float(np.mean([game["opponent_cpu_seconds"] for game in selected])),
                "mean_plies": float(np.mean([game["move_count"] for game in selected])),
                "mean_search_depth": float(np.mean([m.get("search_depth", 0) for game in selected for m in game["moves"] if m.get("actor") == candidate])),
                "mean_search_nodes": float(np.mean([m.get("search_nodes", 0) for game in selected for m in game["moves"] if m.get("actor") == candidate])),
                "mean_simulations": float(np.mean([m.get("simulations", 0) for game in selected for m in game["moves"] if m.get("actor") == candidate])),
                "opponent_mean_search_depth": float(np.mean([m.get("search_depth", 0) for game in selected for m in game["moves"] if m.get("actor") == opponent])),
                "opponent_mean_search_nodes": float(np.mean([m.get("search_nodes", 0) for game in selected for m in game["moves"] if m.get("actor") == opponent])),
                "opponent_mean_cutoffs": float(np.mean([m.get("cutoffs", 0) for game in selected for m in game["moves"] if m.get("actor") == opponent])),
                "timeouts": sum(bool(m.get("timed_out")) for game in selected for m in game["moves"] if m.get("actor") == opponent),
            })
    direct_tasks = [task for task in plan if task["series"] == "G8000_vs_G5000"]
    direct = [by_id[task["game_id"]] for task in direct_tasks]
    direct_summary = summarize_paired_games(direct, seed=_stable_int(EXPERIMENT_ID, "direct"))
    color_summary = {
        color: {"games": len([g for g in direct if g["candidate_color"] == color]),
                "score_rate": float(np.mean([g["points"] for g in direct if g["candidate_color"] == color]))}
        for color in ("B", "W")
    }
    direct_summary.update({"candidate": "G8000", "opponent": "G5000", "games": len(direct),
                           "candidate_black": color_summary["B"], "candidate_white": color_summary["W"],
                           "candidate_cpu_seconds": float(np.mean([g["candidate_cpu_seconds"] for g in direct])),
                           "opponent_cpu_seconds": float(np.mean([g["opponent_cpu_seconds"] for g in direct])),
                           "mean_plies": float(np.mean([g["move_count"] for g in direct])),
                           "candidate_mean_search_depth": float(np.mean([m.get("search_depth", 0) for g in direct for m in g["moves"] if m.get("actor") == "G8000"])),
                           "candidate_mean_search_nodes": float(np.mean([m.get("search_nodes", 0) for g in direct for m in g["moves"] if m.get("actor") == "G8000"])),
                           "candidate_mean_simulations": float(np.mean([m.get("simulations", 0) for g in direct for m in g["moves"] if m.get("actor") == "G8000"])),
                           "opponent_mean_search_depth": float(np.mean([m.get("search_depth", 0) for g in direct for m in g["moves"] if m.get("actor") == "G5000"])),
                           "opponent_mean_search_nodes": float(np.mean([m.get("search_nodes", 0) for g in direct for m in g["moves"] if m.get("actor") == "G5000"])),
                           "opponent_mean_simulations": float(np.mean([m.get("simulations", 0) for g in direct for m in g["moves"] if m.get("actor") == "G5000"]))})
    # Paired improvement uses matching opening, color, seed, and opponent budget.
    improvements = []
    for opponent in BASELINE_OPPONENTS:
        groups = []
        for opening in schedule["openings"][:CORE_OPPONENT_PAIRS[opponent]]:
            color_scores = []
            for color in ("B", "W"):
                lookup = {(t["candidate"], t["opening_group_id"], t["candidate_color"]): by_id[t["game_id"]]
                          for t in plan if t["opponent"] == opponent}
                color_scores.append(lookup[("G8000", opening["opening_group_id"], color)]["points"] -
                                    lookup[("G5000", opening["opening_group_id"], color)]["points"])
            groups.append(float(np.mean(color_scores)))
        rng = np.random.default_rng(_stable_int(EXPERIMENT_ID, "delta", opponent))
        samples = np.mean(np.asarray(groups)[rng.integers(0, len(groups), size=(10000, len(groups)))], axis=1)
        improvements.append({"opponent": opponent, "paired_score_delta": float(np.mean(groups)),
                             "paired_95_ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
                             "opening_groups": len(groups)})
    result = {"experiment_id": EXPERIMENT_ID, "planned_games": 504, "completed_games": len(games),
              "direct_match": direct_summary, "baseline_summaries": summaries,
              "paired_G8000_minus_G5000": improvements,
              "match_cpu_seconds": float(sum(float(game["total_cpu_seconds"]) for game in games)),
              "match_wall_seconds": float(sum(float(game["total_wall_seconds"]) for game in games))}
    atomic_json(root / ARTIFACT / "results/summary.json", result)
    return result


def run_preflight(root: str | Path = ROOT) -> dict[str, Any]:
    """Measure one fixed midgame decision for each frozen search configuration."""
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    registration = json.loads((artifact / "config/registration.json").read_text(encoding="utf-8"))
    models = {name: load_frozen_model(root / registration["checkpoint_paths"][name], name) for name in MODELS}
    board = make_board(BOARD_SIZE, N_IN_ROW, 0)
    oracle = _IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
    rng = random.Random(660607)
    while len(oracle.history) < 25:
        legal = [action for action in board.availables]
        rng.shuffle(legal)
        chosen = None
        color = "B" if board.current_player == 1 else "W"
        for action in legal:
            row, col = divmod(int(action), BOARD_SIZE)
            oracle.place_stone(color, (row, col))
            won = oracle.check_victory((row, col), color)
            oracle.grid[row][col] = None
            oracle.history.pop()
            if not won:
                chosen = action
                break
        if chosen is None:
            raise RuntimeError("could not construct a nonterminal representative midgame")
        row, col = divmod(int(chosen), BOARD_SIZE)
        board.do_move(int(chosen))
        oracle.place_stone(color, (row, col))

    measurements: dict[str, Any] = {}
    for name in MODELS:
        coord, stats = _agent_choice(_make_agent(name, models), board, "B" if board.current_player == 1 else "W", 660608)
        measurements[name] = {"move": list(coord), **stats}
    for opponent in BASELINE_OPPONENTS:
        coord, stats = _agent_choice(_make_agent(opponent, models), board,
                                     "B" if board.current_player == 1 else "W", 660609)
        measurements[opponent] = {"move": list(coord), **stats}

    # Workload uses V6.3 measured game lengths for matching opponents. For stronger C budgets,
    # using the C50 length is conservative in duration; C@5s is explicitly exploratory.
    historical_moves = {"A": 59.0125, "B@50ms": 19.5125, "C@50ms": 20.8125,
                        "C@200ms": 22.0, "C@1s": 24.0, "C@5s": 24.0}
    g_per_move = max(float(measurements[name].get("cpu_seconds", 0.0)) for name in MODELS)
    estimate_cpu = 0.0
    for opponent, pairs in CORE_OPPONENT_PAIRS.items():
        games_per_model = pairs * 2
        turns = historical_moves[opponent]
        estimate_cpu += games_per_model * (turns / 2.0 * g_per_move)
        opponent_per_move = float(measurements[opponent].get("cpu_seconds", 0.0))
        estimate_cpu += games_per_model * (turns / 2.0 * opponent_per_move)
    direct_game_count = DIRECT_PAIRS * 2
    direct_turns = 50.0
    estimate_cpu += direct_game_count * direct_turns * g_per_move
    # 25% overhead covers per-game startup, JSON persistence, auditing and plotting.
    estimate_cpu *= 1.25
    cpu_count = os.cpu_count() or 1
    quota_cpu_count = cpu_count
    try:
        quota_text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_text[0] != "max":
            quota_cpu_count = min(cpu_count, float(quota_text[0]) / float(quota_text[1]))
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    disk = shutil.disk_usage(root)
    memory = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory = {"total_gib": vm.total / 2**30, "available_gib": vm.available / 2**30}
    except ImportError:
        pass
    result = {
        "experiment_id": EXPERIMENT_ID, "probe_position_plies": len(oracle.history),
        "measurements": measurements,
        "resource": {"logical_cpu_count": cpu_count, "estimated_cpu_quota": quota_cpu_count,
                     "memory": memory, "disk_free_gib": disk.free / 2**30},
        "estimate": {
            "optimistic_cpu_hours": estimate_cpu / 3600 * 0.7,
            "central_cpu_hours": estimate_cpu / 3600,
            "conservative_cpu_hours": estimate_cpu / 3600 * 1.7,
            "estimated_artifact_storage_mib": 80,
            "basis": "V6.3 measured game lengths; one 25-ply midgame decision per frozen agent; 25% orchestration overhead; central estimate widened x0.7/x1.7.",
            "preflight_probe_wall_seconds": None,
        },
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(artifact / "results/preflight.json", result)
    return result

