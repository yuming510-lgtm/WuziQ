"""Frozen six-checkpoint G tournament against the V6 traditional opponents.

All 6 x 4 conditions share one pre-registered 20-opening schedule. Each
opening is played twice with the G model colors swapped. Every completed game
is atomically written before the runner advances, making reruns idempotent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard
from . import v5_evaluation as v5_eval
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_8")
MODELS = (5_000, 8_000, 10_000, 12_000, 16_000, 20_000)
OPPONENTS = ("A", "B@50ms", "C@50ms", "C@200ms")
SIMULATIONS = 400
OPENING_SEED = 680_801
OPENING_GROUPS = 20
OPENING_PLIES = 6
BOARD_SIZE = 9
N_IN_ROW = 5
SOURCE_COMMIT = "911242c10c8a374740db072f71a6b5f450a837e6"
G5000_SOURCE_SHA256 = "cc383f0667cfe02b8e8ccbaecf51eccc2c4f088d56c836a4e91e016cfae3ee97"
BOOTSTRAP_REPS = 10_000


def artifact_path(root: str | Path = ROOT) -> Path:
    return Path(root).resolve() / ARTIFACT


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


def _model_path(root: Path, games: int) -> Path:
    return root / ARTIFACT / "models" / f"G{int(games)}.pt"


def _model_node(root: Path, games: int) -> dict[str, Any]:
    path = _model_path(root, games)
    node = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"game_count": games, "games_completed": games,
                "board_size": BOARD_SIZE, "n_in_row": N_IN_ROW,
                "simulations_per_move": SIMULATIONS, "optimizer_updates": games - 1,
                "update_count": games - 1}
    for key, value in expected.items():
        if int(node.get(key, -1)) != value:
            raise ValueError(f"G{games} checkpoint has mismatched {key}: {node.get(key)} != {value}")
    if int(node.get("environment_states", -1)) <= 0 or not isinstance(node.get("model_state"), dict):
        raise ValueError(f"G{games} checkpoint lacks state count or model weights")
    model = ReferencePolicyValueNet(BOARD_SIZE, seed=590_501)
    model.load_policy_param(node["model_state"])
    model.network.eval()
    return node


def verify_models(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    entries = []
    for games in MODELS:
        node = _model_node(root, games)
        path = _model_path(root, games)
        entries.append({"games": games, "path": str(path.relative_to(root)),
                        "sha256": sha_file(path),
                        "environment_states": int(node["environment_states"]),
                        "optimizer_updates": int(node["optimizer_updates"]),
                        "simulations_per_move": int(node["simulations_per_move"]),
                        "board_size": int(node["board_size"]),
                        "n_in_row": int(node["n_in_row"]),
                        "config_sha256": str(node["config_sha256"])})
    if entries[0]["sha256"] != G5000_SOURCE_SHA256:
        raise ValueError("G5000 checkpoint SHA-256 differs from the frozen historical digest")
    if len({row["config_sha256"] for row in entries}) != 1:
        raise ValueError("the six checkpoints do not share the same frozen training configuration")
    result = {"source_commit": SOURCE_COMMIT, "frozen": True,
              "network": "V6 standard AlphaZero reference 4-plane CNN, 32/64/128 channels",
              "board_size": BOARD_SIZE, "n_in_row": N_IN_ROW,
              "simulations_per_move": SIMULATIONS, "models": entries}
    atomic_json(root / ARTIFACT / "config/model_manifest.json", result)
    return result


def _d4_opening_key(moves: list[dict[str, Any]]) -> str:
    return v5_eval.opening_key(moves, BOARD_SIZE)


def _extract_openings(node: Any) -> list[list[dict[str, Any]]]:
    found: list[list[dict[str, Any]]] = []
    if isinstance(node, dict):
        for key in ("opening", "opening_moves", "moves"):
            value = node.get(key)
            if isinstance(value, list) and value and all(isinstance(m, dict) and {"row", "col", "player"} <= set(m) for m in value):
                found.append(value)
        for key in ("development", "blind", "openings", "schedule"):
            value = node.get(key)
            if isinstance(value, list):
                for item in value:
                    found.extend(_extract_openings(item))
            elif isinstance(value, dict):
                found.extend(_extract_openings(value))
        for key, value in node.items():
            if key not in {"opening", "opening_moves", "moves", "development", "blind", "openings", "schedule"} and isinstance(value, (dict, list)):
                found.extend(_extract_openings(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_extract_openings(item))
    return found


def scan_historical_openings(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    roots = sorted({root, *(p for p in root.parent.glob("WuziQ*") if p.is_dir())})
    seen_paths: set[Path] = set()
    keys: set[str] = set()
    parsed_files = 0
    errors = []
    for scan_root in roots:
        for path in scan_root.rglob("*.json"):
            resolved = path.resolve()
            if resolved in seen_paths or ARTIFACT.as_posix() in path.as_posix():
                continue
            seen_paths.add(resolved)
            if not ("opening" in path.name.lower() or "/games/" in path.as_posix() or "/matches/" in path.as_posix()):
                continue
            try:
                node = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append({"path": str(path), "error": type(exc).__name__})
                continue
            parsed_files += 1
            keys.update(_d4_opening_key(moves) for moves in _extract_openings(node) if len(moves) >= OPENING_PLIES)
    return {"scan_roots": [str(p) for p in roots], "parsed_json_files": parsed_files,
            "unique_d4_openings": len(keys), "opening_keys": sorted(keys),
            "parse_errors": errors}


def _validate_opening(opening: dict[str, Any]) -> None:
    moves = opening.get("moves", [])
    if len(moves) != OPENING_PLIES:
        raise ValueError("registered opening has the wrong number of plies")
    board = make_board(BOARD_SIZE, N_IN_ROW, 0)
    oracle = v5_eval._IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
    expected = "B"
    for move in moves:
        player = str(move["player"])
        row, col = int(move["row"]), int(move["col"])
        if player != expected or not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise ValueError("registered opening has an invalid player sequence or coordinate")
        flat = row * BOARD_SIZE + col
        if flat not in board.availables or oracle.get((row, col)) is not None:
            raise ValueError("registered opening contains a repeated or illegal move")
        board.do_move(flat)
        oracle.place_stone(player, (row, col))
        if board.game_end()[0] or oracle.check_victory((row, col), player):
            raise ValueError("registered opening is terminal")
        expected = "W" if expected == "B" else "B"


def prepare_openings(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / ARTIFACT / "config/opening_schedule.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("seed") != OPENING_SEED or len(existing.get("openings", [])) != OPENING_GROUPS
                or existing.get("pairs_per_condition") != OPENING_GROUPS):
            raise ValueError("existing V6.8 opening schedule does not match its frozen registration")
        return existing
    history = scan_historical_openings(root)
    history_keys = set(history["opening_keys"])
    selected_seed = OPENING_SEED
    openings = []
    while selected_seed < OPENING_SEED + 100:
        candidate = v5_eval.generate_opening_pool(size=BOARD_SIZE, n_in_row=N_IN_ROW,
                                                   groups=OPENING_GROUPS, plies=OPENING_PLIES,
                                                   seed=selected_seed)
        keys = [_d4_opening_key(item["moves"]) for item in candidate]
        if len(set(keys)) == OPENING_GROUPS and not (set(keys) & history_keys):
            openings = candidate
            break
        selected_seed += 1
    if not openings:
        raise RuntimeError("could not find a 20-opening pool disjoint from indexed historical data")
    for i, opening in enumerate(openings):
        opening["opening_group_id"] = f"v6.8-9x9-{selected_seed}-{i:03d}"
        opening["seed"] = int(selected_seed + i)
        _validate_opening(opening)
    keys = [_d4_opening_key(item["moves"]) for item in openings]
    result = {"experiment_id": "wuziq-alphazero-g-v6-8-stage-comparison",
              "source_commit": SOURCE_COMMIT, "board_size": BOARD_SIZE, "n_in_row": N_IN_ROW,
              "opening_plies": OPENING_PLIES, "pairs_per_condition": OPENING_GROUPS,
              "games_per_condition": OPENING_GROUPS * 2, "models": list(MODELS),
              "opponents": list(OPPONENTS), "puct_simulations": SIMULATIONS,
              "seed": selected_seed, "same_opening_color_seed_across_all_conditions": True,
              "opening_groups": openings, "historical_d4_overlap": 0,
              "history_scan": {k: v for k, v in history.items() if k != "opening_keys"},
              "opening_schedule_sha256": hashlib.sha256(json.dumps(openings, sort_keys=True).encode()).hexdigest(),
              "opening_key_sha256s": [hashlib.sha256(key.encode()).hexdigest() for key in keys],
              "opponent_protocol": {
                  "A": {"engine": "original rule AI", "time_limit_seconds": None, "artificial_wait": False},
                  "B@50ms": {"engine": "pure Minimax, Alpha-Beta disabled", "time_limit_seconds": 0.05,
                             "max_depth": 4, "candidate_radius": 2, "settings_source": "historical SearchConfig defaults"},
                  "C@50ms": {"engine": "Minimax + Alpha-Beta", "time_limit_seconds": 0.05,
                             "max_depth": 69, "candidate_radius": 2, "timeout_returns_last_complete_depth": True},
                  "C@200ms": {"engine": "Minimax + Alpha-Beta", "time_limit_seconds": 0.2,
                              "max_depth": 69, "candidate_radius": 2, "timeout_returns_last_complete_depth": True}},
              "openings": openings}
    atomic_json(path, result)
    return result


def prepare_experiment(root: str | Path = ROOT) -> dict[str, Any]:
    """Freeze checkpoint identity, opening schedule, resources, and match protocol."""
    root = Path(root).resolve()
    manifest = verify_models(root)
    schedule = prepare_openings(root)
    def read_text(path: str, default: str = "unknown") -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return default
    disk = shutil.disk_usage(root)
    resources = {
        "cpu_count_visible": os.cpu_count(),
        "cpu_quota": read_text("/sys/fs/cgroup/cpu.max"),
        "cpu_stat_before": read_text("/sys/fs/cgroup/cpu.stat"),
        "memory_limit_bytes": read_text("/sys/fs/cgroup/memory.max"),
        "memory_current_bytes": read_text("/sys/fs/cgroup/memory.current"),
        "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    atomic_json(root / ARTIFACT / "config/resources_preflight.json", resources)
    registration = {
        "experiment_id": "wuziq-alphazero-g-v6-8-stage-comparison",
        "branch": "research/alphazero-g-v6-8-stage-comparison",
        "base_commit": SOURCE_COMMIT, "models_frozen": True, "training": False,
        "models": list(MODELS), "opponents": list(OPPONENTS),
        "planned_games": len(MODELS) * len(OPPONENTS) * OPENING_GROUPS * 2,
        "pairs_per_condition": OPENING_GROUPS, "opening_plies": OPENING_PLIES,
        "opening_seed": schedule["seed"], "opening_schedule_sha256": schedule["opening_schedule_sha256"],
        "candidate_simulations": SIMULATIONS,
        "checkpoint_manifest_sha256": sha_file(root / ARTIFACT / "config/model_manifest.json"),
        "opponent_protocol": schedule["opponent_protocol"],
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / ARTIFACT / "config/registration.json", registration)
    return {"registration": registration, "manifest": manifest,
            "schedule": {k: v for k, v in schedule.items() if k != "openings"},
            "resources": resources}


def _store_game(root: Path, row: dict[str, Any]) -> None:
    atomic_json(_game_path(root, row["game_id"]), row)
    _refresh_progress(root, row["game_id"])


def _refresh_progress(root: Path, last_game_id: str | None = None) -> dict[str, Any]:
    games_dir = root / ARTIFACT / "results/games"
    ids = sorted(path.stem for path in games_dir.glob("*.json"))
    total_cpu = sum(float(json.loads(path.read_text(encoding="utf-8")).get("total_cpu_seconds", 0))
                    for path in games_dir.glob("*.json"))
    progress = {
        "experiment_id": "wuziq-alphazero-g-v6-8-stage-comparison",
        "source_commit": SOURCE_COMMIT, "completed_game_ids": ids,
        "completed_games": len(ids), "planned_games": 960,
        "last_game_id": last_game_id, "cumulative_game_cpu_seconds": round(total_cpu, 6),
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(_progress_path(root), progress)
    return progress


def run_preflight(root: str | Path = ROOT) -> dict[str, Any]:
    """Run four already-scheduled games to measure cost; they count in the 960."""
    root = Path(root).resolve()
    frozen = prepare_experiment(root)
    schedule = json.loads((root / ARTIFACT / "config/opening_schedule.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / ARTIFACT / "config/model_manifest.json").read_text(encoding="utf-8"))
    model_hash = manifest["models"][0]["sha256"]
    model_cache: dict[int, ReferencePolicyValueNet] = {}
    rows = []
    for opponent in OPPONENTS:
        opening = schedule["openings"][0]
        color = "B"
        seed = int(schedule["seed"] * 100)
        gid = _game_id(5_000, opponent, 0, color)
        path = _game_path(root, gid)
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            _verify_existing_header(row, games=5_000, opponent=opponent, opening=opening,
                                    color=color, expected_seed=seed, model_sha=model_hash)
        else:
            row = play_match(root=root, games=5_000, opponent=opponent, opening=opening,
                             opening_index=0, candidate_color=color, seed=seed,
                             model_cache=model_cache)
            _store_game(root, row)
        rows.append(row)
    cpu_by_opponent = {r["opponent"]: float(r["total_cpu_seconds"]) for r in rows}
    normal = sum(cpu_by_opponent.values()) * 240
    preflight = {
        "official_pilot_games_included_in_960": [r["game_id"] for r in rows],
        "pilot_cpu_seconds_by_opponent": cpu_by_opponent,
        "pilot_cpu_seconds_total": sum(cpu_by_opponent.values()),
        "estimated_total_cpu_seconds_normal": normal,
        "estimated_cpu_hours": {"optimistic": normal * .75 / 3600,
                                 "normal": normal / 3600,
                                 "conservative": normal * 1.5 / 3600},
        "estimated_wall_hours_at_two_cpu_quota": {
            "optimistic": normal * .75 / 3600 / 2,
            "normal": normal / 3600 / 2,
            "conservative": normal * 1.5 / 3600 / 2},
        "estimate_method": "one scheduled G5000 game per opponent extrapolated to 240 games/opponent; 0.75x/1x/1.5x cost band",
        "estimate_uncertainty": "single game per opponent; A/C search depth and game length cause variance",
        "estimated_completed_games": 960,
        "models_load_verified": len(frozen["manifest"]["models"]),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / ARTIFACT / "results/preflight.json", preflight)
    return preflight


def _project_reference_board(board) -> GomokuBoard:
    converted = GomokuBoard(BOARD_SIZE)
    for move, player in board.states.items():
        converted.place_stone("B" if int(player) == 1 else "W", divmod(int(move), BOARD_SIZE))
    return converted


def _make_candidate(root: Path, games: int, cache: dict[int, ReferencePolicyValueNet]) -> dict[str, Any]:
    if games not in cache:
        node = _model_node(root, games)
        model = ReferencePolicyValueNet(BOARD_SIZE, seed=590_501)
        model.load_policy_param(node["model_state"])
        model.network.eval()
        cache[games] = model
    player = instrumented_player(cache[games].policy_value_fn, c_puct=5.0,
                                n_playout=SIMULATIONS, is_selfplay=False)
    return {"kind": "G", "games": games, "player": player}


def _choose(agent: dict[str, Any], board, color: str, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
    kind = agent["kind"]
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    if kind == "G":
        move, _ = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(move), int(board.width))
        stats = dict(agent["player"].last_search_stats)
        return (row, col), {
            "algorithm": "AlphaZero-reference-PUCT",
            "configured_simulations": SIMULATIONS,
            "root_action_visits": int(stats.get("root_visits", 0)),
            "search_depth": int(stats.get("tree_depth", 0)),
            "search_nodes": int(stats.get("tree_nodes", 0)),
            "network_root_value": stats.get("root_network_value"),
            "cpu_seconds": time.process_time() - started_cpu,
            "elapsed_seconds": time.perf_counter() - started_wall,
            "timed_out": False,
        }
    project = _project_reference_board(board)
    if kind == "A":
        config = SearchConfig(max_depth=69, candidate_radius=2, time_limit_seconds=1e9)
        result = choose_move(project, color, "A", config, int(seed))
    elif kind == "B@50ms":
        config = SearchConfig(time_limit_seconds=0.05)  # historical defaults: depth 4, radius 2
        result = choose_move(project, color, "B", config, int(seed))
    elif kind == "C@50ms":
        config = SearchConfig(max_depth=69, candidate_radius=2, time_limit_seconds=0.05)
        result = choose_move(project, color, "C", config, int(seed))
    elif kind == "C@200ms":
        config = SearchConfig(max_depth=69, candidate_radius=2, time_limit_seconds=0.2)
        result = choose_move(project, color, "C", config, int(seed))
    else:
        raise ValueError(f"unsupported opponent: {kind}")
    return result.move, {
        "algorithm": "rule-A" if kind == "A" else "pure-Minimax" if kind == "B@50ms" else "Alpha-Beta",
        "configured_time_limit_seconds": config.time_limit_seconds if kind != "A" else None,
        "max_depth": config.max_depth,
        "candidate_radius": config.candidate_radius,
        "search_depth": int(result.completed_depth),
        "search_nodes": int(result.nodes),
        "cutoffs": int(result.cutoffs),
        "cpu_seconds": time.process_time() - started_cpu,
        "elapsed_seconds": time.perf_counter() - started_wall,
        "timed_out": bool(result.timed_out),
    }


def _slug(value: str) -> str:
    return value.replace("@", "").replace("-", "")


def _game_path(root: Path, game_id: str) -> Path:
    return root / ARTIFACT / "results/games" / f"{game_id}.json"


def _game_id(games: int, opponent: str, opening_index: int, color: str) -> str:
    return f"G{games}_vs_{_slug(opponent)}_opening_{opening_index:02d}_{color}"


def _verify_existing_header(row: dict[str, Any], *, games: int, opponent: str,
                            opening: dict[str, Any], color: str, expected_seed: int,
                            model_sha: str) -> None:
    expected = {"candidate_games": games, "opponent": opponent,
                "opening_group_id": opening["opening_group_id"],
                "candidate_color": color, "seed": expected_seed,
                "candidate_model_sha256": model_sha}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"existing game conflicts with the frozen schedule: {row.get('game_id')} {key}")
    if int(row.get("move_count", -1)) != len(row.get("moves", [])) or row.get("termination") not in {"five_in_row", "full_board"}:
        raise ValueError("existing game is incomplete")


def play_match(*, root: Path, games: int, opponent: str, opening: dict[str, Any],
               opening_index: int, candidate_color: str, seed: int,
               model_cache: dict[int, ReferencePolicyValueNet]) -> dict[str, Any]:
    if candidate_color not in {"B", "W"}:
        raise ValueError("candidate color must be B or W")
    model_path = _model_path(root, games)
    model_sha = sha_file(model_path)
    candidate = _make_candidate(root, games, model_cache)
    opponent_agent = {"kind": opponent}
    board = make_board(BOARD_SIZE, N_IN_ROW, 0)
    oracle = v5_eval._IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
    stored_moves: list[dict[str, Any]] = []
    expected = "B"
    for move in opening["moves"]:
        color = str(move["player"])
        row, col = int(move["row"]), int(move["col"])
        if color != expected or row * BOARD_SIZE + col not in board.availables or oracle.get((row, col)) is not None:
            raise ValueError("the frozen opening is illegal in replay")
        board.do_move(row * BOARD_SIZE + col)
        oracle.place_stone(color, (row, col))
        if board.game_end()[0] or oracle.check_victory((row, col), color):
            raise ValueError("the frozen opening unexpectedly ends the game")
        stored_moves.append({"player": color, "row": row, "col": col, "opening": True})
        expected = "W" if expected == "B" else "B"
    np.random.seed(int(seed) & 0xFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    game_id = _game_id(games, opponent, opening_index, candidate_color)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    candidate_cpu = opponent_cpu = 0.0
    winner: str | None = None
    termination = ""
    while True:
        color = "B" if int(board.current_player) == 1 else "W"
        is_candidate = color == candidate_color
        agent = candidate if is_candidate else opponent_agent
        move_seed = int(rng.integers(0, 2**31 - 1))
        coord, metrics = _choose(agent, board, color, move_seed)
        row, col = int(coord[0]), int(coord[1])
        flat = row * BOARD_SIZE + col
        if flat not in board.availables or oracle.get((row, col)) is not None:
            raise ValueError(f"{agent['kind']} selected an illegal move {(row, col)}")
        board.do_move(flat)
        oracle.place_stone(color, (row, col))
        actor = f"G{games}" if is_candidate else opponent
        stored_moves.append({"player": color, "row": row, "col": col,
                             "opening": False, "actor": actor, **metrics})
        if is_candidate:
            candidate_cpu += float(metrics["cpu_seconds"])
        else:
            opponent_cpu += float(metrics["cpu_seconds"])
        engine_end, engine_winner = board.game_end()
        oracle_win = oracle.check_victory((row, col), color)
        if bool(engine_end) != bool(oracle_win or oracle.is_full()):
            raise AssertionError("Gomoku engine and independent rule oracle disagree")
        if oracle_win:
            winner, termination = color, "five_in_row"
            if engine_winner != (1 if color == "B" else 2):
                raise AssertionError("engine winner differs from independent rule oracle")
            break
        if oracle.is_full():
            winner, termination = None, "full_board"
            break
    points = 0.5 if winner is None else 1.0 if winner == candidate_color else 0.0
    return {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-8-stage-comparison",
        "game_id": game_id, "candidate": f"G{games}", "candidate_games": games,
        "opponent": opponent, "candidate_model_sha256": model_sha,
        "board_size": BOARD_SIZE, "n_in_row": N_IN_ROW,
        "candidate_config": {"engine": "V6 standard AlphaZero reference", "configured_simulations": SIMULATIONS,
                             "actual_root_action_visits": 399, "c_puct": 5.0,
                             "root_noise": False, "temperature": 1e-3},
        "opponent_config": ({"A": {"engine": "original rule AI", "artificial_wait": False},
                             "B@50ms": {"engine": "pure Minimax", "time_limit_seconds": 0.05,
                                        "max_depth": 4, "candidate_radius": 2, "alpha_beta": False},
                             "C@50ms": {"engine": "Alpha-Beta", "time_limit_seconds": 0.05,
                                        "max_depth": 69, "candidate_radius": 2},
                             "C@200ms": {"engine": "Alpha-Beta", "time_limit_seconds": 0.2,
                                         "max_depth": 69, "candidate_radius": 2}}[opponent]),
        "opening_group_id": opening["opening_group_id"], "opening_index": opening_index,
        "opening_seed": int(opening["seed"]), "opening": opening["moves"],
        "seed": int(seed), "candidate_color": candidate_color, "winner_color": winner,
        "points": points, "termination": termination, "moves": stored_moves,
        "move_count": len(stored_moves), "candidate_cpu_seconds": candidate_cpu,
        "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }


def _expected_games(schedule: dict[str, Any], manifest: dict[str, Any]):
    model_hash = {int(row["games"]): row["sha256"] for row in manifest["models"]}
    for games in MODELS:
        for opponent in OPPONENTS:
            for idx, opening in enumerate(schedule["openings"]):
                for color_index, color in enumerate(("B", "W")):
                    seed = int(schedule["seed"] * 100 + idx * 2 + color_index)
                    yield games, opponent, opening, idx, color, seed, model_hash[games]


def _progress_path(root: Path) -> Path:
    return root / ARTIFACT / "results/progress.json"


def run_batch(root: str | Path = ROOT, *, max_new_games: int = 40) -> dict[str, Any]:
    root = Path(root).resolve()
    if max_new_games < 1:
        raise ValueError("max_new_games must be positive")
    manifest = verify_models(root)
    schedule = prepare_openings(root)
    games_dir = root / ARTIFACT / "results/games"
    games_dir.mkdir(parents=True, exist_ok=True)
    _refresh_progress(root)
    model_cache: dict[int, ReferencePolicyValueNet] = {}
    new_count = 0
    last_game_id = None
    for games, opponent, opening, idx, color, seed, model_sha in _expected_games(schedule, manifest):
        game_id = _game_id(games, opponent, idx, color)
        path = _game_path(root, game_id)
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            _verify_existing_header(row, games=games, opponent=opponent, opening=opening,
                                    color=color, expected_seed=seed, model_sha=model_sha)
            continue
        if new_count >= max_new_games:
            break
        row = play_match(root=root, games=games, opponent=opponent, opening=opening,
                         opening_index=idx, candidate_color=color, seed=seed,
                         model_cache=model_cache)
        _store_game(root, row)
        new_count += 1
        last_game_id = game_id
        progress_path = _progress_path(root)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["last_batch_new_games"] = new_count
        atomic_json(progress_path, progress)
    completed = len(list(games_dir.glob("*.json")))
    _refresh_progress(root, last_game_id)
    result = {"new_games": new_count, "completed_games": completed,
              "planned_games": len(MODELS) * len(OPPONENTS) * OPENING_GROUPS * 2,
              "last_game_id": last_game_id}
    atomic_json(root / ARTIFACT / "results/last_batch.json", result)
    return result


def _bootstrap_ci(values: list[float], *, seed: int) -> list[float]:
    rng = np.random.default_rng(int(seed))
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_REPS, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _summarize(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty condition")
    wins = sum(row["points"] == 1 for row in rows)
    losses = sum(row["points"] == 0 for row in rows)
    draws = len(rows) - wins - losses
    score = float(np.mean([float(row["points"]) for row in rows]))
    black = [row for row in rows if row["candidate_color"] == "B"]
    white = [row for row in rows if row["candidate_color"] == "W"]
    by_opening: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_opening.setdefault(str(row["opening_group_id"]), []).append(row)
    paired = []
    for opening_id, pair in sorted(by_opening.items()):
        if len(pair) != 2 or {row["candidate_color"] for row in pair} != {"B", "W"}:
            raise ValueError(f"incomplete color-swapped pair for opening {opening_id}")
        paired.append(float(np.mean([row["points"] for row in pair])))
    return {"games": len(rows), "paired_openings": len(paired), "wins": int(wins),
            "draws": int(draws), "losses": int(losses),
            "points": float(sum(float(row["points"]) for row in rows)),
            "score_rate": score,
            "black_score_rate": float(np.mean([row["points"] for row in black])),
            "white_score_rate": float(np.mean([row["points"] for row in white])),
            "paired_opening_score_rate": float(np.mean(paired)),
            "paired_opening_bootstrap_ci95": _bootstrap_ci(paired, seed=seed),
            "mean_moves": float(np.mean([row["move_count"] for row in rows])),
            "mean_candidate_cpu_seconds": float(np.mean([row["candidate_cpu_seconds"] for row in rows])),
            "mean_opponent_cpu_seconds": float(np.mean([row["opponent_cpu_seconds"] for row in rows])),
            "mean_candidate_decision_cpu_seconds": float(np.mean([
                m["cpu_seconds"] for row in rows for m in row["moves"]
                if not m.get("opening") and str(m.get("actor", "")).startswith("G") ])),
            "complete": len(rows) == OPENING_GROUPS * 2}


def _replay_game(row: dict[str, Any], schedule: dict[str, Any]) -> None:
    if int(row.get("board_size", -1)) != BOARD_SIZE or int(row.get("n_in_row", -1)) != N_IN_ROW:
        raise ValueError(f"wrong rules in game {row.get('game_id')}")
    opening = schedule["openings"][int(row["opening_index"])]
    if row.get("opening_group_id") != opening["opening_group_id"] or row.get("opening") != opening["moves"]:
        raise ValueError("game opening differs from pre-registered schedule")
    if len(row.get("moves", [])) != int(row.get("move_count", -1)):
        raise ValueError("move_count does not match game record")
    board = make_board(BOARD_SIZE, N_IN_ROW, 0)
    oracle = v5_eval._IndependentRuleBoard(BOARD_SIZE, N_IN_ROW)
    expected_color = "B"
    winner = None
    terminal = False
    for idx, move in enumerate(row["moves"]):
        color = str(move.get("player"))
        coord = (int(move["row"]), int(move["col"]))
        flat = coord[0] * BOARD_SIZE + coord[1]
        if color != expected_color or flat not in board.availables or oracle.get(coord) is not None:
            raise ValueError(f"illegal move at ply {idx + 1} in {row['game_id']}")
        is_opening = idx < OPENING_PLIES
        if bool(move.get("opening")) != is_opening:
            raise ValueError("opening flags do not match registered prefix")
        if is_opening:
            registered = opening["moves"][idx]
            if (color, *coord) != (registered["player"], int(registered["row"]), int(registered["col"])):
                raise ValueError("registered opening sequence was modified")
        else:
            candidate_turn = color == row["candidate_color"]
            expected_actor = f"G{row['candidate_games']}" if candidate_turn else row["opponent"]
            if move.get("actor") != expected_actor:
                raise ValueError("actor/color mismatch")
            if not isinstance(move.get("cpu_seconds"), (int, float)) or float(move["cpu_seconds"]) < 0:
                raise ValueError("missing or invalid per-move time")
            if move.get("search_depth") is None or move.get("search_nodes") is None:
                raise ValueError("per-move search statistics are missing")
            if candidate_turn:
                if int(move.get("configured_simulations", -1)) != SIMULATIONS:
                    raise ValueError("candidate did not use configured PUCT-400")
            elif row["opponent"].startswith("B@"):
                if move.get("algorithm") != "pure-Minimax" or move.get("configured_time_limit_seconds") != 0.05 or int(move.get("max_depth", -1)) != 4:
                    raise ValueError("B@50ms protocol mismatch")
            elif row["opponent"].startswith("C@"):
                expected_seconds = 0.05 if row["opponent"] == "C@50ms" else 0.2
                if (move.get("algorithm") != "Alpha-Beta" or move.get("configured_time_limit_seconds") != expected_seconds
                        or int(move.get("max_depth", -1)) != 69 or int(move.get("candidate_radius", -1)) != 2):
                    raise ValueError("C protocol mismatch")
        board.do_move(flat)
        oracle.place_stone(color, coord)
        engine_end, engine_winner = board.game_end()
        oracle_win = oracle.check_victory(coord, color)
        if bool(engine_end) != bool(oracle_win or oracle.is_full()):
            raise ValueError("rule-engine and independent oracle terminal mismatch")
        expected_color = "W" if color == "B" else "B"
        if oracle_win:
            winner, terminal = color, True
            if engine_winner != (1 if color == "B" else 2):
                raise ValueError("engine winner does not match replay")
            if idx != len(row["moves"]) - 1:
                raise ValueError("moves appear after a terminal win")
            break
        if oracle.is_full():
            winner, terminal = None, True
            if idx != len(row["moves"]) - 1:
                raise ValueError("moves appear after a full-board draw")
            break
    if not terminal:
        raise ValueError("game record ended before a terminal state")
    expected_points = 0.5 if winner is None else 1.0 if winner == row["candidate_color"] else 0.0
    expected_term = "full_board" if winner is None else "five_in_row"
    if row.get("winner_color") != winner or row.get("points") != expected_points or row.get("termination") != expected_term:
        raise ValueError("stored result differs from audited replay")


def audit(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = verify_models(root)
    schedule = prepare_openings(root)
    expected = list(_expected_games(schedule, manifest))
    games_dir = root / ARTIFACT / "results/games"
    found = sorted(games_dir.glob("*.json")) if games_dir.exists() else []
    expected_ids = set()
    by_condition: dict[tuple[int, str], list[dict[str, Any]]] = {}
    expected_meta = {}
    for games, opponent, opening, idx, color, seed, model_sha in expected:
        gid = _game_id(games, opponent, idx, color)
        expected_ids.add(gid)
        expected_meta[gid] = (games, opponent, opening, idx, color, seed, model_sha)
    if len(expected_ids) != 960:
        raise AssertionError(f"schedule should have 960 unique IDs, got {len(expected_ids)}")
    actual_ids = {path.stem for path in found}
    missing, extra = sorted(expected_ids - actual_ids), sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(f"game set mismatch: missing={len(missing)}, extra={len(extra)}")
    for path in found:
        row = json.loads(path.read_text(encoding="utf-8"))
        games, opponent, opening, idx, color, seed, model_sha = expected_meta[path.stem]
        _verify_existing_header(row, games=games, opponent=opponent, opening=opening,
                                color=color, expected_seed=seed, model_sha=model_sha)
        _replay_game(row, schedule)
        by_condition.setdefault((games, opponent), []).append(row)
    summaries = []
    for games in MODELS:
        for opponent in OPPONENTS:
            rows = sorted(by_condition[(games, opponent)], key=lambda r: (r["opening_index"], r["candidate_color"]))
            if len(rows) != 40 or sum(row["candidate_color"] == "B" for row in rows) != 20:
                raise ValueError("condition lacks 40 games or color balance")
            seed = int.from_bytes(hashlib.sha256(f"v6.8:{games}:{opponent}".encode()).digest()[:4], "big")
            summary = _summarize(rows, seed=seed)
            opening_scores = {}
            for opening_id in [o["opening_group_id"] for o in schedule["openings"]]:
                pair = [r for r in rows if r["opening_group_id"] == opening_id]
                opening_scores[opening_id] = float(np.mean([r["points"] for r in pair]))
            summary.update({"candidate": f"G{games}", "candidate_games": games, "opponent": opponent,
                            "candidate_model_sha256": manifest["models"][MODELS.index(games)]["sha256"],
                            "opening_scores": opening_scores})
            atomic_json(root / ARTIFACT / f"results/summaries/G{games}_{_slug(opponent)}.json", summary)
            summaries.append(summary)
    baseline = {(s["opponent"], key): value for s in summaries if s["candidate_games"] == 5_000
                for key, value in s["opening_scores"].items()}
    for summary in summaries:
        deltas = [summary["opening_scores"][key] - baseline[(summary["opponent"], key)]
                  for key in summary["opening_scores"]]
        delta_seed = int.from_bytes(hashlib.sha256(f"v6.8-delta:{summary['candidate_games']}:{summary['opponent']}".encode()).digest()[:4], "big")
        summary["same_opening_delta_vs_G5000"] = float(np.mean(deltas))
        summary["same_opening_delta_vs_G5000_ci95"] = _bootstrap_ci(deltas, seed=delta_seed)
        atomic_json(root / ARTIFACT / f"results/summaries/G{summary['candidate_games']}_{_slug(summary['opponent'])}.json", summary)
    result = {"experiment_id": "wuziq-alphazero-g-v6-8-stage-comparison", "complete": len(found) == 960,
              "planned_games": 960, "audited_games": len(found), "unique_game_ids": len(actual_ids),
              "legal_replays": len(found), "independent_rule_oracle": "v5_evaluation._IndependentRuleBoard",
              "model_count": len(MODELS), "opponent_count": len(OPPONENTS),
              "pairs_per_condition": OPENING_GROUPS, "conditions": len(summaries),
              "color_swapped_conditions": all(s["complete"] and s["games"] == 40 for s in summaries),
              "opening_groups": OPENING_GROUPS,
              "all_conditions_share_same_schedule": True,
              "all_models_use_PUCT400": all(int(m["simulations_per_move"]) == 400 for m in manifest["models"]),
              "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(root / ARTIFACT / "results/audit.json", result)
    return result


def _write_chart(root: Path, summaries: list[dict[str, Any]]) -> None:
    chart_dir = root / ARTIFACT / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
    labels = ["G5000", "G8000", "G10000", "G12000", "G16000", "G20000"]
    colors = {"A": "#2b6cb0", "B@50ms": "#dd6b20", "C@50ms": "#2f855a", "C@200ms": "#c53030"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for opponent in OPPONENTS:
        rows = sorted([s for s in summaries if s["opponent"] == opponent], key=lambda s: s["candidate_games"])
        y = np.asarray([s["score_rate"] for s in rows])
        ci = np.asarray([s["paired_opening_bootstrap_ci95"] for s in rows])
        yerr = np.vstack([np.maximum(0, y - ci[:, 0]), np.maximum(0, ci[:, 1] - y)])
        ax.errorbar(labels, y, yerr=yerr, marker="o", capsize=3, linewidth=2,
                    color=colors[opponent], label=opponent)
    ax.set_ylim(0, 1)
    ax.set_ylabel("得分率（胜1、平0.5、负0）")
    ax.set_xlabel("G累计自我对弈局数")
    ax.set_title("V6.8：六个冻结检查点在统一PUCT-400下的配对对战")
    ax.grid(alpha=.25)
    ax.legend(title="对手")
    fig.tight_layout()
    fig.savefig(chart_dir / "six_stage_score_trend.png", dpi=170)
    plt.close(fig)


def create_snapshot(root: str | Path = ROOT, *, label: str) -> dict[str, Any]:
    """Create a self-verifying archive of only the state needed to resume V6.8."""
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    include = []
    for subdir in ("config", "models", "results"):
        folder = artifact / subdir
        if folder.exists():
            include.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix != ".tmp")
    for path in (artifact / "REPORT.md", root / "src/research/alphazero_g/v6_8_tournament.py",
                 root / "scripts/run_g_v6_8.py", root / "scripts/restore_g_v6_8_snapshot.py"):
        if path.is_file():
            include.append(path)
    include = sorted(set(include))
    files = [{"path": str(path.relative_to(root)), "sha256": sha_file(path), "bytes": path.stat().st_size}
             for path in include]
    progress_path = _progress_path(root)
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"completed_games": 0}
    manifest = {"format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-8-stage-comparison",
                "label": label, "source_commit": SOURCE_COMMIT,
                "completed_games": int(progress.get("completed_games", 0)),
                "planned_games": 960, "files": files,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    output = artifact / "recovery/snapshots" / f"{label}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("SNAPSHOT_MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        for path in include:
            archive.write(path, path.relative_to(root).as_posix())
    os.replace(temp, output)
    return {"path": str(output), "sha256": sha_file(output), "bytes": output.stat().st_size,
            "completed_games": manifest["completed_games"], "file_count": len(files)}


def restore_snapshot_archive(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("restore destination must be empty")
    with zipfile.ZipFile(archive_path, "r") as archive:
        manifest = json.loads(archive.read("SNAPSHOT_MANIFEST.json"))
        archive.extractall(destination)
    checked = 0
    for item in manifest["files"]:
        path = destination / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha_file(path) != item["sha256"]:
            raise ValueError(f"restored snapshot file failed SHA/size check: {item['path']}")
        checked += 1
    progress = destination / ARTIFACT / "results/progress.json"
    actual_games = int(json.loads(progress.read_text(encoding="utf-8")).get("completed_games", 0)) if progress.is_file() else 0
    if actual_games != int(manifest["completed_games"]):
        raise ValueError("restored snapshot progress count differs from manifest")
    return {"valid": True, "label": manifest["label"], "completed_games": actual_games,
            "checked_files": checked, "archive_sha256": sha_file(archive_path)}


def report(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    audit_result = json.loads((root / ARTIFACT / "results/audit.json").read_text(encoding="utf-8"))
    if not audit_result.get("complete"):
        raise ValueError("cannot report until the 960-game audit passes")
    summary_dir = root / ARTIFACT / "results/summaries"
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(summary_dir.glob("*.json"))]
    if len(summaries) != 24:
        raise ValueError("expected 24 complete checkpoint/opponent summaries")
    _write_chart(root, summaries)
    manifest = json.loads((root / ARTIFACT / "config/model_manifest.json").read_text(encoding="utf-8"))
    schedule = json.loads((root / ARTIFACT / "config/opening_schedule.json").read_text(encoding="utf-8"))
    backup_status_path = root / ARTIFACT / "results/drive_backup_status.json"
    backup_status = json.loads(backup_status_path.read_text(encoding="utf-8")) if backup_status_path.is_file() else {}
    by = {(s["candidate_games"], s["opponent"]): s for s in summaries}
    rows = []
    stage_lines = []
    for games in MODELS:
        cells = [f"{by[(games, op)]['score_rate']:.1%}" for op in OPPONENTS]
        rows.append(f"| {games} | " + " | ".join(cells) + " |")
        avg = float(np.mean([by[(games, op)]["score_rate"] for op in OPPONENTS]))
        stage_lines.append((games, avg))
    max_stage = max(stage_lines, key=lambda x: x[1])
    stage_deltas = [(stage_lines[i][0], stage_lines[i + 1][0], stage_lines[i + 1][1] - stage_lines[i][1])
                    for i in range(len(stage_lines) - 1)]
    largest_gain = max(stage_deltas, key=lambda x: x[2])
    largest_drop = min(stage_deltas, key=lambda x: x[2])
    delta_lines = []
    for op in OPPONENTS:
        first, last = by[(5_000, op)], by[(20_000, op)]
        delta = last["same_opening_delta_vs_G5000"]
        ci = last["same_opening_delta_vs_G5000_ci95"]
        delta_lines.append(f"- **{op}**：G5000 {first['score_rate']:.1%} → G20000 {last['score_rate']:.1%}；同开局差 {delta:+.1%}，配对95%自助区间 [{ci[0]:+.1%}, {ci[1]:+.1%}]。")
    table = ["| G累计训练局数 | A | B@50ms | C@50ms | C@200ms |", "|---:|---:|---:|---:|---:|", *rows]
    detail = ["| 模型 | 对手 | 胜-平-负 | 得分率 | 配对95%区间 | 执黑 | 执白 | G CPU秒/局 | 对手CPU秒/局 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for games in MODELS:
        for op in OPPONENTS:
            s = by[(games, op)]
            lo, hi = s["paired_opening_bootstrap_ci95"]
            detail.append(f"| G{games} | {op} | {s['wins']}-{s['draws']}-{s['losses']} | {s['score_rate']:.1%} | [{lo:.1%}, {hi:.1%}] | {s['black_score_rate']:.1%} | {s['white_score_rate']:.1%} | {s['mean_candidate_cpu_seconds']:.2f} | {s['mean_opponent_cpu_seconds']:.2f} |")
    files = root / ARTIFACT
    training_composite = {str(g): round(avg, 6) for g, avg in stage_lines}
    all_games = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((files / "results/games").glob("*.json"))]
    total_cpu_hours = sum(float(row.get("total_cpu_seconds", 0)) for row in all_games) / 3600
    scores_csv = files / "results/scores.csv"
    with scores_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("games,opponent,games_played,wins,draws,losses,score_rate,ci95_low,ci95_high,black_score_rate,white_score_rate,same_opening_delta_vs_G5000,delta_ci95_low,delta_ci95_high,mean_candidate_cpu_seconds,mean_opponent_cpu_seconds\n")
        for s in sorted(summaries, key=lambda x: (x["candidate_games"], OPPONENTS.index(x["opponent"]))):
            lo, hi = s["paired_opening_bootstrap_ci95"]
            dlo, dhi = s["same_opening_delta_vs_G5000_ci95"]
            handle.write(f"{s['candidate_games']},{s['opponent']},{s['games']},{s['wins']},{s['draws']},{s['losses']},{s['score_rate']:.8f},{lo:.8f},{hi:.8f},{s['black_score_rate']:.8f},{s['white_score_rate']:.8f},{s['same_opening_delta_vs_G5000']:.8f},{dlo:.8f},{dhi:.8f},{s['mean_candidate_cpu_seconds']:.6f},{s['mean_opponent_cpu_seconds']:.6f}\n")
    text = f"""# WuziQ AlphaZero G V6.8：六阶段棋力成长对比

## 实验登记与环境

- 基线：V6.7 提交 `{SOURCE_COMMIT}`；新分支：`research/alphazero-g-v6-8-stage-comparison`。
- 本轮仅评测，不训练、不更新权重。6个模型全部冻结，使用相同AlphaZero参考实现、PUCT配置和9×9连五规则。
- 6个阶段：{', '.join('G'+str(x) for x in MODELS)}；每阶段 {len(OPPONENTS)} 个对手，每条件20组开局、换色40局；总计960局。
- 所有条件共用{OPENING_GROUPS}个合法六手开局；按D4对称归一化，与本机可检索历史实验开局重叠数为0。开局表SHA-256：`{schedule['opening_schedule_sha256']}`。
- G模型每步配置400次PUCT模拟（V6参考实现根扩展方式下记录399次根边访问）。温度、根噪声、候选网络均固定，未作训练或调参。
- 资源预检：3个可见逻辑CPU、cgroup最多2核时间、内存上限8GiB；GPU不可用（PyTorch为CPU构建）；960局棋手进程累计CPU约{total_cpu_hours:.2f}小时，按逐局CPU实测求和。

## 检查点身份

| 模型 | 局数 | 累计训练状态 | 更新数 | PUCT/步 | 权重SHA-256 |
|---|---:|---:|---:|---:|---|
""" + "\n".join(f"| G{m['games']} | {m['games']} | {m['environment_states']:,} | {m['optimizer_updates']:,} | {m['simulations_per_move']} | `{m['sha256']}` |" for m in manifest["models"]) + f"""

## 对手设定与结果

- A：原始规则AI，自然计算完成即落子，没有人为等待。
- B@50ms：原纯Minimax，无Alpha-Beta剪枝，50ms/步；沿用历史`SearchConfig`默认深度4、候选半径2。
- C@50ms、C@200ms：Alpha-Beta，`max_depth=69`、`candidate_radius=2`，超时返回最后完整层。
- 区间以20个开局组为配对单位做10,000次bootstrap，描述性95%区间；不是独立样本的普通二项区间。每条件仅40局，作为探索性评测。

### 核心成绩表

""" + "\n".join(table) + "\n\n### 逐项成绩与成本\n\n" + "\n".join(detail) + "\n\n" + f"""### G5000到G20000同开局差异

""" + "\n".join(delta_lines) + f"""

![六阶段得分率曲线](charts/six_stage_score_trend.png)

## 成长趋势解释

四个不同强度对手的得分率等权平均仅作为描述性摘要，不代替分对手结论：{training_composite}。平均值最高的阶段为 G{max_stage[0]}（{max_stage[1]:.1%}）。四对手等权均值最大阶段性上升为G{largest_gain[0]}→G{largest_gain[1]}（{largest_gain[2]:+.1%}），最大阶段性下降为G{largest_drop[0]}→G{largest_drop[1]}（{largest_drop[2]:+.1%}）。四条曲线若有起伏，应优先结合配对区间和执黑执白差异解释；本次20组开局不足以把小幅波动认定为稳定成长或退步。

- 每个对手的G5000→G20000同开局差和95%区间列在上表。区间跨0表示当前样本不足以确认差异方向。
- C@200ms是否仍是瓶颈，按独立得分率曲线和对局记录判断；不能用低预算C或A的成绩代替。
- B@50ms维持V6历史配置（深度4）；因此不能与采用69层截止迭代加深的C对手混为相同搜索深度。
- 每步耗时、搜索深度、节点、剪枝/超时以及PUCT根访问数记录在960份逐局JSON的逐手数据中。

## 完整性验收与复现

- 审计结果：{audit_result['audited_games']}局逐手合法回放，独立规则oracle核对终局；唯一ID {audit_result['unique_game_ids']}；24个条件各40局并按同开局换色；审计状态 `{'通过' if audit_result['complete'] else '失败'}`。
- 原始记录：`results/games/`；开局和模型清单：`config/`；汇总：`results/summaries/`；完整审计：`results/audit.json`。
- 可分析的汇总表：`results/scores.csv`；包含24个条件的胜平负、得分率、配对区间、颜色成绩、同开局G5000差值及CPU成本。
- 复现：`PYTHONPATH=. python scripts/run_g_v6_8.py --max-new-games 40` 重复运行会跳过已完成且配置一致的棋局；完成后 `PYTHONPATH=. python scripts/run_g_v6_8.py --audit` 与 `--report`。
- Drive恢复目录：{backup_status.get('folder_url', '最终恢复链接见results/drive_backup_status.json')}。快照经下载SHA-256核验并在空目录恢复后登记；本地与远端提交信息记在`results/drive_backup_status.json`。
"""
    report_path = files / "REPORT.md"
    report_path.write_text(text, encoding="utf-8")
    return {"report": str(report_path), "scores_csv": str(scores_csv), "audit": audit_result, "composite": training_composite,
            "highest_average_stage": {"games": max_stage[0], "score_rate": max_stage[1]},
            "conditions": len(summaries)}
