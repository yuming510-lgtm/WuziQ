"""Resumable, paired B/C/D 9x9 high-compute Gomoku experiment (V4C)."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from ..ai import SearchConfig, SearchResult, choose_move
from ..gomoku import GomokuBoard, Player
from .checkpoint import append_game, load_game_records, write_json_atomic
from .mcts_d2 import MCTSAgent

_MATCHUPS = (("B", "C"), ("B", "D"), ("C", "D"))
_MODELS = {"B", "C", "D"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def derive_seed(base_seed: int, *parts: object) -> int:
    """Stable unsigned 64-bit seed, independent of interpreter hash randomization."""
    payload = json.dumps([int(base_seed), *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def load_config(config_source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load the immutable V4C schedule and ensure the baseline engine is untouched."""
    if isinstance(config_source, dict):
        config = deepcopy(config_source)
    else:
        config = json.loads(Path(config_source).read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported V4C config schema")
    if config.get("experiment_id") != "wuziq-bcd-high-compute-v4c-2026-10-20":
        raise ValueError("Unexpected V4C experiment ID")
    if int(config.get("board_size", 0)) != 9 or tuple(config.get("models", ())) != ("B", "C", "D"):
        raise ValueError("V4C is frozen to B/C/D on a 9x9 board")
    if tuple(tuple(pair) for pair in config.get("matchups", ())) != _MATCHUPS:
        raise ValueError("V4C requires B-C, B-D, and C-D in that order")
    if int(config.get("games_per_matchup", 0)) != 6 or int(config.get("opening_plies", 0)) != 4:
        raise ValueError("V4C requires six games per matchup and four opening plies")
    if tuple(int(seed) for seed in config.get("opening_seeds", ())) != (731091, 731092, 731093):
        raise ValueError("V4C must use registered opening IDs 731091, 731092, 731093")
    if config.get("budgets_seconds_by_model") != {"B": 1.0, "C": 0.2, "D": 50.0}:
        raise ValueError("V4C budgets are fixed to B=1s, C=200ms, D=50s")
    if config.get("inference_limits", {}).get("planned_games") != 18 or config.get("inference_limits", {}).get("reuse_historical_games") is not False:
        raise ValueError("V4C must contain 18 new games and no historical reuse")
    if int(config["ab_search"].get("max_depth", 0)) < 81:
        raise ValueError("B/C iterative deepening must be allowed through the empty board")
    if float(config["d_search"].get("time_limit_seconds", 0)) != 50.0:
        raise ValueError("D must use the 50-second budget")
    if int(config["d_search"].get("max_depth", 0)) != 4 or not config["d_search"].get("mcts_d2_node_reuse"):
        raise ValueError("D must retain the V4B MCTS configuration and tree reuse")
    catalog = config.get("opening_catalog", {})
    if set(catalog) != {"731091", "731092", "731093"}:
        raise ValueError("V4C opening catalog does not match the registered opening set")
    for seed, stones in catalog.items():
        if len(stones) != 4:
            raise ValueError(f"Opening {seed} must contain exactly four plies")
        board = GomokuBoard(9)
        for index, stone in enumerate(stones):
            if len(stone) != 3 or str(stone[0]) != ("B" if index % 2 == 0 else "W"):
                raise ValueError(f"Opening {seed} must alternate B-W-B-W")
            coord = int(stone[1]), int(stone[2])
            board.place_stone(str(stone[0]), coord)
            if board.check_victory(coord, str(stone[0])):
                raise ValueError(f"Opening {seed} is terminal")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in config.get("engine_sha256", {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise ValueError(f"Frozen V4/V4B engine source changed: {relative}")
    return config


def _opening(config: dict[str, Any], seed: int) -> list[tuple[Player, tuple[int, int]]]:
    return [(str(player), (int(row), int(col))) for player, row, col in config["opening_catalog"][str(seed)]]


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the three registered pairs into six color-balanced games each."""
    frozen = load_config(config)
    schedule: list[dict[str, Any]] = []
    for pair in _MATCHUPS:
        canonical = tuple(sorted(pair))
        for opening_index, opening_seed in enumerate(frozen["opening_seeds"]):
            for color_swap in (False, True):
                black, white = canonical if not color_swap else (canonical[1], canonical[0])
                identifier = f"v4c-9x9-{'-'.join(model.lower() for model in canonical)}-o{opening_seed}-s{int(color_swap)}"
                schedule.append({
                    "experiment_id": frozen["experiment_id"],
                    "game_id": identifier,
                    "board_size": 9,
                    "matchup": canonical,
                    "opening_index": opening_index,
                    "opening_seed": int(opening_seed),
                    "color_swap": color_swap,
                    "black_model": black,
                    "white_model": white,
                    "pair_id": f"v4c-9x9-{'-'.join(model.lower() for model in canonical)}-o{opening_seed}",
                    "game_seed": derive_seed(int(frozen["base_seed"]), "v4c-game", identifier),
                    "opening": [
                        {"player": player, "row": coord[0], "col": coord[1]}
                        for player, coord in _opening(frozen, int(opening_seed))
                    ],
                    "budget_seconds_by_model": {model: float(frozen["budgets_seconds_by_model"][model]) for model in canonical},
                    "source_type": "new",
                })
    identifiers = [row["game_id"] for row in schedule]
    if len(schedule) != 18 or len(identifiers) != len(set(identifiers)):
        raise ValueError("V4C schedule must contain 18 unique game IDs")
    return schedule


def settings_for_model(config: dict[str, Any], model: str, occupied: int) -> SearchConfig:
    model = model.upper()
    if model not in _MODELS:
        raise ValueError(f"Unknown V4C model: {model}")
    if not 0 <= occupied < 81:
        raise ValueError("occupied must leave at least one empty cell")
    if model == "D":
        return SearchConfig(**config["d_search"])
    search = dict(config["ab_search"])
    search["time_limit_seconds"] = float(config["budgets_seconds_by_model"][model])
    search["max_depth"] = 81 - occupied
    return SearchConfig(**search)


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _legal_coordinate(board: GomokuBoard, move: object) -> bool:
    return (
        isinstance(move, (tuple, list)) and len(move) == 2
        and type(move[0]) is int and type(move[1]) is int
        and board.in_bounds(move[0], move[1]) and board.get((move[0], move[1])) is None
    )


def _move_record(result: SearchResult, ply: int, player: Player, model: str, timed_out: bool) -> dict[str, Any]:
    return {
        "ply": ply, "player": player, "model": model,
        "elapsed_seconds": float(result.elapsed_seconds),
        "root_value": int(result.root_value),
        "nodes": int(result.nodes), "completed_depth": int(result.completed_depth),
        "tree_depth": int(result.completed_depth), "cutoffs": int(result.cutoffs),
        "simulations": int(result.simulations), "rollout_plies": int(result.rollout_plies),
        "terminal_simulations": int(result.terminal_simulations),
        "truncated_simulations": int(result.truncated_simulations),
        "root_candidates": int(result.root_candidates), "root_min_visits": int(result.root_min_visits),
        "reused_nodes": int(result.reused_nodes), "timed_out": bool(timed_out), "opening": False,
    }


def _opening_record(player: Player, coord: tuple[int, int], ply: int) -> dict[str, Any]:
    return {
        "ply": ply, "player": player, "model": None, "row": coord[0], "col": coord[1],
        "elapsed_seconds": 0.0, "root_value": 0, "nodes": 0, "completed_depth": 0,
        "tree_depth": 0, "cutoffs": 0, "simulations": 0, "rollout_plies": 0,
        "terminal_simulations": 0, "truncated_simulations": 0, "root_candidates": 0,
        "root_min_visits": 0, "reused_nodes": 0, "timed_out": False, "opening": True,
    }


def run_game(spec: dict[str, Any], config: dict[str, Any], implementation_commit: str = "unknown") -> dict[str, Any]:
    """Run one complete game and retain D's tree across its moves."""
    frozen = load_config(config)
    board = GomokuBoard(9)
    moves: list[dict[str, Any]] = []
    for ply, (player, coord) in enumerate(_opening(frozen, int(spec["opening_seed"]))):
        board.place_stone(player, coord)
        moves.append(_opening_record(player, coord, ply))

    d_agent = MCTSAgent()
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    for _ in range(81 - len(board.history)):
        model = spec["black_model"] if current == "B" else spec["white_model"]
        settings = settings_for_model(frozen, model, len(board.history))
        seed = derive_seed(int(spec["game_seed"]), "move", len(board.history), model, current)
        result = d_agent.choose_move(board, current, settings, seed) if model == "D" else choose_move(board, current, model, settings, seed)
        budget = float(frozen["budgets_seconds_by_model"][model])
        elapsed = float(result.elapsed_seconds)
        timed_out = bool(result.timed_out or elapsed > budget)
        opponent = _other(current)
        opponent_model = spec["black_model"] if opponent == "B" else spec["white_model"]
        move = _move_record(result, len(moves), current, model, timed_out)
        if not _legal_coordinate(board, result.move):
            move.update({
                "attempted_move": list(result.move) if isinstance(result.move, (tuple, list)) else str(result.move),
                "illegal": True,
            })
            moves.append(move)
            winner_color, winner_model, termination = opponent, opponent_model, "illegal_move_forfeit"
            break

        coord = int(result.move[0]), int(result.move[1])
        board.place_stone(current, coord)
        move.update({"row": coord[0], "col": coord[1]})
        moves.append(move)
        if board.check_victory(coord, current):
            winner_color, winner_model, termination = current, model, "five_in_row"
            break
        if board.is_full():
            termination = "full_board"
            break
        current = opponent

    return {
        **spec,
        "matchup": list(spec["matchup"]),
        "implementation_commit": implementation_commit,
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "draw" if winner_color is None else f"{winner_color.lower()}_win",
        "termination": termination,
        "timeout_count": sum(bool(move.get("timed_out")) for move in moves),
        "search_nodes_total": sum(int(move["nodes"]) for move in moves),
        "simulations_total": sum(int(move["simulations"]) for move in moves),
        "rollout_plies_total": sum(int(move["rollout_plies"]) for move in moves),
        "terminal_simulations_total": sum(int(move["terminal_simulations"]) for move in moves),
        "truncated_simulations_total": sum(int(move["truncated_simulations"]) for move in moves),
        "tree_depth_max": max((int(move["tree_depth"]) for move in moves), default=0),
        "thinking_seconds_total": sum(float(move["elapsed_seconds"]) for move in moves),
    }


def _load_unique_games(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = load_game_records(path)
    ids = [str(row["game_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("V4C JSONL contains duplicate game IDs")
    return rows, set(ids)


def run_experiment(
    config_source: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    max_games: int | None = None,
    implementation_commit: str = "unknown",
) -> dict[str, Any]:
    """Resume by unique ID; fsync each complete, replay-validated game before progress."""
    config = load_config(config_source)
    if max_games is not None and max_games < 0:
        raise ValueError("max_games cannot be negative")
    from .bcd_high_compute_v4c_analysis import validate_game_record, validate_records

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing V4C output belongs to a different frozen config")
    if not config_path.exists():
        write_json_atomic(config_path, config)
    config_hash = _sha256(_canonical_json(config).encode("utf-8"))
    records_path = root / "9x9_games.jsonl"
    if not records_path.exists():
        records_path.touch()
        with records_path.open("rb") as handle:
            import os
            os.fsync(handle.fileno())
    records, completed = _load_unique_games(records_path)
    schedule = build_schedule(config)
    by_id = {str(spec["game_id"]): spec for spec in schedule}
    unexpected = completed - set(by_id)
    if unexpected:
        raise ValueError(f"V4C output has unscheduled game IDs: {sorted(unexpected)[:3]}")
    validate_records(records, config, require_complete=False)

    progress_path = root / "progress.json"
    previous_elapsed = 0.0
    first_started = _utc_now()
    if progress_path.exists():
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        if prior.get("experiment_id") != config["experiment_id"] or prior.get("config_sha256") != config_hash:
            raise ValueError("V4C progress metadata does not match config")
        if not set(prior.get("completed_ids", ())).issubset(completed):
            raise ValueError("V4C progress claims results not present in durable records")
        if prior.get("implementation_commit") != implementation_commit:
            raise ValueError("V4C resume code version differs from the previous run")
        previous_elapsed = float(prior.get("elapsed_seconds_total", 0.0))
        first_started = str(prior.get("first_started_at_utc", first_started))

    progress: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "implementation_commit": implementation_commit,
        "planned_games": len(schedule),
        "completed_games": len(completed),
        "completed_ids": sorted(completed),
        "all_games_new": True,
        "first_started_at_utc": first_started,
        "last_started_at_utc": _utc_now(),
        "elapsed_seconds_total": previous_elapsed,
        "updated_at_utc": _utc_now(),
    }
    write_json_atomic(progress_path, progress)
    started = time.perf_counter()
    invocation_games = 0
    for spec in schedule:
        identifier = str(spec["game_id"])
        if identifier in completed:
            continue
        if max_games is not None and invocation_games >= max_games:
            break
        record = run_game(spec, config, implementation_commit=implementation_commit)
        if str(record.get("game_id")) != identifier:
            raise ValueError("V4C runner returned a mismatched unique ID")
        validate_game_record(record, config, spec)
        append_game(records_path, record, completed)
        records.append(record)
        invocation_games += 1
        elapsed_so_far = time.perf_counter() - started
        progress.update({
            "completed_games": len(completed),
            "completed_ids": sorted(completed),
            "last_game_id": identifier,
            "elapsed_seconds_total": previous_elapsed + elapsed_so_far,
            "last_started_at_utc": progress["last_started_at_utc"],
            "updated_at_utc": _utc_now(),
        })
        write_json_atomic(progress_path, progress)
        print(json.dumps({
            "completed": len(completed), "planned": len(schedule), "last_game_id": identifier,
            "winner": record.get("winner_model"), "outcome": record.get("outcome"),
            "thinking_seconds": record.get("thinking_seconds_total"),
            "simulations": record.get("simulations_total"),
        }, ensure_ascii=False), flush=True)

    invocation_elapsed = time.perf_counter() - started
    progress.update({
        "remaining_games": len(schedule) - len(completed),
        "complete": len(completed) == len(schedule),
        "games_completed_this_run": invocation_games,
        "elapsed_seconds_this_run": invocation_elapsed,
        "elapsed_seconds_total": previous_elapsed + invocation_elapsed,
        "completed_at_utc": _utc_now() if len(completed) == len(schedule) else None,
        "updated_at_utc": _utc_now(),
    })
    write_json_atomic(progress_path, progress)
    return progress


def repository_commit(project_root: str | Path | None = None) -> str:
    root = Path(project_root or Path(__file__).resolve().parents[2])
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def load_records(results_dir: str | Path) -> list[dict[str, Any]]:
    return _load_unique_games(Path(results_dir) / "9x9_games.jsonl")[0]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bcd_high_compute_v4c.json")
    parser.add_argument("--output-dir", default="research/v4c/results")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--implementation-commit")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    commit = args.implementation_commit or repository_commit()
    progress = run_experiment(args.config, args.output_dir, max_games=args.max_games, implementation_commit=commit)
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
