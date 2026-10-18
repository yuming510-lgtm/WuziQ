"""Deterministic, resumable paired tournaments for the MCTS V2 study."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Iterable

from ..ai import SearchConfig, SearchResult, choose_move
from ..gomoku import Coordinate, GomokuBoard, Player
from .checkpoint import append_game, load_game_records, write_json_atomic
from .mcts_d2 import MCTSAgent


_MODEL_NAMES = {"A", "B", "C", "D1", "D2"}


def _current_code_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed, independent of Python hash randomization."""

    payload = json.dumps(
        [int(base_seed), *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def paired_opening(
    board_size: int,
    seed: int,
    opening_plies: int = 4,
    color_swap: bool = False,
) -> list[tuple[Player, Coordinate]]:
    """Create a reproducible B-W-B-W opening shared by both games in a pair.

    ``color_swap`` remains accepted for compatibility with saved schedules, but
    the opening itself never changes sides. The paired schedule swaps only the
    model assigned to black and white, which also swaps who makes the fifth ply.
    """

    if board_size < 5:
        raise ValueError("Board size must be at least 5")
    if opening_plies != 4:
        raise ValueError("V2 paired openings use exactly four plies")

    rng = random.Random(seed)
    center = (board_size // 2, board_size // 2)
    opening: list[tuple[Player, Coordinate]] = [("B", center)]
    for ply in range(1, opening_plies):
        nearby: set[Coordinate] = set()
        for _, (row, col) in opening:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    coord = row + dr, col + dc
                    if 0 <= coord[0] < board_size and 0 <= coord[1] < board_size:
                        if all(coord != used for _, used in opening):
                            nearby.add(coord)
        if not nearby:
            nearby = {
                (row, col)
                for row in range(board_size)
                for col in range(board_size)
                if all((row, col) != used for _, used in opening)
            }
        player: Player = "W" if ply % 2 else "B"
        opening.append((player, rng.choice(sorted(nearby))))

    return opening


def _phase_slug(value: object) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    if not slug:
        raise ValueError("phase name must contain letters or digits")
    return slug


def _model_budget_map(phase: dict[str, Any], matchup: tuple[str, str]) -> dict[str, float]:
    supplied = phase.get("budget_seconds_by_model", {})
    if not isinstance(supplied, dict):
        raise ValueError("budget_seconds_by_model must be an object")
    budgets: dict[str, float] = {}
    for model in matchup:
        raw = supplied.get(model, supplied.get("default"))
        if raw is None:
            raise ValueError(f"Missing time budget for model {model}")
        budget = float(raw)
        if budget < 0:
            raise ValueError("time budgets cannot be negative")
        budgets[model] = budget
    return budgets


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand pre-registered phases into unique, paired color-swapped games."""

    experiment_id = str(config["experiment_id"])
    base_seed = int(config["base_seed"])
    opening_plies = int(config.get("opening_plies", 4))
    if opening_plies != 4:
        raise ValueError("opening_plies must be 4 for the V2 opening policy")
    phases = config.get("phases", [])
    if not phases:
        raise ValueError("config must define at least one experiment phase")

    schedule: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for phase in phases:
        phase_name = _phase_slug(phase["name"])
        track = str(phase["track"]).lower()
        if track not in {"fair", "extreme"}:
            raise ValueError("track must be fair or extreme")
        board_size = int(phase["board_size"])
        games_per_matchup = int(phase["games_per_matchup"])
        if board_size < 5:
            raise ValueError("board_size must be at least 5")
        if games_per_matchup < 2 or games_per_matchup % 2:
            raise ValueError("games_per_matchup must be an even number of at least 2")
        matchups = phase.get("matchups", [])
        if not matchups:
            raise ValueError(f"phase {phase_name} has no matchups")
        normalized: list[tuple[str, str]] = []
        for pair in matchups:
            if len(pair) != 2:
                raise ValueError("each matchup must contain exactly two models")
            first, second = (str(model).upper() for model in pair)
            if first not in _MODEL_NAMES or second not in _MODEL_NAMES or first == second:
                raise ValueError(f"invalid matchup: {pair}")
            normalized.append(tuple(sorted((first, second))))
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"phase {phase_name} contains duplicate matchups")

        for pair in sorted(normalized):
            budgets = _model_budget_map(phase, pair)
            if track == "fair" and budgets[pair[0]] != budgets[pair[1]]:
                raise ValueError("fair-track models must receive the same time budget")
            if track == "extreme":
                if set(pair) != {"C", "D2"}:
                    raise ValueError("extreme track is reserved for D2 versus V1 C")
                if budgets["C"] != 0.05:
                    raise ValueError("extreme-track C budget must remain at V1's 0.05 seconds")

            for opening_index in range(games_per_matchup // 2):
                opening_seed = derive_seed(
                    base_seed, "opening", phase_name, board_size, opening_index
                )
                for color_swap in (False, True):
                    black_model, white_model = pair
                    if color_swap:
                        black_model, white_model = white_model, black_model
                    budget_label = "-".join(
                        f"{model.lower()}-{int(round(budgets[model] * 1000)):05d}ms"
                        for model in pair
                    )
                    identifier = (
                        f"{experiment_id}-{board_size}x{board_size}-{track}-"
                        f"{phase_name}-{'-'.join(pair)}-{budget_label}-"
                        f"o{opening_index:04d}-s{int(color_swap)}"
                    )
                    if identifier in seen_ids:
                        raise ValueError(f"Duplicate scheduled game_id: {identifier}")
                    seen_ids.add(identifier)
                    schedule.append(
                        {
                            "game_id": identifier,
                            "experiment_id": experiment_id,
                            "phase_name": phase_name,
                            "track": track,
                            "board_size": board_size,
                            "matchup": pair,
                            "pair_id": (
                                f"{board_size}x{board_size}-{phase_name}-"
                                f"{'-'.join(pair)}-o{opening_index:04d}"
                            ),
                            "opening_index": opening_index,
                            "opening_plies": opening_plies,
                            "color_swap": color_swap,
                            "black_model": black_model,
                            "white_model": white_model,
                            "opening_seed": opening_seed,
                            "game_seed": derive_seed(base_seed, "game", identifier),
                            "budget_seconds_by_model": budgets,
                        }
                    )
    return schedule


def _search_config(config: dict[str, Any], spec: dict[str, Any], model: str) -> SearchConfig:
    values = dict(config.get("search", {}))
    values["time_limit_seconds"] = float(spec["budget_seconds_by_model"][model])
    return SearchConfig(**values)


def _opening_log(player: Player, coord: Coordinate, ply: int) -> dict[str, Any]:
    return {
        "ply": ply,
        "player": player,
        "model": None,
        "row": coord[0],
        "col": coord[1],
        "elapsed_seconds": 0.0,
        "nodes": 0,
        "completed_depth": 0,
        "tree_depth": 0,
        "cutoffs": 0,
        "simulations": 0,
        "rollout_plies": 0,
        "terminal_simulations": 0,
        "truncated_simulations": 0,
        "root_candidates": 0,
        "root_min_visits": 0,
        "reused_nodes": 0,
        "timed_out": False,
        "opening": True,
    }


def _legal_coordinate(board: GomokuBoard, move: object) -> bool:
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        return False
    row, col = move
    return (
        type(row) is int
        and type(col) is int
        and board.in_bounds(row, col)
        and board.get((row, col)) is None
    )


def run_game(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Play one complete game, recording every move and search metric."""

    size = int(spec["board_size"])
    board = GomokuBoard(size)
    opening = paired_opening(
        size,
        int(spec["opening_seed"]),
        int(spec.get("opening_plies", config.get("opening_plies", 4))),
        bool(spec["color_swap"]),
    )
    moves: list[dict[str, Any]] = []
    for ply, (player, coord) in enumerate(opening):
        board.place_stone(player, coord)
        moves.append(_opening_log(player, coord, ply))

    models = tuple(spec["matchup"])
    settings = {model: _search_config(config, spec, model) for model in models}
    agents = {model: MCTSAgent() for model in models if model == "D2"}
    current_player: Player = _other(opening[-1][0])
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"

    for _ in range(size * size - len(board.history)):
        model = spec["black_model"] if current_player == "B" else spec["white_model"]
        budget = settings[model].time_limit_seconds
        seed = derive_seed(
            int(spec["game_seed"]), "move", len(board.history), model, current_player
        )
        if model == "D2":
            result = agents[model].choose_move(board, current_player, settings[model], seed)
        else:
            result = choose_move(board, current_player, model, settings[model], seed)
        elapsed = float(result.elapsed_seconds)
        timed_out = bool(result.timed_out or elapsed > budget)
        opponent_color = _other(current_player)
        opponent_model = (
            spec["black_model"] if opponent_color == "B" else spec["white_model"]
        )
        if not _legal_coordinate(board, result.move):
            moves.append(
                {
                    "ply": len(moves),
                    "player": current_player,
                    "model": model,
                    "attempted_move": (
                        list(result.move)
                        if isinstance(result.move, (tuple, list))
                        else str(result.move)
                    ),
                    "elapsed_seconds": elapsed,
                    "nodes": int(result.nodes),
                    "completed_depth": int(result.completed_depth),
                    "tree_depth": int(result.completed_depth),
                    "cutoffs": int(result.cutoffs),
                    "simulations": int(result.simulations),
                    "rollout_plies": int(result.rollout_plies),
                    "terminal_simulations": int(result.terminal_simulations),
                    "truncated_simulations": int(result.truncated_simulations),
                    "root_candidates": int(result.root_candidates),
                    "root_min_visits": int(result.root_min_visits),
                    "reused_nodes": int(result.reused_nodes),
                    "timed_out": timed_out,
                    "illegal": True,
                    "opening": False,
                }
            )
            winner_color = opponent_color
            winner_model = opponent_model
            termination = "illegal_move_forfeit"
            break

        coord = int(result.move[0]), int(result.move[1])
        board.place_stone(current_player, coord)
        moves.append(
            {
                "ply": len(moves),
                "player": current_player,
                "model": model,
                "row": coord[0],
                "col": coord[1],
                "elapsed_seconds": elapsed,
                "nodes": int(result.nodes),
                "completed_depth": int(result.completed_depth),
                "tree_depth": int(result.completed_depth),
                "cutoffs": int(result.cutoffs),
                "simulations": int(result.simulations),
                "rollout_plies": int(result.rollout_plies),
                "terminal_simulations": int(result.terminal_simulations),
                "truncated_simulations": int(result.truncated_simulations),
                "root_candidates": int(result.root_candidates),
                "root_min_visits": int(result.root_min_visits),
                "reused_nodes": int(result.reused_nodes),
                "timed_out": timed_out,
                "opening": False,
            }
        )
        if board.check_victory(coord, current_player):
            winner_color = current_player
            winner_model = model
            termination = "five_in_row"
            break
        if len(board.history) >= size * size:
            break
        current_player = opponent_color

    normalized_matchup = tuple(sorted(str(model) for model in spec["matchup"]))
    return {
        **spec,
        "matchup": list(normalized_matchup),
        "budget_seconds_by_model": dict(spec["budget_seconds_by_model"]),
        "opening": [
            {"player": player, "row": coord[0], "col": coord[1]}
            for player, coord in opening
        ],
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "draw" if winner_color is None else f"{winner_color.lower()}_win",
        "termination": termination,
        "timeout_count": sum(move.get("timed_out", False) for move in moves),
        "search_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
        "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
        "rollout_plies_total": sum(int(move.get("rollout_plies", 0)) for move in moves),
        "terminal_simulations_total": sum(
            int(move.get("terminal_simulations", 0)) for move in moves
        ),
        "truncated_simulations_total": sum(
            int(move.get("truncated_simulations", 0)) for move in moves
        ),
        "tree_depth_max": max((int(move.get("tree_depth", 0)) for move in moves), default=0),
        "thinking_seconds_total": sum(
            float(move.get("elapsed_seconds", 0.0)) for move in moves
        ),
    }


def _load_config(config_source: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_source, dict):
        config = deepcopy(config_source)
    else:
        config = json.loads(Path(config_source).read_text(encoding="utf-8"))
    current_commit = _current_code_commit()
    recorded_commit = config.get("code_commit")
    if recorded_commit is not None and recorded_commit != current_commit:
        raise ValueError(
            f"Current code commit {current_commit} differs from recorded {recorded_commit}"
        )
    config["code_commit"] = current_commit
    return config


def _record_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*_games.jsonl"))


def _load_records_unique(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    completed: set[str] = set()
    for path in paths:
        for record in load_game_records(path):
            identifier = str(record["game_id"])
            if identifier in completed:
                raise ValueError(f"Duplicate game_id across result files: {identifier}")
            completed.add(identifier)
            records.append(record)
    return records, completed


def _phase_result_path(output_dir: Path, spec: dict[str, Any]) -> Path:
    return output_dir / f"{spec['board_size']}x{spec['board_size']}_{spec['phase_name']}_games.jsonl"


def run_experiment(
    config_source: dict[str, Any] | str | Path,
    output_dir: str | Path,
    max_games: int | None = None,
) -> dict[str, Any]:
    """Run or resume scheduled games, persisting each finished game once."""

    config = _load_config(config_source)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("Existing results use a different experiment config")
    else:
        write_json_atomic(config_path, config)

    schedule = build_schedule(config)
    _, completed_ids = _load_records_unique(_record_files(root))
    scheduled_ids = {str(spec["game_id"]) for spec in schedule}
    unexpected = completed_ids - scheduled_ids
    if unexpected:
        raise ValueError(f"Result directory contains unscheduled game IDs: {sorted(unexpected)[:3]}")
    if max_games is not None and max_games < 0:
        raise ValueError("max_games cannot be negative")

    games_completed_this_run = 0
    last_game_id: str | None = None
    for spec in schedule:
        identifier = str(spec["game_id"])
        if identifier in completed_ids:
            continue
        if max_games is not None and games_completed_this_run >= max_games:
            break
        record = run_game(spec, config)
        append_game(_phase_result_path(root, spec), record, completed_ids)
        games_completed_this_run += 1
        last_game_id = identifier
        write_json_atomic(
            root / "progress.json",
            {
                "experiment_id": config["experiment_id"],
                "code_commit": config["code_commit"],
                "scheduled_games": len(schedule),
                "games_completed": len(completed_ids),
                "remaining_games": len(schedule) - len(completed_ids),
                "last_game_id": last_game_id,
                "complete": len(completed_ids) == len(schedule),
            },
        )

    total = len(completed_ids)
    progress = {
        "experiment_id": config["experiment_id"],
        "code_commit": config["code_commit"],
        "scheduled_games": len(schedule),
        "games_completed": total,
        "games_completed_this_run": games_completed_this_run,
        "games_completed_total": total,
        "remaining_games": len(schedule) - total,
        "complete": total == len(schedule),
    }
    write_json_atomic(root / "progress.json", progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mcts_potential_v2.json")
    parser.add_argument("--output-dir", default="research/v2/results")
    parser.add_argument("--max-games", type=int)
    args = parser.parse_args()
    result = run_experiment(args.config, args.output_dir, max_games=args.max_games)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
