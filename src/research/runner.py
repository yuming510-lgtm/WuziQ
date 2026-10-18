"""Paired, color-swapped AI tournaments with per-game resumable records."""
from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

from ..ai import SearchConfig, choose_move
from ..gomoku import GomokuBoard, Player
from .checkpoint import append_game, load_game_records, write_json_atomic


MODEL_ORDER = ("A", "B", "C", "D")
_OPENING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed without Python's process-randomized hash."""

    payload = json.dumps(
        [int(base_seed), *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")


def game_id(
    board_size: int,
    model_pair: tuple[str, str],
    opening_index: int,
    color_swap: bool,
) -> str:
    first, second = sorted((str(model_pair[0]).upper(), str(model_pair[1]).upper()))
    if first == second:
        raise ValueError("A matchup must contain two different models")
    return f"{board_size}x{board_size}-{first}-{second}-o{opening_index:03d}-s{int(color_swap)}"


def paired_opening(board_size: int, seed: int) -> list[tuple[Player, tuple[int, int]]]:
    """Return a deterministic two-ply opening shared by both color assignments."""

    if board_size < 5:
        raise ValueError("Board size must be at least 5")
    center = (board_size // 2, board_size // 2)
    dr, dc = random.Random(seed).choice(_OPENING_OFFSETS)
    return [("B", center), ("W", (center[0] + dr, center[1] + dc))]


def build_schedule(
    config: dict[str, Any],
    board_sizes: Iterable[int] | None = None,
    model_pairs: Iterable[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Expand phase counts into paired openings with exactly balanced colors."""

    models = tuple(sorted(str(model).upper() for model in config.get("models", ())))
    if len(models) < 2 or len(models) != len(set(models)):
        raise ValueError("models must contain at least two unique identifiers")
    if any(model not in MODEL_ORDER for model in models):
        raise ValueError("models must be selected from A, B, C, and D")
    pairs = list(model_pairs) if model_pairs is not None else list(itertools.combinations(models, 2))
    normalized_pairs = sorted(
        {tuple(sorted((str(pair[0]).upper(), str(pair[1]).upper()))) for pair in pairs}
    )
    if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(models) for pair in normalized_pairs):
        raise ValueError("model_pairs must contain distinct configured models")
    selected_sizes = set(board_sizes) if board_sizes is not None else None
    schedule: list[dict[str, Any]] = []
    base_seed = int(config["base_seed"])
    for phase in config.get("phases", []):
        size = int(phase["board_size"])
        games_per_matchup = int(phase["games_per_matchup"])
        if size < 5:
            raise ValueError("Board size must be at least 5")
        if games_per_matchup < 2 or games_per_matchup % 2:
            raise ValueError("games_per_matchup must be an even number of at least 2")
        if selected_sizes is not None and size not in selected_sizes:
            continue
        for opening_index in range(games_per_matchup // 2):
            opening_seed = derive_seed(base_seed, "opening", size, opening_index)
            for pair in normalized_pairs:
                for color_swap in (False, True):
                    black_model, white_model = pair
                    if color_swap:
                        black_model, white_model = white_model, black_model
                    identifier = game_id(size, pair, opening_index, color_swap)
                    schedule.append(
                        {
                            "game_id": identifier,
                            "board_size": size,
                            "matchup": pair,
                            "opening_index": opening_index,
                            "color_swap": color_swap,
                            "black_model": black_model,
                            "white_model": white_model,
                            "opening_seed": opening_seed,
                            "game_seed": derive_seed(base_seed, "game", identifier),
                        }
                    )
    return schedule


def _search_config(config: dict[str, Any]) -> SearchConfig:
    return SearchConfig(**config.get("search", {}))


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _opening_log(player: Player, coord: tuple[int, int], ply: int) -> dict[str, Any]:
    return {
        "ply": ply,
        "player": player,
        "model": None,
        "row": coord[0],
        "col": coord[1],
        "elapsed_seconds": 0.0,
        "nodes": 0,
        "completed_depth": 0,
        "cutoffs": 0,
        "simulations": 0,
        "timed_out": False,
        "opening": True,
    }


def _legal_coordinate(board: GomokuBoard, move: object) -> bool:
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        return False
    row, col = move
    if type(row) is not int or type(col) is not int:
        return False
    coord = (row, col)
    return board.in_bounds(*coord) and board.get(coord) is None


def run_game(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Play one complete game; timeouts keep legal fallback moves, illegal moves forfeit."""

    size = int(spec["board_size"])
    board = GomokuBoard(size)
    opening = paired_opening(size, int(spec["opening_seed"]))
    moves = []
    for ply, (player, coord) in enumerate(opening):
        board.place_stone(player, coord)
        moves.append(_opening_log(player, coord, ply))

    settings = _search_config(config)
    current_player: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    for _ in range(size * size - len(board.history)):
        model = spec["black_model"] if current_player == "B" else spec["white_model"]
        opponent_color = _other(current_player)
        opponent_model = (
            spec["black_model"] if opponent_color == "B" else spec["white_model"]
        )
        seed = derive_seed(
            int(spec["game_seed"]), "move", len(board.history), model, current_player
        )
        result = choose_move(board, current_player, model, settings, seed)
        elapsed = float(result.elapsed_seconds)
        timed_out = bool(result.timed_out or elapsed > settings.time_limit_seconds)
        if not _legal_coordinate(board, result.move):
            move_record: dict[str, Any] = {
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
                "cutoffs": int(result.cutoffs),
                "simulations": int(result.simulations),
                "timed_out": timed_out,
                "illegal": True,
                "opening": False,
            }
            moves.append(move_record)
            winner_color = opponent_color
            winner_model = opponent_model
            termination = "illegal_move_forfeit"
            break

        coord = (int(result.move[0]), int(result.move[1]))
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
                "cutoffs": int(result.cutoffs),
                "simulations": int(result.simulations),
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
            termination = "full_board"
            break
        current_player = opponent_color

    outcome = "draw" if winner_color is None else f"{winner_color.lower()}_win"
    return {
        **spec,
        "matchup": list(spec["matchup"]),
        "pair_id": f"{size}x{size}-{'-'.join(spec['matchup'])}-o{int(spec['opening_index']):03d}",
        "base_seed": int(config["base_seed"]),
        "time_limit_seconds": settings.time_limit_seconds,
        "max_depth": settings.max_depth,
        "opening": [
            {"player": player, "row": coord[0], "col": coord[1]}
            for player, coord in opening
        ],
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": outcome,
        "termination": termination,
        "timeout_count": sum(move.get("timed_out", False) for move in moves),
        "search_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
        "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
        "thinking_seconds_total": sum(float(move.get("elapsed_seconds", 0)) for move in moves),
    }


def load_all_games(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load all board-size files and reject duplicate IDs across them."""

    root = Path(output_dir)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*_games.jsonl")):
        for record in load_game_records(path):
            game_id_value = str(record["game_id"])
            if game_id_value in seen:
                raise ValueError(f"Duplicate game_id across result files: {game_id_value}")
            seen.add(game_id_value)
            records.append(record)
    return records


def _config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def run_tournament(
    config: dict[str, Any],
    output_dir: str | Path,
    max_games: int | None = None,
    board_sizes: Iterable[int] | None = None,
    model_pairs: Iterable[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Run all scheduled games or the requested batch, skipping completed IDs."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError("Output directory already belongs to a different experiment config")
    else:
        write_json_atomic(config_path, config)

    schedule = build_schedule(config, board_sizes, model_pairs)
    existing = load_all_games(root)
    completed_ids = {str(record["game_id"]) for record in existing}
    completed_counts: dict[str, int] = {}
    for record in existing:
        size_key = str(record["board_size"])
        completed_counts[size_key] = completed_counts.get(size_key, 0) + 1

    produced: list[dict[str, Any]] = []
    for schedule_index, spec in enumerate(schedule):
        if spec["game_id"] in completed_ids:
            continue
        if max_games is not None and len(produced) >= max_games:
            break
        record = run_game(spec, config)
        result_path = root / f"{spec['board_size']}x{spec['board_size']}_games.jsonl"
        append_game(result_path, record, completed_ids)
        produced.append(record)
        size_key = str(spec["board_size"])
        completed_counts[size_key] = completed_counts.get(size_key, 0) + 1
        write_json_atomic(
            root / "progress.json",
            {
                "schema_version": 1,
                "experiment_id": config.get("experiment_id"),
                "base_seed": int(config["base_seed"]),
                "config_sha256": _config_digest(config),
                "next_schedule_index": schedule_index + 1,
                "completed_game_ids": len(completed_ids),
                "completed_games_by_board": completed_counts,
                "last_game_id": record["game_id"],
            },
        )
    return produced


def _parse_matchup(value: str) -> tuple[str, str]:
    pieces = value.upper().replace(" VS ", "-").split("-")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Matchup must look like A-B")
    return pieces[0], pieces[1]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run resumable WuziQ AI tournaments.")
    parser.add_argument("--config", default="configs/four_ai_experiment.json")
    parser.add_argument("--output", default="research/results")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--board-size", type=int, action="append")
    parser.add_argument("--matchup", type=_parse_matchup, action="append")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    produced = run_tournament(
        config,
        args.output,
        max_games=args.max_games,
        board_sizes=args.board_size,
        model_pairs=args.matchup,
    )
    print(
        json.dumps(
            {"new_games": len(produced), "total_games": len(load_all_games(args.output))},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
