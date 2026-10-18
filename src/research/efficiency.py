"""Fixed-depth B/C comparison and equal-time tournament efficiency summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

from ..ai import SearchConfig, choose_move, generate_candidates
from ..gomoku import GomokuBoard, Player
from .metrics import aggregate_results
from .runner import derive_seed, load_all_games


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _would_win(board: GomokuBoard, coord: tuple[int, int], player: Player) -> bool:
    return board.potential_line_length(coord, player) >= 5


def generate_position_suite(
    count: int = 8, board_size: int = 9, seed: int = 5102026
) -> list[dict[str, Any]]:
    """Create deterministic non-terminal positions for exact B/C comparisons."""

    if count < 1:
        raise ValueError("count must be positive")
    if board_size < 5:
        raise ValueError("board_size must be at least 5")
    positions: list[dict[str, Any]] = []
    for index in range(count):
        target_plies = 2 + 2 * (index % 5)
        attempt = 0
        while True:
            attempt += 1
            rng = random.Random(derive_seed(seed, "position", index, attempt))
            board = GomokuBoard(board_size)
            to_move: Player = "B"
            terminal = False
            for _ in range(target_plies):
                candidates = generate_candidates(board, to_move, _other(to_move))
                safe = [move for move in candidates if not _would_win(board, move, to_move)]
                if not safe:
                    terminal = True
                    break
                move = rng.choice(safe)
                board.place_stone(to_move, move)
                to_move = _other(to_move)
            if not terminal:
                positions.append(
                    {
                        "position_id": f"p{index:03d}",
                        "board": board.to_dict(),
                        "to_move": to_move,
                        "seed": derive_seed(seed, "position", index, attempt),
                    }
                )
                break
            if attempt >= 100:
                raise RuntimeError(f"Could not create a non-terminal position {index}")
    return positions


def _move_tuple(move: tuple[int, int]) -> list[int]:
    return [int(move[0]), int(move[1])]


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "move": _move_tuple(result.move),
        "root_value": int(result.root_value),
        "completed_depth": int(result.completed_depth),
        "nodes": int(result.nodes),
        "cutoffs": int(result.cutoffs),
        "elapsed_seconds": float(result.elapsed_seconds),
        "optimal_moves": sorted(_move_tuple(move) for move in result.optimal_moves),
    }


def run_fixed_depth_comparison(
    positions: Iterable[dict[str, Any]],
    depth: int = 2,
    time_limit_seconds: float = 60.0,
    seed: int = 5102026,
) -> list[dict[str, Any]]:
    """Run B and C on identical positions, depth, ordering, and evaluation."""

    if depth < 1:
        raise ValueError("depth must be positive")
    settings = SearchConfig(max_depth=depth, time_limit_seconds=time_limit_seconds)
    results: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        board = GomokuBoard.from_dict(position["board"])
        player = str(position["to_move"])
        if player not in {"B", "W"}:
            raise ValueError("Position to_move must be B or W")
        minimax = choose_move(
            board, player, "B", settings, derive_seed(seed, position["position_id"], "B")
        )
        alpha_beta = choose_move(
            board, player, "C", settings, derive_seed(seed, position["position_id"], "C")
        )
        if minimax.completed_depth != depth or alpha_beta.completed_depth != depth:
            raise TimeoutError(f"Fixed-depth search did not complete at {position['position_id']}")
        if minimax.root_value != alpha_beta.root_value:
            raise AssertionError(f"B/C root values differ at {position['position_id']}")
        if alpha_beta.move not in minimax.optimal_moves:
            raise AssertionError(f"Alpha-Beta move is not minimax-optimal at {position['position_id']}")
        prune_ratio = (
            (minimax.nodes - alpha_beta.nodes) / minimax.nodes if minimax.nodes else 0.0
        )
        results.append(
            {
                "position_id": position["position_id"],
                "board_size": board.size,
                "stone_count": len(board.history),
                "to_move": player,
                "depth": depth,
                "candidate_count": len(generate_candidates(board, player, _other(player))),
                "minimax": _result_summary(minimax),
                "alpha_beta": _result_summary(alpha_beta),
                "same_root_value": True,
                "alpha_beta_move_is_minimax_optimal": True,
                "prune_ratio": prune_ratio,
            }
        )
    return results


def summarize_time_budget_results(
    records: Iterable[dict[str, Any]],
    models: Iterable[str] = ("A", "B", "C", "D"),
    bootstrap_samples: int = 5_000,
    seed: int = 5102026,
) -> dict[str, Any]:
    """Summarize actual score and completed search work under the shared clock cap."""

    return aggregate_results(
        records, models=models, bootstrap_samples=bootstrap_samples, seed=seed
    )


def build_efficiency_report(
    positions: Iterable[dict[str, Any]],
    games: Iterable[dict[str, Any]],
    depth: int,
    time_limit_seconds: float,
    seed: int,
    bootstrap_samples: int = 5_000,
) -> dict[str, Any]:
    position_list = list(positions)
    game_list = list(games)
    fixed = run_fixed_depth_comparison(
        position_list, depth=depth, time_limit_seconds=time_limit_seconds, seed=seed
    )
    return {
        "fixed_depth": {
            "positions": len(fixed),
            "depth": depth,
            "records": fixed,
            "average_nodes_minimax": (
                sum(item["minimax"]["nodes"] for item in fixed) / len(fixed) if fixed else 0.0
            ),
            "average_nodes_alpha_beta": (
                sum(item["alpha_beta"]["nodes"] for item in fixed) / len(fixed) if fixed else 0.0
            ),
            "average_prune_ratio": (
                sum(item["prune_ratio"] for item in fixed) / len(fixed) if fixed else 0.0
            ),
            "all_root_values_match": all(item["same_root_value"] for item in fixed),
            "all_alpha_beta_moves_are_minimax_optimal": all(
                item["alpha_beta_move_is_minimax_optimal"] for item in fixed
            ),
        },
        "same_time_budget": summarize_time_budget_results(
            game_list,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "training": "No neural network training was performed; this is algorithm development and match evaluation.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Minimax with Alpha-Beta pruning.")
    parser.add_argument("--config", default="configs/four_ai_experiment.json")
    parser.add_argument("--games", default="research/results/9x9_games.jsonl")
    parser.add_argument("--output", default="research/results/efficiency.json")
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    positions = generate_position_suite(
        count=args.positions,
        board_size=9,
        seed=int(config["base_seed"]),
    )
    games_path = Path(args.games)
    games = load_all_games(games_path.parent) if games_path.parent.exists() else []
    games = [record for record in games if int(record["board_size"]) == 9]
    report = build_efficiency_report(
        positions,
        games,
        depth=args.depth,
        time_limit_seconds=args.time_limit,
        seed=int(config["base_seed"]),
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"positions": len(positions), "games": len(games), "output": str(output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
