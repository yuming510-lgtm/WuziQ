"""Fixed-position ablations separating D1 compute from D2 engineering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ..ai import SearchConfig, SearchResult, choose_move
from ..gomoku import Coordinate, GomokuBoard, Player
from .mcts_d2 import MCTSAgent
from .mcts_v2 import derive_seed


def ablation_variants() -> dict[str, dict[str, bool]]:
    """Return one-at-a-time D2 feature switches plus the full engine."""

    return {
        "full": {},
        "no_root_tactics": {"mcts_d2_tactical_root": False},
        "no_fair_root": {"mcts_d2_fair_root": False},
        "no_tactical_rollout": {"mcts_d2_tactical_rollout": False},
        "no_node_reuse": {"mcts_d2_node_reuse": False},
    }


def _copy_board(board: GomokuBoard) -> GomokuBoard:
    cloned = GomokuBoard(board.size)
    for player, coord in board.history:
        cloned.place_stone(player, coord)
    return cloned


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _result_row(
    model: str,
    variant: str,
    position_id: str,
    result: SearchResult,
    expected_move: Coordinate | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "variant": variant,
        "position_id": position_id,
        "move": list(result.move),
        "expected_move": list(expected_move) if expected_move is not None else None,
        "expected_move_found": (
            result.move == expected_move if expected_move is not None else None
        ),
        "elapsed_seconds": result.elapsed_seconds,
        "simulations": result.simulations,
        "nodes": result.nodes,
        "tree_depth": result.completed_depth,
        "rollout_plies": result.rollout_plies,
        "terminal_simulations": result.terminal_simulations,
        "truncated_simulations": result.truncated_simulations,
        "root_candidates": result.root_candidates,
        "root_min_visits": result.root_min_visits,
        "reused_nodes": result.reused_nodes,
        "timed_out": result.timed_out,
    }


def _reuse_probe(
    board: GomokuBoard,
    player: Player,
    agent: MCTSAgent,
    first: SearchResult,
    config: SearchConfig,
    seed: int,
) -> tuple[SearchResult | None, bool]:
    """Replay one explored opponent reply and measure the retained subtree."""

    root = agent._root
    if root is None:
        return None, False
    selected = next((child for child in root.children if child.move == first.move), None)
    if selected is None or not selected.children:
        return None, False
    opponent_reply = next(
        (child.move for child in selected.children if child.move is not None), None
    )
    if opponent_reply is None:
        return None, False
    replay = _copy_board(board)
    replay.place_stone(player, first.move)
    replay.place_stone(_other(player), opponent_reply)
    if replay.check_victory(opponent_reply, _other(player)):
        return None, False
    second = agent.choose_move(
        replay, player, config, derive_seed(seed, "reuse-probe", len(board.history))
    )
    return second, True


def run_ablation(
    positions: Iterable[dict[str, Any]],
    budgets_seconds: Iterable[float] = (0.05, 0.2, 1.0),
    simulations: int = 10_000,
    base_seed: int = 5102026,
) -> dict[str, Any]:
    """Run paired fixed-position D1/D2 feature ablations and reuse probes."""

    rows: list[dict[str, Any]] = []
    variants = ablation_variants()
    for position in positions:
        board = position["board"]
        player = position["player"]
        position_id = str(position["position_id"])
        expected_move = position.get("expected_move")
        for budget in budgets_seconds:
            d1_config = SearchConfig(
                time_limit_seconds=float(budget), mcts_simulations=simulations
            )
            d1 = choose_move(
                _copy_board(board),
                player,
                "D1",
                d1_config,
                derive_seed(base_seed, position_id, budget, "D1"),
            )
            rows.append(_result_row("D1", "high-compute-baseline", position_id, d1, expected_move))
            for name, flags in variants.items():
                config = SearchConfig(
                    time_limit_seconds=float(budget),
                    mcts_simulations=simulations,
                    **flags,
                )
                agent = MCTSAgent()
                seed = derive_seed(base_seed, position_id, budget, name)
                first = agent.choose_move(_copy_board(board), player, config, seed)
                rows.append(_result_row("D2", name, position_id, first, expected_move))
                second, replayed = _reuse_probe(
                    board, player, agent, first, config, seed
                )
                if second is not None:
                    rows.append(
                        {
                            **_result_row("D2", f"{name}-reuse-probe", position_id, second, None),
                            "replayed_opponent_reply": replayed,
                        }
                    )
    return {
        "seed": int(base_seed),
        "simulation_cap": int(simulations),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mcts_potential_v2.json")
    parser.add_argument("--output", default="research/v2/pilot/ablation.json")
    args = parser.parse_args()
    from .mcts_v2_benchmark import representative_positions

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = run_ablation(
        representative_positions(int(config["base_seed"])),
        budgets_seconds=config.get("ablation_budgets_seconds", [0.05, 0.2, 1]),
        simulations=int(config.get("ablation_simulations", 10_000)),
        base_seed=int(config["base_seed"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(payload["rows"]), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
