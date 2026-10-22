"""Public V6 registry that keeps E1–E4 fallbacks and budgets comparable."""
from __future__ import annotations

import random
import time
from dataclasses import replace
from typing import Any

from ...ai import SearchConfig, _other, evaluate_board
from ...gomoku import Coordinate, GomokuBoard, Player, select_ai_move
from .e5 import E5Options, E5Search
from .pns import ProofNumberSearch
from .tactics import (
    SearchTimeout,
    TacticalSolver,
    pack_certificate,
    verify_proof_certificate,
)
from .types import V6MoveResult


_TACTICAL_MODELS = {"E1": "VCF", "E2": "VCT", "E3": "TSS"}


def _first_legal(board: GomokuBoard) -> Coordinate:
    center = (board.size // 2, board.size // 2)
    if board.get(center) is None:
        return center
    for row in range(board.size):
        for col in range(board.size):
            if board.grid[row][col] is None:
                return row, col
    raise ValueError("Board has no legal move")


class V6Agent:
    """E1 VCF, E2 VCT, E3 TSS, E4 PNS, or E5 advanced Alpha-Beta."""

    def __init__(
        self,
        model: str,
        *,
        max_attack_depth: int = 8,
        max_nodes: int = 50_000,
        e5_options: E5Options | None = None,
    ) -> None:
        normalized = str(model).upper()
        if normalized not in {"E1", "E2", "E3", "E4", "E5"}:
            raise ValueError(f"Unknown V6 model: {model}")
        self.model = normalized
        self.max_attack_depth = max_attack_depth
        self.max_nodes = max_nodes
        self.e5 = E5Search(e5_options) if normalized == "E5" else None

    @staticmethod
    def verify_proof_certificate(
        board: GomokuBoard,
        attacker: Player,
        certificate: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> bool:
        return verify_proof_certificate(
            board, attacker, certificate, deadline=deadline
        )

    def choose_move(
        self,
        board: GomokuBoard,
        player: Player,
        *,
        time_limit_seconds: float,
        seed: int = 0,
        max_depth: int = 24,
    ) -> V6MoveResult:
        if board.is_full():
            raise ValueError("Board has no legal move")
        started = time.perf_counter()
        budget = max(0.0, float(time_limit_seconds))
        if budget == 0:
            move = _first_legal(board)
            return V6MoveResult(
                move=move,
                root_value=evaluate_board(board, player),
                nodes=0,
                completed_depth=0,
                cutoffs=0,
                simulations=0,
                elapsed_seconds=time.perf_counter() - started,
                timed_out=True,
                proof_status="unknown" if self.model != "E5" else "not_applicable",
                stats={"fallback": "zero-budget-legal"},
            )

        fallback = select_ai_move(
            board, player, _other(player), rng=random.Random(seed)
        )
        # Leave return-path time for certificate replay and a legal fallback;
        # short budgets need proportionally more reserve because threat graph
        # construction and Python call overhead are not interruptible mid-loop.
        reserve = min(0.025, max(0.010, budget * 0.20))
        deadline = started + max(0.0, budget - reserve)
        if time.perf_counter() >= deadline:
            return V6MoveResult(
                fallback, evaluate_board(board, player), 0, 0, 0, 0,
                time.perf_counter() - started, True,
                proof_status="unknown" if self.model != "E5" else "not_applicable",
                stats={"fallback": "A", "deadline_before_search": True},
            )

        if self.model == "E5":
            assert self.e5 is not None
            remaining = max(0.0, budget - (time.perf_counter() - started))
            if remaining <= 0:
                return V6MoveResult(
                    fallback, evaluate_board(board, player), 0, 0, 0, 0,
                    time.perf_counter() - started, True,
                    stats={"fallback": "A", "deadline_before_search": True},
                )
            result = self.e5.choose_move(
                board,
                player,
                SearchConfig(max_depth=max_depth, time_limit_seconds=remaining),
                seed,
                max_depth=max_depth,
            )
            return replace(
                result,
                elapsed_seconds=time.perf_counter() - started,
                timed_out=result.timed_out or time.perf_counter() >= deadline,
            )

        if self.model == "E4":
            pns = ProofNumberSearch(
                deadline,
                max_nodes=self.max_nodes,
                max_attack_depth=self.max_attack_depth,
            )
            root, pns_stats = pns.solve(board, player)
            certificate_root = pns.proof_certificate(root, board, player)
            kind = "PNS"
            certificate = (
                pack_certificate(kind, player, certificate_root)
                if certificate_root is not None
                else None
            )
            proof_move: Coordinate | None = None
            if certificate is not None:
                valid = verify_proof_certificate(
                    board, player, certificate, deadline=deadline
                )
                if valid:
                    proof_move = tuple(map(int, certificate["root"]["move"]))
            timed_out = (
                pns.timed_out
                or pns.limit_hit
                or time.perf_counter() >= deadline
            )
            status = (
                "proven" if proof_move is not None
                else "unknown" if timed_out
                else "not_found_within_bound"
            )
            return V6MoveResult(
                proof_move or fallback,
                evaluate_board(board, player),
                pns_stats.expanded_nodes,
                pns_stats.max_depth,
                0,
                0,
                time.perf_counter() - started,
                timed_out,
                status,
                kind if proof_move is not None else None,
                certificate if proof_move is not None else None,
                {
                    "fallback": "A",
                    "expanded_nodes": pns_stats.expanded_nodes,
                    "generated_nodes": pns_stats.generated_nodes,
                    "defense_expansions": pns_stats.defense_expansions,
                    "proof_number": pns_stats.root_proof,
                    "disproof_number": pns_stats.root_disproof,
                    "node_limit_hit": pns.limit_hit,
                    "certificate_verified": proof_move is not None,
                },
            )

        algorithm = _TACTICAL_MODELS[self.model]
        solver = TacticalSolver(
            algorithm,
            deadline,
            max_attack_depth=self.max_attack_depth,
            max_nodes=self.max_nodes,
        )
        proof_root = solver.solve(board, player)
        certificate = (
            pack_certificate(algorithm, player, proof_root)
            if proof_root is not None
            else None
        )
        proof_move: Coordinate | None = None
        if certificate is not None:
            valid = verify_proof_certificate(
                board, player, certificate, deadline=deadline
            )
            if valid:
                proof_move = tuple(map(int, certificate["root"]["move"]))
        timed_out = (
            solver.timed_out
            or solver.limit_hit
            or time.perf_counter() >= deadline
        )
        status = (
            "proven" if proof_move is not None
            else "unknown" if timed_out
            else "not_found_within_bound"
        )
        stats = {
            "fallback": "A",
            "defense_nodes": solver.stats.defense_nodes,
            "critical_defenses": solver.stats.critical_defenses,
            "max_attack_depth": solver.stats.max_attack_depth,
            "threat_nodes": solver.stats.threat_nodes,
            "dependency_edges": solver.stats.dependency_edges,
            "defense_conflicts": solver.stats.defense_conflicts,
            "node_limit_hit": solver.limit_hit,
            "certificate_verified": proof_move is not None,
        }
        return V6MoveResult(
            proof_move or fallback,
            evaluate_board(board, player),
            solver.stats.nodes,
            solver.stats.max_attack_depth,
            0,
            0,
            time.perf_counter() - started,
            timed_out,
            status,
            algorithm if proof_move is not None else None,
            certificate if proof_move is not None else None,
            stats,
        )
