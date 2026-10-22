"""Proof-number search over threat OR nodes and exhaustive defense AND nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from ...gomoku import Coordinate, GomokuBoard, Player
from .tactics import SearchLimit, SearchTimeout, immediate_wins, legal_moves, vct_attack_moves


_INF = 1_000_000_000


@dataclass
class PNNode:
    move: Coordinate | None
    to_move: Player
    parent: "PNNode | None" = None
    children: list["PNNode"] = field(default_factory=list)
    expanded: bool = False
    proof: int = 1
    disproof: int = 1
    terminal: str | None = None

@dataclass(frozen=True)
class PNSStats:
    expanded_nodes: int
    generated_nodes: int
    defense_expansions: int
    max_depth: int
    root_proof: int
    root_disproof: int


class ProofNumberSearch:
    """Standard most-proving-node PNS with a sound tactical OR move set.

    The attacker is OR and expands only legal VCT-style threat moves. The
    defender is AND and expands every legal move. A zero root proof number is a
    verified forced win in this restricted attacker strategy space; no-proof
    results are not reported as a game-theoretic loss.
    """

    def __init__(
        self,
        deadline: float,
        *,
        max_nodes: int = 50_000,
        max_attack_depth: int = 8,
    ) -> None:
        self.deadline = deadline
        self.max_nodes = max_nodes
        self.max_attack_depth = max_attack_depth
        self.expanded_nodes = 0
        self.generated_nodes = 1
        self.defense_expansions = 0
        self.max_depth = 0
        self.timed_out = False
        self.limit_hit = False

    def _check(self) -> None:
        if time.perf_counter() >= self.deadline:
            self.timed_out = True
            raise SearchTimeout
        if self.generated_nodes >= self.max_nodes:
            self.limit_hit = True
            raise SearchLimit

    @staticmethod
    def _is_or(node: PNNode, attacker: Player) -> bool:
        return node.to_move == attacker

    def _update(self, node: PNNode, attacker: Player) -> None:
        if not node.children:
            return
        if self._is_or(node, attacker):
            node.proof = min(child.proof for child in node.children)
            node.disproof = min(_INF, sum(child.disproof for child in node.children))
        else:
            node.proof = min(_INF, sum(child.proof for child in node.children))
            node.disproof = min(child.disproof for child in node.children)

    def _expand(self, node: PNNode, board: GomokuBoard, attacker: Player, depth: int) -> None:
        self._check()
        node.expanded = True
        self.expanded_nodes += 1
        self.max_depth = max(self.max_depth, depth)
        moves = vct_attack_moves(board, attacker, self._check) if self._is_or(node, attacker) else legal_moves(board)
        if not self._is_or(node, attacker):
            self.defense_expansions += len(moves)
        player = node.to_move
        other = "W" if player == "B" else "B"
        try:
            for move in moves:
                self._check()
                board.place_stone(player, move)
                try:
                    if board.check_victory(move, player):
                        attacker_won = player == attacker
                        child = PNNode(
                            move,
                            other,
                            parent=node,
                            expanded=True,
                            proof=0 if attacker_won else _INF,
                            disproof=_INF if attacker_won else 0,
                            terminal="attacker_win" if attacker_won else "defender_win",
                        )
                    elif len(board.history) >= board.size * board.size:
                        child = PNNode(
                            move, other, parent=node, expanded=True,
                            proof=_INF, disproof=0, terminal="draw",
                        )
                    else:
                        child = PNNode(move, other, parent=node)
                    self.generated_nodes += 1
                    node.children.append(child)
                finally:
                    board.undo()
        except (SearchTimeout, SearchLimit):
            node.children.clear()
            node.expanded = False
            node.proof = node.disproof = 1
            raise
        if not node.children:
            node.proof, node.disproof = _INF, 0
            node.terminal = "no_proof_moves"

    def _select(self, root: PNNode, attacker: Player, board: GomokuBoard) -> tuple[PNNode, list[PNNode]]:
        node = root
        path = [root]
        depth = 0
        while node.expanded and node.children and node.proof not in (0, _INF) and node.disproof not in (0, _INF):
            is_or = self._is_or(node, attacker)
            key = (lambda child: (child.proof, child.disproof, child.move or (-1, -1))) if is_or else (
                lambda child: (child.disproof, child.proof, child.move or (-1, -1))
            )
            child = min(node.children, key=key)
            if child.move is None:
                break
            board.place_stone(node.to_move, child.move)
            node = child
            path.append(node)
            depth += 1
        return node, path

    def solve(self, board: GomokuBoard, attacker: Player) -> tuple[PNNode, PNSStats]:
        root = PNNode(None, attacker)
        self.expanded_nodes = 0
        self.generated_nodes = 1
        self.defense_expansions = self.max_depth = 0
        self.timed_out = self.limit_hit = False
        while root.proof != 0 and root.disproof != 0:
            try:
                self._check()
                leaf, path = self._select(root, attacker, board)
                depth = len(path) - 1
                try:
                    if leaf.terminal is not None:
                        pass
                    elif depth > 2 * self.max_attack_depth + 1:
                        leaf.proof, leaf.disproof = _INF, 0
                        leaf.terminal = "depth_bound"
                        leaf.expanded = True
                    else:
                        self._expand(leaf, board, attacker, depth)
                    self._update(leaf, attacker)
                    for ancestor in reversed(path[:-1]):
                        self._update(ancestor, attacker)
                finally:
                    for _ in path[1:]:
                        board.undo()
            except (SearchTimeout, SearchLimit):
                break
        return root, self.stats(root)

    def stats(self, root: PNNode) -> PNSStats:
        return PNSStats(
            self.expanded_nodes,
            self.generated_nodes,
            self.defense_expansions,
            self.max_depth,
            root.proof,
            root.disproof,
        )

    def proof_certificate(
        self,
        root: PNNode,
        board: GomokuBoard,
        attacker: Player,
    ) -> dict[str, Any] | None:
        if root.proof != 0:
            return None

        def build_or(node: PNNode, state: GomokuBoard) -> dict[str, Any] | None:
            children = [child for child in node.children if child.proof == 0]
            if not children:
                return None
            chosen = min(children, key=lambda child: child.move or (-1, -1))
            move = chosen.move
            if move is None:
                return None
            state.place_stone(attacker, move)
            try:
                if state.check_victory(move, attacker):
                    return {"move": list(move), "terminal": "win"}
                if not chosen.children:
                    return None
                defense_node = chosen
                defenses = legal_moves(state)
                by_move = {child.move: child for child in defense_node.children}
                if set(by_move) != set(defenses) or len(by_move) != len(defense_node.children):
                    return None
                exceptions: list[dict[str, Any]] = []
                defender = "W" if attacker == "B" else "B"
                wins_before = set(immediate_wins(state, attacker))
                counter_wins = set(immediate_wins(state, defender))
                if counter_wins:
                    return None
                for defense in defenses:
                    child = by_move[defense]
                    if wins_before and (defense not in wins_before or len(wins_before) > 1):
                        if child.proof != 0:
                            return None
                        continue
                    state.place_stone(defender, defense)
                    try:
                        proof = build_or(child, state)
                        if proof is None:
                            return None
                        exceptions.append({"move": list(defense), "proof": proof})
                    finally:
                        state.undo()
                return {
                    "move": list(move),
                    "defense_coverage": len(defenses),
                    "exceptions": exceptions,
                }
            finally:
                state.undo()

        return build_or(root, board)
