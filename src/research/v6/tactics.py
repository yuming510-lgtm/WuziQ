"""Proof-safe forcing-threat search for the V6 tactical Gomoku players.

Attackers search a restricted set of threat moves, while defender nodes cover
every legal reply. A found proof is therefore sound; a missing proof says only
that the bounded search found none.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Any, Callable

from ...gomoku import Coordinate, GomokuBoard, Player


class SearchTimeout(Exception):
    """Internal deadline signal. It is never converted into a proof."""


class SearchLimit(Exception):
    """Internal node-limit signal. It is never converted into a proof."""


@dataclass(frozen=True)
class TacticalStats:
    nodes: int
    defense_nodes: int
    critical_defenses: int
    max_attack_depth: int
    threat_nodes: int = 0
    dependency_edges: int = 0
    defense_conflicts: int = 0


@dataclass(frozen=True)
class Threat:
    gain: Coordinate
    pattern: str
    cost_squares: tuple[Coordinate, ...]
    rest_squares: tuple[Coordinate, ...]


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


@lru_cache(maxsize=16)
def _five_windows(size: int) -> tuple[tuple[Coordinate, ...], ...]:
    windows: list[tuple[Coordinate, ...]] = []
    for row in range(size):
        for col in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row, end_col = row + 4 * dr, col + 4 * dc
                if 0 <= end_row < size and 0 <= end_col < size:
                    windows.append(tuple((row + i * dr, col + i * dc) for i in range(5)))
    return tuple(windows)


def legal_moves(board: GomokuBoard) -> list[Coordinate]:
    return [
        (row, col)
        for row in range(board.size)
        for col in range(board.size)
        if board.grid[row][col] is None
    ]


def immediate_wins(
    board: GomokuBoard,
    player: Player,
    check: Callable[[], None] | None = None,
) -> list[Coordinate]:
    windows, own, opponent, empties = _line_counts(board, player, check)
    wins: set[Coordinate] = set()
    for index, _window in enumerate(windows):
        if check:
            check()
        if opponent[index] == 0 and own[index] == 4 and len(empties[index]) == 1:
            wins.add(empties[index][0])
    return sorted(wins)


def _line_counts(
    board: GomokuBoard,
    player: Player,
    check: Callable[[], None] | None = None,
) -> tuple[list[tuple[Coordinate, ...]], list[int], list[int], list[list[Coordinate]]]:
    opponent = _other(player)
    windows = _five_windows(board.size)
    own_counts: list[int] = []
    opponent_counts: list[int] = []
    empties: list[list[Coordinate]] = []
    for index, window in enumerate(windows):
        if check and index % 16 == 0:
            check()
        own = 0
        other = 0
        empty: list[Coordinate] = []
        for coord in window:
            value = board.grid[coord[0]][coord[1]]
            own += value == player
            other += value == opponent
            if value is None:
                empty.append(coord)
        own_counts.append(own)
        opponent_counts.append(other)
        empties.append(empty)
    return list(windows), own_counts, opponent_counts, empties


def vcf_attack_moves(
    board: GomokuBoard,
    attacker: Player,
    check: Callable[[], None] | None = None,
) -> list[Coordinate]:
    """Legal gains that can create an immediate winning continuation."""
    wins = set(immediate_wins(board, attacker, check))
    windows, own, opponent, empties = _line_counts(board, attacker, check)
    candidates = set(wins)
    for index, _window in enumerate(windows):
        if opponent[index] == 0 and own[index] == 3 and len(empties[index]) == 2:
            candidates.update(empties[index])
    return _ordered_attack_gains(board, attacker, candidates, check)


def live_three_attack_moves(
    board: GomokuBoard,
    attacker: Player,
    check: Callable[[], None] | None = None,
) -> list[Coordinate]:
    """Moves creating a legal three-plus-two window with two four continuations."""
    windows, own, opponent, empties = _line_counts(board, attacker, check)
    followups: dict[Coordinate, set[Coordinate]] = {}
    for index, _window in enumerate(windows):
        if opponent[index] != 0 or own[index] != 2 or len(empties[index]) != 3:
            continue
        for gain in empties[index]:
            followups.setdefault(gain, set()).update(c for c in empties[index] if c != gain)
    return sorted(
        move for move, continuations in followups.items()
        if len(continuations) >= 2 and board.get(move) is None
    )


def has_live_three(board: GomokuBoard, attacker: Player) -> bool:
    _windows, own, opponent, empties = _line_counts(board, attacker)
    return any(
        opponent[index] == 0 and own[index] == 3 and len(empties[index]) == 2
        for index in range(len(own))
    )


def vct_attack_moves(
    board: GomokuBoard,
    attacker: Player,
    check: Callable[[], None] | None = None,
) -> list[Coordinate]:
    """Legal immediate wins, four threats, and open-three threat gains."""
    candidates = set(vcf_attack_moves(board, attacker, check)) | set(live_three_attack_moves(board, attacker, check))
    return _ordered_attack_gains(board, attacker, candidates, check)


def _ordered_attack_gains(
    board: GomokuBoard,
    attacker: Player,
    candidates: set[Coordinate],
    check: Callable[[], None] | None = None,
) -> list[Coordinate]:
    ranked: list[tuple[tuple[int, int, int, int, int], Coordinate]] = []
    center = board.size // 2
    for move in candidates:
        if check:
            check()
        board.place_stone(attacker, move)
        try:
            windows, own, opponent, empties = _line_counts(board, attacker, check)
            four_cells: set[Coordinate] = set()
            three_windows = 0
            for index, _window in enumerate(windows):
                if opponent[index] != 0:
                    continue
                if own[index] == 4 and len(empties[index]) == 1:
                    four_cells.add(empties[index][0])
                elif own[index] == 3 and len(empties[index]) == 2:
                    three_windows += 1
            is_win = int(board.check_victory(move, attacker))
            ranked.append((
                (
                    is_win,
                    len(four_cells),
                    three_windows,
                    -abs(move[0] - center) - abs(move[1] - center),
                    -move[0] * board.size - move[1],
                ),
                move,
            ))
        finally:
            board.undo()
    ranked.sort(reverse=True)
    return [move for _score, move in ranked]


def _threat_after_move(
    board: GomokuBoard,
    attacker: Player,
    gain: Coordinate,
    check: Callable[[], None] | None = None,
) -> Threat:
    board.place_stone(attacker, gain)
    try:
        windows, own, opponent, empties = _line_counts(board, attacker, check)
        immediate: set[Coordinate] = set()
        rest: set[Coordinate] = set()
        live: set[Coordinate] = set()
        for index, window in enumerate(windows):
            if opponent[index] != 0:
                continue
            if own[index] == 4 and len(empties[index]) == 1:
                immediate.add(empties[index][0])
                rest.update(empties[index])
            elif own[index] == 3 and len(empties[index]) == 2:
                live.update(empties[index])
                rest.update(empties[index])
        if immediate:
            return Threat(gain, "four", tuple(sorted(immediate)), tuple(sorted(rest)))
        return Threat(gain, "three", tuple(sorted(live)), tuple(sorted(rest)))
    finally:
        board.undo()


def build_threat_graph(
    board: GomokuBoard,
    attacker: Player,
    check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build a shallow TSS gain/cost/rest dependency graph.

    The graph is an attack move ordering aid. It does not prune defense nodes,
    and is never accepted as a win proof by itself.
    """
    gains = vct_attack_moves(board, attacker, check)
    threats = [_threat_after_move(board, attacker, gain, check) for gain in gains]
    edges: set[tuple[Coordinate, Coordinate]] = set()
    for threat in threats:
        if check:
            check()
        board.place_stone(attacker, threat.gain)
        try:
            next_gains = set(vct_attack_moves(board, attacker, check))
            for next_gain in next_gains:
                if next_gain != threat.gain and next_gain in threat.rest_squares:
                    edges.add((threat.gain, next_gain))
        finally:
            board.undo()
    conflicts: set[tuple[Coordinate, Coordinate]] = set()
    for i, left in enumerate(threats):
        if not left.cost_squares:
            continue
        for right in threats[i + 1 :]:
            if right.cost_squares and set(left.cost_squares).isdisjoint(right.cost_squares):
                conflicts.add(tuple(sorted((left.gain, right.gain))))
    return {
        "threats": threats,
        "dependencies": sorted(edges),
        "defense_conflicts": sorted(conflicts),
    }


def _attack_moves(
    board: GomokuBoard,
    attacker: Player,
    algorithm: str,
    graph: dict[str, Any] | None = None,
    check: Callable[[], None] | None = None,
) -> list[Coordinate]:
    if algorithm == "VCF":
        return vcf_attack_moves(board, attacker, check)
    if algorithm == "VCT":
        return vct_attack_moves(board, attacker, check)
    if algorithm == "TSS":
        graph = graph or build_threat_graph(board, attacker, check)
        threats: list[Threat] = graph["threats"]
        outgoing: dict[Coordinate, int] = {}
        for source, _target in graph["dependencies"]:
            outgoing[source] = outgoing.get(source, 0) + 1
        conflict_count: dict[Coordinate, int] = {}
        for left, right in graph["defense_conflicts"]:
            conflict_count[left] = conflict_count.get(left, 0) + 1
            conflict_count[right] = conflict_count.get(right, 0) + 1
        ordered = sorted(
            threats,
            key=lambda threat: (
                -outgoing.get(threat.gain, 0),
                -conflict_count.get(threat.gain, 0),
                -len(threat.cost_squares),
                threat.gain,
            ),
        )
        return [threat.gain for threat in ordered]
    raise ValueError(f"Unsupported tactical algorithm {algorithm}")


def _position_key(board: GomokuBoard, player: Player, depth: int) -> tuple[Any, ...]:
    return (board.size, player, depth, tuple(cell for row in board.grid for cell in row))


class TacticalSolver:
    """Depth-limited OR/AND forcing search with complete defense expansion."""

    def __init__(
        self,
        algorithm: str,
        deadline: float,
        *,
        max_attack_depth: int = 8,
        max_nodes: int = 100_000,
    ) -> None:
        if algorithm not in {"VCF", "VCT", "TSS"}:
            raise ValueError("TacticalSolver algorithm must be VCF, VCT or TSS")
        self.algorithm = algorithm
        self.deadline = deadline
        self.max_attack_depth = max_attack_depth
        self.max_nodes = max_nodes
        self.nodes = 0
        self.defense_nodes = 0
        self.critical_defenses = 0
        self.max_depth_seen = 0
        self.timed_out = False
        self.limit_hit = False
        self._memo: dict[tuple[Any, ...], dict[str, Any] | None] = {}
        self.threat_nodes = 0
        self.dependency_edges = 0
        self.defense_conflicts = 0

    def _check(self) -> None:
        if time.perf_counter() >= self.deadline:
            self.timed_out = True
            raise SearchTimeout
        if self.nodes >= self.max_nodes:
            self.limit_hit = True
            raise SearchLimit

    def solve(self, board: GomokuBoard, attacker: Player) -> dict[str, Any] | None:
        self._memo.clear()
        self.nodes = self.defense_nodes = self.critical_defenses = 0
        self.max_depth_seen = self.threat_nodes = self.dependency_edges = self.defense_conflicts = 0
        self.timed_out = self.limit_hit = False
        try:
            return self._solve_or(board, attacker, self.max_attack_depth, 0)
        except (SearchTimeout, SearchLimit):
            return None

    def _solve_or(
        self,
        board: GomokuBoard,
        attacker: Player,
        depth: int,
        attack_ply: int,
    ) -> dict[str, Any] | None:
        self._check()
        self.nodes += 1
        self.max_depth_seen = max(self.max_depth_seen, attack_ply)
        key = _position_key(board, attacker, depth)
        if key in self._memo:
            proof = self._memo[key]
            return None if proof is None else dict(proof)

        direct = immediate_wins(board, attacker, self._check)
        if direct:
            proof = {"move": list(direct[0]), "terminal": "win"}
            self._memo[key] = proof
            return dict(proof)
        if depth <= 0:
            self._memo[key] = None
            return None

        graph = None
        if self.algorithm == "TSS":
            graph = build_threat_graph(board, attacker, self._check)
            self.threat_nodes += len(graph["threats"])
            self.dependency_edges += len(graph["dependencies"])
            self.defense_conflicts += len(graph["defense_conflicts"])
        attacks = _attack_moves(board, attacker, self.algorithm, graph, self._check)
        self._check()
        opponent = _other(attacker)
        for attack in attacks:
            self._check()
            board.place_stone(attacker, attack)
            try:
                if board.check_victory(attack, attacker):
                    proof = {"move": list(attack), "terminal": "win"}
                    self._memo[key] = proof
                    return dict(proof)
                own_wins = immediate_wins(board, attacker, self._check)
                if self.algorithm == "VCF" and not own_wins:
                    continue
                if self.algorithm != "VCF" and not own_wins and not has_live_three(board, attacker):
                    continue
                defenses = legal_moves(board)
                if not defenses:
                    continue
                # These exact current win squares are all defenses that can
                # suppress the one-ply win. Any other legal reply leaves an
                # attacker win available. Also rule out every immediate
                # counter-win before using that reduction.
                counter_wins = set(immediate_wins(board, opponent, self._check))
                branches: list[dict[str, Any]] = []
                success = not counter_wins
                self.defense_nodes += len(defenses)
                for defense in defenses:
                    if not success:
                        break
                    self._check()
                    if defense in counter_wins:
                        success = False
                        break
                    if own_wins and (defense not in own_wins or len(own_wins) > 1):
                        continue
                    board.place_stone(opponent, defense)
                    try:
                        self.critical_defenses += 1
                        child = self._solve_or(board, attacker, depth - 1, attack_ply + 1)
                        if child is None:
                            success = False
                            break
                        branches.append({"move": list(defense), "proof": child})
                    finally:
                        board.undo()
                if success:
                    proof = {
                        "move": list(attack),
                        "defense_coverage": len(defenses),
                        "exceptions": branches,
                    }
                    self._memo[key] = proof
                    return dict(proof)
            finally:
                board.undo()
        self._memo[key] = None
        return None

    @property
    def stats(self) -> TacticalStats:
        return TacticalStats(
            self.nodes, self.defense_nodes, self.critical_defenses, self.max_depth_seen,
            self.threat_nodes, self.dependency_edges, self.defense_conflicts,
        )


def verify_proof_certificate(
    board: GomokuBoard,
    attacker: Player,
    certificate: dict[str, Any],
    *,
    deadline: float | None = None,
) -> bool:
    """Independently replay a compact proof and enumerate every legal defense."""
    try:
        def check() -> None:
            if deadline is not None and time.perf_counter() >= deadline:
                raise SearchTimeout

        if certificate.get("version") != 1 or certificate.get("attacker") != attacker:
            return False
        algorithm = str(certificate.get("algorithm"))
        root = certificate.get("root")
        if algorithm not in {"VCF", "VCT", "TSS", "PNS"} or not isinstance(root, dict):
            return False

        def verify_or(state: GomokuBoard, node: dict[str, Any]) -> bool:
            move_raw = node.get("move")
            if not isinstance(move_raw, (list, tuple)) or len(move_raw) != 2:
                return False
            move = int(move_raw[0]), int(move_raw[1])
            if not state.in_bounds(*move) or state.get(move) is not None:
                return False
            check()
            allowed = vcf_attack_moves(state, attacker, check) if algorithm == "VCF" else vct_attack_moves(state, attacker, check)
            if move not in allowed:
                return False
            state.place_stone(attacker, move)
            try:
                if node.get("terminal") == "win":
                    return state.check_victory(move, attacker)
                if state.check_victory(move, attacker):
                    return False
                wins_before_defense = immediate_wins(state, attacker, check)
                if algorithm == "VCF" and not wins_before_defense:
                    return False
                if algorithm != "VCF" and not wins_before_defense and not has_live_three(state, attacker):
                    return False
                defenses = legal_moves(state)
                if int(node.get("defense_coverage", -1)) != len(defenses):
                    return False
                exceptions_raw = node.get("exceptions")
                if not isinstance(exceptions_raw, list):
                    return False
                exceptions: dict[Coordinate, dict[str, Any]] = {}
                for branch in exceptions_raw:
                    coordinate = tuple(map(int, branch.get("move", ())))
                    if len(coordinate) != 2 or coordinate in exceptions:
                        return False
                    proof = branch.get("proof")
                    if not isinstance(proof, dict):
                        return False
                    exceptions[coordinate] = proof
                defender = _other(attacker)
                counter_wins = set(immediate_wins(state, defender, check))
                if counter_wins:
                    return False
                expected_exceptions: set[Coordinate] = set()
                for defense in defenses:
                    if defense in counter_wins:
                        return False
                    if wins_before_defense and (defense not in wins_before_defense or len(wins_before_defense) > 1):
                        if defense in exceptions:
                            return False
                        continue
                    expected_exceptions.add(defense)
                    state.place_stone(defender, defense)
                    try:
                        child = exceptions.get(defense)
                        if child is None or not verify_or(state, child):
                            return False
                    finally:
                        state.undo()
                return set(exceptions) == expected_exceptions
            finally:
                state.undo()

        return verify_or(board, root)
    except (KeyError, TypeError, ValueError, IndexError, RecursionError, SearchTimeout):
        return False


def pack_certificate(algorithm: str, attacker: Player, root: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "algorithm": algorithm, "attacker": attacker, "root": root}
