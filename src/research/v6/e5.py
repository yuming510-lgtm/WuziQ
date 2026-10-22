"""PVS/TT/Zobrist Alpha-Beta player with reproducible component ablations."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any, Iterable

from ...ai import WIN_SCORE, SearchConfig, _SearchTimeout as CoreSearchTimeout, _candidate_moves, _other, evaluate_board
from ...gomoku import Coordinate, GomokuBoard, Player, select_ai_move
from .tactics import SearchTimeout, immediate_wins
from .types import V6MoveResult


@dataclass(frozen=True)
class E5Options:
    pvs: bool = True
    transposition_table: bool = True
    move_ordering: bool = True
    tactical_extensions: bool = True
    extension_limit: int = 2
    table_limit: int = 200_000


@dataclass(frozen=True)
class TTEntry:
    depth: int
    score: int
    bound: str
    move: Coordinate | None


class Zobrist:
    """Deterministic 64-bit position hashing, including side-to-move and size."""

    def __init__(self, size: int, seed: int = 604_2026) -> None:
        self.size = size
        rng = random.Random(seed + size * 1_000_003)
        self.stones = {
            player: [[rng.getrandbits(64) for _ in range(size)] for _ in range(size)]
            for player in ("B", "W")
        }
        self.side = {player: rng.getrandbits(64) for player in ("B", "W")}
        self.size_key = rng.getrandbits(64)

    def position(self, board: GomokuBoard, to_move: Player) -> int:
        value = self.size_key ^ self.side[to_move]
        for row in range(board.size):
            for col in range(board.size):
                player = board.grid[row][col]
                if player is not None:
                    value ^= self.stones[player][row][col]
        return value

    def after_move(self, key: int, player: Player, coord: Coordinate) -> int:
        other = _other(player)
        return key ^ self.stones[player][coord[0]][coord[1]] ^ self.side[player] ^ self.side[other]


class E5Search:
    """Iterative-deepening negamax with PVS, bound-aware TT and bounded extensions."""

    def __init__(self, options: E5Options | None = None) -> None:
        self.options = options or E5Options()
        self._tt: dict[int, TTEntry] = {}
        self._deadline = math.inf
        self._nodes = 0
        self._cutoffs = 0
        self._tt_hits = 0
        self._pvs_researches = 0
        self._extensions = 0
        self._killer_moves: dict[int, list[Coordinate]] = {}
        self._history: dict[tuple[Player, Coordinate], int] = {}
        self._zobrist: Zobrist | None = None

    def _check(self) -> None:
        if time.perf_counter() >= self._deadline:
            raise SearchTimeout

    def _win_cells(self, board: GomokuBoard, player: Player, moves: list[Coordinate]) -> set[Coordinate]:
        result: set[Coordinate] = set()
        for move in moves:
            if board.potential_line_length(move, player) >= 5:
                result.add(move)
        return result

    def _ordered_moves(
        self,
        board: GomokuBoard,
        player: Player,
        moves: list[Coordinate],
        depth: int,
        tt_move: Coordinate | None,
    ) -> list[Coordinate]:
        if not self.options.move_ordering:
            return moves
        opponent = _other(player)
        killers = self._killer_moves.get(depth, ())
        def priority(move: Coordinate) -> tuple[int, int, int, int, int, int]:
            self._check()
            own = board.potential_line_length(move, player)
            other = board.potential_line_length(move, opponent)
            tactical = 2 if own >= 5 else 1 if other >= 5 else 0
            details = board.line_details(move, player)
            open_threes = sum(length == 3 and int(forward) + int(backward) == 2 for length, forward, backward in details)
            history = self._history.get((player, move), 0)
            center_distance = abs(move[0] - board.size // 2) + abs(move[1] - board.size // 2)
            return (
                int(move == tt_move),
                tactical,
                open_threes,
                int(move in killers),
                history,
                -center_distance,
            )
        return sorted(moves, key=lambda move: (*priority(move), -move[0], -move[1]), reverse=True)

    def _has_tactical_extension(self, board: GomokuBoard, player: Player, moves: list[Coordinate]) -> bool:
        if self._win_cells(board, player, moves):
            return True
        return bool(self._win_cells(board, _other(player), moves))

    def _negamax(
        self,
        board: GomokuBoard,
        player: Player,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        key: int,
        extensions_left: int,
    ) -> int:
        self._check()
        self._nodes += 1
        alpha_original, beta_original = alpha, beta
        entry = self._tt.get(key) if self.options.transposition_table else None
        tt_move: Coordinate | None = None
        if entry is not None:
            tt_move = entry.move
            if entry.depth >= depth:
                self._tt_hits += 1
                if entry.bound == "EXACT":
                    return entry.score
                if entry.bound == "LOWER":
                    alpha = max(alpha, entry.score)
                elif entry.bound == "UPPER":
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    self._cutoffs += 1
                    return entry.score

        moves = _candidate_moves(board, player, _other(player), 2, self._deadline)
        if depth <= 0:
            if self.options.tactical_extensions and extensions_left > 0 and self._has_tactical_extension(board, player, moves):
                depth = 1
                extensions_left -= 1
                self._extensions += 1
            else:
                return evaluate_board(board, player)
        if not moves:
            return evaluate_board(board, player)
        moves = self._ordered_moves(board, player, moves, ply, tt_move)
        best = -WIN_SCORE * 2
        best_move: Coordinate | None = None
        for index, move in enumerate(moves):
            self._check()
            board.place_stone(player, move)
            child_key = self._zobrist.after_move(key, player, move)  # type: ignore[union-attr]
            try:
                if board.check_victory(move, player):
                    score = WIN_SCORE - ply
                else:
                    next_player = _other(player)
                    if self.options.pvs and index > 0:
                        score = -self._negamax(
                            board, next_player, depth - 1, -alpha - 1, -alpha,
                            ply + 1, child_key, extensions_left,
                        )
                        if alpha < score < beta:
                            self._pvs_researches += 1
                            score = -self._negamax(
                                board, next_player, depth - 1, -beta, -alpha,
                                ply + 1, child_key, extensions_left,
                            )
                    else:
                        score = -self._negamax(
                            board, next_player, depth - 1, -beta, -alpha,
                            ply + 1, child_key, extensions_left,
                        )
            finally:
                board.undo()
            if score > best:
                best, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self._cutoffs += 1
                if move not in self._killer_moves.setdefault(ply, []):
                    self._killer_moves[ply] = ([move] + self._killer_moves[ply])[:2]
                self._history[(player, move)] = self._history.get((player, move), 0) + depth * depth
                break
        if self.options.transposition_table:
            if self._tt.get(key) is None or depth >= self._tt[key].depth:
                bound = "UPPER" if best <= alpha_original else "LOWER" if best >= beta_original else "EXACT"
                if len(self._tt) >= self.options.table_limit:
                    # Deterministic bounded replacement: discard the shallowest entry.
                    shallow = min(self._tt, key=lambda item: (self._tt[item].depth, item))
                    del self._tt[shallow]
                self._tt[key] = TTEntry(depth, best, bound, best_move)
        return best

    def choose_move(
        self,
        board: GomokuBoard,
        player: Player,
        config: SearchConfig,
        seed: int,
        *,
        max_depth: int = 24,
    ) -> V6MoveResult:
        started = time.perf_counter()
        fallback = select_ai_move(board, player, _other(player), rng=random.Random(seed))
        reserve = min(0.025, max(0.010, config.time_limit_seconds * 0.20))
        self._deadline = started + max(0.0, config.time_limit_seconds - reserve)
        self._nodes = self._cutoffs = self._tt_hits = self._pvs_researches = self._extensions = 0
        self._killer_moves = {}
        self._history = {}
        self._zobrist = Zobrist(board.size)
        best_move = fallback
        best_value = evaluate_board(board, player)
        completed_depth = 0
        iterations: list[dict[str, Any]] = []
        timed_out = False
        if config.time_limit_seconds <= 0:
            return V6MoveResult(
                fallback, best_value, 0, 0, 0, 0, time.perf_counter() - started, True,
                stats={"tt_hits": 0, "pvs_researches": 0, "extensions": 0, "tt_size": len(self._tt), "depth_iterations": []},
            )
        # Exact one-ply tactics take precedence over an interrupted iterative
        # deepening pass. This keeps even the 50ms E5 setting from missing an
        # immediate win or the sole legal block just because ordering ran long.
        root_wins = immediate_wins(board, player)
        if root_wins:
            return V6MoveResult(
                root_wins[0], WIN_SCORE, 0, 1, 0, 0,
                time.perf_counter() - started, False,
                stats={
                    "tt_hits": 0, "pvs_researches": 0, "extensions": 0,
                    "tt_size": len(self._tt), "depth_iterations": [],
                    "root_tactical": "immediate_win",
                },
            )
        forced_blocks = immediate_wins(board, _other(player))
        if len(forced_blocks) == 1:
            return V6MoveResult(
                forced_blocks[0], -WIN_SCORE, 0, 1, 0, 0,
                time.perf_counter() - started, False,
                stats={
                    "tt_hits": 0, "pvs_researches": 0, "extensions": 0,
                    "tt_size": len(self._tt), "depth_iterations": [],
                    "root_tactical": "forced_block",
                },
            )
        if self.options.transposition_table and len(self._tt) > self.options.table_limit:
            self._tt.clear()
        try:
            root_moves = _candidate_moves(board, player, _other(player), 2, self._deadline)
            if not root_moves:
                return V6MoveResult(
                    fallback, best_value, 0, 0, 0, 0, time.perf_counter() - started, False,
                    stats={"tt_hits": 0, "pvs_researches": 0, "extensions": 0, "tt_size": len(self._tt), "depth_iterations": []},
                )
            for depth in range(1, max_depth + 1):
                self._check()
                iteration_start = time.perf_counter()
                root_key = self._zobrist.position(board, player)
                entry = self._tt.get(root_key) if self.options.transposition_table else None
                moves = self._ordered_moves(board, player, root_moves, 0, entry.move if entry else None)
                alpha, beta = -WIN_SCORE * 2, WIN_SCORE * 2
                value_best = -WIN_SCORE * 2
                move_best = moves[0]
                for index, move in enumerate(moves):
                    self._check()
                    board.place_stone(player, move)
                    child_key = self._zobrist.after_move(root_key, player, move)
                    try:
                        if board.check_victory(move, player):
                            value = WIN_SCORE - 1
                        elif self.options.pvs and index > 0:
                            value = -self._negamax(
                                board, _other(player), depth - 1, -alpha - 1, -alpha,
                                1, child_key, self.options.extension_limit,
                            )
                            if alpha < value < beta:
                                self._pvs_researches += 1
                                value = -self._negamax(
                                    board, _other(player), depth - 1, -beta, -alpha,
                                    1, child_key, self.options.extension_limit,
                                )
                        else:
                            value = -self._negamax(
                                board, _other(player), depth - 1, -beta, -alpha,
                                1, child_key, self.options.extension_limit,
                            )
                    finally:
                        board.undo()
                    if value > value_best:
                        value_best, move_best = value, move
                    alpha = max(alpha, value_best)
                self._check()
                best_move, best_value, completed_depth = move_best, value_best, depth
                iterations.append({
                    "depth": depth,
                    "root_value": value_best,
                    "move": list(move_best),
                    "nodes": self._nodes,
                    "elapsed_seconds": time.perf_counter() - iteration_start,
                })
                if value_best >= WIN_SCORE - depth:
                    break
        except (SearchTimeout, CoreSearchTimeout):
            timed_out = True
        return V6MoveResult(
            best_move,
            best_value,
            self._nodes,
            completed_depth,
            self._cutoffs,
            0,
            time.perf_counter() - started,
            timed_out,
            stats={
                "tt_hits": self._tt_hits,
                "pvs_researches": self._pvs_researches,
                "extensions": self._extensions,
                "tt_size": len(self._tt),
                "depth_iterations": iterations,
                "zobrist_root": self._zobrist.position(board, player),
                "pvs": self.options.pvs,
                "transposition_table": self.options.transposition_table,
                "move_ordering": self.options.move_ordering,
                "tactical_extensions": self.options.tactical_extensions,
            },
        )


ABLATION_VARIANTS: tuple[tuple[str, E5Options], ...] = (
    ("E5_base", E5Options(False, False, False, False)),
    ("E5_PVS_only", E5Options(True, False, False, False)),
    ("E5_TT_only", E5Options(False, True, False, False)),
    ("E5_ordering_only", E5Options(False, False, True, False)),
    ("E5_extensions_only", E5Options(False, False, False, True)),
    ("E5_all", E5Options(True, True, True, True)),
)


def run_ablation(
    positions: Iterable[tuple[str, GomokuBoard, Player]],
    *,
    depth: int = 3,
    time_limit_seconds: float = 30.0,
    seed: int = 510,
) -> list[dict[str, Any]]:
    """Compare each component on identical positions, fixed depth and budget."""
    rows: list[dict[str, Any]] = []
    for position_id, board, player in positions:
        for variant, options in ABLATION_VARIANTS:
            result = E5Search(options).choose_move(
                board, player, SearchConfig(max_depth=depth, time_limit_seconds=time_limit_seconds),
                seed, max_depth=depth,
            )
            rows.append({
                "position_id": position_id,
                "variant": variant,
                "move": list(result.move),
                "root_value": result.root_value,
                "nodes": result.nodes,
                "cutoffs": result.cutoffs,
                "completed_depth": result.completed_depth,
                "elapsed_seconds": result.elapsed_seconds,
                "timed_out": result.timed_out,
                **result.stats,
            })
    return rows
