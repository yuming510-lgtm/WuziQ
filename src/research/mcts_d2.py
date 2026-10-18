"""Tactical, terminal-rollout UCT for the V2 pure-MCTS study."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Optional

from ..ai import (
    WIN_SCORE,
    SearchConfig,
    SearchResult,
    _SearchTimeout,
    _check_deadline,
    _first_legal_move,
    _other,
    generate_candidates,
)
from ..gomoku import Coordinate, GomokuBoard, Player


@dataclass
class _Node:
    move: Coordinate | None
    player_just_moved: Player
    parent: Optional["_Node"] = None
    children: list["_Node"] = field(default_factory=list)
    unexpanded: list[Coordinate] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0
    initialized: bool = False


def _uct_child(node: _Node, exploration: float) -> _Node:
    unvisited = [child for child in node.children if child.visits == 0]
    if unvisited:
        return unvisited[0]
    log_parent = math.log(max(1, node.visits))
    return max(
        node.children,
        key=lambda child: (
            child.wins / child.visits
            + exploration * math.sqrt(log_parent / child.visits),
            -child.move[0],
            -child.move[1],
        ),
    )


def _local_pool(board: GomokuBoard, radius: int) -> set[Coordinate]:
    if not board.history:
        return {(board.size // 2, board.size // 2)}
    pool: set[Coordinate] = set()
    for _, (row, col) in board.history:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                coord = row + dr, col + dc
                if board.in_bounds(*coord) and board.get(coord) is None:
                    pool.add(coord)
    return pool


def _add_neighborhood(
    pool: set[Coordinate], board: GomokuBoard, coord: Coordinate, radius: int
) -> None:
    row, col = coord
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            candidate = row + dr, col + dc
            if board.in_bounds(*candidate) and board.get(candidate) is None:
                pool.add(candidate)


def _rollout(
    board: GomokuBoard,
    to_move: Player,
    root_player: Player,
    rng: random.Random,
    config: SearchConfig,
    deadline: float,
) -> tuple[float, int]:
    """Roll to a real win or full board; interrupted trials are not scored."""

    pool = _local_pool(board, config.candidate_radius)
    plies = 0
    while len(board.history) < board.size * board.size:
        _check_deadline(deadline)
        moves = sorted(coord for coord in pool if board.get(coord) is None)
        if not moves:
            moves = [
                (row, col)
                for row in range(board.size)
                for col in range(board.size)
                if board.get((row, col)) is None
            ]
        if not moves:
            return 0.5, plies

        opponent = _other(to_move)
        if config.mcts_d2_tactical_rollout:
            winning = [
                coord
                for coord in moves
                if board.potential_line_length(coord, to_move) >= 5
            ]
            blocking = [] if winning else [
                coord
                for coord in moves
                if board.potential_line_length(coord, opponent) >= 5
            ]
            choices = winning or blocking or moves
        else:
            choices = moves

        coord = rng.choice(choices)
        pool.discard(coord)
        board.place_stone(to_move, coord)
        plies += 1
        if board.check_victory(coord, to_move):
            return (1.0 if to_move == root_player else 0.0), plies
        if len(board.history) >= board.size * board.size:
            return 0.5, plies
        _add_neighborhood(pool, board, coord, config.candidate_radius)
        to_move = opponent
    return 0.5, plies


def _tree_node_count(root: _Node) -> int:
    count = 0
    pending = [root]
    while pending:
        node = pending.pop()
        count += 1
        pending.extend(node.children)
    return count


class MCTSAgent:
    """Pure UCT agent; a runner may retain it to reuse a matching subtree."""

    def __init__(self) -> None:
        self._root: _Node | None = None
        self._history: tuple[tuple[Player, Coordinate], ...] = ()
        self._last_move: Coordinate | None = None
        self._root_player: Player | None = None

    def _reuse_root(
        self, board: GomokuBoard, player: Player, enabled: bool
    ) -> tuple[_Node | None, int]:
        if (
            not enabled
            or self._root is None
            or self._last_move is None
            or self._root_player is None
            or len(board.history) < len(self._history)
            or tuple(board.history[: len(self._history)]) != self._history
        ):
            return None, 0

        suffix = board.history[len(self._history) :]
        if not suffix or suffix[0] != (self._root_player, self._last_move):
            return None, 0
        node = next(
            (child for child in self._root.children if child.move == suffix[0][1]), None
        )
        if node is None:
            return None, 0
        for side, coord in suffix[1:]:
            if side != _other(node.player_just_moved):
                return None, 0
            node = next((child for child in node.children if child.move == coord), None)
            if node is None:
                return None, 0
        if _other(node.player_just_moved) != player:
            return None, 0
        reused = _tree_node_count(node)
        node.parent = None
        return node, reused

    def choose_move(
        self,
        board: GomokuBoard,
        player: Player,
        config: SearchConfig,
        seed: int,
    ) -> SearchResult:
        started = time.perf_counter()
        fallback = _first_legal_move(board)
        if config.time_limit_seconds == 0:
            result = SearchResult(
                fallback,
                0,
                0,
                0,
                0,
                0,
                time.perf_counter() - started,
                True,
                (fallback,),
            )
            self._remember(board, player, result.move, None)
            return result

        deadline = started + max(0.0, config.time_limit_seconds - 0.001)
        rng = random.Random(seed)
        candidates = generate_candidates(
            board, player, _other(player), radius=config.candidate_radius
        )
        if not candidates:
            raise ValueError("Board has no legal move")

        if config.mcts_d2_tactical_root:
            winning = [
                coord
                for coord in candidates
                if board.potential_line_length(coord, player) >= 5
            ]
            if winning:
                move = min(winning)
                result = SearchResult(
                    move,
                    WIN_SCORE,
                    len(candidates),
                    0,
                    0,
                    0,
                    time.perf_counter() - started,
                    time.perf_counter() > deadline,
                    tuple(winning),
                    root_candidates=len(candidates),
                )
                self._remember(board, player, move, None)
                return result
            opponent = _other(player)
            threats = [
                coord
                for coord in candidates
                if board.potential_line_length(coord, opponent) >= 5
            ]
            if len(threats) == 1:
                move = threats[0]
                result = SearchResult(
                    move,
                    0,
                    len(candidates),
                    0,
                    0,
                    0,
                    time.perf_counter() - started,
                    time.perf_counter() > deadline,
                    (move,),
                    root_candidates=len(candidates),
                )
                self._remember(board, player, move, None)
                return result

        root, reused_nodes = self._reuse_root(
            board, player, config.mcts_d2_node_reuse
        )
        if root is None:
            root = _Node(None, _other(player), initialized=True)
            if config.mcts_d2_fair_root:
                rng.shuffle(candidates)
                root.children = [
                    _Node(coord, player, parent=root) for coord in candidates
                ]
            else:
                root.unexpanded = list(candidates)
        elif config.mcts_d2_fair_root:
            known = {child.move for child in root.children}
            missing = [coord for coord in candidates if coord not in known]
            rng.shuffle(missing)
            root.children.extend(
                _Node(coord, player, parent=root) for coord in missing
            )
            root.initialized = True
            root.unexpanded.clear()

        simulations = 0
        nodes = 0
        max_depth = 0
        rollout_plies_total = 0
        terminal_simulations = 0
        truncated_simulations = 0
        timed_out = False
        base_history_length = len(board.history)

        while simulations < config.mcts_simulations:
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            node = root
            path = [root]
            winner: Player | None = None
            terminal = False
            iteration_nodes = 0
            try:
                while node.children and (
                    (
                        node is root
                        and config.mcts_d2_fair_root
                        and any(child.visits == 0 for child in node.children)
                    )
                    or (node.initialized and not node.unexpanded)
                ):
                    _check_deadline(deadline)
                    if (
                        node is root
                        and config.mcts_d2_fair_root
                        and any(child.visits == 0 for child in node.children)
                    ):
                        child = next(child for child in node.children if child.visits == 0)
                    else:
                        child = _uct_child(node, config.uct_constant)
                    assert child.move is not None
                    to_move = _other(node.player_just_moved)
                    board.place_stone(to_move, child.move)
                    iteration_nodes += 1
                    if board.check_victory(child.move, to_move):
                        winner = to_move
                        terminal = True
                    elif len(board.history) >= board.size * board.size:
                        terminal = True
                    node = child
                    path.append(node)
                    if terminal:
                        break

                if not terminal:
                    if not node.initialized:
                        to_move = _other(node.player_just_moved)
                        node.unexpanded = generate_candidates(
                            board,
                            to_move,
                            _other(to_move),
                            radius=config.candidate_radius,
                        )
                        node.initialized = True
                    if node.unexpanded:
                        _check_deadline(deadline)
                        index = rng.randrange(len(node.unexpanded))
                        coord = node.unexpanded.pop(index)
                        to_move = _other(node.player_just_moved)
                        board.place_stone(to_move, coord)
                        iteration_nodes += 1
                        child = _Node(coord, to_move, parent=node, initialized=True)
                        node.children.append(child)
                        node = child
                        path.append(node)
                        if board.check_victory(coord, to_move):
                            winner = to_move
                            terminal = True
                        elif len(board.history) >= board.size * board.size:
                            terminal = True
                        else:
                            next_player = _other(to_move)
                            child.unexpanded = generate_candidates(
                                board,
                                next_player,
                                to_move,
                                radius=config.candidate_radius,
                            )

                if not terminal:
                    rollout_value, rollout_plies = _rollout(
                        board,
                        _other(node.player_just_moved),
                        player,
                        rng,
                        config,
                        deadline,
                    )
                    iteration_nodes += rollout_plies
                    rollout_plies_total += rollout_plies
                else:
                    rollout_value = (
                        0.5
                        if winner is None
                        else 1.0 if winner == player else 0.0
                    )

                simulations += 1
                terminal_simulations += 1
                nodes += iteration_nodes
                max_depth = max(max_depth, len(path) - 1)
                cursor: _Node | None = node
                while cursor is not None:
                    cursor.visits += 1
                    cursor.wins += (
                        rollout_value
                        if cursor.player_just_moved == player
                        else 1.0 - rollout_value
                    )
                    cursor = cursor.parent
            except _SearchTimeout:
                timed_out = True
                break
            finally:
                while len(board.history) > base_history_length:
                    board.undo()

        visited = [child for child in root.children if child.visits and child.move]
        if not visited:
            move = candidates[0]
            root_value = 0
            optimal_moves = (move,)
        else:
            best = max(
                visited,
                key=lambda child: (
                    child.visits,
                    child.wins / child.visits,
                    -child.move[0],
                    -child.move[1],
                ),
            )
            assert best.move is not None
            move = best.move
            root_value = round((2 * best.wins / best.visits - 1) * WIN_SCORE)
            optimal_moves = tuple(
                sorted(
                    child.move
                    for child in visited
                    if child.visits == best.visits
                    and child.wins / child.visits == best.wins / best.visits
                    and child.move is not None
                )
            )

        root_count = len(candidates)
        by_move = {child.move: child.visits for child in root.children}
        min_visits = min(by_move.get(coord, 0) for coord in candidates)
        result = SearchResult(
            move=move,
            root_value=root_value,
            nodes=nodes,
            completed_depth=max_depth,
            cutoffs=0,
            simulations=simulations,
            elapsed_seconds=time.perf_counter() - started,
            timed_out=timed_out,
            optimal_moves=optimal_moves,
            rollout_plies=rollout_plies_total,
            terminal_simulations=terminal_simulations,
            truncated_simulations=truncated_simulations,
            root_candidates=root_count,
            root_min_visits=min_visits,
            reused_nodes=reused_nodes,
        )
        self._remember(board, player, move, root)
        return result

    def _remember(
        self,
        board: GomokuBoard,
        player: Player,
        move: Coordinate,
        root: _Node | None,
    ) -> None:
        self._root = root
        self._history = tuple(board.history)
        self._last_move = move
        self._root_player = player
