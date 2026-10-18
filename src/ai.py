"""Deterministic, budgeted Gomoku move selectors for four research models."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import random
import time
from typing import Optional

from .gomoku import Coordinate, GomokuBoard, Player, select_ai_move


WIN_SCORE = 1_000_000_000_000
_INFINITY = WIN_SCORE * 2
_MODEL_NAMES = {"A", "B", "C", "D", "D1"}
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_WINDOW_WEIGHTS = (0, 1, 8, 100, 2_000, 0)
_DEADLINE_RESERVE_SECONDS = 0.001


@dataclass(frozen=True)
class SearchConfig:
    """Shared search limits and settings for the four models."""

    max_depth: int = 4
    time_limit_seconds: float = 0.05
    mcts_simulations: int = 10_000
    mcts_rollout_max_plies: int = 24
    candidate_radius: int = 2
    uct_constant: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.time_limit_seconds < 0:
            raise ValueError("time_limit_seconds cannot be negative")
        if self.mcts_simulations < 1:
            raise ValueError("mcts_simulations must be at least 1")
        if self.mcts_rollout_max_plies < 1:
            raise ValueError("mcts_rollout_max_plies must be at least 1")
        if self.candidate_radius < 0:
            raise ValueError("candidate_radius cannot be negative")
        if self.uct_constant < 0:
            raise ValueError("uct_constant cannot be negative")


@dataclass(frozen=True)
class SearchResult:
    """Move plus comparable counters collected while choosing it."""

    move: Coordinate
    root_value: int
    nodes: int
    completed_depth: int
    cutoffs: int
    simulations: int
    elapsed_seconds: float
    timed_out: bool
    # For C this is a proven subset of root optima: the selected move is exact,
    # while unverified equal-value fail-low bounds and other ties are omitted.
    optimal_moves: tuple[Coordinate, ...] = field(default_factory=tuple)
    rollout_plies: int = 0
    terminal_simulations: int = 0
    truncated_simulations: int = 0
    root_candidates: int = 0
    root_min_visits: int = 0
    reused_nodes: int = 0

    @property
    def score(self) -> int:
        """Alias used by reports that call the root evaluation a score."""

        return self.root_value


class _SearchTimeout(Exception):
    """Internal control flow for a consumed wall-clock budget."""


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise _SearchTimeout


def _first_legal_move(board: GomokuBoard) -> Coordinate:
    if board.is_full():
        raise ValueError("Board has no legal move")
    center = (board.size // 2, board.size // 2)
    if board.get(center) is None:
        return center
    for row in range(board.size):
        for col in range(board.size):
            if board.get((row, col)) is None:
                return row, col
    raise ValueError("Board has no legal move")


def _would_win(board: GomokuBoard, coord: Coordinate, player: Player) -> bool:
    return board.potential_line_length(coord, player) >= 5


def _adjacent_stones(board: GomokuBoard, coord: Coordinate) -> int:
    row, col = coord
    return sum(
        1
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc)
        and board.in_bounds(row + dr, col + dc)
        and board.get((row + dr, col + dc)) is not None
    )


def _candidate_moves(
    board: GomokuBoard,
    player: Player,
    opponent: Player,
    radius: int,
    deadline: float | None = None,
) -> list[Coordinate]:
    if board.is_full():
        return []

    occupied = [
        (row, col)
        for row in range(board.size)
        for col in range(board.size)
        if board.get((row, col)) is not None
    ]
    if not occupied:
        return [(board.size // 2, board.size // 2)]

    nearby: set[Coordinate] = set()
    for row, col in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                coord = (row + dr, col + dc)
                if board.in_bounds(*coord) and board.get(coord) is None:
                    nearby.add(coord)

    if not nearby:
        nearby = {
            (row, col)
            for row in range(board.size)
            for col in range(board.size)
            if board.get((row, col)) is None
        }

    def priority(coord: Coordinate) -> tuple[int, int, int, int, int, int, int]:
        _check_deadline(deadline)
        own_run = board.potential_line_length(coord, player)
        opposing_run = board.potential_line_length(coord, opponent)
        own_win = int(own_run >= 5)
        block_win = int(opposing_run >= 5)
        adjacency = _adjacent_stones(board, coord)
        center_distance = abs(coord[0] - board.size // 2) + abs(
            coord[1] - board.size // 2
        )
        return (
            -own_win,
            -block_win,
            -max(own_run, opposing_run),
            -adjacency,
            center_distance,
            coord[0],
            coord[1],
        )

    return sorted(nearby, key=priority)


def generate_candidates(
    board: GomokuBoard,
    player: Player,
    opponent: Player,
    radius: int = 2,
) -> list[Coordinate]:
    """Return legal local moves in a stable, shared B/C ordering."""

    if radius < 0:
        raise ValueError("radius cannot be negative")
    return _candidate_moves(board, player, opponent, radius)


@lru_cache(maxsize=32)
def _five_windows(size: int) -> tuple[tuple[Coordinate, ...], ...]:
    windows: list[tuple[Coordinate, ...]] = []
    for row in range(size):
        for col in range(size):
            for dr, dc in _DIRECTIONS:
                end_row = row + 4 * dr
                end_col = col + 4 * dc
                if 0 <= end_row < size and 0 <= end_col < size:
                    windows.append(
                        tuple((row + step * dr, col + step * dc) for step in range(5))
                    )
    return tuple(windows)


def evaluate_board(board: GomokuBoard, perspective: Player) -> int:
    """Score unobstructed five-cell windows from ``perspective``."""

    opponent = _other(perspective)
    score = 0
    for window in _five_windows(board.size):
        own_count = 0
        opponent_count = 0
        for row, col in window:
            cell = board.grid[row][col]
            own_count += cell == perspective
            opponent_count += cell == opponent
        if own_count and not opponent_count:
            score += _WINDOW_WEIGHTS[own_count]
        elif opponent_count and not own_count:
            score -= _WINDOW_WEIGHTS[opponent_count]
    return score


def _minimax_value(
    board: GomokuBoard,
    to_move: Player,
    root_player: Player,
    depth: int,
    last_move: Coordinate | None,
    last_player: Player | None,
    config: SearchConfig,
    deadline: float,
    alpha: int,
    beta: int,
    use_alpha_beta: bool,
    counters: list[int],
) -> int:
    _check_deadline(deadline)
    counters[0] += 1
    if last_move is not None and last_player is not None:
        if board.check_victory(last_move, last_player):
            return WIN_SCORE if last_player == root_player else -WIN_SCORE
    if depth == 0 or len(board.history) >= board.size * board.size:
        return evaluate_board(board, root_player)

    moves = _candidate_moves(
        board, to_move, _other(to_move), config.candidate_radius, deadline
    )
    if not moves:
        return evaluate_board(board, root_player)

    maximizing = to_move == root_player
    best = -_INFINITY if maximizing else _INFINITY
    for coord in moves:
        _check_deadline(deadline)
        board.place_stone(to_move, coord)
        try:
            value = _minimax_value(
                board,
                _other(to_move),
                root_player,
                depth - 1,
                coord,
                to_move,
                config,
                deadline,
                alpha,
                beta,
                use_alpha_beta,
                counters,
            )
        finally:
            board.undo()
        if maximizing:
            best = max(best, value)
            if use_alpha_beta:
                alpha = max(alpha, best)
        else:
            best = min(best, value)
            if use_alpha_beta:
                beta = min(beta, best)
        if use_alpha_beta and beta <= alpha:
            counters[1] += 1
            break
    return best


def _search_root(
    board: GomokuBoard,
    root_player: Player,
    depth: int,
    root_moves: list[Coordinate],
    config: SearchConfig,
    deadline: float,
    use_alpha_beta: bool,
    counters: list[int],
) -> tuple[int, tuple[Coordinate, ...]]:
    best_value = -_INFINITY
    best_moves: list[Coordinate] = []
    alpha = -_INFINITY
    beta = _INFINITY
    for coord in root_moves:
        _check_deadline(deadline)
        board.place_stone(root_player, coord)
        try:
            value = _minimax_value(
                board,
                _other(root_player),
                root_player,
                depth - 1,
                coord,
                root_player,
                config,
                deadline,
                alpha,
                beta,
                use_alpha_beta,
                counters,
            )
        finally:
            board.undo()
        if value > best_value:
            best_value = value
            best_moves = [coord]
        elif value == best_value and not use_alpha_beta:
            # B has exact values for every root move; C omits equal bounds unless re-searched.
            best_moves.append(coord)
        if use_alpha_beta:
            alpha = max(alpha, best_value)
    return best_value, tuple(best_moves)


def _search_minimax(
    board: GomokuBoard,
    player: Player,
    config: SearchConfig,
    seed: int,
    use_alpha_beta: bool,
) -> SearchResult:
    started = time.perf_counter()
    deadline = started + max(
        0.0, config.time_limit_seconds - _DEADLINE_RESERVE_SECONDS
    )
    fallback = _first_legal_move(board)
    if config.time_limit_seconds == 0:
        return SearchResult(fallback, evaluate_board(board, player), 0, 0, 0, 0, 0.0, True)

    counters = [0, 0]
    completed_depth = 0
    best_move = fallback
    root_value = evaluate_board(board, player)
    optimal_moves: tuple[Coordinate, ...] = (fallback,)
    timed_out = False
    try:
        root_moves = _candidate_moves(
            board, player, _other(player), config.candidate_radius, deadline
        )
        if not root_moves:
            raise ValueError("Board has no legal move")
        best_move = root_moves[0]
        optimal_moves = (best_move,)
        for depth in range(1, config.max_depth + 1):
            value, best_moves = _search_root(
                board,
                player,
                depth,
                root_moves,
                config,
                deadline,
                use_alpha_beta,
                counters,
            )
            completed_depth = depth
            root_value = value
            optimal_moves = best_moves
            best_move = best_moves[0]
    except _SearchTimeout:
        timed_out = True

    return SearchResult(
        move=best_move,
        root_value=root_value,
        nodes=counters[0],
        completed_depth=completed_depth,
        cutoffs=counters[1],
        simulations=0,
        elapsed_seconds=time.perf_counter() - started,
        timed_out=timed_out,
        optimal_moves=optimal_moves,
    )


@dataclass
class _MCTSNode:
    move: Coordinate | None
    player_just_moved: Player
    parent: Optional["_MCTSNode"] = None
    children: list["_MCTSNode"] = field(default_factory=list)
    unexpanded: list[Coordinate] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0


def _uct_child(node: _MCTSNode, exploration: float) -> _MCTSNode:
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


def _rollout_moves(board: GomokuBoard, radius: int) -> list[Coordinate]:
    """Build the same local candidate set without the tree's expensive ordering."""

    occupied = [coord for _, coord in board.history]
    if not occupied:
        return [(board.size // 2, board.size // 2)]
    nearby: set[Coordinate] = set()
    for row, col in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                coord = (row + dr, col + dc)
                if board.in_bounds(*coord) and board.get(coord) is None:
                    nearby.add(coord)
    if nearby:
        return sorted(nearby)
    return [
        (row, col)
        for row in range(board.size)
        for col in range(board.size)
        if board.get((row, col)) is None
    ]


def _rollout(
    board: GomokuBoard,
    to_move: Player,
    root_player: Player,
    rng: random.Random,
    config: SearchConfig,
    deadline: float,
) -> tuple[float, int, bool]:
    plies = 0
    max_plies = min(
        board.size * board.size - len(board.history), config.mcts_rollout_max_plies
    )
    for _ in range(max_plies):
        _check_deadline(deadline)
        opponent = _other(to_move)
        moves = _rollout_moves(board, config.candidate_radius)
        if not moves:
            return 0.5, plies, True
        winning = [coord for coord in moves if board.check_victory(coord, to_move)]
        blocking = [] if winning else [
            coord for coord in moves if board.check_victory(coord, opponent)
        ]
        choices = winning or blocking or moves
        coord = rng.choice(choices)
        board.place_stone(to_move, coord)
        plies += 1
        if board.check_victory(coord, to_move):
            return (1.0 if to_move == root_player else 0.0), plies, True
        if len(board.history) >= board.size * board.size:
            return 0.5, plies, True
        to_move = opponent
    score = evaluate_board(board, root_player)
    estimate = 0.5 + 0.5 * math.tanh(score / 2_500.0)
    return estimate, plies, False


def _search_mcts(
    board: GomokuBoard,
    player: Player,
    config: SearchConfig,
    seed: int,
) -> SearchResult:
    started = time.perf_counter()
    deadline = started + max(
        0.0, config.time_limit_seconds - _DEADLINE_RESERVE_SECONDS
    )
    fallback = _first_legal_move(board)
    if config.time_limit_seconds == 0:
        return SearchResult(fallback, evaluate_board(board, player), 0, 0, 0, 0, 0.0, True)

    rng = random.Random(seed)
    root = _MCTSNode(None, _other(player))
    try:
        root.unexpanded = _candidate_moves(
            board, player, _other(player), config.candidate_radius, deadline
        )
    except _SearchTimeout:
        return SearchResult(
            fallback,
            evaluate_board(board, player),
            0,
            0,
            0,
            0,
            time.perf_counter() - started,
            True,
            (fallback,),
        )
    if not root.unexpanded:
        raise ValueError("Board has no legal move")
    fallback = root.unexpanded[0]
    simulations = 0
    nodes = 0
    max_depth = 0
    rollout_plies_total = 0
    terminal_simulations = 0
    truncated_simulations = 0
    root_candidate_count = len(root.unexpanded)
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
            while not node.unexpanded and node.children:
                _check_deadline(deadline)
                to_move = _other(node.player_just_moved)
                child = _uct_child(node, config.uct_constant)
                assert child.move is not None
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

            if not terminal and node.unexpanded:
                _check_deadline(deadline)
                to_move = _other(node.player_just_moved)
                index = rng.randrange(len(node.unexpanded))
                coord = node.unexpanded.pop(index)
                board.place_stone(to_move, coord)
                iteration_nodes += 1
                child = _MCTSNode(coord, to_move, parent=node)
                node.children.append(child)
                node = child
                path.append(node)
                if board.check_victory(coord, to_move):
                    winner = to_move
                    terminal = True
                elif len(board.history) >= board.size * board.size:
                    terminal = True
                else:
                    node.unexpanded = _candidate_moves(
                        board,
                        _other(to_move),
                        to_move,
                        config.candidate_radius,
                        deadline,
                    )

            if not terminal:
                rollout_value, rollout_plies, rollout_terminal = _rollout(
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
                rollout_plies = 0
                rollout_terminal = True
                rollout_value = (
                    0.5
                    if winner is None
                    else 1.0 if winner == player else 0.0
                )

            simulations += 1
            if rollout_terminal:
                terminal_simulations += 1
            else:
                truncated_simulations += 1
            nodes += iteration_nodes
            max_depth = max(max_depth, len(path) - 1)
            cursor: _MCTSNode | None = node
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

    visited_children = [child for child in root.children if child.visits]
    if not visited_children:
        best_move = fallback
        root_value = evaluate_board(board, player)
        optimal_moves = (fallback,)
    else:
        best_child = max(
            visited_children,
            key=lambda child: (
                child.visits,
                child.wins / child.visits,
                -child.move[0],
                -child.move[1],
            ),
        )
        assert best_child.move is not None
        best_move = best_child.move
        root_value = round((2 * best_child.wins / best_child.visits - 1) * WIN_SCORE)
        optimal_moves = tuple(
            sorted(
                child.move
                for child in visited_children
                if child.visits == best_child.visits
                and child.wins / child.visits == best_child.wins / best_child.visits
                and child.move is not None
            )
        )

    return SearchResult(
        move=best_move,
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
        root_candidates=root_candidate_count,
        root_min_visits=min(
            [child.visits for child in root.children]
            + [0] * max(0, root_candidate_count - len(root.children))
        ),
    )


def choose_move(
    board: GomokuBoard,
    player: Player,
    model: str,
    config: SearchConfig | None = None,
    seed: int = 0,
) -> SearchResult:
    """Choose a legal move with model A, B, C, or D and return search metrics."""

    normalized_model = str(model).upper()
    if normalized_model not in _MODEL_NAMES:
        raise ValueError(f"Unknown AI model: {model}")
    settings = config or SearchConfig()
    opponent = _other(player)
    if board.is_full():
        raise ValueError("Board has no legal move")

    if normalized_model == "A":
        started = time.perf_counter()
        if settings.time_limit_seconds == 0:
            move = _first_legal_move(board)
            elapsed = time.perf_counter() - started
            return SearchResult(
                move, evaluate_board(board, player), 0, 0, 0, 0, elapsed, True, (move,)
            )
        move = select_ai_move(board, player, opponent, rng=random.Random(seed))
        elapsed = time.perf_counter() - started
        return SearchResult(
            move=move,
            root_value=evaluate_board(board, player),
            nodes=sum(cell is None for row in board.grid for cell in row),
            completed_depth=0,
            cutoffs=0,
            simulations=0,
            elapsed_seconds=elapsed,
            timed_out=elapsed > settings.time_limit_seconds,
            optimal_moves=(move,),
        )
    if normalized_model in {"B", "C"}:
        return _search_minimax(
            board,
            player,
            settings,
            seed,
            use_alpha_beta=normalized_model == "C",
        )
    if normalized_model == "D1":
        remaining_cells = board.size * board.size - len(board.history)
        d1_settings = replace(settings, mcts_rollout_max_plies=max(1, remaining_cells))
        return _search_mcts(board, player, d1_settings, seed)
    return _search_mcts(board, player, settings, seed)
