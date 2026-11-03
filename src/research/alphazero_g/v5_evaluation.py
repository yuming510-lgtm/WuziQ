"""Opening generation, independent-rule matches, and paired summaries for V5."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard
from ..imitation_f.model import load_f_checkpoint
from .puct import PUCTConfig, PUCTSearch
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board


def other(player: str) -> str:
    return "W" if player == "B" else "B"


def _d4(coord: tuple[int, int], size: int, transform: int) -> tuple[int, int]:
    row, col = coord
    if transform >= 4:
        col = size - 1 - col
        transform -= 4
    for _ in range(transform):
        row, col = size - 1 - col, row
    return row, col


def opening_key(moves: list[dict[str, Any]], size: int) -> str:
    forms = []
    for transform in range(8):
        forms.append(tuple(
            (str(item["player"]), *_d4((int(item["row"]), int(item["col"])), size, transform))
            for item in moves
        ))
    return repr(min(forms))


def generate_opening_pool(
    *, size: int, n_in_row: int, groups: int, plies: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    seen: set[str] = set()
    openings: list[dict[str, Any]] = []
    attempt = 0
    while len(openings) < groups:
        attempt += 1
        if attempt > groups * 1000:
            raise RuntimeError("could not generate the requested unique legal openings")
        board = make_board(size, n_in_row, 0)
        moves = []
        terminal = False
        for _ in range(plies):
            if not board.availables:
                break
            move = int(rng.choice(board.availables))
            row, col = divmod(move, size)
            player = "B" if board.current_player == 1 else "W"
            board.do_move(move)
            moves.append({"player": player, "row": row, "col": col})
            if board.game_end()[0]:
                terminal = True
                break
        if len(moves) != plies or terminal:
            continue
        key = opening_key(moves, size)
        if key in seen:
            continue
        seen.add(key)
        group = f"v5-{size}x{size}-{seed}-{len(openings):03d}"
        openings.append({"opening_group_id": group, "seed": int(seed + len(openings)), "moves": moves})
    return openings


def _project_board(board) -> GomokuBoard:
    if int(board.width) != 9:
        raise ValueError("frozen G0/A/B opponents are registered only for 9x9")
    converted = GomokuBoard(9)
    for move, player_number in board.states.items():
        coord = divmod(int(move), 9)
        player = "B" if int(player_number) == 1 else "W"
        converted.place_stone(player, coord)
    return converted


def _agent_choice(
    agent: dict[str, Any], board, color: str, seed: int
) -> tuple[tuple[int, int], dict[str, Any]]:
    kind = agent["kind"]
    started_cpu = time.process_time()
    if kind == "reference":
        move, _policy = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(move), int(board.width))
        result = dict(agent["player"].last_search_stats)
        result["simulations"] = int(agent["player"].mcts._n_playout - 1)
        result["root_action_visits"] = int(result.get("root_visits", 0))
        result["search_depth"] = int(result.get("tree_depth", 0))
        result["search_nodes"] = int(result.get("tree_nodes", 0))
        return (row, col), {**result, "cpu_seconds": time.process_time() - started_cpu}
    if kind == "g0":
        project = _project_board(board)
        result = agent["search"].search(project, color, int(agent.get("simulations", 64)),
                               add_root_noise=False, temperature=0.0, seed=int(seed))
        return result.move, {
            "simulations": result.simulations,
            "root_action_visits": int(result.root_visits.sum()),
            "search_depth": result.max_depth,
            "search_nodes": result.tree_nodes,
            "network_evaluations": result.network_evaluations,
            "cpu_seconds": time.process_time() - started_cpu,
        }
    if kind in {"A", "B"}:
        project = _project_board(board)
        config = SearchConfig(time_limit_seconds=0.05 if kind == "B" else 1e9)
        result = choose_move(project, color, kind, config, int(seed))
        return result.move, {
            "simulations": result.simulations,
            "root_action_visits": 0,
            "search_depth": result.completed_depth,
            "search_nodes": result.nodes,
            "cutoffs": result.cutoffs,
            "cpu_seconds": time.process_time() - started_cpu,
            "timed_out": result.timed_out,
        }
    raise ValueError(f"unknown V5 evaluation agent: {kind}")


def _make_agent(
    name: str,
    *,
    reference_model: ReferencePolicyValueNet | None = None,
    g0_checkpoint: str | Path | None = None,
    simulations: int = 64,
    evaluation_root_visits: bool = True,
) -> dict[str, Any]:
    if name == "reference":
        if reference_model is None:
            raise ValueError("reference model is required")
        # The upstream loop uses its first playout to expand the root. Add one
        # root expansion so formal evaluations have exactly N root-edge visits.
        n_playout = int(simulations) + (1 if evaluation_root_visits else 0)
        player = instrumented_player(reference_model.policy_value_fn,
                                     c_puct=5.0, n_playout=n_playout, is_selfplay=False)
        return {"kind": "reference", "player": player}
    if name == "G0" or name.startswith("G0-"):
        if g0_checkpoint is None:
            raise ValueError("G0 checkpoint is required")
        model = load_f_checkpoint(g0_checkpoint, device="cpu")
        model.eval()
        search = PUCTSearch(model, config=PUCTConfig(c_puct=5.0, root_noise_fraction=0.0), device="cpu")
        return {"kind": "g0", "model": model, "search": search, "simulations": int(simulations)}
    if name in {"A", "B"}:
        return {"kind": name}
    raise ValueError(f"unknown V5 model name {name!r}")


def _agent_kind(name: str) -> str:
    if name.lower().startswith("ref"):
        return "reference"
    if name.startswith("G0-") or name == "G0":
        return "G0"
    if name in {"A", "A-rule"}:
        return "A"
    if name in {"B", "B-50ms"}:
        return "B"
    raise ValueError(f"unknown V5 model identity {name!r}")


class _IndependentRuleBoard:
    """Small plain-grid oracle supporting both 6x6/four and 9x9/five."""

    def __init__(self, size: int, n_in_row: int) -> None:
        self.size = int(size)
        self.n_in_row = int(n_in_row)
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.history: list[tuple[str, tuple[int, int]]] = []

    def get(self, coord: tuple[int, int]):
        return self.grid[coord[0]][coord[1]]

    def place_stone(self, player: str, coord: tuple[int, int]) -> None:
        row, col = coord
        if not (0 <= row < self.size and 0 <= col < self.size) or self.get(coord) is not None:
            raise ValueError("illegal move in independent rule board")
        self.grid[row][col] = player
        self.history.append((player, coord))

    def check_victory(self, coord: tuple[int, int], player: str) -> bool:
        row, col = coord
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                rr, cc = row + sign * dr, col + sign * dc
                while 0 <= rr < self.size and 0 <= cc < self.size and self.grid[rr][cc] == player:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= self.n_in_row:
                return True
        return False

    def is_full(self) -> bool:
        return all(cell is not None for row in self.grid for cell in row)


def play_match(
    *,
    game_id: str,
    candidate_name: str,
    opponent_name: str,
    opening: dict[str, Any],
    candidate_color: str,
    seed: int,
    reference_model: ReferencePolicyValueNet | None = None,
    candidate_reference_model: ReferencePolicyValueNet | None = None,
    opponent_reference_model: ReferencePolicyValueNet | None = None,
    candidate_g0_checkpoint: str | Path | None = None,
    opponent_g0_checkpoint: str | Path | None = None,
    simulations: int = 64,
    board_size: int = 9,
    n_in_row: int = 5,
) -> dict[str, Any]:
    if candidate_color not in {"B", "W"}:
        raise ValueError("candidate_color must be B or W")
    candidate_reference_model = candidate_reference_model or reference_model
    opponent_reference_model = opponent_reference_model or reference_model
    candidate_kind = _agent_kind(candidate_name)
    opponent_kind = _agent_kind(opponent_name)
    candidate = _make_agent(candidate_kind, reference_model=candidate_reference_model,
                            g0_checkpoint=candidate_g0_checkpoint, simulations=simulations)
    opponent = _make_agent(opponent_kind, reference_model=opponent_reference_model,
                           g0_checkpoint=opponent_g0_checkpoint, simulations=simulations)

    board = make_board(board_size, n_in_row, 0)
    opening_moves = opening.get("moves", [])
    expected = "B"
    stored_moves: list[dict[str, Any]] = []
    independent = _IndependentRuleBoard(int(board_size), int(n_in_row))
    for item in opening_moves:
        color = str(item["player"])
        if color != expected:
            raise ValueError("opening colors must alternate from black")
        row, col = int(item["row"]), int(item["col"])
        flat = row * int(board.width) + col
        if flat not in board.availables:
            raise ValueError("opening contains an illegal move")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        if board.game_end()[0] or independent.check_victory((row, col), color):
            raise ValueError("opening is terminal")
        stored_moves.append({"player": color, "row": row, "col": col, "opening": True})
        expected = other(expected)

    np.random.seed(int(seed) & 0xFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    winner: str | None = None
    termination = ""
    agent_cpu = opponent_cpu = 0.0
    while True:
        color = "B" if board.current_player == 1 else "W"
        is_candidate = color == candidate_color
        agent = candidate if is_candidate else opponent
        agent_name = candidate_name if is_candidate else opponent_name
        move_seed = int(rng.integers(0, 2**31 - 1))
        decision_started = time.perf_counter()
        coord, metrics = _agent_choice(agent, board, color, move_seed)
        elapsed = time.perf_counter() - decision_started
        if is_candidate:
            agent_cpu += float(metrics.get("cpu_seconds", 0.0))
        else:
            opponent_cpu += float(metrics.get("cpu_seconds", 0.0))
        row, col = coord
        flat = row * int(board.width) + col
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError(f"{agent_name} selected an illegal move {coord}")
        board.do_move(flat)
        independent.place_stone(color, coord)
        stored_moves.append({"player": color, "row": row, "col": col,
                             "opening": False, "actor": agent_name,
                             "elapsed_seconds": elapsed, **metrics})
        engine_end, engine_winner = board.game_end()
        project_end = independent.check_victory(coord, color)
        if bool(engine_end) != bool(project_end or independent.is_full()):
            raise AssertionError("reference and independent rules disagree during match replay")
        if project_end:
            winner, termination = color, "five_in_row"
            if engine_winner != board.current_player ^ 3:
                raise AssertionError("reference board winner differs from independent rule replay")
            break
        if independent.is_full():
            winner, termination = None, "full_board"
            break
    points = 0.5 if winner is None else 1.0 if winner == candidate_color else 0.0
    return {
        "format_version": 1,
        "game_id": game_id,
        "source": "evaluation",
        "candidate": candidate_name,
        "opponent": opponent_name,
        "board_size": int(board_size),
        "n_in_row": int(n_in_row),
        "candidate_color": candidate_color,
        "winner_color": winner,
        "points": points,
        "termination": termination,
        "opening_group_id": opening["opening_group_id"],
        "opening": opening_moves,
        "seed": int(seed),
        "simulations_per_reference_turn": int(simulations),
        "reference_configured_playouts": int(simulations + 1),
        "moves": stored_moves,
        "move_count": len(stored_moves),
        "candidate_cpu_seconds": agent_cpu,
        "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }


def paired_bootstrap_ci(
    pair_scores: list[float], *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    if not pair_scores:
        return (float("nan"), float("nan"))
    values = np.asarray(pair_scores, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[indexes].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize_paired_games(games: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    by_opening: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        by_opening.setdefault(str(game["opening_group_id"]), []).append(game)
    pair_scores = []
    color_scores = {"B": [], "W": []}
    wins = draws = losses = 0
    for group, rows in sorted(by_opening.items()):
        if len(rows) != 2 or {row["candidate_color"] for row in rows} != {"B", "W"}:
            raise ValueError(f"opening {group} does not contain a color-swapped pair")
        pair_scores.append(float(np.mean([float(row["points"]) for row in rows])))
        for row in rows:
            color_scores[row["candidate_color"]].append(float(row["points"]))
            if row["points"] == 1:
                wins += 1
            elif row["points"] == 0.5:
                draws += 1
            else:
                losses += 1
    total = wins + draws + losses
    lo, hi = paired_bootstrap_ci(pair_scores, seed=seed)
    return {
        "games": total,
        "paired_openings": len(pair_scores),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": wins + draws / 2.0,
        "score_rate": (wins + draws / 2.0) / total if total else 0.0,
        "paired_opening_score_rate": float(np.mean(pair_scores)) if pair_scores else 0.0,
        "paired_opening_bootstrap_ci95": [lo, hi],
        "black_score_rate": float(np.mean(color_scores["B"])) if color_scores["B"] else None,
        "white_score_rate": float(np.mean(color_scores["W"])) if color_scores["W"] else None,
        "mean_moves": float(np.mean([row["move_count"] for row in games])) if games else None,
        "mean_candidate_cpu_seconds": float(np.mean([row["candidate_cpu_seconds"] for row in games])) if games else None,
        "mean_opponent_cpu_seconds": float(np.mean([row["opponent_cpu_seconds"] for row in games])) if games else None,
    }


def _canonical_position(board, to_move: int) -> str:
    size = int(board.width)
    grid = [[board.states.get(r * size + c, 0) for c in range(size)] for r in range(size)]
    forms = []
    matrix = np.asarray(grid, dtype=np.int8)
    for reflect in (False, True):
        base = np.fliplr(matrix) if reflect else matrix
        for turns in range(4):
            transformed = np.rot90(base, turns)
            forms.append((int(to_move), transformed.tobytes().hex()))
    return min(repr(x) for x in forms)


def _winning_actions(board, player_number: int) -> list[int]:
    wins = []
    for action in board.availables:
        clone = make_board(board.width, board.n_in_row, 0)
        clone.states = dict(board.states)
        clone.availables = list(board.availables)
        clone.current_player = int(player_number)
        clone.last_move = int(board.last_move)
        clone.do_move(int(action))
        ended, winner = clone.game_end()
        if ended and winner == player_number:
            wins.append(int(action))
    return wins


def generate_six_by_six_tactics(*, seed: int = 560605, per_category: int = 32) -> list[dict[str, Any]]:
    """Make a held-out, D4-deduplicated set from exact 6x6/connect-four rules."""
    rng = np.random.default_rng(int(seed))
    wanted = {"direct_win": int(per_category), "unique_block": int(per_category)}
    found: dict[str, list[dict[str, Any]]] = {name: [] for name in wanted}
    seen: set[str] = set()
    attempts = 0
    while any(len(found[name]) < wanted[name] for name in wanted):
        attempts += 1
        if attempts > 10000:
            break
        board = make_board(6, 4, 0)
        history = []
        for _ in range(int(rng.integers(8, 25))):
            to_move = int(board.current_player)
            opponent = 3 - to_move
            direct = _winning_actions(board, to_move)
            threats = _winning_actions(board, opponent)
            category = None
            targets: list[int] = []
            if direct:
                category, targets = "direct_win", direct
            elif len(threats) == 1:
                category, targets = "unique_block", threats
            if category and len(found[category]) < wanted[category]:
                key = _canonical_position(board, to_move)
                if key not in seen:
                    seen.add(key)
                    player_name = "B" if to_move == 1 else "W"
                    puzzle = {
                        "puzzle_id": f"v5-6x6-{category}-{len(found[category]):03d}",
                        "category": category,
                        "board_size": 6,
                        "n_in_row": 4,
                        "to_move": player_name,
                        "moves": list(history),
                        "target_actions": [int(a) for a in targets],
                        "position_key": key,
                        "source": "synthetic legal position; exact rules used for scoring only",
                    }
                    found[category].append(puzzle)
            if not board.availables:
                break
            move = int(rng.choice(board.availables))
            row, col = divmod(move, 6)
            color = "B" if to_move == 1 else "W"
            board.do_move(move)
            history.append({"player": color, "row": row, "col": col})
            if board.game_end()[0]:
                break
    return found["direct_win"] + found["unique_block"]


def _puzzle_board(puzzle: dict[str, Any]):
    board = make_board(int(puzzle["board_size"]), int(puzzle["n_in_row"]), 0)
    for move in puzzle["moves"]:
        board.do_move(int(move["row"]) * int(puzzle["board_size"]) + int(move["col"]))
    expected = "B" if board.current_player == 1 else "W"
    if expected != puzzle["to_move"]:
        raise ValueError("puzzle side to move is inconsistent with its history")
    if board.game_end()[0]:
        raise ValueError("puzzle position is terminal")
    return board


def evaluate_reference_tactics(
    model: ReferencePolicyValueNet,
    puzzles: list[dict[str, Any]],
    *,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    rows = []
    for puzzle in puzzles:
        board = _puzzle_board(puzzle)
        priors, _value = model.policy_value_fn(board)
        policy_move = int(max(priors, key=lambda row: (float(row[1]), -int(row[0])))[0])
        player = instrumented_player(model.policy_value_fn, c_puct=5.0,
                                     n_playout=int(simulations) + 1, is_selfplay=False)
        result_move, _ = player.get_action(board, temp=1e-3, return_prob=1)
        targets = set(map(int, puzzle["target_actions"]))
        rows.append({
            "puzzle_id": puzzle["puzzle_id"], "category": puzzle["category"],
            "network_move": policy_move, "network_correct": policy_move in targets,
            "puct_move": int(result_move), "puct_correct": int(result_move) in targets,
            "target_actions": sorted(targets),
            "tree_depth": int(player.last_search_stats.get("tree_depth", 0)),
            "tree_nodes": int(player.last_search_stats.get("tree_nodes", 0)),
            "root_action_visits": int(player.last_search_stats.get("root_visits", 0)),
        })
    categories = sorted({row["category"] for row in rows})
    return {
        "simulations": int(simulations),
        "puzzles": len(rows),
        "network_accuracy": float(np.mean([row["network_correct"] for row in rows])) if rows else None,
        "puct_accuracy": float(np.mean([row["puct_correct"] for row in rows])) if rows else None,
        "by_category": {
            category: {
                "puzzles": sum(row["category"] == category for row in rows),
                "network_accuracy": float(np.mean([row["network_correct"] for row in rows if row["category"] == category])),
                "puct_accuracy": float(np.mean([row["puct_correct"] for row in rows if row["category"] == category])),
            } for category in categories
        },
        "rows": rows,
    }


def build_historical_midgame_pool(root: str | Path, *, maximum: int = 96) -> list[dict[str, Any]]:
    """Extract legal state prefixes only; never copy historical G0 action labels."""
    root = Path(root)
    sources = [
        root / "research/g_v3/results/selfplay/G0-seed-510901.jsonl",
        root / "research/g_v3/results/selfplay/G0-seed-510902.jsonl",
        root / "research/g_v4/results/selfplay/A-G0-s540402.jsonl",
        root / "research/g_v4_2/results/games/selfplay/history_pool",
    ]
    source_candidates: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    existing_sources = [source for source in sources if source.exists()]
    quota = max(1, int(np.ceil(maximum / max(1, len(existing_sources)))))
    for source in sources:
        if not source.exists():
            continue
        game_records = []
        if source.is_file():
            for line in source.read_text(encoding="utf-8").splitlines():
                if line:
                    game_records.append(json.loads(line))
        elif source.is_dir():
            for path in sorted(source.glob("*.json")):
                game_records.append(json.loads(path.read_text(encoding="utf-8")))
        bucket = source_candidates.setdefault(str(source.relative_to(root)), [])
        for game in game_records:
            if len(bucket) >= quota:
                break
            moves = game.get("moves", [])
            if len(moves) < 12:
                continue
            for ply in (8, 10, 12, 14, 16, 18, 20, 22):
                if ply >= len(moves):
                    continue
                board = make_board(9, 5, 0)
                prefix = []
                valid = True
                for move in moves[:ply]:
                    color = str(move["player"])
                    row, col = int(move["row"]), int(move["col"])
                    action = row * 9 + col
                    expected = "B" if board.current_player == 1 else "W"
                    if color != expected or action not in board.availables:
                        valid = False; break
                    board.do_move(action)
                    prefix.append({"player": color, "row": row, "col": col})
                    if board.game_end()[0]:
                        valid = False; break
                if not valid or len(prefix) != ply:
                    continue
                key = _canonical_position(board, board.current_player)
                if key in seen:
                    continue
                seen.add(key)
                bucket.append({
                    "moves": prefix,
                    "source_game_id": str(game.get("game_id", "unknown")),
                    "source_file": str(source.relative_to(root)),
                    "source_ply": ply,
                    "position_key": key,
                })
                if len(bucket) >= quota:
                    break
    # Interleave origins to prevent the first historical file from dominating.
    candidates: list[dict[str, Any]] = []
    max_rows = max((len(rows) for rows in source_candidates.values()), default=0)
    for index in range(max_rows):
        for source_name in sorted(source_candidates):
            rows = source_candidates[source_name]
            if index < len(rows) and len(candidates) < maximum:
                row = dict(rows[index])
                row["opening_group_id"] = f"v5-midgame-{len(candidates):04d}"
                candidates.append(row)
    return candidates
