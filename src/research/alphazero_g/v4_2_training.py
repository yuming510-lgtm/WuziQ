"""Training-only G0 self-play, including a frozen historical-G0 opponent pool."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Any

import numpy as np
import torch

from ...gomoku import GomokuBoard, Player
from ..imitation_f.model import PolicyValueNet
from .puct import PUCTConfig, PUCTSearch, temperature_policy
from .replay import _validate_game


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _canonical_board(board: GomokuBoard, to_move: Player) -> str:
    opponent = _other(to_move)
    return "".join("1" if cell == to_move else "2" if cell == opponent else "."
                   for row in board.grid for cell in row)


def prefix_selfplay_game(game: dict[str, Any], count: int) -> dict[str, Any]:
    """Keep exactly the first `count` completed-game policy/value rows for a state budget."""
    rows = game.get("experience_rows")
    total = len(game.get("moves", [])) - len(game.get("opening", []))
    if not isinstance(rows, list) or count < 1 or count > len(rows) or len(rows) != total:
        raise ValueError("invalid completed self-play state prefix")
    clipped = copy.deepcopy(game)
    clipped["experience_rows"] = copy.deepcopy(rows[:count])
    clipped["budgeted_environment_steps"] = int(count)
    clipped["completion_overrun_steps"] = int(total - count)
    clipped["replay_prefix"] = count < total
    return clipped


class V4SampleReplay:
    """Uniform-by-game replay compatible with GTrainer and exact checkpoint resume."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("replay capacity must be a positive integer")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(int(seed))
        self._games: list[dict[str, Any]] = []
        self._ids: dict[str, str] = {}
        self._positions = 0

    @property
    def position_count(self) -> int:
        return self._positions

    @property
    def game_count(self) -> int:
        return len(self._games)

    def add_game(self, game: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> bool:
        game_id = game.get("game_id")
        source_rows = game.get("experience_rows") if rows is None else rows
        if not isinstance(game_id, str) or not game_id or not isinstance(source_rows, list) or not source_rows:
            raise ValueError("replay requires a unique complete game and nonempty sample rows")
        payload = {"game_id": game_id, "rows": copy.deepcopy(source_rows),
                   "winner_color": game.get("winner_color"), "arm_id": game.get("arm_id")}
        digest = hashlib.sha256(
            __import__("json").dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if game_id in self._ids:
            if self._ids[game_id] != digest:
                raise ValueError("conflicting self-play game id in V4.2 replay")
            return False
        self._games.append(payload)
        self._ids[game_id] = digest
        self._positions += len(source_rows)
        while self._positions > self.capacity and self._games:
            old = self._games.pop(0)
            self._positions -= len(old["rows"])
            del self._ids[old["game_id"]]
        return True

    def sample(self, batch_size: int, *, mode: str = "uniform_game") -> list[dict[str, Any]]:
        if batch_size < 1 or not self._games:
            raise ValueError("cannot draw a batch from an empty replay")
        if mode not in {"uniform_game", "uniform_state", "winner_balanced"}:
            raise ValueError("unsupported replay sampling mode")
        samples: list[dict[str, Any]] = []
        if mode == "uniform_state":
            all_rows = [row for game in self._games for row in game["rows"]]
            indexes = self._rng.integers(0, len(all_rows), size=int(batch_size))
            return [copy.deepcopy(all_rows[int(index)]) for index in indexes]
        if mode == "winner_balanced":
            categories = {
                color: [game for game in self._games if game["winner_color"] == color]
                for color in ("B", "W")
            }
            if categories["B"] and categories["W"]:
                for _ in range(int(batch_size)):
                    color = "B" if self._rng.random() < 0.5 else "W"
                    game = categories[color][int(self._rng.integers(0, len(categories[color])))]
                    row = game["rows"][int(self._rng.integers(0, len(game["rows"]))) ]
                    samples.append(copy.deepcopy(row))
                return samples
        games = self._rng.integers(0, len(self._games), size=int(batch_size))
        for index in games:
            rows = self._games[int(index)]["rows"]
            samples.append(copy.deepcopy(rows[int(self._rng.integers(0, len(rows))) ]))
        return samples

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1, "capacity": self.capacity,
                "games": copy.deepcopy(self._games), "rng_state": copy.deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1 or int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("incompatible V4.2 replay checkpoint")
        rebuilt = V4SampleReplay(self.capacity, seed=0)
        for game in state.get("games", []):
            wrapper = {"game_id": game["game_id"], "winner_color": game.get("winner_color"),
                       "arm_id": game.get("arm_id")}
            rebuilt.add_game(wrapper, game["rows"])
        rebuilt._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._games, self._ids, self._positions = rebuilt._games, rebuilt._ids, rebuilt._positions


def validate_history_pool_game(game: dict[str, Any], learner_color: Player) -> None:
    """Replay a two-G0-model game and verify that only learner turns carry targets."""
    if game.get("completed") is not True or game.get("source") != "self_play" or game.get("opponent_pool") != "historical_g0":
        raise ValueError("history-pool training record is not a completed G0 self-play game")
    board = GomokuBoard(9)
    expected: Player = "B"
    rows_by_ply = {int(row["ply"]): row for row in game.get("experience_rows", [])}
    if len(rows_by_ply) != len(game.get("experience_rows", [])):
        raise ValueError("history-pool game has duplicate learner states")
    observed_winner: Player | None = None
    observed_termination = ""
    expected_learner_plies: set[int] = set()
    for ply, move in enumerate(game.get("moves", [])):
        if move.get("player") != expected:
            raise ValueError("history-pool moves do not alternate")
        if ply < len(game.get("opening", [])):
            opening = game["opening"][ply]
            if (int(move["row"]), int(move["col"])) != (int(opening["row"]), int(opening["col"])):
                raise ValueError("history-pool game differs from its registered opening")
        elif move.get("actor") == "agent":
            if expected != learner_color:
                raise ValueError("historical opponent actor was mislabeled as the current learner")
            row = rows_by_ply.get(ply)
            if row is None or row.get("player") != expected or row.get("canonical_board") != _canonical_board(board, expected):
                raise ValueError("history-pool learner sample does not match its pre-move board")
        elif ply in rows_by_ply:
            raise ValueError("historical opponent positions must not enter the training targets")
        elif ply >= len(game.get("opening", [])) and expected == learner_color:
            raise ValueError("current G0 learner move is missing its search target")
        elif ply >= len(game.get("opening", [])) and move.get("actor") != "historical_g0":
            raise ValueError("the non-learner self-play side must be a registered historical G0 model")
        coord = (int(move["row"]), int(move["col"]))
        if ply >= len(game.get("opening", [])):
            if int(move.get("simulations", 0)) != int(game.get("simulations_per_move", 0)):
                raise ValueError("history-pool game has an incorrect MCTS simulation count")
            if expected == learner_color:
                expected_learner_plies.add(ply)
        board.place_stone(expected, coord)
        if board.check_victory(coord, expected):
            observed_winner, observed_termination = expected, "five_in_row"
            if ply != len(game["moves"]) - 1:
                raise ValueError("history-pool game continued after a win")
        elif board.is_full():
            observed_winner, observed_termination = None, "full_board"
            if ply != len(game["moves"]) - 1:
                raise ValueError("history-pool game continued after a draw")
        expected = _other(expected)
    if observed_winner != game.get("winner_color") or observed_termination != game.get("termination"):
        raise ValueError("history-pool result does not replay from the complete move list")
    if set(rows_by_ply) != expected_learner_plies:
        raise ValueError("history-pool must record every current-learner state and no opponent states")
    for row in rows_by_ply.values():
        if (row.get("player") != learner_color or row.get("source") != "self_play"
                or row.get("game_complete") is not True or row.get("game_id") != game.get("game_id")
                or row.get("arm_id") != game.get("arm_id")):
            raise ValueError("history-pool rows must belong to the current G0 learner only")
        policy = np.asarray(row.get("policy_target"), dtype=np.float64)
        canonical = row.get("canonical_board", "")
        if policy.shape != (81,) or not np.isfinite(policy).all() or np.any(policy < 0):
            raise ValueError("history-pool policy target must be finite and 81 cells")
        if not math.isclose(float(policy.sum()), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("history-pool visit target must sum to one")
        if not isinstance(canonical, str) or len(canonical) != 81 or set(canonical) - {"1", "2", "."}:
            raise ValueError("history-pool canonical board is malformed")
        occupied = np.fromiter((cell != "." for cell in canonical), dtype=bool, count=81)
        if np.any(policy[occupied] != 0):
            raise ValueError("history-pool target assigns visits to occupied cells")
        expected_value = 0.0 if observed_winner is None else 1.0 if row["player"] == observed_winner else -1.0
        if row.get("value_target") != expected_value:
            raise ValueError("history-pool value target does not match the terminal winner")


def play_history_pool_game(
    learner: PolicyValueNet,
    opponent_model: PolicyValueNet,
    *,
    game_id: str,
    opening_group_id: str,
    arm_id: str,
    opening: list[dict[str, Any]],
    learner_color: Player,
    simulations: int,
    seed: int,
    config: PUCTConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    if learner_color not in {"B", "W"}:
        raise ValueError("learner color must be B or W")
    board = GomokuBoard(9)
    expected: Player = "B"
    moves = []
    for item in opening:
        if item.get("player") != expected:
            raise ValueError("training opening must alternate from black")
        coord = (int(item["row"]), int(item["col"]))
        board.place_stone(expected, coord)
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("training opening must be legal and nonterminal")
        moves.append({"player": expected, "row": coord[0], "col": coord[1], "opening": True})
        expected = _other(expected)
    rng = np.random.default_rng(int(seed))
    learner_search, opponent_search = PUCTSearch(learner, config, device), PUCTSearch(opponent_model, config, device)
    learner.eval(); opponent_model.eval()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    learner_rows: list[dict[str, Any]] = []
    total_simulations = learner_simulations = opponent_simulations = 0
    max_depth = learner_max_depth = opponent_max_depth = 0
    winner: Player | None = None
    termination = ""
    while True:
        learner_turn = expected == learner_color
        search = learner_search if learner_turn else opponent_search
        result = search.search(
            board, expected, int(simulations), add_root_noise=learner_turn,
            temperature=1.0 if learner_turn else 0.0,
            seed=int(rng.integers(0, 2**63 - 1)),
        )
        coord = result.move
        if learner_turn:
            ply = len(moves)
            learner_rows.append({
                "source": "self_play", "arm_id": arm_id, "game_id": game_id,
                "sample_id": f"{game_id}:ply:{ply}", "opening_group_id": opening_group_id,
                "ply": ply, "player": expected, "canonical_board": _canonical_board(board, expected),
                "policy_target": result.visit_policy.tolist(), "value_target": None,
                "game_complete": False, "simulations": int(result.simulations),
                "root_value": float(result.root_value), "policy_entropy": float(result.visit_entropy),
                "search_max_depth": int(result.max_depth),
            })
            learner_simulations += result.simulations
            learner_max_depth = max(learner_max_depth, result.max_depth)
            actor, model_id = "agent", arm_id
        else:
            opponent_simulations += result.simulations
            opponent_max_depth = max(opponent_max_depth, result.max_depth)
            actor, model_id = "historical_g0", str(getattr(opponent_model, "checkpoint_metadata", {}).get("candidate_id", "G0-history"))
        moves.append({"player": expected, "row": coord[0], "col": coord[1], "opening": False,
                      "actor": actor, "model_id": model_id, "simulations": result.simulations,
                      "nodes": result.tree_nodes, "search_depth": result.max_depth,
                      "root_value": result.root_value, "elapsed_seconds": result.elapsed_seconds})
        total_simulations += result.simulations
        max_depth = max(max_depth, result.max_depth)
        board.place_stone(expected, coord)
        if board.check_victory(coord, expected):
            winner, termination = expected, "five_in_row"
            break
        if board.is_full():
            termination = "full_board"
            break
        expected = _other(expected)
    for row in learner_rows:
        row["value_target"] = 0.0 if winner is None else 1.0 if row["player"] == winner else -1.0
        row["game_complete"] = True
    record = {
        "format_version": 1, "source": "self_play", "opponent_pool": "historical_g0",
        "arm_id": arm_id, "game_id": game_id, "opening_group_id": opening_group_id,
        "opening": copy.deepcopy(opening), "learner_color": learner_color,
        "simulations_per_move": int(simulations), "completed": True,
        "termination": termination, "winner_color": winner,
        "outcome": "draw" if winner is None else "B_win" if winner == "B" else "W_win",
        "move_count": len(moves), "moves": moves, "experience_rows": learner_rows,
        "stats": {
            "simulations": total_simulations, "learner_simulations": learner_simulations,
            "historical_g0_simulations": opponent_simulations,
            "max_search_depth": max_depth, "learner_max_search_depth": learner_max_depth,
            "historical_g0_max_search_depth": opponent_max_depth,
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
    }
    validate_history_pool_game(record, learner_color)
    return record
