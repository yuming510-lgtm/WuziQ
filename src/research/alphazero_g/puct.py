"""AlphaZero-style PUCT using policy priors and value evaluations from a net."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable

import numpy as np
import torch

from ...gomoku import GomokuBoard, Player
from ..imitation_f.model import PolicyValueNet, encode_board, select_device


def _other(player: Player) -> Player:
    if player == "B":
        return "W"
    if player == "W":
        return "B"
    raise ValueError("player must be 'B' or 'W'")


def terminal_value(winner: Player | None, to_move: Player) -> float:
    """Return a terminal result from the perspective of the next player."""
    if to_move not in ("B", "W"):
        raise ValueError("to_move must be 'B' or 'W'")
    if winner is None:
        return 0.0
    if winner not in ("B", "W"):
        raise ValueError("winner must be 'B', 'W', or None")
    return 1.0 if winner == to_move else -1.0


@dataclass(frozen=True)
class PUCTConfig:
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.25
    prior_smoothing_fraction: float = 0.0
    leaf_value_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and nonnegative")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not math.isfinite(self.root_noise_fraction) or not 0 <= self.root_noise_fraction <= 1:
            raise ValueError("root_noise_fraction must be in [0,1]")
        if not math.isfinite(self.prior_smoothing_fraction) or not 0 <= self.prior_smoothing_fraction <= 1:
            raise ValueError("prior_smoothing_fraction must be in [0,1]")
        if not math.isfinite(self.leaf_value_scale) or not 0 <= self.leaf_value_scale <= 1:
            raise ValueError("leaf_value_scale must be in [0,1]")


@dataclass
class PUCTNode:
    prior: float
    to_move: Player
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[int, "PUCTNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expand(self, priors: np.ndarray) -> None:
        if self.expanded:
            raise ValueError("node is already expanded")
        if priors.shape != (81,) or not np.isfinite(priors).all() or np.any(priors < 0):
            raise ValueError("priors must be a finite nonnegative 81-cell vector")
        if not np.isclose(float(priors.sum()), 1.0, atol=1e-6):
            raise ValueError("priors must sum to one")
        for action in np.flatnonzero(priors > 0):
            self.children[int(action)] = PUCTNode(
                prior=float(priors[action]), to_move=_other(self.to_move)
            )
        self.expanded = True


def backup_path(path: Iterable[PUCTNode], leaf_value: float) -> None:
    """Back up a leaf value, changing player perspective at each parent edge."""
    nodes = list(path)
    if not nodes:
        raise ValueError("backup path cannot be empty")
    if not math.isfinite(float(leaf_value)) or not -1.0 <= float(leaf_value) <= 1.0:
        raise ValueError("leaf_value must be finite and in [-1,1]")
    value = float(leaf_value)
    for node in reversed(nodes):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def temperature_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert nonnegative root visits to a normalized action distribution."""
    values = np.asarray(visits, dtype=np.float64)
    if values.shape != (81,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("visits must be a finite nonnegative 81-cell vector")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    total = values.sum()
    if total <= 0:
        raise ValueError("visits must contain at least one visit")
    if temperature <= 1e-8:
        result = np.zeros(81, dtype=np.float64)
        result[int(np.argmax(values))] = 1.0
        return result
    positive = values > 0
    weights = np.zeros_like(values)
    weights[positive] = np.exp(np.log(values[positive]) / float(temperature))
    denominator = weights.sum()
    if not math.isfinite(float(denominator)) or denominator <= 0:
        raise ValueError("visit weights could not be normalized")
    return weights / denominator


@dataclass(frozen=True)
class PUCTResult:
    move: tuple[int, int]
    visit_policy: np.ndarray
    temperature_zero_policy: np.ndarray
    root_visits: np.ndarray
    root_priors: np.ndarray
    root_value: float
    simulations: int
    expanded_nodes: int
    tree_nodes: int
    network_evaluations: int
    max_depth: int
    visit_entropy: float
    elapsed_seconds: float


def _terminal_winner(board: GomokuBoard) -> tuple[bool, Player | None]:
    if board.history:
        last_player, last_coord = board.history[-1]
        if board.check_victory(last_coord, last_player):
            return True, last_player
    if board.is_full():
        return True, None
    return False, None


class PUCTSearch:
    """PUCT tree search whose only nonterminal evaluator is the policy/value net."""

    def __init__(
        self,
        model: PolicyValueNet,
        config: PUCTConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.config = config or PUCTConfig()
        self.device = select_device(str(device))
        self.model.to(self.device)

    def _evaluate(self, board: GomokuBoard, to_move: Player) -> tuple[np.ndarray, float]:
        encoded = encode_board(board, to_move)
        legal = encoded[2].reshape(-1).numpy().astype(bool)
        if not legal.any():
            raise ValueError("cannot evaluate a terminal full board")
        with torch.inference_mode():
            logits, value = self.model(encoded.unsqueeze(0).to(self.device))
            if logits.shape != (1, 81) or value.shape != (1,):
                raise ValueError("policy/value model must return [1,81] logits and [1] value")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(value).all()):
                raise ValueError("policy/value model returned non-finite output")
            logits = logits[0].to(dtype=torch.float64)
            legal_device = torch.as_tensor(legal, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~legal_device, float("-inf"))
            logits = logits - logits[legal_device].max()
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        probabilities[~legal] = 0.0
        probabilities[legal] = np.maximum(probabilities[legal], 1e-12)
        probabilities /= probabilities.sum()
        smoothing = self.config.prior_smoothing_fraction
        if smoothing:
            probabilities *= 1.0 - smoothing
            probabilities[legal] += smoothing / int(legal.sum())
        leaf_value = float(value[0].clamp(-1, 1).cpu().item()) * self.config.leaf_value_scale
        return probabilities, leaf_value

    def _select_child(self, node: PUCTNode) -> tuple[int, PUCTNode]:
        if not node.children:
            raise ValueError("expanded node has no legal children")
        scale = math.sqrt(max(1, node.visit_count))
        action, child = max(
            node.children.items(),
            key=lambda item: (
                -item[1].mean_value
                + self.config.c_puct * item[1].prior * scale / (1 + item[1].visit_count),
                -item[0],
            ),
        )
        return action, child

    def search(
        self,
        board: GomokuBoard,
        to_move: Player,
        simulations: int,
        *,
        add_root_noise: bool = False,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> PUCTResult:
        if board.size != 9:
            raise ValueError("AlphaZero G V1 requires a 9x9 board")
        if to_move not in ("B", "W"):
            raise ValueError("to_move must be 'B' or 'W'")
        if isinstance(simulations, bool) or int(simulations) != simulations or simulations < 1:
            raise ValueError("simulations must be a positive integer")
        if not math.isfinite(float(temperature)) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if add_root_noise and self.config.root_noise_fraction > 0:
            noise_enabled = True
        else:
            noise_enabled = False
        is_terminal, _ = _terminal_winner(board)
        if is_terminal:
            raise ValueError("cannot search from a terminal root position")
        if board.history and board.history[-1][0] == to_move:
            raise ValueError("to_move must alternate after the last recorded move")

        started = time.perf_counter()
        rng = np.random.default_rng(int(seed))
        working = GomokuBoard.from_dict(board.to_dict())
        prior_training = self.model.training
        self.model.eval()
        try:
            base_priors, _ = self._evaluate(working, to_move)
            root = PUCTNode(prior=1.0, to_move=to_move)
            root_priors = base_priors.copy()
            if noise_enabled:
                legal_actions = np.flatnonzero(root_priors > 0)
                noise = rng.dirichlet(np.full(len(legal_actions), self.config.dirichlet_alpha))
                root_priors[legal_actions] = (
                    (1.0 - self.config.root_noise_fraction) * root_priors[legal_actions]
                    + self.config.root_noise_fraction * noise
                )
            root.expand(root_priors)
            network_evaluations = 1
            expanded_nodes = 1
            tree_nodes = 1 + len(root.children)
            max_depth = 0

            for _ in range(int(simulations)):
                node = root
                path = [root]
                applied: list[tuple[int, int]] = []
                try:
                    while True:
                        action, child = self._select_child(node)
                        move = divmod(action, 9)
                        player = node.to_move
                        working.place_stone(player, move)
                        applied.append(move)
                        path.append(child)
                        max_depth = max(max_depth, len(path) - 1)

                        if working.check_victory(move, player):
                            leaf_value = terminal_value(player, child.to_move)
                            break
                        if working.is_full():
                            leaf_value = 0.0
                            break
                        node = child
                        if not node.expanded:
                            priors, leaf_value = self._evaluate(working, node.to_move)
                            network_evaluations += 1
                            node.expand(priors)
                            expanded_nodes += 1
                            tree_nodes += len(node.children)
                            break
                    backup_path(path, leaf_value)
                finally:
                    for _move in reversed(applied):
                        working.undo()

            visits = np.zeros(81, dtype=np.int64)
            for action, child in root.children.items():
                visits[action] = child.visit_count
            policy = temperature_policy(visits, float(temperature))
            deterministic_policy = temperature_policy(visits, 0.0)
            action = int(rng.choice(np.arange(81), p=policy))
            entropy = -float(np.sum(policy[policy > 0] * np.log(policy[policy > 0])))
            return PUCTResult(
                move=divmod(action, 9),
                visit_policy=policy,
                temperature_zero_policy=deterministic_policy,
                root_visits=visits,
                root_priors=root_priors,
                root_value=root.mean_value,
                simulations=int(simulations),
                expanded_nodes=expanded_nodes,
                tree_nodes=tree_nodes,
                network_evaluations=network_evaluations,
                max_depth=max_depth,
                visit_entropy=entropy,
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            if prior_training:
                self.model.train()
