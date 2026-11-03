"""Compatibility adapter around the independent Junxiao Song reference code."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from third_party.AlphaZero_Gomoku.game import Board
from third_party.AlphaZero_Gomoku.mcts_alphaZero import MCTSPlayer
from third_party.AlphaZero_Gomoku.policy_value_net_pytorch import PolicyValueNet as UpstreamNet
from third_party.AlphaZero_Gomoku.policy_value_net_pytorch import set_learning_rate


class ReferencePolicyValueNet:
    """Use upstream architecture/optimizer while adapting the old PyTorch API."""

    def __init__(self, board_size: int, seed: int | None = None) -> None:
        self.board_size = int(board_size)
        if seed is None:
            self.upstream = UpstreamNet(self.board_size, self.board_size, use_gpu=False)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                self.upstream = UpstreamNet(self.board_size, self.board_size, use_gpu=False)
        self.network = self.upstream.policy_value_net
        self.optimizer = self.upstream.optimizer

    def policy_value_fn(self, board: Board):
        # The vendored function, including its state encoder and legal move
        # filtering, is retained. inference_mode avoids building unused graphs.
        with torch.inference_mode():
            return self.upstream.policy_value_fn(board)

    def policy_value(self, state_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            return self.upstream.policy_value(state_batch)

    def train_step(
        self,
        state_batch: np.ndarray,
        mcts_probs: np.ndarray,
        winner_batch: np.ndarray,
        lr: float,
    ) -> tuple[float, float]:
        """Upstream training objective with the documented modern `.item()` fix."""
        states = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32)
        targets = torch.as_tensor(np.asarray(mcts_probs), dtype=torch.float32)
        winners = torch.as_tensor(np.asarray(winner_batch), dtype=torch.float32)
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, float(lr))
        log_probs, values = self.network(states)
        value_loss = F.mse_loss(values.view(-1), winners)
        policy_loss = -torch.mean(torch.sum(targets * log_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=1))
        return float(loss.detach().item()), float(entropy.detach().item())

    def get_policy_param(self) -> dict[str, torch.Tensor]:
        return self.network.state_dict()

    def load_policy_param(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.network.load_state_dict(state_dict, strict=True)


def instrumented_player(
    policy_value_fn: Callable[[Board], Any],
    *,
    c_puct: float,
    n_playout: int,
    is_selfplay: bool,
) -> MCTSPlayer:
    """Create the original MCTSPlayer and expose otherwise-private tree stats."""
    player = MCTSPlayer(
        policy_value_fn,
        c_puct=float(c_puct),
        n_playout=int(n_playout),
        is_selfplay=int(bool(is_selfplay)),
    )
    original = player.mcts.get_move_probs
    original_policy = player.mcts._policy
    capture: dict[str, Any] = {"first_eval": True, "first_leaf_value": None,
                               "root_value": None, "root_state": None}

    def measured_policy(state):
        action_priors, value = original_policy(state)
        if capture["first_eval"]:
            capture["first_leaf_value"] = float(value)
            if capture["root_state"] is not None and np.array_equal(
                state.current_state(), capture["root_state"]
            ):
                capture["root_value"] = float(value)
            capture["first_eval"] = False
        return action_priors, value

    player.mcts._policy = measured_policy

    def tree_stats(root) -> dict[str, int]:
        max_depth = 0
        nodes = 0
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            nodes += 1
            max_depth = max(max_depth, depth)
            stack.extend((child, depth + 1) for child in node._children.values())
        return {
            "root_visits": sum(child._n_visits for child in root._children.values()),
            "root_children": len(root._children),
            "tree_nodes": nodes,
            "tree_depth": max_depth,
        }

    def measured_get_move_probs(state, temp=1e-3):
        import time

        capture["first_eval"] = True
        capture["first_leaf_value"] = None
        capture["root_value"] = None
        capture["root_state"] = state.current_state().copy()
        started = time.process_time()
        acts, probs = original(state, temp)
        player.last_search_stats = {
            **tree_stats(player.mcts._root),
            "configured_playouts": int(player.mcts._n_playout),
            "root_network_value": capture["root_value"],
            "first_leaf_value": capture["first_leaf_value"],
            "search_cpu_seconds": time.process_time() - started,
        }
        return acts, probs

    player.mcts.get_move_probs = measured_get_move_probs
    player.last_search_stats = {}
    return player


def make_board(size: int, n_in_row: int, start_player: int = 0) -> Board:
    board = Board(width=int(size), height=int(size), n_in_row=int(n_in_row))
    board.init_board(int(start_player))
    return board
