"""Compact 9x9 CNN policy/value model and G-compatible inference interface."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...gomoku import GomokuBoard


def select_device(device: str = "auto") -> torch.device:
    requested = str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return selected


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.bn1(self.conv1(value)))
        residual = self.bn2(self.conv2(residual))
        return self.activation(value + residual)


class PolicyValueNet(nn.Module):
    """Small residual CNN returning 81 policy logits and a scalar value."""

    def __init__(
        self,
        board_size: int = 9,
        channels: int = 32,
        blocks: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if board_size != 9:
            raise ValueError("F V1 is currently frozen to a 9x9 board")
        if channels < 4 or blocks < 1:
            raise ValueError("channels must be >=4 and blocks must be positive")
        if seed is None:
            self._initialize(board_size, channels, blocks)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                self._initialize(board_size, channels, blocks)
        self.board_size = board_size
        self.channels = channels
        self.blocks = blocks

    def _initialize(self, board_size: int, channels: int, blocks: int) -> None:
        self.input_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 32)
        self.value_fc2 = nn.Linear(32, 1)
        self.activation = nn.ReLU(inplace=True)
        self.value_activation = nn.Tanh()

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if planes.ndim != 4 or planes.shape[1:] != (3, self.board_size, self.board_size):
            raise ValueError(f"input must have shape [batch,3,{self.board_size},{self.board_size}]")
        hidden = self.activation(self.input_bn(self.input_conv(planes)))
        hidden = self.trunk(hidden)
        policy = self.policy_fc(self.policy_conv(hidden).flatten(start_dim=1))
        value = self.value_conv(hidden).flatten(start_dim=1)
        value = self.activation(self.value_fc1(value))
        value = self.value_activation(self.value_fc2(value)).squeeze(-1)
        return policy, value


def encode_board(board: GomokuBoard, to_move: str) -> torch.Tensor:
    """Encode current player stones, opponent stones, and legal actions."""
    if board.size != 9 or to_move not in ("B", "W"):
        raise ValueError("F V1 expects a 9x9 board and to_move in {'B','W'}")
    opponent = "W" if to_move == "B" else "B"
    planes = torch.zeros((3, board.size, board.size), dtype=torch.float32)
    for row in range(board.size):
        for col in range(board.size):
            stone = board.grid[row][col]
            if stone == to_move:
                planes[0, row, col] = 1.0
            elif stone == opponent:
                planes[1, row, col] = 1.0
            else:
                planes[2, row, col] = 1.0
    return planes


@torch.inference_mode()
def predict_policy_value(
    model: PolicyValueNet,
    board: GomokuBoard,
    to_move: str,
    device: str | torch.device = "auto",
) -> tuple[torch.Tensor, float]:
    """Return legal-masked probabilities and value from the side-to-move view."""
    encoded = encode_board(board, to_move)
    legal = encoded[2].reshape(-1).bool()
    if not bool(legal.any()):
        raise ValueError("board has no legal move")
    selected_device = select_device(str(device))
    model.to(selected_device)
    prior_training = model.training
    model.eval()
    logits, value = model(encoded.unsqueeze(0).to(selected_device))
    logits = logits[0].masked_fill(~legal.to(selected_device), float("-inf"))
    probabilities = torch.softmax(logits, dim=-1).to("cpu")
    result_value = float(value[0].clamp(-1.0, 1.0).cpu().item())
    if prior_training:
        model.train()
    return probabilities, result_value


def save_f_checkpoint(
    path: str | Path, model: PolicyValueNet, metadata: dict[str, Any] | None = None
) -> str:
    """Atomically persist loadable model weights and caller provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "architecture": {
            "board_size": model.board_size,
            "channels": model.channels,
            "blocks": model.blocks,
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": dict(metadata or {}),
    }
    with temp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_f_checkpoint(
    path: str | Path, device: str = "auto"
) -> PolicyValueNet:
    """Load a validated F checkpoint on CPU or GPU for later inference/RL use."""
    selected_device = select_device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported F checkpoint format")
    architecture = payload.get("architecture", {})
    model = PolicyValueNet(
        board_size=int(architecture.get("board_size", 0)),
        channels=int(architecture.get("channels", 0)),
        blocks=int(architecture.get("blocks", 0)),
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("F checkpoint is missing model weights")
    model.load_state_dict(state_dict, strict=True)
    model.to(selected_device)
    model.eval()
    model.checkpoint_metadata = payload.get("metadata", {})
    return model
