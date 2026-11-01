"""Self-play-only policy/value optimization and exact-resume checkpoints."""
from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..imitation_f.model import PolicyValueNet, load_f_checkpoint, save_f_checkpoint, select_device
from .replay import ReplayBuffer, augment_sample

F2_SHA256 = "50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257"


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(__file__).resolve().parents[3]
    return root / candidate


def initialize_arm(
    arm_id: str,
    seed: int,
    f2_checkpoint: str | Path | None = None,
    *,
    device: str = "cpu",
) -> PolicyValueNet:
    """Construct a registered random arm or hash-verified F2 warm start."""
    if arm_id == "G0":
        model = PolicyValueNet(board_size=9, channels=32, blocks=2, seed=int(seed))
        model.to(select_device(device))
        model.checkpoint_metadata = {
            "arm_id": "G0", "initialization": "seeded_random", "seed": int(seed),
        }
        return model
    if arm_id != "G-F2":
        raise ValueError("arm_id must be 'G0' or 'G-F2'")
    source = _repository_path(f2_checkpoint or "research/f_v2/checkpoints/best_f2.pt")
    if not source.is_file():
        raise FileNotFoundError(f"frozen F2 checkpoint not found: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != F2_SHA256:
        raise ValueError(f"F2 checkpoint hash mismatch: {digest}")
    model = load_f_checkpoint(source, device=device)
    if (model.board_size, model.channels, model.blocks) != (9, 32, 2):
        raise ValueError("F2 checkpoint architecture does not match the registered G V1 network")
    model.checkpoint_metadata = {
        "arm_id": "G-F2", "initialization": "frozen_f2", "checkpoint_sha256": digest,
    }
    return model


class GTrainer:
    """One arm's AdamW optimizer, schedule, augmentation stream and progress."""

    def __init__(
        self,
        model: PolicyValueNet,
        *,
        seed: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        expected_updates: int = 100,
        device: str = "cpu",
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if isinstance(expected_updates, bool) or int(expected_updates) != expected_updates or expected_updates < 1:
            raise ValueError("expected_updates must be a positive integer")
        self.device = select_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(expected_updates), eta_min=float(learning_rate) * 0.2
        )
        self.expected_updates = int(expected_updates)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.update_count = 0

    def train_updates(self, replay: ReplayBuffer, *, updates: int, batch_size: int) -> dict[str, float | int]:
        if isinstance(updates, bool) or int(updates) != updates or updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if updates and replay.position_count <= 0:
            raise ValueError("self-play replay must contain rows before training")
        if self.update_count + int(updates) > self.expected_updates:
            raise ValueError("training update count exceeds registered schedule")
        self.model.train()
        policy_losses: list[float] = []
        value_losses: list[float] = []
        total_losses: list[float] = []
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for _ in range(int(updates)):
            batch = replay.sample(int(batch_size))
            augmented: list[dict[str, Any]] = []
            for sample in batch:
                augmented.append(augment_sample(
                    sample,
                    rotations=int(self.rng.integers(0, 4)),
                    reflect=bool(self.rng.integers(0, 2)),
                ))
            planes = np.zeros((len(augmented), 3, 9, 9), dtype=np.float32)
            legal = np.zeros((len(augmented), 81), dtype=np.bool_)
            targets = np.zeros((len(augmented), 81), dtype=np.float32)
            values = np.zeros((len(augmented),), dtype=np.float32)
            for index, sample in enumerate(augmented):
                cells = np.asarray(list(sample["canonical_board"])).reshape(9, 9)
                planes[index, 0] = cells == "1"
                planes[index, 1] = cells == "2"
                planes[index, 2] = cells == "."
                legal[index] = cells.reshape(-1) == "."
                targets[index] = np.asarray(sample["policy_target"], dtype=np.float32)
                values[index] = float(sample["value_target"])
            x = torch.from_numpy(planes).to(self.device)
            legal_mask = torch.from_numpy(legal).to(self.device)
            policy_target = torch.from_numpy(targets).to(self.device)
            value_target = torch.from_numpy(values).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits, predicted_values = self.model(x)
            # A finite floor avoids the undefined ``0 * -inf`` product in the
            # soft-target cross entropy while keeping illegal actions at zero
            # probability in float32.
            masked_logits = logits.masked_fill(~legal_mask, -1e9)
            policy_loss = -(policy_target * torch.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
            value_loss = F.mse_loss(predicted_values, value_target)
            loss = policy_loss + value_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite self-play training loss")
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.update_count += 1
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
        return {
            "updates": int(updates),
            "update_count": self.update_count,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "cpu_seconds": time.process_time() - cpu_started,
            "wall_seconds": time.perf_counter() - wall_started,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "update_count": self.update_count,
            "expected_updates": self.expected_updates,
            "seed": self.seed,
            "augmentation_rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1 or int(state.get("expected_updates", -1)) != self.expected_updates:
            raise ValueError("incompatible G trainer state")
        self.model.load_state_dict(state["model_state"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.update_count = int(state["update_count"])
        self.rng.bit_generator.state = state["augmentation_rng_state"]


def save_g_checkpoint(path: str | Path, model: PolicyValueNet, metadata: dict[str, Any] | None = None) -> str:
    """Write an ordinary F-format policy/value checkpoint loadable by G and F APIs."""
    return save_f_checkpoint(path, model, metadata={"model_family": "G", **dict(metadata or {})})


def load_g_checkpoint(path: str | Path, device: str = "cpu") -> PolicyValueNet:
    model = load_f_checkpoint(path, device=device)
    metadata = dict(getattr(model, "checkpoint_metadata", {}))
    if metadata.get("model_family") != "G":
        raise ValueError("checkpoint metadata does not identify a G model")
    return model


def save_training_state(
    path: str | Path,
    trainer: GTrainer,
    replay: ReplayBuffer,
    *,
    progress: dict[str, Any],
) -> str:
    """Atomically save optimizer, schedule, RNG streams, replay and progress."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "trainer": trainer.state_dict(),
        "replay": replay.state_dict(),
        "progress": dict(progress),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_training_state(
    path: str | Path,
    trainer: GTrainer,
    replay: ReplayBuffer,
    *,
    restore_global_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported G training-state format")
    trainer.load_state_dict(payload["trainer"])
    replay.load_state_dict(payload["replay"])
    if restore_global_rng:
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if torch.cuda.is_available() and payload["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
    return dict(payload["progress"])
