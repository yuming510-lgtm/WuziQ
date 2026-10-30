"""Deterministic, resumable policy/value imitation training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from datetime import datetime, timezone
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .model import PolicyValueNet, save_f_checkpoint, select_device


def _rng_payload() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_training_state(path: str | Path, state: dict[str, Any]) -> str:
    """Atomically save a full optimizer/RNG/progress state and return its hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["rng_state"] = _rng_payload()
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_state(
    path: str | Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
) -> dict[str, Any]:
    """Restore model, optimizer, scheduler and every supported RNG stream."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    rng = state.get("rng_state", {})
    if "python_rng_state" in rng:
        random.setstate(rng["python_rng_state"])
    if "numpy_rng_state" in rng:
        np.random.set_state(rng["numpy_rng_state"])
    if "torch_rng_state" in rng:
        torch.set_rng_state(rng["torch_rng_state"])
    if torch.cuda.is_available() and rng.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(rng["cuda_rng_state_all"])
    return state


def _read_rows(source: Any) -> list[dict[str, Any]]:
    if isinstance(source, list):
        return source
    path = Path(source)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fingerprint_rows(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_rows(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, policies, values = [], [], []
    for row in rows:
        board_text = row.get("canonical_board")
        if board_text is None:
            board_text = str(row["state_key"]).split(":", 1)[1]
        if len(board_text) != 81 or any(cell not in ".12" for cell in board_text):
            raise ValueError("dataset row has an invalid canonical 9x9 board")
        cells = np.fromiter((ord(cell) for cell in board_text), dtype=np.int32, count=81)
        own = (cells == ord("1")).astype(np.float32)
        opponent = (cells == ord("2")).astype(np.float32)
        legal = (cells == ord(".")).astype(np.float32)
        features.append(np.stack((own, opponent, legal)).reshape(3, 9, 9))
        policy = np.asarray(row["policy_target"], dtype=np.float32)
        if policy.shape != (81,) or not np.isfinite(policy).all() or np.any(policy < 0):
            raise ValueError("dataset row has an invalid policy target")
        if not np.isclose(policy.sum(), 1.0, atol=1e-5):
            raise ValueError("policy target must sum to one")
        if np.any(policy[legal == 0] > 1e-7):
            raise ValueError("policy target assigns mass to an illegal square")
        if not legal.any():
            raise ValueError("full-board rows cannot have a policy target")
        value = float(row["value_target"])
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("dataset value target must lie in [-1,1]")
        policies.append(policy)
        values.append(value)
    return (
        torch.from_numpy(np.stack(features)),
        torch.from_numpy(np.stack(policies)),
        torch.tensor(values, dtype=torch.float32),
    )


def _augment(x: torch.Tensor, policy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    augmented_x, augmented_policy = [], []
    for index in range(x.shape[0]):
        rotations = int(torch.randint(0, 4, ()).item())
        reflect = bool(torch.randint(0, 2, ()).item())
        plane = torch.rot90(x[index], rotations, dims=(-2, -1))
        target = policy[index].reshape(9, 9)
        target = torch.rot90(target, rotations, dims=(-2, -1))
        if reflect:
            plane = torch.flip(plane, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        augmented_x.append(plane)
        augmented_policy.append(target.reshape(81))
    return torch.stack(augmented_x), torch.stack(augmented_policy)


@torch.inference_mode()
def _evaluate(model: PolicyValueNet, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device,
              batch_size: int) -> dict[str, float]:
    x, target_policy, target_value = data
    model.eval()
    losses = []
    top1 = []
    target_mass = []
    absolute_values = []
    squared_values = []
    legal_predictions = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        policy_target = target_policy[start:start + batch_size].to(device)
        value_target = target_value[start:start + batch_size].to(device)
        logits, value = model(xb)
        legal = xb[:, 2].reshape(-1, 81).bool()
        logits = logits.masked_fill(~legal, -1e9)
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        losses.extend((-(policy_target * log_probabilities).sum(dim=-1)).cpu().tolist())
        predictions = probabilities.argmax(dim=-1)
        top1.extend((predictions == policy_target.argmax(dim=-1)).float().cpu().tolist())
        target_mass.extend((probabilities * policy_target).sum(dim=-1).cpu().tolist())
        legal_predictions.extend(legal.gather(1, predictions[:, None]).squeeze(1).float().cpu().tolist())
        error = (value.clamp(-1, 1) - value_target)
        absolute_values.extend(error.abs().cpu().tolist())
        squared_values.extend(error.square().cpu().tolist())
    return {
        "policy_loss": float(np.mean(losses)),
        "top1_accuracy": float(np.mean(top1)),
        "target_probability_mass": float(np.mean(target_mass)),
        "value_mae": float(np.mean(absolute_values)),
        "value_mse": float(np.mean(squared_values)),
        "legal_prediction_rate": float(np.mean(legal_predictions)),
    }


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def train(
    config: dict[str, Any],
    dataset: dict[str, Any],
    checkpoint_dir: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Train and return the best validation-policy checkpoint."""
    session_started = time.perf_counter()
    session_started_at_utc = datetime.now(timezone.utc).isoformat()
    seed = int(config.get("seed", 731501))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, int(config.get("cpu_threads", min(3, os.cpu_count() or 1)))))
    device = select_device(str(config.get("device", "auto")))
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("dataset must include train and validation partitions")
    train_rows = _read_rows(dataset["train"])
    validation_rows = _read_rows(dataset["validation"])
    dataset_fingerprints = {
        "train": _fingerprint_rows(train_rows),
        "validation": _fingerprint_rows(validation_rows),
    }
    config_fingerprint = _fingerprint_config(config)
    train_data = _encode_rows(train_rows)
    val_data = _encode_rows(validation_rows)
    if len(train_data[0]) == 0 or len(val_data[0]) == 0:
        raise ValueError("training and validation sets must both be nonempty")

    model = PolicyValueNet(
        board_size=9,
        channels=int(config.get("channels", 32)),
        blocks=int(config.get("blocks", 2)),
        seed=seed,
    ).to(device)
    learning_rate = float(config.get("learning_rate", 0.001))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(config.get("weight_decay", 1e-4))
    )
    epochs = int(config.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(config.get("minimum_learning_rate", learning_rate * 0.05))
    )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "training_state.pt"
    best_path = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "training_history.jsonl"
    batch_size = int(config.get("batch_size", 64))
    checkpoint_interval = max(1, int(config.get("checkpoint_every_updates", 25)))
    patience = max(1, int(config.get("patience", 5)))

    state: dict[str, Any] = {
        "epoch": 0, "sampler_order": [], "sampler_cursor": 0, "update_step": 0,
        "history": [], "best_policy_loss": float("inf"), "best_epoch": -1,
        "bad_epochs": 0, "best_model_state": None,
        "dataset_fingerprints": dataset_fingerprints,
        "config_fingerprint": config_fingerprint,
        "training_started_at_utc": session_started_at_utc,
        "elapsed_training_seconds": 0.0,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "dataset_counts": {"train": len(train_data[0]), "validation": len(val_data[0])},
        "config": dict(config),
    }
    elapsed_before_session = 0.0
    if resume and state_path.exists():
        existing = torch.load(state_path, map_location="cpu", weights_only=False)
        if existing.get("dataset_fingerprints") != dataset_fingerprints:
            raise ValueError("training resume dataset fingerprint mismatch")
        if existing.get("config_fingerprint") != config_fingerprint:
            raise ValueError("training resume config fingerprint mismatch")
        state = load_training_state(state_path, model, optimizer, scheduler)
        elapsed_before_session = float(state.get("elapsed_training_seconds", 0.0))
        session_started_at_utc = str(state.get("training_started_at_utc", session_started_at_utc))
        history = list(state.get("history", []))
        if history_path.exists():
            _atomic_jsonl(history_path, history)
    else:
        history = []
        save_training_state(state_path, state)

    epoch = int(state["epoch"])
    sampler_order = list(state.get("sampler_order", []))
    cursor = int(state.get("sampler_cursor", 0))
    step = int(state.get("update_step", 0))
    best_loss = float(state.get("best_policy_loss", float("inf")))
    best_epoch = int(state.get("best_epoch", -1))
    bad_epochs = int(state.get("bad_epochs", 0))
    best_model_state = state.get("best_model_state")
    model.train()

    def save_current() -> None:
        state.update({
            "epoch": epoch, "sampler_order": sampler_order, "sampler_cursor": cursor,
            "update_step": step, "history": history, "best_policy_loss": best_loss,
            "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "best_model_state": best_model_state,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "config": dict(config),
            "dataset_counts": {"train": len(train_data[0]), "validation": len(val_data[0])},
            "dataset_fingerprints": dataset_fingerprints,
            "config_fingerprint": config_fingerprint,
            "training_started_at_utc": session_started_at_utc,
            "elapsed_training_seconds": elapsed_before_session + time.perf_counter() - session_started,
            "last_checkpoint_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        save_training_state(state_path, state)

    while epoch < epochs and bad_epochs < patience:
        if not sampler_order:
            sampler_order = torch.randperm(len(train_data[0])).tolist()
            cursor = 0
            save_current()
        while cursor < len(sampler_order):
            selected = sampler_order[cursor:cursor + batch_size]
            xb = train_data[0][selected]
            policy_target = train_data[1][selected]
            value_target = train_data[2][selected]
            xb, policy_target = _augment(xb, policy_target)
            xb, policy_target, value_target = xb.to(device), policy_target.to(device), value_target.to(device)
            logits, value = model(xb)
            legal = xb[:, 2].reshape(-1, 81).bool()
            if torch.any(policy_target.masked_select(~legal) > 1e-7):
                raise ValueError("training policy target assigns mass to an occupied square")
            logits = logits.masked_fill(~legal, -1e9)
            policy_loss = -(policy_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, value_target)
            loss = policy_loss + float(config.get("value_loss_weight", 1.0)) * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 5.0)))
            optimizer.step()
            cursor += len(selected)
            step += 1
            if step % checkpoint_interval == 0:
                save_current()

        model.eval()
        validation = _evaluate(model, val_data, device, batch_size)
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1, "updates": step,
            "train_policy_loss_last_batch": float(policy_loss.detach().cpu().item()),
            "train_value_loss_last_batch": float(value_loss.detach().cpu().item()),
            "validation": validation, "learning_rate": learning_rate_used,
        }
        history.append(row)
        _atomic_jsonl(history_path, history)
        if validation["policy_loss"] < best_loss - float(config.get("minimum_improvement", 1e-5)):
            best_loss = validation["policy_loss"]
            best_epoch = epoch + 1
            bad_epochs = 0
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            metadata = {
                "seed": seed, "config": dict(config), "best_epoch": best_epoch,
                "validation": validation, "dataset_counts": state["dataset_counts"],
                "device": str(device),
                "dataset_fingerprints": dataset_fingerprints,
                "config_fingerprint": config_fingerprint,
            }
            save_f_checkpoint(best_path, model, metadata)
        else:
            bad_epochs += 1
        scheduler.step()
        epoch += 1
        sampler_order = []
        cursor = 0
        model.train()
        save_current()

    if best_model_state is not None:
        model.load_state_dict(best_model_state, strict=True)
    else:
        save_f_checkpoint(best_path, model, {"seed": seed, "no_validation_improvement": True})
        best_epoch = max(1, epoch)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["training_complete"] = epoch >= epochs or bad_epochs >= patience
    state["best_epoch"] = best_epoch
    save_training_state(state_path, state)
    return {
        "model": model,
        "updates": step,
        "best_epoch": best_epoch,
        "best_checkpoint": best_path,
        "state_checkpoint": state_path,
        "history_path": history_path,
        "history": history,
        "device": str(device),
        "elapsed_seconds": float(state.get("elapsed_training_seconds", elapsed_before_session + time.perf_counter() - session_started)),
    }
