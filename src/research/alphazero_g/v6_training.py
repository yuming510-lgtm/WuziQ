"""V6 continuation adapter: preserve V5 training code, isolate V6 outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from .v5_training import train_reference_arm
from .v6_registration import (
    MILESTONES,
    SIMULATIONS_PER_MOVE,
    TARGET_STATES,
    validate_v5_resume_checkpoint,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with source.open("rb") as input_file, temp.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temp, target)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_v6_continuation(
    root: str | Path,
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically stage V5's full training state into V6 once."""
    root = Path(root).resolve()
    source_root = Path(source_root or root).resolve()
    validation = validate_v5_resume_checkpoint(source_root)
    source_dir = source_root / "research/g_v5/checkpoints/reference400_empty"
    target_dir = root / "research/g_v6/checkpoints/reference400_empty"
    target_state = target_dir / "training_state.pt"
    target_node0 = target_dir / "node_0.pt"
    target_node1500 = target_dir / "node_1500.pt"
    source_state = source_dir / "training_state.pt"

    if target_state.exists():
        staged = torch.load(target_state, map_location="cpu", weights_only=False)
        if (staged.get("arm_id") != "reference400_empty"
                or staged.get("config_sha256") != validation["v5_config_sha256"]
                or int(staged.get("board_size", 0)) != 9
                or int(staged.get("environment_states", 0)) < validation["environment_states"]):
            raise ValueError("existing V6 continuation checkpoint is incompatible or behind V5")
        if int(staged["environment_states"]) == validation["environment_states"]:
            if _sha(target_state) != validation["training_state_sha256"]:
                raise ValueError("V6 initial copy differs from the registered V5 state")
    else:
        if target_dir.exists() and any(target_dir.iterdir()):
            raise ValueError("V6 checkpoint directory has artifacts but no resumable training state")
        _atomic_copy(source_state, target_state)

    for name in ("node_0.pt", "node_1500.pt"):
        source = source_dir / name
        target = target_dir / name
        if not target.exists():
            _atomic_copy(source, target)
        elif _sha(source) != _sha(target):
            raise ValueError(f"V6 inherited milestone changed: {name}")

    result = {
        **validation,
        "resume_validated": True,
        "staged_checkpoint": "research/g_v6/checkpoints/reference400_empty/training_state.pt",
        "staged_checkpoint_sha256": _sha(target_state),
        "v5_source_unchanged": _sha(source_state) == validation["training_state_sha256"],
    }
    _atomic_json(root / "research/g_v6/results/resume_validation.json", result)
    return result


def train_v6(
    root: str | Path,
    *,
    source_root: str | Path | None = None,
    target_states: int = TARGET_STATES,
    stop_at_cpu_seconds: float | None = None,
) -> dict[str, Any]:
    """Continue the V5 400-simulation arm to a cumulative state target."""
    root = Path(root).resolve()
    source_root = Path(source_root or root).resolve()
    prepared = prepare_v6_continuation(root, source_root=source_root)
    v5_config_path = source_root / "research/g_v5/config/experiment.json"
    config = json.loads(v5_config_path.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    before_cpu = time.process_time()
    settings = train_reference_arm(
        root,
        arm="reference400_empty",
        board_size=9,
        n_in_row=5,
        simulations=SIMULATIONS_PER_MOVE,
        target_states=int(target_states),
        milestones=[node for node in MILESTONES if node > prepared["environment_states"]],
        seed=int(config["nine_by_nine"]["training_seed"]),
        config=config,
        stop_at_cpu_seconds=stop_at_cpu_seconds,
        artifact_dir="research/g_v6",
    )
    session_cpu = time.process_time() - before_cpu
    result = {
        **settings,
        "simulations_per_move": SIMULATIONS_PER_MOVE,
        "target_cumulative_states": int(target_states),
        "v5_origin_states": int(prepared["environment_states"]),
        "v5_origin_games": int(prepared["game_count"]),
        "v5_origin_cpu_seconds": float(prepared["v5_cumulative_training_cpu_seconds"]),
        "v6_session_training_cpu_seconds": session_cpu,
        "v6_training_cpu_seconds": max(0.0, float(settings["cumulative_cpu_seconds"])
                                        - float(prepared["v5_cumulative_training_cpu_seconds"])),
    }
    _atomic_json(root / "research/g_v6/results/training_summary.json", result)
    return result

