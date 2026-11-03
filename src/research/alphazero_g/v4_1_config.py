"""Frozen registration and identity checks for the G V4.1 experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .v3_schedule import validate_v3_schedule

BRANCH = "research/alphazero-g-v4-1"
BASE_COMMIT = "d3c434d5d31a2e2c052aa0bd4b50c0424e88ca7a"
V3_CHAMPION_SHA256 = "ed6e2e6b5a25f355db5b31a9c58ba22158af2b345f86b3bcd3add9c3a49da7f9"
ARM_IDS = ("control_v4", "state_uniform", "winner_balanced", "low_lr")


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def config_sha256(config: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def load_v4_1_config(root: str | Path, *, require_branch: bool = True) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config_path = root_path / "research/g_v4_1/config/experiment.json"
    schedule_path = root_path / "research/g_v4_1/config/openings.json"
    if not config_path.is_file() or not schedule_path.is_file():
        raise FileNotFoundError("G V4.1 preregistration and opening schedule are required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1 or config.get("experiment_id") != "wuziq-alphazero-g-v4-1":
        raise ValueError("unsupported G V4.1 config")
    if config.get("branch") != BRANCH or config.get("base_commit") != BASE_COMMIT:
        raise ValueError("G V4.1 branch/base identity differs from registration")
    if config.get("config_sha256") != config_sha256(config):
        raise ValueError("G V4.1 config SHA-256 mismatch")
    if require_branch:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root_path,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if branch != BRANCH:
            raise ValueError(f"G V4.1 requires branch {BRANCH}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=root_path,
            capture_output=True,
        )
        if ancestor.returncode != 0:
            raise ValueError("G V4.1 base commit is not an ancestor of HEAD")
    if set(config.get("arms", {})) != set(ARM_IDS):
        raise ValueError("G V4.1 must contain exactly the four preregistered arms")
    training = config.get("training", {})
    if int(training.get("maximum_total_new_states", -1)) != 40_000:
        raise ValueError("G V4.1 state budget exceeds the preregistered 40K cap")
    if int(training.get("new_states_per_arm", -1)) != 10_000:
        raise ValueError("each G V4.1 arm must use exactly the registered 10K budget")
    common: dict[str, Any] | None = None
    for arm_id, arm in config["arms"].items():
        if arm_id not in ARM_IDS or arm.get("selfplay_opponent") != "current_g0":
            raise ValueError("only self-trained G0 models may enter V4.1 self-play")
        shared = {key: value for key, value in arm.items() if key not in {"sampling_mode", "learning_rate"}}
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError("G V4.1 arms differ in more than one registered factor")
    expected = {
        "control_v4": ("uniform_game", 1e-4),
        "state_uniform": ("uniform_state", 1e-4),
        "winner_balanced": ("winner_balanced", 1e-4),
        "low_lr": ("uniform_game", 3e-5),
    }
    for arm_id, (sampling, learning_rate) in expected.items():
        arm = config["arms"][arm_id]
        if arm.get("sampling_mode") != sampling or float(arm.get("learning_rate", -1)) != learning_rate:
            raise ValueError(f"G V4.1 arm {arm_id} differs from its preregistered single factor")
    checkpoint = root_path / str(config["initial_checkpoint"])
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != V3_CHAMPION_SHA256:
        raise ValueError("frozen V3 G0@20K initialization checkpoint is missing or changed")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    if schedule.get("schedule_sha256") != config.get("opening_schedule_sha256"):
        raise ValueError("G V4.1 opening schedule differs from preregistration")
    if schedule.get("development_groups") != 12 or schedule.get("blind_groups") != 20:
        raise ValueError("G V4.1 development or blind opening quota changed")
    return config
