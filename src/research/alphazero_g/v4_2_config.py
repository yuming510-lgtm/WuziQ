"""Frozen registration rules for the G V4.2 search/training experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

BRANCH = "research/alphazero-g-v4-2"
BASE_COMMIT = "4dd0c449c7a4afe1a4f68b3e2a234cdd70e186df"
ARM_IDS = ("baseline", "search_target", "history_pool")


def config_sha256(config: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_v4_2_config(
    config: dict[str, Any],
    *,
    require_branch: bool = True,
    root: str | Path | None = None,
) -> dict[str, Any]:
    if config.get("format_version") != 1 or config.get("experiment_id") != "wuziq-alphazero-g-v4-2":
        raise ValueError("unsupported V4.2 experiment registration")
    if config.get("branch") != BRANCH or config.get("base_commit") != BASE_COMMIT:
        raise ValueError("V4.2 branch/base identity differs from registration")
    if config.get("config_sha256") != config_sha256(config):
        raise ValueError("V4.2 config SHA-256 mismatch")
    if require_branch:
        if root is None:
            raise ValueError("repository root is required for branch identity validation")
        root_path = Path(root).resolve()
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root_path,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=root_path,
            capture_output=True,
        )
        if branch != BRANCH or ancestor.returncode != 0:
            raise ValueError("V4.2 requires its registered branch and base ancestor")

    compute = config.get("compute", {})
    cap = float(compute.get("cpu_hour_cap", -1))
    if not 0 < cap <= 5.0:
        raise ValueError("V4.2 CPU hard cap must be positive and no more than 5 hours")
    training = config.get("training", {})
    per_arm = int(training.get("states_per_arm", -1))
    max_states = int(training.get("maximum_total_new_states", -1))
    if max_states > 40_000:
        raise ValueError("V4.2 total new-state budget cannot exceed 40,000")
    if per_arm < 1 or max_states != per_arm * len(config.get("arms", {})):
        raise ValueError("V4.2 training budgets must be equal across registered arms")
    arms = config.get("arms")
    if not isinstance(arms, dict) or len(arms) not in (2, 3) or "baseline" not in arms:
        raise ValueError("V4.2 must register two or three arms including baseline")
    if set(arms) - set(ARM_IDS):
        raise ValueError("V4.2 contains an unknown training arm")
    common = {key: value for key, value in arms["baseline"].items()
              if key not in {"search_variant", "opponent_pool"}}
    for arm_id, arm in arms.items():
        shared = {key: value for key, value in arm.items()
                  if key not in {"search_variant", "opponent_pool"}}
        if shared != common:
            raise ValueError("V4.2 arms differ in an unregistered training factor")
        differences = {key for key in set(arm) | set(arms["baseline"])
                       if arm.get(key) != arms["baseline"].get(key)}
        allowed = set() if arm_id == "baseline" else (
            {"search_variant"} if arm_id == "search_target" else {"opponent_pool"}
        )
        if differences != allowed:
            raise ValueError(f"V4.2 arm {arm_id} must change exactly one registered factor")
        if arm.get("opponent_pool") not in {"current", "historical_g0"}:
            raise ValueError("training opponents must be current or historical G0 only")
    if arms["baseline"].get("search_variant") != "baseline" or arms["baseline"].get("opponent_pool") != "current":
        raise ValueError("V4.2 baseline must preserve the original PUCT/current-selfplay setup")
    if "history_pool" in arms and arms["history_pool"].get("opponent_pool") != "historical_g0":
        raise ValueError("history_pool must use only registered historical G0 opponents")
    if "search_target" in arms and arms["search_target"].get("search_variant") != "selected_diagnostic":
        raise ValueError("search_target must use the development-selected diagnostic configuration")
    candidates = config.get("historical_candidates", [])
    if any(candidate.get("training_family") != "G0" for candidate in candidates):
        raise ValueError("V4.2 training pool may contain G0 checkpoints only")
    candidate_ids = {candidate.get("candidate_id") for candidate in candidates}
    history_ids = training.get("history_opponents", [])
    if len(history_ids) != len(set(history_ids)) or set(history_ids) - candidate_ids:
        raise ValueError("history opponent pool must reference unique registered G0 checkpoints")
    forbidden = set(config.get("forbidden_training_inputs", []))
    if forbidden != {"A", "B", "C", "E", "F", "F1", "F2", "teacher_labels", "handcrafted_tactical_reward"}:
        raise ValueError("V4.2 must explicitly forbid teacher and hand-crafted training inputs")
    return config


def load_v4_2_config(root: str | Path, *, require_branch: bool = True) -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = root_path / "research/g_v4_2/config/experiment.json"
    if not path.is_file():
        raise FileNotFoundError("V4.2 experiment registration is missing")
    return validate_v4_2_config(json.loads(path.read_text(encoding="utf-8")),
                                require_branch=require_branch, root=root_path)
