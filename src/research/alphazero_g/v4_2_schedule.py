"""Pre-registered V4.2 openings, held out by all D4 board symmetries."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .v3_schedule import build_v3_schedule, validate_v3_schedule


def _canonical_schedule_hash(schedule: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_v4_2_schedule(
    *,
    seed: int,
    train_groups_per_arm: int,
    development_groups: int,
    blind_groups: int,
    excluded_openings: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    generated = build_v3_schedule(
        [int(seed)], train_groups_per_seed=int(train_groups_per_arm),
        development_groups=int(development_groups), blind_groups=int(blind_groups),
        excluded_openings=excluded_openings,
    )
    schedule = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v4-2",
        "seed": int(seed),
        "train_groups_per_arm": int(train_groups_per_arm),
        "development_groups_count": int(development_groups),
        "blind_groups_count": int(blind_groups),
        "excluded_opening_digests": sorted(generated["excluded_opening_digests"]),
        "train_openings": generated["train_openings_by_seed"][str(int(seed))],
        "development_openings": generated["development_openings"],
        "blind_openings": generated["blind_openings"],
    }
    schedule["schedule_sha256"] = _canonical_schedule_hash(schedule)
    validate_v4_2_schedule(schedule)
    return schedule


def validate_v4_2_schedule(schedule: dict[str, Any]) -> None:
    if schedule.get("format_version") != 1 or schedule.get("experiment_id") != "wuziq-alphazero-g-v4-2":
        raise ValueError("unsupported V4.2 opening schedule")
    if schedule.get("schedule_sha256") != _canonical_schedule_hash(schedule):
        raise ValueError("V4.2 opening schedule SHA-256 mismatch")
    core = {
        "format_version": 1,
        "seeds": [int(schedule["seed"])],
        "train_groups_per_seed": int(schedule["train_groups_per_arm"]),
        "development_groups": int(schedule["development_groups_count"]),
        "blind_groups": int(schedule["blind_groups_count"]),
        "excluded_opening_digests": list(schedule["excluded_opening_digests"]),
        "train_openings_by_seed": {str(int(schedule["seed"])): schedule["train_openings"]},
        "development_openings": schedule["development_openings"],
        "blind_openings": schedule["blind_openings"],
    }
    core["schedule_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    validate_v3_schedule(core)
    groups = [*schedule["train_openings"], *schedule["development_openings"], *schedule["blind_openings"]]
    if len({str(group["opening_group_id"]) for group in groups}) != len(groups):
        raise ValueError("V4.2 opening group IDs must be unique")
