from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from src.research.alphazero_g.v6_registration import (
    EXPERIMENT_ID,
    V5_COMMIT,
    V6_BRANCH,
    build_opening_schedule,
    make_registration,
    validate_opening_schedule,
    validate_v5_resume_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _canonical_hash(value: dict[str, Any], key: str) -> str:
    body = dict(value)
    body.pop(key, None)
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def register_v6(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != V6_BRANCH:
        raise ValueError("V6 registration must run on its independent research branch")
    baseline = subprocess.run(["git", "rev-parse", "research/alphazero-g-v5"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    if baseline != V5_COMMIT:
        raise ValueError("V5 baseline moved from the frozen V6 registration commit")

    output = root / "research/g_v6"
    config_path = output / "config/registration.json"
    if config_path.exists():
        registered = json.loads(config_path.read_text(encoding="utf-8"))
        if registered.get("registration_sha256") != _canonical_hash(registered, "registration_sha256"):
            raise ValueError("existing V6 registration hash mismatch")
        schedule_path = output / "config/opening_schedule.json"
        tactics_path = output / "config/tactics.json"
        if _sha(schedule_path) != registered["opening_schedule_sha256"]:
            raise ValueError("existing V6 opening schedule hash mismatch")
        if _sha(tactics_path) != registered["tactics_sha256"]:
            raise ValueError("existing V6 tactics hash mismatch")
        return registered
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("commit V6 implementation before freezing registration")

    schedule = build_opening_schedule()
    validate_opening_schedule(schedule, root)
    schedule_path = output / "config/opening_schedule.json"
    _atomic_json(schedule_path, schedule)
    tactics_source = root / "research/g_v4_2/config/tactical_puzzles.json"
    tactics_path = output / "config/tactics.json"
    tactics_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tactics_source, tactics_path)
    tactics = json.loads(tactics_path.read_text(encoding="utf-8"))
    if not tactics or any(int(item.get("board_size", 9)) != 9 for item in tactics):
        raise ValueError("V6 tactical set must contain frozen 9x9-only evaluation items")

    resume = validate_v5_resume_checkpoint(root)
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip()
    cpu_quota = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    memory_limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    disk = shutil.disk_usage(root)
    registration = make_registration(root)
    registration.update({
        "registered_code_commit": current_commit,
        "source_reference_manifest_sha256": _sha(root / "research/g_v5/reference/source_manifest.json"),
        "v5_opening_schedule_sha256": _sha(root / "research/g_v5/config/opening_schedule.json"),
        "opening_schedule_sha256": _sha(schedule_path),
        "tactics_sha256": _sha(tactics_path),
        "tactics_count": len(tactics),
        "resume_validation": resume,
        "resource_snapshot": {
            "logical_cpus": os.cpu_count(),
            "cpu_cgroup_max": cpu_quota,
            "memory_cgroup_max_bytes": int(memory_limit) if memory_limit.isdigit() else memory_limit,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "preflight_required_before_training": True,
        "drive_recovery_folder_name": "WuziQ G V6 Recovery",
        "drive_snapshot_policy": "one latest verified V6 snapshot; upload/download/hash/empty-restore before removing prior V6 snapshot",
    })
    registration["registration_sha256"] = _canonical_hash(registration, "registration_sha256")
    _atomic_json(config_path, registration)
    _atomic_json(output / "results/resume_validation.json", resume)
    _atomic_json(output / "results/progress.json", {
        "format_version": 1, "experiment_id": EXPERIMENT_ID,
        "phase": "registered", "complete": False,
        "v6_cumulative_cpu_seconds": 0.0, "cpu_cap_seconds": 36_000,
        "registration_sha256": registration["registration_sha256"],
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return registration


if __name__ == "__main__":
    print(json.dumps(register_v6(), ensure_ascii=False, indent=2))
