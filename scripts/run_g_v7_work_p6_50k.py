#!/usr/bin/env python3
"""Run one resumable, exactly bounded segment of the registered V7 P6 job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import psutil
import torch

from src.research.alphazero_g import v6_7_training
from src.research.alphazero_g.v7_work_p6_50k import (
    ARTIFACT, BASE_ARCHIVE_SHA256, BASE_CHECKPOINT_COMMIT, BASE_COMMIT, BASE_GAMES, BASE_STATE_SHA256,
    ROOT, audit_run, run_one, _atomic_json,
)


EXPERIMENT_ID = "wuziq-alphazero-g-v7-work-p6-50k"
RUN_ID = "V7_WORK_P6_50K"
TARGET_ADDED_GAMES = 30_000
BASELINE = ROOT / ARTIFACT / "inputs/G20000_training_state.pt"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def preflight() -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    proc = psutil.Process()
    disk = psutil.disk_usage(str(ROOT))
    mem = psutil.virtual_memory()
    validation = v6_7_training.validate_training_state(
        BASELINE, root=ROOT, expected_games=BASE_GAMES,
    )
    if validation["sha256"] != BASE_STATE_SHA256:
        raise ValueError("the Work baseline checkpoint SHA-256 does not match the registered G20000 state")
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    return {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
        "checked_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": __import__("numpy").__version__,
        "logical_cpu_count": os.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
        "affinity_cpu_ids": affinity, "affinity_cpu_count": len(affinity) if affinity else None,
        "cpu_quota": {p: _read(p) for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us")},
        "cpu_stat": {p: _read(p) for p in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat")},
        "memory_total_bytes": mem.total, "memory_available_bytes": mem.available,
        "memory_limit": {p: _read(p) for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")},
        "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
        "gpu": {"cuda_available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else []},
        "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_thread_environment": {k: os.environ[k] for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
        "github": {"repository": "yuming510-lgtm/WuziQ", "branch": git_branch, "head": git_head,
                   "base_commit": BASE_COMMIT, "code_fetch": "success; local baseline commit is present"},
        "google_drive": {"source_folder_id": "1bkakjo5fdhgELAzyuOohDE8wynVBY1fh",
                         "target_recovery_root": "https://drive.google.com/drive/folders/1BrIaJHAmdtTn1BHbQDWz7ReKrwLCXpSX",
                         "source_archive_sha256": BASE_ARCHIVE_SHA256,
                         "checkpoint_source_commit": BASE_CHECKPOINT_COMMIT,
                         "source_download": "success; both original recovery shards downloaded",
                         "empty_directory_restore_and_full_state_audit": "success"},
        "baseline_audit": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-stop-games", type=int,
                        help="cumulative number of new games to commit before this resumable segment exits")
    parser.add_argument("--audit-after", action="store_true")
    args = parser.parse_args()
    config_dir = ROOT / ARTIFACT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = config_dir / "preflight.json"
    if not preflight_path.exists():
        _atomic_json(preflight_path, preflight())
    config = {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
        "baseline_games": BASE_GAMES, "target_games": 50_000,
        "added_games": TARGET_ADDED_GAMES, "baseline_training_state_sha256": BASE_STATE_SHA256,
        "baseline_archive_sha256": BASE_ARCHIVE_SHA256,
        "training_code_base_commit": BASE_COMMIT,
        "checkpoint_source_commit": BASE_CHECKPOINT_COMMIT,
        "drive_root_url": "https://drive.google.com/drive/folders/1BrIaJHAmdtTn1BHbQDWz7ReKrwLCXpSX",
        "workers": 6, "start_method": "fork", "trainer_processes": 1,
        "torch_intraop_threads": 1, "torch_interop_threads": 1,
        "blas_threads": 1, "board_size": 9, "n_in_row": 5, "simulations_per_move": 400,
        "training_update_calls_per_committed_game": 1,
        "training_config_path": "research/g_v6_7/config/reference_training_config.json",
        "user_requested_no_teacher_or_manual_tactical_rules": True,
        "checkpoint_interval_games": 250, "checkpoint_interval_aggregate_cpu_hours": 3,
        "drive_recovery_root": "https://drive.google.com/drive/folders/1BrIaJHAmdtTn1BHbQDWz7ReKrwLCXpSX",
        "evaluation_protocol": "V6.8 same opening schedule, color pairings and per-game seeds",
        "training_schedule": "asynchronous self-play; each game uses a frozen model_version and is updated once upon central commit",
    }
    _atomic_json(config_dir / "training_registration.json", config)
    if args.segment_stop_games is None:
        print(json.dumps({"preflight": json.loads(preflight_path.read_text()), "registration": config}, ensure_ascii=False))
        return
    if not 1 <= args.segment_stop_games <= TARGET_ADDED_GAMES:
        raise SystemExit("--segment-stop-games must be within 1..30000")
    run_dir = ROOT / ARTIFACT / "runs" / RUN_ID
    summary = run_one(
        run_id=RUN_ID, workers=6, games=TARGET_ADDED_GAMES,
        segment_stop_games=args.segment_stop_games,
        baseline_checkpoint=BASELINE,
        output_root=ROOT / ARTIFACT / "runs", root=ROOT, start_method="fork",
    )
    result = {"summary": summary}
    if args.audit_after:
        result["audit"] = audit_run(run_dir, baseline_checkpoint=BASELINE, root=ROOT)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
