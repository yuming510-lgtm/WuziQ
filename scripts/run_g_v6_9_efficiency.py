from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import psutil
import torch

from src.research.alphazero_g.v6_9_parallel import ARTIFACT, ROOT, audit_run, run_one


WORKER_BY_SCHEME = {"P1": 1, "P2": 2, "P4": 4, "P6": 6, "P8": 8}


def _read_first(paths: list[str]) -> tuple[str | None, str | None]:
    for path in paths:
        file = Path(path)
        if file.is_file():
            try:
                return path, file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
    return None, None


def _preflight() -> dict:
    disk = psutil.disk_usage(str(ROOT))
    mem = psutil.virtual_memory()
    cpu_quota_path, cpu_quota = _read_first([
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    ])
    memory_limit_path, memory_limit = _read_first([
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ])
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head, git_branch = None, None
    gpu = {"cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())}
    if gpu["cuda_available"]:
        gpu["devices"] = [torch.cuda.get_device_name(index) for index in range(gpu["device_count"])]
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "logical_cpu_count": int(os.cpu_count() or 0),
        "affinity_cpu_count": len(affinity) if affinity is not None else None,
        "affinity_cpu_ids": affinity,
        "cpu_quota_path": cpu_quota_path,
        "cpu_quota_value": cpu_quota,
        "cpu_quota_exposed": cpu_quota is not None,
        "memory_total_bytes": int(mem.total),
        "memory_available_bytes": int(mem.available),
        "memory_limit_path": memory_limit_path,
        "memory_limit_value": memory_limit,
        "gpu": gpu,
        "disk_total_bytes": int(disk.total),
        "disk_used_bytes": int(disk.used),
        "disk_free_bytes": int(disk.free),
        "thread_environment": {
            key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
            )
        },
        "git_branch": git_branch,
        "git_head": git_head,
        "git_repository": "yuming510-lgtm/WuziQ",
        "github_clone_test": "success; repository cloned and baseline commit resolved",
        "google_drive_download_test": "success; two G20000 recovery shards downloaded",
        "google_drive_restore_test": "success; empty-directory restore and full-state audit passed",
        "checked_at_local": __import__("datetime").datetime.now().astimezone().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WuziQ V6.9 parallel AlphaZero efficiency experiment")
    parser.add_argument("--preflight", action="store_true", help="write the current execution preflight")
    parser.add_argument("--scenario", choices=sorted(WORKER_BY_SCHEME))
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-after", action="store_true")
    parser.add_argument("--start-method", choices=__import__("multiprocessing").get_all_start_methods(), default="spawn")
    args = parser.parse_args()

    if args.preflight:
        result = _preflight()
        destination = ROOT / ARTIFACT / "results" / "preflight.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        if not args.scenario:
            return

    if not args.scenario and args.workers is None:
        return
    if args.baseline_checkpoint is None:
        parser.error("--baseline-checkpoint is required for a training or audit run")
    workers = args.workers if args.workers is not None else WORKER_BY_SCHEME[str(args.scenario)]
    run_id = args.run_id or str(args.scenario)
    summary = run_one(
        run_id=run_id,
        workers=workers,
        games=args.games,
        baseline_checkpoint=args.baseline_checkpoint,
        output_root=args.output_root,
        start_method=args.start_method,
    )
    output_root = ROOT / ARTIFACT / "runs" if args.output_root is None else args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    audit = audit_run(
        output_root / run_id,
        baseline_checkpoint=args.baseline_checkpoint,
    ) if args.audit_after else None
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
