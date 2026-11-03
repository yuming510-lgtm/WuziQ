from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.research.alphazero_g.v5_reference import ReferencePolicyValueNet
from src.research.alphazero_g.v5_training import _run_game


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _resource_info() -> dict:
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    mem_max = _read("/sys/fs/cgroup/memory.max")
    if cpu_max and cpu_max != "max":
        quota, period = map(int, cpu_max.split())
        quota_cores = quota / period
    else:
        quota_cores = float(os.cpu_count() or 1)
    if mem_max and mem_max != "max":
        memory_bytes = int(mem_max)
    else:
        memory_bytes = None
    disk = subprocess.run(["df", "-B1", "/workspace"], capture_output=True, text=True, check=True).stdout.splitlines()[-1].split()
    return {
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "cpu_quota_cores": quota_cores,
        "memory_limit_bytes": memory_bytes,
        "workspace_disk_available_bytes": int(disk[3]),
        "gpu_available": bool(torch.cuda.is_available()),
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
    }


def _train_batch_seconds(size: int, repeats: int = 5) -> float:
    model = ReferencePolicyValueNet(size, seed=991 + size)
    states = np.random.default_rng(size).normal(size=(512, 4, size, size)).astype(np.float32)
    probs = np.random.default_rng(size + 1).random(size=(512, size * size), dtype=np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    values = np.random.default_rng(size + 2).uniform(-1, 1, size=512).astype(np.float32)
    start = time.process_time()
    for _ in range(repeats):
        model.train_step(states, probs, values, 0.002)
    return (time.process_time() - start) / repeats


def run_preflight() -> dict:
    cpu_started = time.process_time()
    torch.set_num_threads(1)
    bench = []
    for size, row, simulations in ((6, 4, 400), (9, 5, 64), (9, 5, 200), (9, 5, 400)):
        model = ReferencePolicyValueNet(size, seed=7700 + size + simulations)
        record, states, _probs, _values = _run_game(
            size=size, n_in_row=row, simulations=simulations, model=model,
            game_id=f"preflight-{size}-{simulations}", seed=8800 + simulations,
        )
        bench.append({
            "board_size": size,
            "n_in_row": row,
            "simulations_configured": simulations,
            "complete_game_plies": int(record["training_state_count"]),
            "selfplay_cpu_seconds": float(record["runner_cpu_seconds"]),
            "selfplay_cpu_seconds_per_state": float(record["runner_cpu_seconds"] / max(1, len(states))),
            "mean_search_cpu_seconds_per_state": float(record["mean_search_cpu_seconds"]),
            "mean_search_depth": float(record["mean_tree_depth"]),
        })
    batch = {str(size): _train_batch_seconds(size) for size in (6, 9)}

    def estimate(copies: float) -> float:
        six = 20_000 * bench[0]["selfplay_cpu_seconds_per_state"]
        six_games = 20_000 / max(1, bench[0]["complete_game_plies"])
        six += six_games * 5 * batch["6"]
        nine64 = 8_000 * bench[1]["selfplay_cpu_seconds_per_state"]
        nine64 += (8_000 / bench[1]["complete_game_plies"]) * 5 * batch["9"]
        nine200 = 3_000 * bench[2]["selfplay_cpu_seconds_per_state"]
        nine200 += (3_000 / bench[2]["complete_game_plies"]) * 5 * batch["9"]
        nine400 = 1_500 * bench[3]["selfplay_cpu_seconds_per_state"]
        nine400 += (1_500 / bench[3]["complete_game_plies"]) * 5 * batch["9"]
        midgame64 = 3_000 * bench[1]["selfplay_cpu_seconds_per_state"]
        midgame64 += (3_000 / bench[1]["complete_game_plies"]) * 5 * batch["9"]
        # Reserve 1.5 CPU hours for repeated paired evaluation, audits, and figures.
        return (six + nine64 + nine200 + nine400 + midgame64) / 3600 * copies + 1.5

    estimates = {
        "optimistic_cpu_hours": estimate(0.65),
        "normal_cpu_hours": estimate(1.25),
        "conservative_cpu_hours": estimate(2.0),
        "basis": "one complete self-play game per cell and batch-512 optimizer throughput; project each route to registered state counts, then add 1.5 CPU hours for paired evaluation/audit; factors cover game-length/CPU variance",
    }
    result = {
        "format_version": 1,
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resources": _resource_info(),
        "torch_threads": torch.get_num_threads(),
        "benchmark_games": bench,
        "batch512_optimizer_seconds_per_update": batch,
        "estimates": estimates,
        "cpu_hour_hard_cap": 10.0,
    }
    result["preflight_cpu_seconds"] = time.process_time() - cpu_started
    return result


def main() -> None:
    report = run_preflight()
    path = Path(__file__).resolve().parents[1] / "research/g_v5/results/preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
