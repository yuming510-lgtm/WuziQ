"""Resource inspection and measured, category-weighted V4 runtime preflight."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from .abc_search_v4 import (
    EXPECTED_BUDGETS_MS,
    EXPECTED_MODELS,
    _canonical_hash,
    _run_one,
    load_config,
)
from .checkpoint import write_json_atomic


_REPRESENTATIVE_IDS = (
    "opening_08",
    "midgame_24",
    "late_58",
    "attack_open_four",
    "defense_block_four",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resource_snapshot(path: str | Path) -> dict[str, Any]:
    """Measure visible CPU quota, memory limits, load, and workspace disk."""

    cpu_model = "unknown"
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    meminfo: dict[str, int] = {}
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text(errors="replace").splitlines():
            key, _, rest = line.partition(":")
            if rest.strip().endswith("kB"):
                meminfo[key] = int(rest.strip().split()[0]) * 1024
    cpu_quota = None
    cpu_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_path.exists():
        quota, period = cpu_path.read_text().split()[:2]
        cpu_quota = None if quota == "max" else int(quota) / int(period)
    memory_limit = None
    memory_path = Path("/sys/fs/cgroup/memory.max")
    if memory_path.exists():
        raw = memory_path.read_text().strip()
        memory_limit = None if raw == "max" else int(raw)
    disk = shutil.disk_usage(Path(path))
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "cgroup_cpu_quota_cores": cpu_quota,
        "load_average_1_5_15m": load_average,
        "memory_total_bytes": meminfo.get("MemTotal"),
        "memory_available_bytes": meminfo.get("MemAvailable"),
        "cgroup_memory_limit_bytes": memory_limit,
        "workspace_disk_total_bytes": disk.total,
        "workspace_disk_used_bytes": disk.used,
        "workspace_disk_free_bytes": disk.free,
    }


def representative_positions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select one frozen position from each required search-phase category."""

    indexed = {str(row["position_id"]): row for row in config["positions"]}
    missing = set(_REPRESENTATIVE_IDS) - set(indexed)
    if missing:
        raise ValueError(f"V4 config lacks preflight positions: {sorted(missing)}")
    return [indexed[position_id] for position_id in _REPRESENTATIVE_IDS]


def estimate_full_runtime(
    config: dict[str, Any], sample_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Estimate full wall time and JSONL storage from measured category/budget cells."""

    category_counts = Counter(str(row["category"]) for row in config["positions"])
    samples: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in sample_runs:
        model = str(row["model"])
        budget = int(row["budget_ms"])
        category = str(row["category"])
        if model not in EXPECTED_MODELS or budget not in EXPECTED_BUDGETS_MS:
            raise ValueError("Preflight sample contains an unsupported model or budget")
        samples[(category, budget, model)].append(float(row["elapsed_seconds"]))

    by_model_budget: dict[str, dict[str, Any]] = {model: {} for model in EXPECTED_MODELS}
    estimated_seconds = 0.0
    for model in EXPECTED_MODELS:
        for budget in EXPECTED_BUDGETS_MS:
            category_estimates: dict[str, float] = {}
            seconds = 0.0
            for category, position_count in sorted(category_counts.items()):
                values = samples.get((category, budget, model))
                if not values:
                    raise ValueError(f"Missing preflight sample: {category}/{model}/{budget}ms")
                average = sum(values) / len(values)
                category_estimates[category] = average
                seconds += average * position_count
            by_model_budget[model][str(budget)] = {
                "estimated_seconds": seconds,
                "mean_seconds_per_position_by_category": category_estimates,
            }
            estimated_seconds += seconds
    sample_sizes = [len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1 for row in sample_runs]
    average_record_bytes = sum(sample_sizes) / max(1, len(sample_sizes))
    official_runs = len(config["positions"]) * len(EXPECTED_MODELS) * len(EXPECTED_BUDGETS_MS)
    estimated_storage = int(average_record_bytes * official_runs * 1.25)
    return {
        "preflight_sample_runs": len(sample_runs),
        "official_runs": official_runs,
        "position_count": len(config["positions"]),
        "estimated_official_compute_seconds": estimated_seconds,
        "estimated_official_compute_minutes": estimated_seconds / 60.0,
        "estimated_total_seconds_including_preflight": estimated_seconds,
        "estimated_result_storage_bytes": estimated_storage,
        "estimated_average_record_bytes": average_record_bytes,
        "by_model_budget": by_model_budget,
        "estimate_note": (
            "Category-weighted planning estimate from one measured representative position per "
            "category and budget. Wall time includes sequential search/decision time; process and "
            "filesystem overhead are small additional costs."
        ),
    }


def run_preflight(
    config_path: str | Path,
    output_path: str | Path,
    *,
    resource_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute 105 measured sample decisions and write resource/runtime estimates."""

    config = load_config(config_path)
    root = Path(resource_path or Path(config_path).resolve().parents[1])
    resources_before = resource_snapshot(root)
    positions = representative_positions(config)
    code_commit = "unknown"
    try:
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(config_path).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    started_wall = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for position in positions:
        for model in EXPECTED_MODELS:
            for budget_ms in EXPECTED_BUDGETS_MS:
                result = _run_one(
                    config,
                    _canonical_hash(config),
                    code_commit,
                    position,
                    model,
                    budget_ms,
                )
                result["preflight_id"] = (
                    f"{position['position_id']}__{model}__{budget_ms:05d}ms"
                )
                samples.append(result)
    elapsed_wall = time.perf_counter() - started_wall
    resources_after = resource_snapshot(root)
    estimate = estimate_full_runtime(config, samples)
    estimate["preflight_wall_seconds"] = elapsed_wall
    estimate["estimated_total_seconds_including_preflight"] += elapsed_wall
    estimate["estimated_total_minutes_including_preflight"] = (
        estimate["estimated_total_seconds_including_preflight"] / 60.0
    )
    estimate["estimated_storage_fits_current_disk"] = (
        estimate["estimated_result_storage_bytes"] < resources_after["workspace_disk_free_bytes"]
    )
    output = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "started_at_utc": _utc_now(),
        "resources_before": resources_before,
        "resources_after": resources_after,
        "estimate": estimate,
        "sample_runs": samples,
    }
    write_json_atomic(output_path, output)
    return output
