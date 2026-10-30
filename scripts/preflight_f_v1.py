#!/usr/bin/env python3
"""Measure pilot generation/annotation and CNN training throughput before the run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.dataset import HISTORICAL_9X9_SOURCES
from src.research.imitation_f.generation import run_generation
from src.research.imitation_f.model import PolicyValueNet, select_device


def _memory_snapshot() -> dict:
    data = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable"):
                data[name] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass
    return data


def _benchmark_updates(device: torch.device, updates: int = 100, batch_size: int = 64) -> float:
    model = PolicyValueNet(channels=32, blocks=2, seed=731501).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.rand(batch_size, 3, 9, 9, device=device)
    target = torch.softmax(torch.randn(batch_size, 81, device=device), dim=-1)
    value_target = torch.rand(batch_size, device=device) * 2 - 1
    start = time.perf_counter()
    model.train()
    for _ in range(updates):
        logits, value = model(x)
        legal = x[:, 2].reshape(-1, 81) > 0.1
        masked = logits.masked_fill(~legal, -1e9)
        loss = -(target * torch.log_softmax(masked, dim=-1)).sum(-1).mean() + (value - value_target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / updates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "research/f_v1/config/generation.json"
    output_path = args.output or root / "research/f_v1/results/preflight.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = root / "research/f_v1/data/historical_split_manifest.json"
    if not split_path.exists():
        raise FileNotFoundError("historical split must be frozen before the preflight")
    device = select_device("auto")
    torch.set_num_threads(3)
    step_seconds = _benchmark_updates(device)
    pilot_config = dict(config)
    pilot_config["games"] = config["games"][:2]
    pilot_config["maximum_games"] = len(pilot_config["games"])
    pilot_dir = Path("/tmp/wuziq-f-v1-pilot")
    if pilot_dir.exists():
        shutil.rmtree(pilot_dir)
    pilot_started = time.perf_counter()
    pilot = run_generation(pilot_config, pilot_dir, split_path)
    pilot_seconds = time.perf_counter() - pilot_started
    pilot_rows_path = pilot_dir / "generated_annotations.jsonl"
    pilot_rows = [json.loads(line) for line in pilot_rows_path.read_text().splitlines() if line.strip()]
    pilot_unique = len({row["state_key"] for row in pilot_rows})
    games_per_hour = 3600 * len(pilot_config["games"]) / max(0.001, pilot_seconds)
    labels_per_second = len(pilot_rows) / max(0.001, pilot_seconds)
    data_dir = root / "research/f_v1/data"
    current_train = sum(1 for _ in (data_dir / "train.jsonl").open())
    required = max(0, int(json.loads((root / "research/f_v1/config/training.json").read_text())["target_train_positions"]) - current_train)
    avg_unique_per_game = pilot_unique / max(1, len(pilot_config["games"]))
    estimate_games = min(int(config["maximum_games"]), int((required + max(1, avg_unique_per_game) - 1) // max(1, avg_unique_per_game)))
    planned_games = int(config["maximum_games"])
    generation_seconds = pilot_seconds / max(1, len(pilot_config["games"])) * int(config["maximum_games"])
    dataset_rows = max(current_train, current_train + int(planned_games * avg_unique_per_game))
    updates = 20 * ((dataset_rows + 63) // 64)
    training_seconds = updates * step_seconds
    # Reserve a factor for validation passes, JSON/data preparation and periodic checkpoint I/O.
    central_seconds = generation_seconds + training_seconds * 1.35 + 30
    payload = {
        "resource_snapshot": {
            "cpu_logical": os.cpu_count(),
            "memory_bytes": _memory_snapshot(),
            "disk_free_bytes": shutil.disk_usage(root).free,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "torch_version": torch.__version__,
            "historical_9x9_source_count": len(HISTORICAL_9X9_SOURCES),
        },
        "pilot": {
            "games": len(pilot_config["games"]),
            "wall_seconds": pilot_seconds,
            "games_per_hour": games_per_hour,
            "annotations": len(pilot_rows),
            "unique_canonical_positions": pilot_unique,
            "labels_per_second": labels_per_second,
        },
        "training_benchmark": {"updates": 100, "batch_size": 64, "seconds_per_update": step_seconds},
        "estimate": {
            "required_unique_train_positions": required,
            "estimated_games_to_target": estimate_games,
            "maximum_games": int(config["maximum_games"]),
            "estimated_dataset_train_rows": dataset_rows,
            "estimated_training_updates": updates,
            "generation_seconds_at_pilot_rate": generation_seconds,
            "training_seconds_without_validation": training_seconds,
            "optimistic_seconds": central_seconds * 0.65,
            "central_seconds": central_seconds,
            "conservative_seconds": central_seconds * 2.5,
            "calculation": f"pilot wall/game × {planned_games} planned games + 20×ceil(train rows/64) × measured seconds/update × 1.35 + 30 seconds overhead",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
