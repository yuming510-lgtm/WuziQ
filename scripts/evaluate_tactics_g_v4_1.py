#!/usr/bin/env python3
"""Run exact rule-labelled tactical probes on the frozen V4.1 checkpoints."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.research.alphazero_g.training import load_g_checkpoint
from src.research.alphazero_g.v3_evaluation import build_v3_tactical_puzzles, evaluate_v3_tactics
from src.research.alphazero_g.v4_1_config import ARM_IDS, load_v4_1_config
from src.research.alphazero_g.v4_1_runner import _cpu_used, _record_cpu


def main() -> None:
    config = load_v4_1_config(ROOT)
    torch.set_num_threads(1)
    checkpoints = {"G0-V3-20K": ROOT / config["initial_checkpoint"]}
    checkpoints.update({arm: ROOT / "research/g_v4_1/checkpoints" / arm / "node_10000.pt"
                        for arm in ARM_IDS})
    puzzles = build_v3_tactical_puzzles()
    results = []
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for checkpoint_id, path in checkpoints.items():
        model = load_g_checkpoint(path, device="cpu")
        for mode in ("network", "puct"):
            row = evaluate_v3_tactics(model, mode=mode, puzzles=puzzles, simulations=64)
            row.update({
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            results.append(row)
    elapsed_cpu, elapsed_wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    ledger = ROOT / "research/g_v4_1/results/cpu_ledger.json"
    if _cpu_used(ROOT) + elapsed_cpu > 6 * 3600:
        raise RuntimeError("G V4.1 six CPU-hour hard cap reached during tactical evaluation")
    _record_cpu(ROOT, "tactics:all-checkpoints", elapsed_cpu, elapsed_wall, "tactical_evaluation")
    payload = {
        "format_version": 1,
        "label_source": "exact five-in-a-row rules only; no teacher or training labels",
        "puzzle_count": len(puzzles),
        "mother_position_count": 4,
        "symmetry_variants_per_mother_position": 8,
        "independent_blind_data_used": False,
        "elapsed_cpu_seconds": elapsed_cpu,
        "elapsed_wall_seconds": elapsed_wall,
        "results": results,
    }
    destination = ROOT / "research/g_v4_1/results/evaluation/tactics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({"puzzles": len(puzzles), "elapsed_cpu_seconds": elapsed_cpu,
                      "results": [{"checkpoint_id": row["checkpoint_id"], "mode": row["mode"],
                                   "correct": row["correct"], "puzzles": row["puzzles"],
                                   "categories": row["categories"]} for row in results]},
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
