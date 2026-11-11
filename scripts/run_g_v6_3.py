from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.research.alphazero_g.v6_3_evaluation import run_all
from src.research.alphazero_g.v6_3_training import (
    ARTIFACT, SNAPSHOT_INTERVAL_CPU_SECONDS, SNAPSHOT_INTERVAL_GAMES,
    prepare_experiment, run_training_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="WuziQ AlphaZero G V6.3 runner")
    parser.add_argument("phase", choices=("prepare", "train-batch", "challenge"))
    parser.add_argument("--batch-games", type=int, default=SNAPSHOT_INTERVAL_GAMES)
    parser.add_argument("--batch-cpu-seconds", type=float, default=SNAPSHOT_INTERVAL_CPU_SECONDS)
    parser.add_argument("--match-cpu-soft-limit", type=float, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        result = prepare_experiment(ROOT)
    elif args.phase == "train-batch":
        result = run_training_batch(ROOT, batch_games=args.batch_games,
                                    batch_cpu_seconds=args.batch_cpu_seconds)
    else:
        result = run_all(ROOT, cpu_soft_limit_seconds=args.match_cpu_soft_limit)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
