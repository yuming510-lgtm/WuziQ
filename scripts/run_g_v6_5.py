from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.research.alphazero_g.v6_5_training import prepare_experiment, preflight, run_training_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="WuziQ AlphaZero G V6.5 continuation")
    parser.add_argument("phase", choices=("prepare", "preflight", "train-batch"))
    parser.add_argument("--source-state", default=None)
    parser.add_argument("--source-node", default=None)
    parser.add_argument("--drive-folder-id", default=None)
    parser.add_argument("--batch-games", type=int, default=250)
    parser.add_argument("--batch-cpu-seconds", type=float, default=10800)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "prepare":
        result = prepare_experiment(
            ROOT, source_state_path=args.source_state,
            source_node_path=args.source_node, drive_folder_id=args.drive_folder_id,
        )
    elif args.phase == "preflight":
        result = preflight(ROOT)
    else:
        result = run_training_batch(
            ROOT, batch_games=args.batch_games, batch_cpu_seconds=args.batch_cpu_seconds,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
