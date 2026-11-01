#!/usr/bin/env python3
"""Run one registered checkpoint on the fixed G V2 development openings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v2_selection import run_v2_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("research/g_v2/config/experiment.json"))
    parser.add_argument("--candidate", choices=("F2", "lr_1e4", "lr_3e5", "lr_1e3"), required=True)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponents", nargs="+", choices=("F2", "A", "B"), default=("F2", "A", "B"))
    parser.add_argument("--modes", nargs="+", choices=("network", "puct"), default=("network", "puct"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    if config.get("registered") is not True:
        raise ValueError("development evaluation requires the registered G V2 config")
    schedule_path = root / config["opening_schedule_path"]
    schedule = json.loads(schedule_path.read_text())
    if args.checkpoint:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    elif args.candidate == "F2":
        checkpoint = root / "research/f_v2/checkpoints/best_f2.pt"
    else:
        checkpoint = root / "research/g_v2/checkpoints" / args.candidate / f"node_{args.node}.pt"
    matches = run_v2_development(
        root, checkpoint, args.candidate, args.node, schedule, int(config["seed"]),
        opponents=tuple(args.opponents), modes=tuple(args.modes),
        cpu_cap_seconds=float(config.get("hard_cpu_hour_cap", 2.0)) * 3600.0,
    )
    print(json.dumps({"candidate": args.candidate, "node": args.node, "games": len(matches)}, sort_keys=True))


if __name__ == "__main__":
    main()
