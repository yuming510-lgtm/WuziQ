#!/usr/bin/env python3
"""Freeze the hash-locked V3 comparison after throughput and BN evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_config import register_v3_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[510901, 510902])
    parser.add_argument("--states-per-seed", type=int, choices=(10000, 30000, 50000), default=30000)
    parser.add_argument("--cpu-hour-cap", type=float, default=6.0)
    parser.add_argument("--bn-policy", choices=("update", "freeze"), default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.bn_policy is None:
        probe = json.loads((root / "research/g_v3/results/bn_probe/probe.json").read_text())
        bn_policy = probe["selection_rule"]["selected_common_mode"]
    else:
        bn_policy = args.bn_policy
    config = register_v3_budget(
        root,
        seeds=list(args.seeds),
        states_per_seed=args.states_per_seed,
        cpu_hour_cap=args.cpu_hour_cap,
        bn_policy=bn_policy,
    )
    print(json.dumps({
        "path": "research/g_v3/config/experiment.json",
        "config_sha256": config["config_sha256"],
        "states_per_seed": config["states_per_seed"],
        "seeds": config["seeds"],
        "bn_policy": config["bn_policy"],
        "cpu_hour_cap": config["hard_cpu_hour_cap"],
        "runtime_estimate": config["runtime_estimate"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
