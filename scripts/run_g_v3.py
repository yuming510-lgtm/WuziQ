#!/usr/bin/env python3
"""Run/resume registered G V3 training and paired milestone evaluations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_budget import BudgetExhausted
from src.research.alphazero_g.v3_evaluation import run_v3_development, run_v3_tactical_stage, run_v3_blind
from src.research.alphazero_g.v3_runner import (
    ARMS, completed_v3_node_progress, initialize_v3_starts, load_v3_config,
    run_v3_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--through-node", type=int, default=None,
                        help="stop after this registered milestone; repeated calls resume exactly")
    parser.add_argument("--max-games-per-call", type=int, default=None,
                        help="optional bounded yield for checkpoint/recovery testing")
    parser.add_argument("--skip-blind", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_v3_config(root)
    seeds = list(map(int, config["seeds"]))
    nodes = list(map(int, config["stage_nodes"]))
    target_node = nodes[-1] if args.through_node is None else int(args.through_node)
    if target_node not in nodes:
        raise SystemExit(f"--through-node must be one of {nodes}")
    initialized_starts = initialize_v3_starts(root)
    summary = {"status": "running", "through_node": target_node, "training": [], "evaluation": []}
    summary["initialized_route_seeds"] = len(initialized_starts)
    for node in nodes:
        if node > target_node:
            break
        for seed in seeds:
            for arm in ARMS:
                result = completed_v3_node_progress(root, arm=arm, seed=seed, node=node)
                if result is None:
                    result = run_v3_arm(
                        root, arm=arm, seed=seed, until_node=node,
                        stop_after_games=args.max_games_per_call,
                    )
                summary["training"].append({
                    "arm": arm, "seed": seed, "node": node,
                    "budgeted_environment_steps": result["budgeted_environment_steps"],
                    "game_count": result["game_count"],
                    "optimizer_updates": result["optimizer_updates"],
                    "completed": result["completed"],
                    "budget_exhausted": result.get("budget_exhausted"),
                })
                if not result["completed"]:
                    summary["status"] = "incomplete_training"
                    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
                    return
            if node == nodes[0]:
                try:
                    dev_rows = run_v3_development(root, node=0, seed=seed)
                    tactic_rows = run_v3_tactical_stage(root, node=0, seed=seed)
                    summary["evaluation"].extend([
                        {"node": 0, "seed": seed, "type": "development", "new_records": len(dev_rows)},
                        {"node": 0, "seed": seed, "type": "tactical", "new_records": len(tactic_rows)},
                    ])
                except BudgetExhausted as exc:
                    summary["status"] = "incomplete_cpu_budget"
                    summary["budget_exhausted"] = str(exc)
                    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
                    return
            try:
                dev_rows = run_v3_development(root, node=node, seed=seed)
                tactic_rows = run_v3_tactical_stage(root, node=node, seed=seed)
                summary["evaluation"].extend([
                    {"node": node, "seed": seed, "type": "development", "new_records": len(dev_rows)},
                    {"node": node, "seed": seed, "type": "tactical", "new_records": len(tactic_rows)},
                ])
            except BudgetExhausted as exc:
                summary["status"] = "incomplete_cpu_budget"
                summary["budget_exhausted"] = str(exc)
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
                return
    if target_node == nodes[-1] and not args.skip_blind:
        try:
            blind_rows = run_v3_blind(root)
            summary["blind_games_new"] = len(blind_rows)
        except BudgetExhausted as exc:
            summary["status"] = "incomplete_cpu_budget"
            summary["budget_exhausted"] = str(exc)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return
    summary["status"] = "complete_training_and_registered_evaluations" if target_node == nodes[-1] else "milestone_complete"
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
