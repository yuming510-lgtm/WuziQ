#!/usr/bin/env python3
"""Run the pre-registered low-cost BatchNorm comparison before V3 training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_bn import run_bn_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[510901, 510902])
    args = parser.parse_args()
    root = args.root.resolve()
    config = {
        "seeds": list(args.seeds),
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "bn_probe_selection": {
            "states_per_route": 256,
            "updates": 16,
            "opening_groups": 8,
            "tie_score_rate_margin": 0.125,
        },
    }
    result = run_bn_probe(root, config)
    print(json.dumps({
        "selection_rule": result["selection_rule"],
        "wall_seconds": result["wall_seconds"],
        "cpu_seconds": result["cpu_seconds"],
        "routes": {
            arm: {
                "budgeted_states": row["budgeted_states"],
                "selfplay_games": row["selfplay_game_count"],
                "scores": {mode: item["puct_score_rate_vs_own_initial"] for mode, item in row["modes"].items()},
            } for arm, row in result["routes"].items()
        },
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
