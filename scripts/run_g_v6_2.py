from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_2_evaluation import run_final_blind, run_stage
from src.research.alphazero_g.v6_2_training import ROOT, train_to_games
from src.research.alphazero_g.v6_2_evaluation import _atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume the registered G V6.2 experiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train-through-games", type=int, choices=(1250, 1500, 1750, 2000, 2500, 3000))
    group.add_argument("--evaluate-games", type=int, choices=(1250, 1500, 1750, 2000, 2500, 3000))
    group.add_argument("--finalize-blind", action="store_true")
    parser.add_argument("--session-cpu-hours", type=float, default=9.0)
    args = parser.parse_args()
    if args.session_cpu_hours <= 0 or args.session_cpu_hours > 9.5:
        parser.error("each execution session must be >0 and <=9.5 CPU hours")
    if args.train_through_games is not None:
        result = train_to_games(ROOT, target_games=args.train_through_games,
                                session_cpu_seconds=args.session_cpu_hours * 3600)
    elif args.evaluate_games is not None:
        result = run_stage(ROOT, game_count=args.evaluate_games)
        if args.evaluate_games == 2_000:
            primary = result["comparisons"]["400"]["G20K"]
            primary_complete = bool(primary.get("complete") and int(primary.get("paired_openings", 0)) >= 80)
            passed = (primary_complete
                      and float(primary["score_rate"]) > 0.55
                      and float(primary["paired_opening_bootstrap_ci95"][0]) > 0.50
                      and int(primary["paired_openings"]) >= 80)
            gate = {
                "experiment_id": "wuziq-alphazero-g-v6-2-9x9-selfplay-scale",
                "primary": "2,000-game candidate vs frozen G20K, PUCT-400, 80 paired openings",
                "score_rate": float(primary["score_rate"]),
                "paired_ci95": primary["paired_opening_bootstrap_ci95"],
                "paired_openings": int(primary["paired_openings"]),
                "primary_complete": primary_complete,
                "threshold_score_rate_strictly_above": 0.55,
                "threshold_ci95_lower_strictly_above": 0.50,
                "passed": bool(passed),
                "decision": "advance to 2,500 and 3,000 games" if passed else "stop training at 2,000 games; gate not met",
            }
            _atomic_json(Path(ROOT) / "research/g_v6_2/results/gate_2000.json", gate)
            result["advancement_gate"] = gate
    else:
        result = run_final_blind(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
