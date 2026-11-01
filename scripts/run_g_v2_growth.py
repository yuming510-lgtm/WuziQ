#!/usr/bin/env python3
"""Resume registered G V2 training, development selection and final blind evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v2_blind import run_v2_blind, summarize_v2_results
from src.research.alphazero_g.v2_selection import _atomic_json, run_v2_growth


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("research/g_v2/config/experiment.json"))
    parser.add_argument("--growth-only", action="store_true", help="finish training/development but defer the blind run")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    if config.get("registered") is not True:
        raise ValueError("G V2 growth requires the registered experiment configuration")
    schedule = json.loads((root / config["opening_schedule_path"]).read_text())
    state = run_v2_growth(root, config)
    output: dict[str, Any] = {"growth": state}
    if not args.growth_only:
        best = state.get("best_candidate_for_blind")
        if not best or not best.get("checkpoint"):
            raise RuntimeError("no development-selected candidate is available for blind evaluation")
        g_checkpoint = Path(best["checkpoint"])
        if not g_checkpoint.is_absolute():
            g_checkpoint = root / g_checkpoint
        f2_checkpoint = root / "research/f_v2/checkpoints/best_f2.pt"
        matches = run_v2_blind(
            root, config, schedule, {"F2": f2_checkpoint, "G": g_checkpoint},
        )
        summary = summarize_v2_results(matches)
        summary.update({
            "selected_candidate": best,
            "development_champion": state["champion"],
            "blind_games": len(matches),
            "registered_blind_games": int(config["evaluation"]["blind_games"]),
        })
        summary_path = root / "research/g_v2/results/evaluation/blind_summary.json"
        _atomic_json(summary_path, summary)
        output["blind_summary"] = summary
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return output


if __name__ == "__main__":
    main()
