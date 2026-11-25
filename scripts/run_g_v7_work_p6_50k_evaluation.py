#!/usr/bin/env python3
"""Run resumable 160-game V6.8-protocol evaluation blocks for each V7 stage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import hashlib
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import torch

from src.research.alphazero_g import v7_work_p6_50k_evaluation as evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-games", type=int, default=160)
    parser.add_argument("--stage-games", type=int,
                        help="after the idempotent batch, summarize all four 40-game conditions at this milestone")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = {}
    if args.register:
        result["registration"] = evaluation.prepare_experiment()
    if args.audit:
        result["audit"] = evaluation.audit()
        result["registration"] = evaluation.prepare_experiment()
    elif not args.register:
        result["batch"] = evaluation.run_batch(max_new_games=args.batch_games)
        if args.stage_games is not None:
            games = args.stage_games
            if games not in evaluation.MODELS:
                raise SystemExit(f"--stage-games must be one of {evaluation.MODELS}")
            schedule = json.loads((evaluation.ROOT / evaluation.ARTIFACT / "config/opening_schedule.json").read_text())
            stage_summaries = []
            for opponent in evaluation.OPPONENTS:
                rows = []
                for idx, opening in enumerate(schedule["openings"]):
                    for color in ("B", "W"):
                        gid = evaluation._game_id(games, opponent, idx, color)
                        path = evaluation._game_path(evaluation.ROOT, gid)
                        row = json.loads(path.read_text(encoding="utf-8"))
                        evaluation._verify_existing_header(
                            row, games=games, opponent=opponent, opening=opening, color=color,
                            expected_seed=int(schedule["seed"] * 100 + idx * 2 + (color == "W")),
                            model_sha=evaluation.sha_file(evaluation._model_path(evaluation.ROOT, games)),
                        )
                        evaluation._replay_game(row, schedule)
                        rows.append(row)
                if len(rows) != 40:
                    raise ValueError(f"G{games} vs {opponent} has {len(rows)} games, expected 40")
                seed = int.from_bytes(hashlib.sha256(f"v7-work-p6:{games}:{opponent}".encode()).digest()[:4], "big")
                summary = evaluation._summarize(rows, seed=seed)
                summary.update({"experiment_id": evaluation.EXPERIMENT_ID, "candidate": f"G{games}",
                                "candidate_games": games, "opponent": opponent,
                                "opening_schedule_sha256": schedule["opening_schedule_sha256"],
                                "candidate_model_sha256": evaluation.sha_file(evaluation._model_path(evaluation.ROOT, games)),
                                "legal_replays": len(rows)})
                out = evaluation.ROOT / evaluation.ARTIFACT / f"results/summaries/G{games}_{evaluation._slug(opponent)}.json"
                evaluation.atomic_json(out, summary)
                stage_summaries.append(summary)
            stage = {"experiment_id": evaluation.EXPERIMENT_ID, "candidate": f"G{games}",
                     "candidate_games": games, "games": sum(x["games"] for x in stage_summaries),
                     "all_conditions_complete": all(x["complete"] for x in stage_summaries),
                     "scores": {x["opponent"]: x["score_rate"] for x in stage_summaries},
                     "summaries": stage_summaries,
                     "opening_schedule_sha256": schedule["opening_schedule_sha256"]}
            evaluation.atomic_json(evaluation.ROOT / evaluation.ARTIFACT / f"results/stages/G{games}_stage_summary.json", stage)
            result["stage_summary"] = stage
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
