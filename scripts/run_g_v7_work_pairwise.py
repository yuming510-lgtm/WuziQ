#!/usr/bin/env python3
"""New-opening PUCT-400 head-to-head validation: G50K vs G12K and G20K."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from src.research.alphazero_g import v5_evaluation
from src.research.alphazero_g import v7_work_p6_50k_evaluation as evaluation


ROOT = evaluation.ROOT
PAIRWISE_SEED = 681_101
OPENINGS = 40
OPPONENT_MODELS = (12_000, 20_000)
OUT = ROOT / evaluation.ARTIFACT / "results/pairwise"


def _atomic_json(path: Path, value: dict) -> None:
    evaluation.atomic_json(path, value)


def prepare_schedule() -> dict:
    standard = json.loads((ROOT / evaluation.ARTIFACT / "config/opening_schedule.json").read_text())
    used = {evaluation._d4_opening_key(o["moves"]) for o in standard["openings"]}
    seed = PAIRWISE_SEED
    while seed < PAIRWISE_SEED + 100:
        pool = v5_evaluation.generate_opening_pool(size=9, n_in_row=5, groups=OPENINGS,
                                                    plies=6, seed=seed)
        keys = {evaluation._d4_opening_key(o["moves"]) for o in pool}
        if len(keys) == OPENINGS and not keys.intersection(used):
            break
        seed += 1
    else:
        raise RuntimeError("could not generate a new 40-opening pool disjoint from V6.8")
    for i, opening in enumerate(pool):
        opening["opening_group_id"] = f"v7-work-p6-pair-{seed}-{i:03d}"
        opening["seed"] = seed + i
    schedule = {
        "experiment_id": "wuziq-alphazero-g-v7-work-p6-50k-pairwise",
        "seed": seed, "opening_plies": 6, "opening_groups": OPENINGS,
        "games_per_condition": OPENINGS * 2, "candidate": "G50000",
        "opponents": [f"G{x}" for x in OPPONENT_MODELS],
        "board_size": 9, "n_in_row": 5, "simulations_per_move": 400,
        "opening_schedule_sha256": hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
        "disjoint_from_v6_8_schedule_under_d4": True, "openings": pool,
    }
    _atomic_json(OUT / "config/opening_schedule.json", schedule)
    return schedule


def run() -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    schedule = prepare_schedule()
    OUT.mkdir(parents=True, exist_ok=True)
    model_cache = {}
    summaries = []
    for opponent_games in OPPONENT_MODELS:
        opponent = f"G{opponent_games}"
        rows = []
        for idx, opening in enumerate(schedule["openings"]):
            for color_index, color in enumerate(("B", "W")):
                seed = int(schedule["seed"] * 100 + idx * 2 + color_index)
                expected_id = evaluation._game_id(50_000, opponent, idx, color)
                path = OUT / "games" / f"{expected_id}.json"
                if path.is_file():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if (row.get("experiment_id") != schedule["experiment_id"]
                            or row.get("opponent") != opponent or row.get("seed") != seed
                            or row.get("candidate_color") != color
                            or row.get("opening_group_id") != opening["opening_group_id"]):
                        raise ValueError(f"existing pairwise match conflicts with frozen schedule: {path.name}")
                else:
                    row = evaluation.play_match(
                        root=ROOT, games=50_000, opponent=opponent, opening=opening,
                        opening_index=idx, candidate_color=color, seed=seed,
                        model_cache=model_cache,
                    )
                    row["experiment_id"] = schedule["experiment_id"]
                    row["opponent_model_sha256"] = evaluation.sha_file(evaluation._model_path(ROOT, opponent_games))
                    row["pairwise_opening_schedule_sha256"] = schedule["opening_schedule_sha256"]
                    _atomic_json(path, row)
                evaluation._replay_game(row, schedule)
                rows.append(row)
        summary = evaluation._summarize(rows, seed=int(schedule["seed"] + opponent_games))
        summary.update({"experiment_id": schedule["experiment_id"], "candidate": "G50000",
                       "candidate_games": 50_000, "opponent": opponent,
                       "paired_openings": OPENINGS, "games": OPENINGS * 2,
                       "complete": len(rows) == OPENINGS * 2,
                       "opening_schedule_sha256": schedule["opening_schedule_sha256"],
                       "candidate_model_sha256": evaluation.sha_file(evaluation._model_path(ROOT, 50_000)),
                       "opponent_model_sha256": evaluation.sha_file(evaluation._model_path(ROOT, opponent_games)),
                       "games_replayed_and_legal": len(rows)})
        _atomic_json(OUT / "summaries" / f"G50000_vs_{opponent}.json", summary)
        summaries.append(summary)
    audit = {"experiment_id": schedule["experiment_id"], "complete": all(s["complete"] for s in summaries),
             "planned_games": 160, "audited_games": sum(s["games_replayed_and_legal"] for s in summaries),
             "unique_game_ids": len(list((OUT / "games").glob("*.json"))),
             "opening_groups_per_opponent": OPENINGS, "paired_openings_new_vs_v6_8": True,
             "all_matches_puct400_for_both_models": True}
    _atomic_json(OUT / "audit.json", audit)
    return {"summaries": summaries, "audit": audit}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
