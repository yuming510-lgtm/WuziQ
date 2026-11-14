from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_4_ladder import (
    ARTIFACT, BASE_COMMIT, DEFAULT_OPENING_SEED, EXPERIMENT_ID, MODEL_SHA256,
    OPPONENTS, ROOT as REPO_ROOT, SNAPSHOT_SOURCE_DRIVE_FOLDER,
    SNAPSHOT_SOURCE_SEQUENCE, SNAPSHOT_SOURCE_SHA256, TRAINING_STATE_SHA256,
    build_schedule, canonical_json, collect_historical_openings, schedule_games,
    validate_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and validate the G5000 V6.4 opening schedule")
    parser.add_argument("--historical-root", action="append", default=[],
                        help="additional root to scan for prior V6.3 JSON/JSONL opening records")
    parser.add_argument("--seed", type=int, default=DEFAULT_OPENING_SEED)
    args = parser.parse_args()
    roots = [REPO_ROOT, *(Path(value).resolve() for value in args.historical_root)]
    historical_by_key = {}
    for root in roots:
        for moves in collect_historical_openings(root):
            from src.research.alphazero_g.v5_evaluation import opening_key
            historical_by_key[opening_key(moves, 9)] = moves
    historical = [historical_by_key[key] for key in sorted(historical_by_key)]
    schedule = build_schedule(seed=args.seed, historical_openings=historical)
    audit = validate_schedule(schedule, historical_openings=historical)
    output_dir = REPO_ROOT / ARTIFACT / "config"
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "opening_schedule.json"
    schedule_path.write_text(json.dumps(schedule, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    historical_digest = hashlib.sha256(canonical_json(sorted(historical_by_key)).encode("utf-8")).hexdigest()
    registration = {
        "experiment_id": EXPERIMENT_ID,
        "branch": "research/g5000-ladder-v6-4",
        "base_commit": BASE_COMMIT,
        "board_size": 9,
        "n_in_row": 5,
        "candidate": {"name": "G5000", "model_sha256": MODEL_SHA256,
                      "training_state_sha256": TRAINING_STATE_SHA256,
                      "search": {"type": "frozen AlphaZero PUCT", "simulations": 400,
                                 "configured_playouts": 401, "c_puct": 5.0}},
        "opponents": [dict(row) for row in OPPONENTS],
        "alpha_beta_settings": {"max_depth": 69, "candidate_radius": 2},
        "opening_groups": 3,
        "opening_plies": 6,
        "planned_games": 24,
        "cpu_snapshot_interval_seconds": 3600,
        "source_drive_snapshot": {"sequence": SNAPSHOT_SOURCE_SEQUENCE,
                                  "sha256": SNAPSHOT_SOURCE_SHA256,
                                  "folder_url": SNAPSHOT_SOURCE_DRIVE_FOLDER,
                                  "training_state_sha256": TRAINING_STATE_SHA256},
        "historical_opening_scan": {
            "roots": [str(path) for path in roots],
            "unique_d4_opening_keys": len(historical_by_key),
            "sha256": historical_digest,
            "overlap": audit["historical_d4_overlap"],
        },
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_path": str(schedule_path.relative_to(REPO_ROOT)),
        "game_ids": [row["game_id"] for row in schedule_games(schedule)],
        "opening_seed": schedule["opening_seed"],
        "preflight_resource_assumptions": "CPU-only; values recorded by preflight_g_v6_4.py before games",
    }
    (output_dir / "registration.json").write_text(
        json.dumps(registration, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registration": registration, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
