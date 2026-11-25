#!/usr/bin/env python3
"""Replay-audit one closed V7 WORK P6 self-play segment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g import v6_5_audit, v6_5_training
from src.research.alphazero_g.v7_work_p6_50k import (
    ARTIFACT, EXPERIMENT_ID, SIMULATIONS, _atomic_json,
)

RUN_ID = "V7_WORK_P6_50K"


def audit(chunk_start: int, chunk_stop: int | None = None) -> dict:
    run_dir = ROOT / ARTIFACT / "runs" / RUN_ID
    progress = json.loads((run_dir / "results/progress.json").read_text())
    if progress.get("status") not in {"checkpointed", "complete"}:
        raise ValueError("chunk audit requires a clean saved segment boundary")
    total_added = int(progress["games_completed"])
    chunk_stop = total_added if chunk_stop is None else int(chunk_stop)
    if not 0 <= chunk_start < chunk_stop <= total_added:
        raise ValueError("chunk range must be a non-empty committed prefix interval")

    metrics_path = run_dir / "results/per_game.jsonl"
    all_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(all_rows) != total_added:
        raise ValueError("per-game row count differs from committed progress")
    rows = [row for row in all_rows if chunk_start <= int(row["ordinal"]) < chunk_stop]
    expected = chunk_stop - chunk_start
    ids = [str(row["game_id"]) for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise ValueError("chunk game IDs or ordinals are missing or duplicated")

    raw_dir = run_dir / "raw/selfplay/reference400_empty"
    replayed_states = 0
    for row in rows:
        game_id = str(row["game_id"])
        record_path = raw_dir / f"{game_id}.json"
        sample_path = raw_dir / f"{game_id}.npz"
        if not record_path.is_file() or not sample_path.is_file():
            raise FileNotFoundError(f"missing raw record or training samples for {game_id}")
        record = json.loads(record_path.read_text())
        if str(record.get("game_id")) != game_id or record.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"game identity mismatch for {game_id}")
        check = v6_5_audit.audit_game_record(record, sample_path)
        if int(record.get("simulations_configured_per_move", -1)) != SIMULATIONS:
            raise ValueError(f"{game_id} did not use PUCT-400")
        if int(record["seed"]) != int(row["seed"]):
            raise ValueError(f"{game_id} seed differs between metric and game record")
        if int(row["training_update_calls"]) != 1 or int(row["updates_after"]) != int(row["updates_before"]) + 1:
            raise ValueError(f"{game_id} did not receive exactly one optimizer update")
        if int(record.get("parallel_training", {}).get("weight_lag_updates", -1)) != int(row["weight_lag_updates"]):
            raise ValueError(f"{game_id} weight-lag record is inconsistent")
        replayed_states += int(check["move_count"])

    state_path = run_dir / "checkpoints/training_state.pt"
    state = v6_5_training.validate_training_state(state_path, root=ROOT)
    added_updates = int(state["optimizer_updates"]) - 19_999
    if int(state["game_count"]) != 20_000 + total_added or added_updates != total_added:
        raise ValueError("checkpoint game/update counters do not match all committed rows")
    if int(progress["optimizer_update_calls"]) != total_added:
        raise ValueError("central optimizer call count differs from committed games")

    result = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "range_added_games": [chunk_start, chunk_stop],
        "games": expected,
        "unique_game_ids": len(set(ids)),
        "legal_terminal_games": expected,
        "puct400_games": expected,
        "optimizer_update_calls": expected,
        "replayed_training_states": replayed_states,
        "cumulative_games": int(state["game_count"]),
        "optimizer_updates": int(state["optimizer_updates"]),
        "checkpoint_sha256": state["sha256"],
        "checkpoint_valid": bool(state["valid"]),
        "valid": True,
    }
    audit_path = run_dir / f"results/audit_chunk_{chunk_start+1:05d}_{chunk_stop:05d}.json"
    _atomic_json(audit_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-start", type=int, required=True)
    parser.add_argument("--chunk-stop", type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.chunk_start, args.chunk_stop), sort_keys=True))


if __name__ == "__main__":
    main()
