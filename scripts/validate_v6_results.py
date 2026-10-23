#!/usr/bin/env python3
"""Replay and audit every V6 game, route, ID, and persisted progress claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import load_game_records
from src.research.v6.config import canonical, load_config, sha256
from src.research.v6.schedule import build_schedule
from src.research.v6.state import derive_state
from src.research.v6.reuse import reconstruct_frozen_schedule
from src.research.v6.validation import validate_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/experiment.json")
    parser.add_argument("--results-dir", default="research/v6/results")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    results = ROOT / args.results_dir
    _base_schedule, routes = build_schedule(config)
    saved_schedule = json.loads((results / "schedule.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads((results / "reuse_manifest.json").read_text(encoding="utf-8"))
    try:
        schedule, _verified_manifest = reconstruct_frozen_schedule(
            config, saved_schedule, saved_manifest
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    records = load_game_records(results / "9x9_games.jsonl")
    replay = validate_records(records, schedule, require_complete_groups=True, verify_proofs=True)
    state = derive_state(records, schedule, routes)
    progress = json.loads((results / "progress.json").read_text(encoding="utf-8"))
    if progress.get("config_sha256") != sha256(canonical(config)):
        raise SystemExit("progress config hash mismatch")
    if progress.get("schedule_sha256") != sha256(canonical(schedule)):
        raise SystemExit("progress schedule hash mismatch")
    if progress.get("completed_ids") != sorted(row["game_id"] for row in records):
        raise SystemExit("progress completed IDs disagree with durable game records")
    if progress.get("completed_games") != len(records):
        raise SystemExit("progress completed game count mismatch")
    if progress.get("complete") != state["complete"]:
        raise SystemExit("progress completion flag disagrees with rebuilt route state")
    if not state["complete"]:
        raise SystemExit(f"V6 route evaluation still has {len(state['pending_group_ids'])} pending groups")
    report = {
        "valid": True,
        "replay": replay,
        "complete": state["complete"],
        "routes": len(state["routes"]),
        "route_status_counts": {
            status: sum(route["status"] == status for route in state["routes"].values())
            for status in sorted({route["status"] for route in state["routes"].values()})
        },
        "groups": len(state["groups"]),
        "group_status_counts": {
            status: sum(group["status"] == status for group in state["groups"].values())
            for status in sorted({group["status"] for group in state["groups"].values()})
        },
        "actual_games": len(records),
        "implementation_commit": progress.get("implementation_commit"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
