"""Exactly-once V6.2 tournament runner with game-level crash recovery."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from ..checkpoint import append_game, artifact_lock, load_game_records, write_json_atomic
from ..v6.config import ROOT, canonical, load_config as load_v6_config, sha256, v5_1_engine_settings
from ..v6.runner import play_game
from .config import load_v6_2_config
from .schedule import build_schedule
from .validation import validate_v6_2_records, validate_v6_2_game_record


OUTPUT_DEFAULT = ROOT / "research/v6_2/results"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _progress(
    *, config: dict[str, Any], schedule: list[dict[str, Any]], records: list[dict[str, Any]],
    commit: str, started_at: str, started: float, prior_elapsed: float,
    current_game_id: str | None, games_this_run: int,
) -> dict[str, Any]:
    validation = validate_v6_2_records(records, schedule, require_complete=False)
    updated = _utc_now()
    complete = len(records) == len(schedule)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "baseline_commit": config["baseline_commit"],
        "implementation_commit": commit,
        "config_sha256": sha256(canonical(config)),
        "schedule_sha256": sha256(canonical(schedule)),
        "completed_ids": sorted(str(row["game_id"]) for row in records),
        "completed_games": len(records),
        "completed_groups": validation["groups"],
        "current_game_id": current_game_id,
        "games_completed_this_run": games_this_run,
        "elapsed_seconds_total": prior_elapsed + time.perf_counter() - started,
        "elapsed_wall_seconds_total": max(
            0.0,
            (datetime.fromisoformat(updated.replace("Z", "+00:00"))
             - datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds(),
        ),
        "first_started_at_utc": started_at,
        "updated_at_utc": updated,
        "completed_at_utc": updated if complete else None,
        "complete": complete,
    }


def run_championship(
    config_source: str | Path | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    *,
    game_runner: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], dict[str, Any]] | None = None,
    implementation_commit: str | None = None,
    initialize_only: bool = False,
) -> dict[str, Any]:
    """Run/resume all six games; an interrupted in-progress game is rerun from its opening."""
    frozen = load_v6_2_config(config_source)
    root = Path(output_dir) if output_dir is not None else OUTPUT_DEFAULT
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    schedule_path = root / "schedule.json"
    records_path = root / "9x9_games.jsonl"
    progress_path = root / "progress.json"
    schedule = build_schedule(frozen)

    if config_path.exists() and canonical(json.loads(config_path.read_text(encoding="utf-8"))) != canonical(frozen):
        raise ValueError("Existing V6.2 results belong to a different frozen config")
    if schedule_path.exists() and canonical(json.loads(schedule_path.read_text(encoding="utf-8"))) != canonical(schedule):
        raise ValueError("Existing V6.2 schedule differs from its deterministic preregistration")
    write_json_atomic(config_path, frozen)
    write_json_atomic(schedule_path, schedule)

    commit = str(implementation_commit or _repository_commit())
    v6_config = load_v6_config()
    engine_config = v5_1_engine_settings(v6_config)
    runner = play_game if game_runner is None else game_runner
    records = load_game_records(records_path)
    validation = validate_v6_2_records(records, schedule, require_complete=False)
    existing_ids = {str(row["game_id"]) for row in records}

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress.get("experiment_id") != frozen["experiment_id"]
                or progress.get("config_sha256") != sha256(canonical(frozen))
                or progress.get("schedule_sha256") != sha256(canonical(schedule))):
            raise ValueError("V6.2 progress does not match the frozen config or schedule")
        if progress.get("implementation_commit") != commit:
            raise ValueError("V6.2 resume implementation commit differs from the pinned run")
        if not set(progress.get("completed_ids", ())).issubset(existing_ids):
            raise ValueError("V6.2 progress claims a completed game missing from durable JSONL")
        started_at = str(progress["first_started_at_utc"])
        prior_elapsed = float(progress.get("elapsed_seconds_total", 0.0))
    else:
        started_at, prior_elapsed = _utc_now(), 0.0
    started = time.perf_counter()
    games_this_run = 0
    progress = _progress(
        config=frozen, schedule=schedule, records=records, commit=commit,
        started_at=started_at, started=started, prior_elapsed=prior_elapsed,
        current_game_id=None, games_this_run=0,
    )
    write_json_atomic(progress_path, progress)

    if initialize_only:
        return {
            "complete": bool(progress.get("complete")),
            "games_completed_this_run": 0,
            "games": len(records),
            "groups": validation["groups"],
            "champions": validation["champions"],
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "implementation_commit": commit,
        }

    for spec in schedule:
        game_id = str(spec["game_id"])
        if game_id in existing_ids:
            continue
        write_json_atomic(
            progress_path,
            _progress(
                config=frozen, schedule=schedule, records=records, commit=commit,
                started_at=started_at, started=started, prior_elapsed=prior_elapsed,
                current_game_id=game_id, games_this_run=games_this_run,
            ),
        )
        record = runner(spec, engine_config, v6_config, commit)
        validate_v6_2_game_record(record, spec)
        with artifact_lock(root / ".v6_2_checkpoint.lock"):
            append_game(records_path, record, existing_ids)
            records.append(record)
            games_this_run += 1
            progress = _progress(
                config=frozen, schedule=schedule, records=records, commit=commit,
                started_at=started_at, started=started, prior_elapsed=prior_elapsed,
                current_game_id=None, games_this_run=games_this_run,
            )
            write_json_atomic(progress_path, progress)
        print(json.dumps({
            "event": "game_completed", "game_id": game_id,
            "winner_model": record.get("winner_model"),
            "moves": record.get("move_count"),
            "thinking_seconds": record.get("thinking_seconds_total"),
            "games_completed": len(records),
        }, ensure_ascii=False), flush=True)

    final = validate_v6_2_records(records, schedule, require_complete=True)
    progress = _progress(
        config=frozen, schedule=schedule, records=records, commit=commit,
        started_at=started_at, started=started, prior_elapsed=prior_elapsed,
        current_game_id=None, games_this_run=games_this_run,
    )
    write_json_atomic(progress_path, progress)
    return {
        "complete": True,
        "games_completed_this_run": games_this_run,
        "games": len(records),
        "groups": final["groups"],
        "champions": final["champions"],
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "implementation_commit": commit,
        "standings": final["standings"],
    }
