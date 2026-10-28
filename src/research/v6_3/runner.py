"""Exactly-once V6.3 C versus E5 runner with conditional tiebreak recovery."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from ..checkpoint import append_game, artifact_lock, load_game_records, write_json_atomic
from ..v6.config import canonical, load_config as load_v6_config, sha256, v5_1_engine_settings
from ..v6.runner import play_game
from .config import ROOT, load_v6_3_config
from .schedule import build_schedule
from .validation import validate_v6_3_game_record, validate_v6_3_records


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _repository_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _progress_payload(
    *, config: dict[str, Any], schedule: list[dict[str, Any]], records: list[dict[str, Any]],
    commit: str, started_at: str, started: float, prior_elapsed: float,
    current_game_id: str | None, games_this_run: int,
) -> dict[str, Any]:
    validation = validate_v6_3_records(records, schedule, require_complete=False)
    now = _utc_now()
    elapsed_wall = max(
        0.0,
        (datetime.fromisoformat(now.replace("Z", "+00:00"))
         - datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds(),
    )
    complete = bool(validation["complete"])
    if validation["initial_games"] < 4:
        next_action = "play_initial_games"
    elif validation["tiebreak_status"] == "required":
        next_action = "play_conditional_tiebreak"
    else:
        next_action = "complete"
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source_v6_2_commit": config["source_v6_2_commit"],
        "implementation_commit": commit,
        "config_sha256": sha256(canonical(config)),
        "schedule_sha256": sha256(canonical(schedule)),
        "completed_ids": sorted(str(row["game_id"]) for row in records),
        "completed_games": len(records),
        "initial_games": validation["initial_games"],
        "conditional_games": validation["conditional_games"],
        "initial_points": validation["initial_points"],
        "tiebreak_status": validation["tiebreak_status"],
        "next_action": next_action,
        "current_game_id": current_game_id,
        "games_completed_this_run": games_this_run,
        "elapsed_seconds_total": prior_elapsed + time.perf_counter() - started,
        "elapsed_wall_seconds_total": elapsed_wall,
        "first_started_at_utc": started_at,
        "updated_at_utc": now,
        "completed_at_utc": now if complete else None,
        "complete": complete,
    }


def run_championship(
    config_source: str | Path | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    *,
    game_runner: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], dict[str, Any]] | None = None,
    implementation_commit: str | None = None,
    max_games_this_run: int | None = None,
    initialize_only: bool = False,
) -> dict[str, Any]:
    """Run four initial games, then run the registered two only if first-stage points tie."""
    config = load_v6_3_config(config_source)
    root = Path(output_dir) if output_dir is not None else ROOT / "research/v6_3/results"
    root.mkdir(parents=True, exist_ok=True)
    config_path, schedule_path = root / "config.json", root / "schedule.json"
    records_path, progress_path = root / "9x9_games.jsonl", root / "progress.json"
    schedule = build_schedule(config)
    if config_path.exists() and canonical(json.loads(config_path.read_text(encoding="utf-8"))) != canonical(config):
        raise ValueError("Existing V6.3 results use a different frozen config")
    if schedule_path.exists() and canonical(json.loads(schedule_path.read_text(encoding="utf-8"))) != canonical(schedule):
        raise ValueError("Existing V6.3 results use a different preregistered schedule")
    write_json_atomic(config_path, config)
    write_json_atomic(schedule_path, schedule)
    records_path.touch(exist_ok=True)

    commit = str(implementation_commit or _repository_commit())
    v6_config = load_v6_config()
    engine_config = v5_1_engine_settings(v6_config)
    runner = play_game if game_runner is None else game_runner
    records = load_game_records(records_path)
    validation = validate_v6_3_records(records, schedule, require_complete=False)
    existing_ids = {str(row["game_id"]) for row in records}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("experiment_id", config["experiment_id"]),
            ("config_sha256", sha256(canonical(config))),
            ("schedule_sha256", sha256(canonical(schedule))),
            ("implementation_commit", commit),
        ):
            if progress.get(key) != expected:
                raise ValueError(f"V6.3 progress does not match frozen {key}")
        if not set(progress.get("completed_ids", ())).issubset(existing_ids):
            raise ValueError("V6.3 progress claims a game missing from durable JSONL")
        started_at = str(progress["first_started_at_utc"])
        prior_elapsed = float(progress.get("elapsed_seconds_total", 0.0))
    else:
        started_at, prior_elapsed = _utc_now(), 0.0
    started = time.perf_counter()
    games_this_run = 0
    write_json_atomic(progress_path, _progress_payload(
        config=config, schedule=schedule, records=records, commit=commit,
        started_at=started_at, started=started, prior_elapsed=prior_elapsed,
        current_game_id=None, games_this_run=0,
    ))
    if initialize_only:
        return {
            "complete": validation["complete"], "games": len(records),
            "tiebreak_status": validation["tiebreak_status"],
            "games_completed_this_run": 0, "implementation_commit": commit,
        }

    if max_games_this_run is not None and max_games_this_run < 1:
        raise ValueError("max_games_this_run must be positive")
    stopped_early = False
    for spec in schedule:
        game_id = str(spec["game_id"])
        if game_id in existing_ids:
            continue
        current = validate_v6_3_records(records, schedule, require_complete=False)
        if spec["stage"] == "conditional_tiebreak" and current["tiebreak_status"] != "required":
            break
        if spec["stage"] == "initial" and current["initial_games"] >= 4:
            continue
        write_json_atomic(progress_path, _progress_payload(
            config=config, schedule=schedule, records=records, commit=commit,
            started_at=started_at, started=started, prior_elapsed=prior_elapsed,
            current_game_id=game_id, games_this_run=games_this_run,
        ))
        record = runner(spec, engine_config, v6_config, commit)
        validate_v6_3_game_record(record, spec)
        with artifact_lock(root / ".v6_3_checkpoint.lock"):
            append_game(records_path, record, existing_ids)
            records.append(record)
            existing_ids.add(game_id)
            games_this_run += 1
            validation = validate_v6_3_records(records, schedule, require_complete=False)
            write_json_atomic(progress_path, _progress_payload(
                config=config, schedule=schedule, records=records, commit=commit,
                started_at=started_at, started=started, prior_elapsed=prior_elapsed,
                current_game_id=None, games_this_run=games_this_run,
            ))
        print(json.dumps({
            "event": "game_completed", "game_id": game_id,
            "winner_model": record.get("winner_model"), "moves": record.get("move_count"),
            "thinking_seconds": record.get("thinking_seconds_total"),
            "games_completed": len(records), "tiebreak_status": validation["tiebreak_status"],
        }, ensure_ascii=False), flush=True)
        if max_games_this_run is not None and games_this_run >= max_games_this_run:
            stopped_early = True
            break

    final = validate_v6_3_records(records, schedule, require_complete=not stopped_early)
    payload = _progress_payload(
        config=config, schedule=schedule, records=records, commit=commit,
        started_at=started_at, started=started, prior_elapsed=prior_elapsed,
        current_game_id=None, games_this_run=games_this_run,
    )
    write_json_atomic(progress_path, payload)
    return {
        "complete": final["complete"],
        "games_completed_this_run": games_this_run,
        "games": final["games"],
        "tiebreak_status": final["tiebreak_status"],
        "standings": final["standings"],
        "leaders": final["leaders"],
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "implementation_commit": commit,
    }

