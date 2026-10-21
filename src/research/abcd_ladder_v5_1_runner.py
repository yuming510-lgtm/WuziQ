"""Resumable V5.1 tournament runner with exact V5 game reuse."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from .abcd_ladder_v5 import _project_root, run_game as run_v5_game
from .abcd_ladder_v5_analysis import validate_game_record as validate_v5_game
from .abcd_ladder_v5_1 import build_schedule, engine_config, load_config
from .abcd_ladder_v5_1_reuse import derive_state, prepare_v5_reuse
from .checkpoint import append_game, load_game_records, write_json_atomic

_V5_OUTPUT = "research/v5/results"
_V5_CONFIG = "configs/abcd_ladder_v5.json"
_RECORD_TOTALS = (
    "winner_color", "winner_model", "outcome", "termination", "timeout_count",
    "search_nodes_total", "simulations_total", "tree_depth_max", "thinking_seconds_total",
    "moves", "move_count", "implementation_commit",
)
_LEGACY_FIELDS = (
    "experiment_id", "group_id", "board_size", "pair", "matchup", "route_ids",
    "challenge_level", "challenger_model", "shared_baseline", "opening_index",
    "opening_seed", "color_swap", "black_model", "white_model", "pair_id", "game_seed",
    "opening", "budget_seconds_by_model", "challenger_budget_label", "opponent_budget_label",
    "source_type",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _wall_seconds(started_at: str, ended_at: str | None = None) -> float:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat((ended_at or _utc_now()).replace("Z", "+00:00"))
    return max(0.0, (end - start).total_seconds())


def repository_commit(root: str | Path | None = None) -> str:
    project_root = Path(root or _project_root())
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_v51_record(record: dict[str, Any], spec: dict[str, Any], engine: dict[str, Any]) -> None:
    if str(record.get("game_id")) != str(spec["game_id"]):
        raise ValueError("V5.1 runner returned a mismatched game ID")
    if str(record.get("group_id")) != str(spec["group_id"]):
        raise ValueError("V5.1 runner returned a mismatched group ID")
    for key, expected in spec.items():
        if record.get(key) != expected:
            raise ValueError(f"V5.1 game {spec['game_id']} disagrees with frozen field {key}")
    if not record.get("implementation_commit"):
        raise ValueError(f"V5.1 game {spec['game_id']} has no implementation commit")

    # Reuse the V5, fully tested turn-by-turn replay validator. The adapter
    # supplies only legacy route labels it requires; all game facts are taken
    # from this V5.1 record and its immutable scheduled spec.
    compatibility = deepcopy(spec)
    challenger = str(spec["pair"][0])
    opponent = str(spec["pair"][1])
    compatibility.update({
        "experiment_id": str(engine["experiment_id"]),
        "route_ids": [ref["route_id"] for ref in spec["route_refs"]],
        "challenge_level": 0,
        "challenger_model": challenger,
        "shared_baseline": False,
        "challenger_budget_label": str(spec["model_versions"][challenger]),
        "opponent_budget_label": str(spec["model_versions"][opponent]),
        "source_type": str(spec.get("source_type", "new")),
    })
    compat_record = deepcopy(record)
    for key in _LEGACY_FIELDS:
        compat_record[key] = compatibility.get(key)
    validate_v5_game(compat_record, engine, compatibility)


def _rekey_v5_record(source: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(spec)
    for key in _RECORD_TOTALS:
        if key not in source:
            raise ValueError(f"V5 source game {source.get('game_id')} lacks {key}")
        result[key] = deepcopy(source[key])
    result["implementation_commit"] = str(source["implementation_commit"])
    return result


def _prepare_initial(
    frozen: dict[str, Any], output: Path, reuse_v5: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule_path = output / "schedule.json"
    manifest_path = output / "reuse_manifest.json"
    if schedule_path.exists() or manifest_path.exists():
        if not schedule_path.is_file() or not manifest_path.is_file():
            raise ValueError("V5.1 output has only one of schedule.json or reuse_manifest.json")
        saved_schedule, saved_manifest = _load_json(schedule_path), _load_json(manifest_path)
        if reuse_v5:
            expected_schedule, expected_manifest = prepare_v5_reuse(
                build_schedule(frozen), _project_root() / _V5_OUTPUT, frozen
            )
        else:
            expected_schedule = build_schedule(frozen)
            expected_manifest = {
                "source_experiment_id": None, "source_commit": None, "source_config_sha256": None,
                "engine_sha256": frozen["engine_sha256"], "reused_groups": 0, "reused_games": 0,
                "groups": [], "rejections": [], "source_validation": None,
            }
        if _canonical(saved_schedule) != _canonical(expected_schedule):
            raise ValueError("Saved V5.1 schedule differs from its deterministic source")
        if _canonical(saved_manifest) != _canonical(expected_manifest):
            raise ValueError("Saved V5.1 reuse manifest differs from its deterministic source")
        return saved_schedule, saved_manifest

    schedule = build_schedule(frozen)
    if reuse_v5:
        schedule, manifest = prepare_v5_reuse(
            schedule, _project_root() / _V5_OUTPUT, frozen
        )
    else:
        manifest = {
            "source_experiment_id": None, "source_commit": None, "source_config_sha256": None,
            "engine_sha256": frozen["engine_sha256"], "reused_groups": 0, "reused_games": 0,
            "groups": [], "rejections": [], "source_validation": None,
        }
    write_json_atomic(schedule_path, schedule)
    write_json_atomic(manifest_path, manifest)
    return schedule, manifest


def run_experiment(
    config_source: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    reuse_v5: bool = True,
    max_games: int | None = None,
    max_groups: int | None = None,
    implementation_commit: str | None = None,
    game_runner: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run active groups in frozen schedule order; completed records are never rerun."""
    frozen = load_config(config_source)
    for value, name in ((max_games, "max_games"), (max_groups, "max_groups")):
        if value is not None and value < 0:
            raise ValueError(f"{name} cannot be negative")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    engine = engine_config(frozen)
    commit = str(implementation_commit or repository_commit())
    config_path = root / "config.json"
    schedule_path = root / "schedule.json"
    manifest_path = root / "reuse_manifest.json"
    records_path = root / "9x9_games.jsonl"
    progress_path = root / "progress.json"
    config_bytes = _canonical(frozen)
    if config_path.exists() and _canonical(_load_json(config_path)) != config_bytes:
        raise ValueError("Existing V5.1 output belongs to a different frozen config")
    if not config_path.exists():
        write_json_atomic(config_path, frozen)
    schedule, reuse_manifest = _prepare_initial(frozen, root, reuse_v5)
    schedule_bytes = _canonical(schedule)
    manifest_bytes = _canonical(reuse_manifest)
    schedule_sha, manifest_sha = _sha(schedule_bytes), _sha(manifest_bytes)
    frozen_records = build_schedule(frozen)
    if [(g["pair"], g["model_versions"], g["opening_seed"]) for g in schedule] != [
        (g["pair"], g["model_versions"], g["opening_seed"]) for g in frozen_records
    ] and int(reuse_manifest.get("reused_groups", 0)) == 0:
        raise ValueError("V5.1 schedule differs from the deterministic frozen schedule")
    schedule_ids = {str(game["game_id"]): game for group in schedule for game in group["games"]}
    if len(schedule_ids) != sum(len(group["games"]) for group in schedule):
        raise ValueError("V5.1 schedule contains duplicate game IDs")
    records = load_game_records(records_path)
    existing_ids = {str(row["game_id"]) for row in records}
    for record in records:
        expected = schedule_ids.get(str(record["game_id"]))
        if expected is None:
            raise ValueError(f"V5.1 results contain unscheduled game {record['game_id']}")
        _validate_v51_record(record, expected, engine)
    if game_runner is None:
        game_runner = run_v5_game

    previous_seconds = 0.0
    first_started = _utc_now()
    if progress_path.exists():
        prior = _load_json(progress_path)
        if (prior.get("experiment_id") != frozen["experiment_id"]
                or prior.get("config_sha256") != _sha(config_bytes)
                or prior.get("schedule_sha256") != schedule_sha
                or prior.get("reuse_manifest_sha256") != manifest_sha):
            raise ValueError("V5.1 progress does not match frozen config, schedule or reuse manifest")
        if prior.get("implementation_commit") != commit:
            raise ValueError("V5.1 resume implementation commit differs from prior run")
        if not set(prior.get("completed_ids", ())).issubset(existing_ids):
            raise ValueError("V5.1 progress claims games missing from durable JSONL")
        previous_seconds = float(prior.get("elapsed_seconds_total", 0.0))
        first_started = str(prior.get("first_started_at_utc", first_started))
    if not records_path.exists():
        records_path.touch()
        with records_path.open("rb") as handle:
            os.fsync(handle.fileno())

    started = time.perf_counter()
    games_this_run = groups_this_run = 0

    def persist_progress() -> dict[str, Any]:
        current_state = derive_state(records, frozen, schedule)
        payload = {
            "schema_version": 1,
            "experiment_id": frozen["experiment_id"],
            "config_sha256": _sha(config_bytes),
            "schedule_sha256": schedule_sha,
            "reuse_manifest_sha256": manifest_sha,
            "implementation_commit": commit,
            "engine_sha256": dict(frozen["engine_sha256"]),
            "maximum_routes": 66,
            "maximum_groups": 168,
            "maximum_games": 336,
            "completed_games": len(existing_ids),
            "completed_groups": current_state["completed_groups"],
            "completed_ids": sorted(existing_ids),
            "pending_group_id": current_state["pending_group_id"],
            "routes": current_state["routes"],
            "groups": current_state["groups"],
            "complete": current_state["complete"],
            "games_completed_this_run": games_this_run,
            "groups_completed_this_run": groups_this_run,
            "elapsed_seconds_total": previous_seconds + time.perf_counter() - started,
            "elapsed_wall_seconds_total": _wall_seconds(first_started),
            "first_started_at_utc": first_started,
            "last_started_at_utc": _utc_now(),
            "completed_at_utc": _utc_now() if current_state["complete"] else None,
            "updated_at_utc": _utc_now(),
        }
        write_json_atomic(progress_path, payload)
        return payload

    progress = persist_progress()
    v5_source_rows: dict[str, dict[str, Any]] = {}
    if int(reuse_manifest.get("reused_groups", 0)):
        source_path = _project_root() / _V5_OUTPUT / "9x9_games.jsonl"
        for source_row in load_game_records(source_path):
            v5_source_rows[str(source_row["game_id"])] = source_row
    by_group = {str(group["group_id"]): group for group in schedule}

    while not progress["complete"]:
        if max_groups is not None and groups_this_run >= max_groups:
            break
        group_id = progress["pending_group_id"]
        if group_id is None:
            break
        group = by_group[group_id]
        present = {str(row["game_id"]) for row in records if str(row["group_id"]) == group_id}
        for spec in group["games"]:
            identifier = str(spec["game_id"])
            if identifier in present:
                continue
            if max_games is not None and games_this_run >= max_games:
                break
            if spec.get("source_type") == "reused_v5":
                source_id = str(spec["source_game_id"])
                source = v5_source_rows.get(source_id)
                if source is None:
                    raise ValueError(f"Reusable V5 source game is missing: {source_id}")
                record = _rekey_v5_record(source, spec)
            else:
                record = game_runner(spec, engine, commit)
            _validate_v51_record(record, spec, engine)
            append_game(records_path, record, existing_ids)
            records.append(record)
            present.add(identifier)
            games_this_run += 1
            progress = persist_progress()
            print(json.dumps({
                "completed_games": len(existing_ids), "completed_groups": progress["completed_groups"],
                "last_game_id": identifier, "winner": record.get("winner_model"),
                "source_type": spec.get("source_type"), "pending_group_id": progress["pending_group_id"],
            }, ensure_ascii=False), flush=True)
        if max_games is not None and games_this_run >= max_games:
            break
        progress = persist_progress()
        if group_id in progress["groups"] and progress["groups"][group_id].get("status") == "played":
            groups_this_run += 1
            progress = persist_progress()
        elif progress["pending_group_id"] == group_id:
            break

    progress = persist_progress()
    return progress
