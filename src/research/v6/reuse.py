"""Replay-validated reuse of exact complete empirical V5.1 pairs."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ..abcd_ladder_v5_1 import (
    build_schedule as build_v51_schedule,
    engine_config as v51_engine_config,
    load_config as load_v51_config,
)
from ..abcd_ladder_v5_1_runner import _validate_v51_record
from ..checkpoint import load_game_records
from .config import ROOT, canonical, sha256


def _group_key(group: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(group["pair"]),
        canonical(group["model_versions"]),
        canonical(group["budget_seconds_by_model"]),
        int(group["opening_seed"]),
        canonical(group["opening"]),
    )


def empty_reuse_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic manifest used when history reuse is disabled."""
    return {
        "source_experiment_id": None,
        "source_commit": None,
        "source_config_sha256": None,
        "source_schedule_sha256": None,
        "source_records_sha256": None,
        "source_record_count": 0,
        "source_full_groups": 0,
        "source_partial_games_excluded": [],
        "v6_reused_groups": 0,
        "v6_reused_games": 0,
        "engine_sha256": config["v5_1_engine_sha256"],
        "groups": [],
        "rejections": [],
        "selection_rule": "No history import requested.",
    }


def reconstruct_frozen_schedule(
    config: dict[str, Any],
    saved_schedule: list[dict[str, Any]],
    saved_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild and verify the schedule including its exact reuse decoration."""
    from .schedule import build_schedule

    base_schedule, _routes = build_schedule(config)
    if saved_manifest.get("source_experiment_id") is None:
        expected_schedule = base_schedule
        expected_manifest = empty_reuse_manifest(config)
    else:
        expected_schedule, expected_manifest, _records = prepare_v5_1_reuse(
            base_schedule, config
        )
    if canonical(saved_schedule) != canonical(expected_schedule):
        raise ValueError("Saved V6 schedule differs from its deterministic source and reuse manifest")
    if canonical(saved_manifest) != canonical(expected_manifest):
        raise ValueError("Saved V6 reuse manifest differs from its exact source audit")
    return expected_schedule, expected_manifest


def prepare_v5_1_reuse(
    schedule: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    source_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Return frozen schedule, exact source manifest, and normalized reused games."""
    source = Path(source_dir) if source_dir is not None else ROOT / str(config["reuse_source"]["results_dir"])
    source_config_path = source / "config.json"
    source_schedule_path = source / "schedule.json"
    source_games_path = source / "9x9_games.jsonl"
    source_progress_path = source / "progress.json"
    required = (source_config_path, source_schedule_path, source_games_path, source_progress_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("V5.1 reuse requires config, schedule, progress, and raw game records")
    source_config = load_v51_config(source_config_path)
    if source_config["experiment_id"] != config["reuse_source"]["experiment_id"]:
        raise ValueError("V5.1 reuse source experiment ID does not match V6 config")
    if source_config["engine_sha256"] != config["v5_1_engine_sha256"]:
        raise ValueError("V5.1 source engine hashes differ from V6 pins")
    source_progress = json.loads(source_progress_path.read_text(encoding="utf-8"))
    if not source_progress.get("complete"):
        raise ValueError("V5.1 source task does not have a completed route evaluation")
    old_schedule = json.loads(source_schedule_path.read_text(encoding="utf-8"))
    old_by_game = {
        str(game["game_id"]): game
        for group in old_schedule for game in group["games"]
    }
    old_by_group = {str(group["group_id"]): group for group in old_schedule}
    source_rows = load_game_records(source_games_path)
    engine = v51_engine_config(source_config)
    source_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        spec = old_by_game.get(str(row.get("game_id")))
        if spec is None:
            raise ValueError(f"V5.1 source has an unscheduled game: {row.get('game_id')}")
        _validate_v51_record(row, spec, engine)
        source_rows_by_group[str(row["group_id"])].append(row)

    v6_groups = deepcopy(schedule)
    candidate_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for group in v6_groups:
        if set(group["pair"]).issubset({"A", "B", "C", "D"}):
            candidate_groups[_group_key(group)].append(group)
    reused_records: list[dict[str, Any]] = []
    reused_manifest: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    used_v6_groups: set[str] = set()
    unpaired_source: list[str] = []
    for source_group_id, old_rows in sorted(source_rows_by_group.items()):
        if len(old_rows) != 2:
            unpaired_source.extend(str(row["game_id"]) for row in old_rows)
            rejections.append({"source_group_id": source_group_id, "reason": "incomplete_color_pair"})
            continue
        old_group = old_by_group[source_group_id]
        matches = candidate_groups.get(_group_key(old_group), [])
        if len(matches) != 1:
            rejections.append({"source_group_id": source_group_id, "reason": "no_unique_frozen_v6_match"})
            continue
        v6_group = matches[0]
        if v6_group["group_id"] in used_v6_groups:
            raise ValueError(f"Multiple V5.1 groups map to V6 group {v6_group['group_id']}")
        old_games = {bool(game["color_swap"]): game for game in old_group["games"]}
        old_records = {bool(row["color_swap"]): row for row in old_rows}
        new_games = {bool(game["color_swap"]): game for game in v6_group["games"]}
        if set(old_games) != {False, True} or set(old_records) != {False, True}:
            rejections.append({"source_group_id": source_group_id, "reason": "invalid_color_pair"})
            continue
        exact = True
        for swap in (False, True):
            old_game, new_game, old_row = old_games[swap], new_games[swap], old_records[swap]
            new_game["game_seed"] = int(old_game["game_seed"])
            required_pairs = (
                (old_game["black_model"], new_game["black_model"]),
                (old_game["white_model"], new_game["white_model"]),
                (old_game["budget_seconds_by_model"], new_game["budget_seconds_by_model"]),
                (old_game["opening_seed"], new_game["opening_seed"]),
                (old_game["opening"], new_game["opening"]),
                (old_game["color_swap"], new_game["color_swap"]),
                (old_game["game_seed"], old_row["game_seed"]),
                (old_game["game_seed"], new_game["game_seed"]),
            )
            if any(left != right for left, right in required_pairs):
                exact = False
                break
        if not exact:
            rejections.append({"source_group_id": source_group_id, "reason": "game_configuration_or_seed_mismatch"})
            continue
        used_v6_groups.add(str(v6_group["group_id"]))
        v6_group["source_type"] = "reused_v5_1"
        v6_group["reuse_source"] = {
            "experiment_id": source_config["experiment_id"],
            "source_group_id": source_group_id,
            "source_commit": source_progress.get("implementation_commit"),
        }
        v6_group["source_type"] = "reused_v5_1"
        for game in v6_group["games"]:
            game["source_type"] = "reused_v5_1"
            game["source_game_id"] = str(old_records[bool(game["color_swap"])]["game_id"])
            game["source_group_id"] = source_group_id
            game["source_experiment_id"] = source_config["experiment_id"]
            game["source_implementation_commit"] = str(
                old_records[bool(game["color_swap"])]["implementation_commit"]
            )
            # The V5.1 deterministic seed is part of exact match identity.
            game["game_seed"] = int(old_records[bool(game["color_swap"])]["game_seed"])
            source_row = old_records[bool(game["color_swap"])]
            record = deepcopy(game)
            for key in (
                "winner_color", "winner_model", "outcome", "termination", "timeout_count",
                "search_nodes_total", "simulations_total", "tree_depth_max",
                "thinking_seconds_total", "move_count", "moves",
            ):
                if key not in source_row:
                    raise ValueError(f"V5.1 source record {source_row['game_id']} lacks {key}")
                record[key] = deepcopy(source_row[key])
            record["implementation_commit"] = str(source_row["implementation_commit"])
            record["source_game_id"] = str(source_row["game_id"])
            record["source_group_id"] = source_group_id
            record["source_experiment_id"] = source_config["experiment_id"]
            record["source_implementation_commit"] = str(source_row["implementation_commit"])
            record["source_type"] = "reused_v5_1"
            record["timeout_count"] = int(source_row["timeout_count"])
            record["cutoffs_total"] = sum(int(move.get("cutoffs", 0)) for move in record["moves"])
            record["search_depth_max"] = max(
                (int(move.get("completed_depth", 0)) for move in record["moves"]),
                default=0,
            )
            record["proof_count"] = 0
            record["proof_verified_count"] = 0
            reused_records.append(record)
        reused_manifest.append({
            "v6_group_id": str(v6_group["group_id"]),
            "source_group_id": source_group_id,
            "source_experiment_id": source_config["experiment_id"],
            "game_ids": [
                {"v6_game_id": str(game["game_id"]), "source_game_id": str(game["source_game_id"]),
                 "color_swap": bool(game["color_swap"]), "game_seed": int(game["game_seed"])}
                for game in v6_group["games"]
            ],
        })
    manifest = {
        "source_experiment_id": source_config["experiment_id"],
        "source_commit": source_progress.get("implementation_commit"),
        "source_config_sha256": sha256(source_config_path.read_bytes()),
        "source_schedule_sha256": sha256(source_schedule_path.read_bytes()),
        "source_records_sha256": sha256(source_games_path.read_bytes()),
        "source_record_count": len(source_rows),
        "source_full_groups": sum(len(rows) == 2 for rows in source_rows_by_group.values()),
        "source_partial_games_excluded": unpaired_source,
        "v6_reused_groups": len(reused_manifest),
        "v6_reused_games": len(reused_records),
        "engine_sha256": config["v5_1_engine_sha256"],
        "groups": reused_manifest,
        "rejections": rejections,
        "selection_rule": "All complete groups were matched by configuration before outcomes were read; source outcomes never influence eligibility.",
    }
    if len(reused_manifest) != len({row["source_group_id"] for row in reused_manifest}):
        raise ValueError("V6 reuse manifest contains duplicate source group mappings")
    return v6_groups, manifest, reused_records
