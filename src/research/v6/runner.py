"""Resumable full-game executor for the V6 independent-version challenge ladder."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from ...ai import SearchConfig, choose_move
from ...gomoku import Coordinate, GomokuBoard, Player, select_ai_move
from .config import ROOT, canonical, load_config, sha256, v5_1_engine_settings
from .engine import V6Agent
from .reuse import prepare_v5_1_reuse
from .schedule import build_schedule
from .state import derive_state
from .validation import validate_game_record
from ..abcd_ladder_v5 import derive_seed, settings_for_model
from ..checkpoint import (
    append_game,
    artifact_lock,
    load_game_records,
    write_json_atomic,
)
from ..mcts_d2 import MCTSAgent


OUTPUT_DEFAULT = ROOT / "research/v6/results"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _elapsed_wall(started: str) -> float:
    return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds())


def repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _legal(board: GomokuBoard, move: object) -> bool:
    return (
        isinstance(move, (tuple, list))
        and len(move) == 2
        and type(move[0]) is int
        and type(move[1]) is int
        and board.in_bounds(move[0], move[1])
        and board.get((move[0], move[1])) is None
    )


def _opening_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "ply": index,
            "player": opening["player"],
            "model": None,
            "row": int(opening["row"]),
            "col": int(opening["col"]),
            "elapsed_seconds": 0.0,
            "root_value": 0,
            "nodes": 0,
            "completed_depth": 0,
            "tree_depth": 0,
            "cutoffs": 0,
            "simulations": 0,
            "rollout_plies": 0,
            "terminal_simulations": 0,
            "truncated_simulations": 0,
            "root_candidates": 0,
            "root_min_visits": 0,
            "reused_nodes": 0,
            "timed_out": False,
            "opening": True,
            "proof_status": "not_applicable",
            "proof_kind": None,
            "proof_certificate": None,
            "stats": {},
        }
        for index, opening in enumerate(spec["opening"])
    ]


def _decision_record(
    result: Any,
    *,
    ply: int,
    player: Player,
    model: str,
    coord: Coordinate | None,
    timed_out: bool,
) -> dict[str, Any]:
    stats = dict(getattr(result, "stats", {}) or {})
    move: dict[str, Any] = {
        "ply": ply,
        "player": player,
        "model": model,
        "elapsed_seconds": float(result.elapsed_seconds),
        "root_value": int(result.root_value),
        "nodes": int(result.nodes),
        "completed_depth": int(result.completed_depth),
        "tree_depth": int(stats.get("max_attack_depth", result.completed_depth)),
        "cutoffs": int(result.cutoffs),
        "simulations": int(result.simulations),
        "rollout_plies": int(getattr(result, "rollout_plies", 0)),
        "terminal_simulations": int(getattr(result, "terminal_simulations", 0)),
        "truncated_simulations": int(getattr(result, "truncated_simulations", 0)),
        "root_candidates": int(getattr(result, "root_candidates", 0)),
        "root_min_visits": int(getattr(result, "root_min_visits", 0)),
        "reused_nodes": int(getattr(result, "reused_nodes", 0)),
        "timed_out": bool(timed_out),
        "opening": False,
        "proof_status": str(getattr(result, "proof_status", "not_applicable")),
        "proof_kind": getattr(result, "proof_kind", None),
        "proof_certificate": deepcopy(getattr(result, "proof_certificate", None)),
        "stats": stats,
    }
    if coord is not None:
        move["row"], move["col"] = coord
    return move


def play_game(
    spec: dict[str, Any],
    engine_config: dict[str, Any],
    v6_config: dict[str, Any],
    implementation_commit: str,
) -> dict[str, Any]:
    """Play one legal game; one persistent D agent and one E agent per family."""
    board = GomokuBoard(int(spec["board_size"]))
    moves = _opening_records(spec)
    for opening in spec["opening"]:
        coord = int(opening["row"]), int(opening["col"])
        board.place_stone(str(opening["player"]), coord)
    d_agent = MCTSAgent()
    e_agents = {
        family: V6Agent(
            family,
            max_attack_depth=int(v6_config["tactical_limits"]["max_attack_depth"]),
            max_nodes=int(
                v6_config["tactical_limits"]["pns_max_nodes"]
                if family == "E4"
                else v6_config["tactical_limits"]["max_nodes"]
            ),
        )
        for family in ("E1", "E2", "E3", "E4", "E5")
        if family in spec["pair"]
    }
    current: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    for _ in range(board.size * board.size - len(board.history)):
        model = spec["black_model"] if current == "B" else spec["white_model"]
        budget = spec["budget_seconds_by_model"][model]
        seed = derive_seed(int(spec["game_seed"]), "move", len(board.history), model, current)
        if model in {"A", "B", "C", "D"}:
            settings = settings_for_model(engine_config, model, len(board.history), budget)
            result = d_agent.choose_move(board, current, settings, seed) if model == "D" else choose_move(
                board, current, model, settings, seed
            )
        else:
            if budget is None:
                raise ValueError(f"V6 search model {model} has no budget")
            result = e_agents[model].choose_move(
                board,
                current,
                time_limit_seconds=float(budget),
                seed=seed,
                max_depth=int(v6_config["tactical_limits"]["e5_max_depth"]),
            )
        elapsed = float(result.elapsed_seconds)
        timed_out = bool(result.timed_out or (budget is not None and elapsed > float(budget)))
        opponent = _other(current)
        opponent_model = spec["black_model"] if opponent == "B" else spec["white_model"]
        if not _legal(board, result.move):
            move = _decision_record(
                result, ply=len(moves), player=current, model=model, coord=None, timed_out=timed_out
            )
            move.update({"illegal": True, "attempted_move": str(result.move)})
            moves.append(move)
            winner_color, winner_model, termination = opponent, opponent_model, "illegal_move_forfeit"
            break
        coord = int(result.move[0]), int(result.move[1])
        move = _decision_record(
            result, ply=len(moves), player=current, model=model, coord=coord, timed_out=timed_out
        )
        moves.append(move)
        board.place_stone(current, coord)
        if board.check_victory(coord, current):
            winner_color, winner_model, termination = current, model, "five_in_row"
            break
        if board.is_full():
            termination = "full_board"
            break
        current = opponent

    return {
        **deepcopy(spec),
        "implementation_commit": implementation_commit,
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "draw" if winner_color is None else f"{winner_color.lower()}_win",
        "termination": termination,
        "timeout_count": sum(bool(move.get("timed_out")) for move in moves),
        "search_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
        "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
        "tree_depth_max": max((int(move.get("tree_depth", 0)) for move in moves), default=0),
        "search_depth_max": max((int(move.get("completed_depth", 0)) for move in moves), default=0),
        "cutoffs_total": sum(int(move.get("cutoffs", 0)) for move in moves),
        "thinking_seconds_total": sum(float(move.get("elapsed_seconds", 0)) for move in moves),
        "proof_count": sum(move.get("proof_status") == "proven" for move in moves),
        "proof_verified_count": sum(
            move.get("proof_status") == "proven" and bool(move.get("proof_certificate"))
            for move in moves
        ),
        "source_type": str(spec.get("source_type", "new")),
        "source_game_id": spec.get("source_game_id"),
    }


def _empty_reuse_manifest(config: dict[str, Any]) -> dict[str, Any]:
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


def _write_progress(
    path: Path,
    *,
    frozen: dict[str, Any],
    schedule: list[dict[str, Any]],
    reuse_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    route_specs: dict[str, dict[str, Any]],
    commit: str,
    first_started: str,
    started: float,
    prior_elapsed: float,
    games_this_run: int,
    groups_this_run: int,
    current_group: str | None,
    latest_game: str | None,
) -> dict[str, Any]:
    state = derive_state(records, schedule, route_specs)
    payload = {
        "schema_version": 1,
        "experiment_id": frozen["experiment_id"],
        "implementation_commit": commit,
        "config_sha256": sha256(canonical(frozen)),
        "schedule_sha256": sha256(canonical(schedule)),
        "reuse_manifest_sha256": sha256(canonical(reuse_manifest)),
        "completed_ids": sorted(str(row["game_id"]) for row in records),
        "completed_games": len(records),
        "completed_groups": state["completed_groups"],
        "inferred_groups": state["inferred_groups"],
        "pending_group_id": state["pending_group_id"],
        "current_group_id": current_group,
        "latest_game_id": latest_game,
        "routes": state["routes"],
        "groups": state["groups"],
        "complete": state["complete"],
        "games_completed_this_run": games_this_run,
        "groups_completed_this_run": groups_this_run,
        "elapsed_seconds_total": prior_elapsed + time.perf_counter() - started,
        "elapsed_wall_seconds_total": _elapsed_wall(first_started),
        "first_started_at_utc": first_started,
        "last_started_at_utc": _utc_now(),
        "completed_at_utc": _utc_now() if state["complete"] else None,
        "updated_at_utc": _utc_now(),
    }
    write_json_atomic(path, payload)
    return payload


def run_experiment(
    config_source: str | Path | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    *,
    reuse_v5_1: bool = True,
    max_groups: int | None = None,
    implementation_commit: str | None = None,
    game_runner: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or resume all active V6 routes, persisting each finished game."""
    frozen = load_config(config_source)
    if max_groups is not None and max_groups < 0:
        raise ValueError("max_groups cannot be negative")
    root = Path(output_dir) if output_dir is not None else OUTPUT_DEFAULT
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    schedule_path = root / "schedule.json"
    reuse_path = root / "reuse_manifest.json"
    records_path = root / "9x9_games.jsonl"
    progress_path = root / "progress.json"
    v6_config_bytes = canonical(frozen)
    if config_path.exists() and canonical(json.loads(config_path.read_text())) != v6_config_bytes:
        raise ValueError("Existing V6 output belongs to a different frozen config")
    if not config_path.exists():
        write_json_atomic(config_path, frozen)

    fresh_schedule, route_specs = build_schedule(frozen)
    if reuse_v5_1:
        expected_schedule, expected_manifest, reusable_records = prepare_v5_1_reuse(
            fresh_schedule, frozen
        )
    else:
        expected_schedule = fresh_schedule
        expected_manifest = _empty_reuse_manifest(frozen)
        reusable_records = []
    if schedule_path.exists() or reuse_path.exists():
        if not schedule_path.is_file() or not reuse_path.is_file():
            raise ValueError("V6 output has only one of schedule.json or reuse_manifest.json")
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        reuse_manifest = json.loads(reuse_path.read_text(encoding="utf-8"))
        if canonical(schedule) != canonical(expected_schedule):
            raise ValueError("Saved V6 schedule differs from its deterministic source")
        if canonical(reuse_manifest) != canonical(expected_manifest):
            raise ValueError("Saved V6 reuse manifest differs from its exact source audit")
    else:
        schedule, reuse_manifest = expected_schedule, expected_manifest
        write_json_atomic(schedule_path, schedule)
        write_json_atomic(reuse_path, reuse_manifest)

    engine_config = v5_1_engine_settings(frozen)
    records = load_game_records(records_path)
    by_game = {str(game["game_id"]): game for group in schedule for game in group["games"]}
    existing_ids = {str(row["game_id"]) for row in records}
    for record in records:
        spec = by_game.get(str(record["game_id"]))
        if spec is None:
            raise ValueError(f"V6 output has an unscheduled game: {record['game_id']}")
        validate_game_record(record, spec, verify_proofs=True)
    if not records and reusable_records:
        with artifact_lock(root / ".v6_checkpoint.lock"):
            for record in reusable_records:
                spec = by_game[str(record["game_id"])]
                validate_game_record(record, spec, verify_proofs=False)
                append_game(records_path, record, existing_ids)
                records.append(record)
    elif records and reusable_records:
        missing = {str(row["game_id"]) for row in reusable_records} - existing_ids
        if missing:
            raise ValueError("V6 output is missing one or more frozen V5.1 reuse records")
    if game_runner is None:
        game_runner = play_game

    commit = str(implementation_commit or repository_commit())
    first_started = _utc_now()
    prior_elapsed = 0.0
    progress: dict[str, Any] | None = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected_sha = sha256(canonical(frozen))
        if (progress.get("experiment_id") != frozen["experiment_id"]
                or progress.get("config_sha256") != expected_sha
                or progress.get("schedule_sha256") != sha256(canonical(schedule))
                or progress.get("reuse_manifest_sha256") != sha256(canonical(reuse_manifest))):
            raise ValueError("V6 progress does not match frozen configuration/schedule/reuse manifest")
        if progress.get("implementation_commit") != commit:
            raise ValueError("V6 resume implementation commit differs from the pinned run")
        if not set(progress.get("completed_ids", ())).issubset(existing_ids):
            raise ValueError("V6 progress claims completed games missing from durable JSONL")
        first_started = str(progress.get("first_started_at_utc", first_started))
        prior_elapsed = float(progress.get("elapsed_seconds_total", 0.0))

    started = time.perf_counter()
    current_state = derive_state(records, schedule, route_specs)
    progress = _write_progress(
        progress_path, frozen=frozen, schedule=schedule, reuse_manifest=reuse_manifest,
        records=records, route_specs=route_specs, commit=commit, first_started=first_started,
        started=started, prior_elapsed=prior_elapsed, games_this_run=0, groups_this_run=0,
        current_group=None, latest_game=None,
    )
    groups_this_run = 0
    games_this_run = 0
    by_group = {str(group["group_id"]): group for group in schedule}
    for group in schedule:
        if max_groups is not None and groups_this_run >= max_groups:
            break
        group_id = str(group["group_id"])
        current_state = derive_state(records, schedule, route_specs)
        group_state = current_state["groups"].get(group_id, {})
        partial_pair = 0 < int(group_state.get("games", 0)) < 2
        if not partial_pair and group_state.get("status") in {
            "played", "played_after_route_stop", "inferred_dominated",
            "inferred_dominated_partial_observation", "skipped_no_active_route",
        }:
            continue
        specs = group["games"]
        existing_group_ids = {
            str(row["game_id"]) for row in records if str(row["group_id"]) == group_id
        }
        active_ids = {
            str(game["game_id"]) for game in specs if str(game["game_id"]) not in existing_group_ids
        }
        if not active_ids:
            continue
        # Inference is allowed only when every active route is dominated. A
        # reusable historical pair always wins over inference if it exists.
        if group_state.get("status") not in {"pending", "incomplete_pair"} and not partial_pair:
            continue
        if group.get("source_type") == "reused_v5_1":
            raise ValueError(f"V6 exact reuse group {group_id} was not imported")
        for spec in specs:
            game_id = str(spec["game_id"])
            if game_id in existing_ids:
                continue
            record = game_runner(spec, engine_config, frozen, commit)
            validate_game_record(record, spec, verify_proofs=True)
            with artifact_lock(root / ".v6_checkpoint.lock"):
                append_game(records_path, record, existing_ids)
                records.append(record)
                games_this_run += 1
                _write_progress(
                    progress_path, frozen=frozen, schedule=schedule, reuse_manifest=reuse_manifest,
                    records=records, route_specs=route_specs, commit=commit,
                    first_started=first_started, started=started, prior_elapsed=prior_elapsed,
                    games_this_run=games_this_run, groups_this_run=groups_this_run,
                    current_group=group_id, latest_game=game_id,
                )
            print(json.dumps({
                "event": "game_completed",
                "game_id": game_id,
                "group_id": group_id,
                "winner_model": record.get("winner_model"),
                "source_type": record.get("source_type"),
                "games_this_run": games_this_run,
                "durable_games": len(existing_ids),
            }, ensure_ascii=False), flush=True)
        final_state = derive_state(records, schedule, route_specs)
        group_status = final_state["groups"].get(group_id, {}).get("status")
        if group_status not in {
            "played", "played_after_route_stop", "inferred_dominated",
            "inferred_dominated_partial_observation", "skipped_no_active_route",
        }:
            raise ValueError(f"V6 paired group {group_id} did not reach a terminal state")
        groups_this_run += 1
        progress = _write_progress(
            progress_path, frozen=frozen, schedule=schedule, reuse_manifest=reuse_manifest,
            records=records, route_specs=route_specs, commit=commit,
            first_started=first_started, started=started, prior_elapsed=prior_elapsed,
            games_this_run=games_this_run, groups_this_run=groups_this_run,
            current_group=None, latest_game=None,
        )
        print(json.dumps({
            "event": "group_completed",
            "group_id": group_id,
            "status": group_status,
            "completed_groups": progress["completed_groups"],
            "inferred_groups": progress["inferred_groups"],
            "pending_group_id": progress["pending_group_id"],
            "groups_this_run": groups_this_run,
        }, ensure_ascii=False), flush=True)
    final_state = derive_state(records, schedule, route_specs)
    progress = _write_progress(
        progress_path, frozen=frozen, schedule=schedule, reuse_manifest=reuse_manifest,
        records=records, route_specs=route_specs, commit=commit,
        first_started=first_started, started=started, prior_elapsed=prior_elapsed,
        games_this_run=games_this_run, groups_this_run=groups_this_run,
        current_group=None, latest_game=None,
    )
    return progress
