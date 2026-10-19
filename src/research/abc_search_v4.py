"""Resumable seven-budget ABC search measurements for fixed 9x9 positions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from ..ai import SearchConfig, choose_move
from ..gomoku import GomokuBoard, Player
from .checkpoint import write_json_atomic


EXPECTED_BUDGETS_MS = (50, 200, 1_000, 5_000, 10_000, 20_000, 50_000)
EXPECTED_MODELS = ("A", "B", "C")
EXPECTED_CATEGORIES = {"opening", "midgame", "late_game", "attack", "defense"}
_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _expected_budget_values(config: dict[str, Any]) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in config.get("budgets_ms", ()))
    if budgets != EXPECTED_BUDGETS_MS:
        raise ValueError(f"V4 budgets must be exactly {list(EXPECTED_BUDGETS_MS)}")
    return budgets


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate every frozen position and the required seven-budget matrix."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported V4 benchmark config schema")
    if int(config.get("board_size", 0)) != 9:
        raise ValueError("V4 benchmark requires 9x9 positions")
    if tuple(config.get("models", ())) != EXPECTED_MODELS:
        raise ValueError(f"V4 models must be exactly {list(EXPECTED_MODELS)}")
    _expected_budget_values(config)
    if int(config.get("base_seed", -1)) < 0:
        raise ValueError("base_seed must be a nonnegative integer")

    positions = config.get("positions")
    if not isinstance(positions, list) or len(positions) < 12:
        raise ValueError("V4 benchmark requires at least 12 frozen positions")
    ids: set[str] = set()
    categories: set[str] = set()
    for position in positions:
        position_id = str(position.get("position_id", ""))
        if not position_id or position_id in ids:
            raise ValueError("Position IDs must be nonempty and unique")
        ids.add(position_id)
        category = str(position.get("category", ""))
        if category not in EXPECTED_CATEGORIES:
            raise ValueError(f"Unknown position category for {position_id}: {category}")
        categories.add(category)
        moves = position.get("moves")
        if not isinstance(moves, list) or not moves or len(moves) >= 81:
            raise ValueError(f"Position {position_id} needs between 1 and 80 moves")
        board = GomokuBoard(9)
        for index, move in enumerate(moves):
            expected_player = "B" if index % 2 == 0 else "W"
            player = str(move.get("player", ""))
            if player != expected_player:
                raise ValueError(f"Position {position_id} has nonalternating move order")
            coordinate = (int(move["row"]), int(move["col"]))
            board.place_stone(player, coordinate)
            if board.check_victory(coordinate, player):
                raise ValueError(f"Position {position_id} contains a terminal win")
        expected_moves = position.get("expected_best_moves")
        if expected_moves is not None:
            if not isinstance(expected_moves, list) or not expected_moves:
                raise ValueError(f"Position {position_id} has an empty tactical expectation")
            occupied = {
                (int(move["row"]), int(move["col"])) for move in moves
            }
            normalized_expected: set[tuple[int, int]] = set()
            for coordinate in expected_moves:
                if (
                    not isinstance(coordinate, list)
                    or len(coordinate) != 2
                    or any(int(value) < 0 or int(value) >= 9 for value in coordinate)
                ):
                    raise ValueError(f"Position {position_id} has an invalid tactical coordinate")
                cell = tuple(map(int, coordinate))
                if cell in occupied or cell in normalized_expected:
                    raise ValueError(f"Position {position_id} has an occupied/duplicate tactical move")
                normalized_expected.add(cell)
        expected_to_move = "B" if len(moves) % 2 == 0 else "W"
        if position.get("to_move") != expected_to_move:
            raise ValueError(f"Position {position_id} has incorrect to_move value")
    missing_categories = EXPECTED_CATEGORIES - categories
    if missing_categories:
        raise ValueError(f"V4 positions are missing categories: {sorted(missing_categories)}")

    search = config.get("search")
    if not isinstance(search, dict) or int(search.get("candidate_radius", -1)) < 0:
        raise ValueError("V4 search config needs a nonnegative candidate_radius")
    if int(search.get("max_depth", 0)) < 1:
        raise ValueError("V4 search max_depth must be positive")
    return config


def make_run_id(position_id: str, model: str, budget_ms: int) -> str:
    """Return a stable, readable unique key for one matrix cell."""

    normalized_model = str(model).upper()
    if normalized_model not in EXPECTED_MODELS:
        raise ValueError(f"Unknown V4 model: {model}")
    if int(budget_ms) not in EXPECTED_BUDGETS_MS:
        raise ValueError(f"Unknown V4 budget: {budget_ms}")
    return f"{position_id}__{normalized_model}__{int(budget_ms):05d}ms"


def expected_run_ids(config: dict[str, Any]) -> list[str]:
    """Build all scheduled run IDs in stable position/model/budget order."""

    budgets = _expected_budget_values(config)
    return [
        make_run_id(str(position["position_id"]), model, budget)
        for position in config["positions"]
        for model in EXPECTED_MODELS
        for budget in budgets
    ]


def derive_seed(base_seed: int, position_id: str, model: str) -> int:
    """Derive a deterministic seed shared by all budgets for one model/position."""

    material = f"{int(base_seed)}|{position_id}|{model.upper()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def load_run_records(
    path: str | Path, expected_ids: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Read unique JSONL records and truncate only a corrupt final tail."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    raw = file_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed = None if expected_ids is None else set(expected_ids)
    valid_offset = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_offset += len(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                with file_path.open("r+b") as handle:
                    handle.truncate(valid_offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f"Invalid JSONL record before final line in {file_path}") from exc
        run_id = str(record.get("run_id", "")) if isinstance(record, dict) else ""
        if not run_id:
            raise ValueError(f"Run record in {file_path} is missing run_id")
        if run_id in seen:
            raise ValueError(f"Duplicate run_id in {file_path}: {run_id}")
        if allowed is not None and run_id not in allowed:
            raise ValueError(f"Run ID outside the configured matrix: {run_id}")
        seen.add(run_id)
        records.append(record)
        valid_offset += len(line)
    return records


def append_run_record(
    path: str | Path, record: dict[str, Any], completed_ids: set[str]
) -> None:
    """Append one unique result and fsync it before returning."""

    run_id = str(record.get("run_id", ""))
    if not run_id:
        raise ValueError("Run record is missing run_id")
    if run_id in completed_ids:
        raise ValueError(f"Duplicate run_id: {run_id}")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    with file_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    completed_ids.add(run_id)


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _restore_position(position: dict[str, Any]) -> tuple[GomokuBoard, Player]:
    board = GomokuBoard(9)
    for move in position["moves"]:
        board.place_stone(str(move["player"]), (int(move["row"]), int(move["col"])))
    return board, str(position["to_move"])


def _run_one(
    config: dict[str, Any],
    config_hash: str,
    code_commit: str,
    position: dict[str, Any],
    model: str,
    budget_ms: int,
) -> dict[str, Any]:
    board, player = _restore_position(position)
    original_history = list(board.history)
    seed = derive_seed(int(config["base_seed"]), position["position_id"], model)
    depth_limit = board.size * board.size - len(board.history)
    settings = SearchConfig(
        **{
            **config["search"],
            "max_depth": depth_limit,
            "time_limit_seconds": budget_ms / 1_000.0,
        }
    )
    started_at = _utc_now()
    result = choose_move(board, player, model, settings, seed=seed)
    elapsed = result.elapsed_seconds
    if board.history != original_history:
        raise RuntimeError(f"{model} modified frozen board {position['position_id']}")
    if board.get(result.move) is not None or not board.in_bounds(*result.move):
        raise RuntimeError(f"{model} returned an illegal move on {position['position_id']}")
    measured_nodes = result.nodes if model in {"B", "C"} else 0
    depth_rows = [
        {
            "depth": row.depth,
            "root_value": row.root_value,
            "move": list(row.move),
            "nodes": row.nodes,
            "cutoffs": row.cutoffs,
            "elapsed_seconds": row.elapsed_seconds,
        }
        for row in result.depth_iterations
    ]
    completed_depth_nodes = sum(row["nodes"] for row in depth_rows)
    completed_depth_cutoffs = sum(row["cutoffs"] for row in depth_rows)
    measured_cutoffs = int(result.cutoffs if model in {"B", "C"} else 0)
    return {
        "run_id": make_run_id(position["position_id"], model, budget_ms),
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "code_commit": code_commit,
        "position_id": position["position_id"],
        "category": position["category"],
        "plies": len(board.history),
        "board_size": 9,
        "to_move": player,
        "model": model,
        "budget_ms": int(budget_ms),
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "move": list(result.move),
        "root_value": int(result.root_value),
        "completed_depth": int(result.completed_depth),
        "max_depth_limit": depth_limit,
        "depth_iterations": depth_rows,
        "nodes": int(measured_nodes),
        "completed_depth_nodes": completed_depth_nodes,
        "partial_iteration_nodes": max(0, measured_nodes - completed_depth_nodes),
        "cutoffs": measured_cutoffs,
        "completed_depth_cutoffs": completed_depth_cutoffs,
        "partial_iteration_cutoffs": max(0, measured_cutoffs - completed_depth_cutoffs),
        "nodes_per_second": measured_nodes / elapsed if elapsed > 0 else 0.0,
        "root_candidates": int(result.root_candidates),
        "elapsed_seconds": float(elapsed),
        "timed_out": bool(result.timed_out or elapsed > budget_ms / 1_000.0),
        "tactical_expected_moves": position.get("expected_best_moves"),
        "tactical_move_matches_expected": (
            list(result.move) in position["expected_best_moves"]
            if position.get("expected_best_moves") is not None
            else None
        ),
    }


def run_matrix(
    config_path: str | Path,
    results_dir: str | Path,
    max_new_runs: int | None = None,
) -> dict[str, Any]:
    """Run uncompleted matrix rows sequentially, persisting every result immediately."""

    config = load_config(config_path)
    if max_new_runs is not None and max_new_runs < 0:
        raise ValueError("max_new_runs cannot be negative")
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / "config.json"
    if config_copy.exists() and json.loads(config_copy.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output belongs to a different V4 config")
    if not config_copy.exists():
        write_json_atomic(config_copy, config)
    config_hash = _canonical_hash(config)
    all_ids = expected_run_ids(config)
    records_path = output / "search_runs.jsonl"
    records = load_run_records(records_path, all_ids)
    from .abc_search_v4_analysis import validate_runs

    validate_runs(records, config, require_complete=False)
    completed = {str(record["run_id"]) for record in records}
    planned = len(all_ids)
    skipped = len(completed)
    record_code_commits = {str(record["code_commit"]) for record in records}
    if len(record_code_commits) > 1:
        raise ValueError("Existing runs contain mixed code commits")
    current_code_commit = _code_commit()
    stored_code_commit = next(iter(record_code_commits), current_code_commit)
    if completed and len(completed) < planned and stored_code_commit != current_code_commit:
        raise ValueError(
            "Existing partial results use a different code commit; refusing to mix revisions"
        )

    progress_path = output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema_version") != 1
            or progress.get("experiment_id") != config["experiment_id"]
            or progress.get("config_sha256") != config_hash
            or int(progress.get("planned_runs", -1)) != planned
        ):
            raise ValueError("Existing progress metadata does not match the configured matrix")
        progress_ids = progress.get("completed_ids")
        if not isinstance(progress_ids, list) or any(not isinstance(item, str) for item in progress_ids):
            raise ValueError("Existing progress completed_ids must be a string list")
        if len(progress_ids) != len(set(progress_ids)):
            raise ValueError("Existing progress contains duplicate run IDs")
        if not set(progress_ids).issubset(completed):
            raise ValueError("Progress references runs missing from the durable JSONL")
        if int(progress.get("completed_runs", -1)) != len(progress_ids):
            raise ValueError("Progress completed run count does not match its IDs")
        if records and progress.get("code_commit") != stored_code_commit:
            raise ValueError("Progress code commit does not match durable run records")

    next_progress = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "code_commit": stored_code_commit,
        "planned_runs": planned,
        "completed_runs": len(completed),
        "completed_ids": sorted(completed),
        "updated_at_utc": _utc_now(),
    }
    write_json_atomic(output / "progress.json", next_progress)

    started = time.perf_counter()
    new_runs = 0
    code_commit = stored_code_commit
    budget_by_id: dict[str, tuple[dict[str, Any], str, int]] = {}
    for position in config["positions"]:
        for model in EXPECTED_MODELS:
            for budget_ms in EXPECTED_BUDGETS_MS:
                budget_by_id[make_run_id(position["position_id"], model, budget_ms)] = (
                    position,
                    model,
                    budget_ms,
                )

    for run_id in all_ids:
        if run_id in completed:
            continue
        if max_new_runs is not None and new_runs >= max_new_runs:
            break
        position, model, budget_ms = budget_by_id[run_id]
        record = _run_one(config, config_hash, code_commit, position, model, budget_ms)
        append_run_record(records_path, record, completed)
        new_runs += 1
        write_json_atomic(
            output / "progress.json",
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "config_sha256": config_hash,
                "code_commit": code_commit,
                "planned_runs": planned,
                "completed_runs": len(completed),
                "completed_ids": sorted(completed),
                "last_run_id": run_id,
                "updated_at_utc": _utc_now(),
            },
        )

    return {
        "experiment_id": config["experiment_id"],
        "planned_runs": planned,
        "new_runs": new_runs,
        "skipped_runs": skipped,
        "completed_runs": len(completed),
        "remaining_runs": planned - len(completed),
        "elapsed_seconds": time.perf_counter() - started,
        "records_path": str(records_path),
    }
