"""Resumable, paired ABCD 9x9 final match for research V4B."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

from ..ai import SearchConfig, SearchResult, choose_move
from ..gomoku import GomokuBoard, Player
from .checkpoint import append_game, load_game_records, write_json_atomic
from .mcts_d2 import MCTSAgent

_MATCHUPS = (("A", "B"), ("A", "C"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D"))
_MODELS = {"A", "B", "C", "D"}
_V3_MODEL_MAP = {"A": "A", "D3": "D"}
_EXPECTED_V3_MCTS_SHA256 = "9e5b552a6172f47db63bafe6c765b38be7aa519b7dfd11cc9c4a7de1af7d4991"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_historical_games(path: str | Path) -> list[dict[str, Any]]:
    """Read legacy JSONL without invoking checkpoint repair or modifying source data."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(source.read_bytes().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid historical JSONL at {source}:{line_number}: {exc}") from exc
        if not isinstance(row, dict) or not row.get("game_id"):
            raise ValueError(f"Invalid historical JSONL record at {source}:{line_number}: missing game_id")
        identifier = str(row["game_id"])
        if identifier in seen:
            raise ValueError(f"Duplicate historical game_id {identifier} in {source}")
        seen.add(identifier)
        rows.append(row)
    return rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def derive_seed(base_seed: int, *parts: object) -> int:
    """Stable unsigned 64-bit seed independent of Python hash randomization."""

    payload = json.dumps([int(base_seed), *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def load_config(config_source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and strictly validate the frozen V4B experiment specification."""

    if isinstance(config_source, dict):
        config = deepcopy(config_source)
    else:
        config = json.loads(Path(config_source).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported V4B config schema")
    if int(config.get("board_size", 0)) != 9 or tuple(config.get("models", ())) != ("A", "B", "C", "D"):
        raise ValueError("V4B is frozen to models A/B/C/D on a 9x9 board")
    if tuple(tuple(pair) for pair in config.get("matchups", ())) != _MATCHUPS:
        raise ValueError("V4B must include the six registered ABCD matchups in order")
    if int(config.get("games_per_matchup", 0)) != 6 or int(config.get("opening_plies", 0)) != 4:
        raise ValueError("V4B requires six games per matchup and four opening plies")
    opening_seeds = tuple(int(seed) for seed in config.get("opening_seeds", ()))
    if opening_seeds != (731091, 731092, 731093):
        raise ValueError("V4B must use the three pre-registered V3 opening seeds")
    catalog = config.get("opening_catalog", {})
    if set(catalog) != {str(seed) for seed in opening_seeds}:
        raise ValueError("V4B opening catalog does not match the registered seed set")
    for seed in opening_seeds:
        moves = catalog[str(seed)]
        if len(moves) != 4:
            raise ValueError(f"Opening {seed} must contain four moves")
        board = GomokuBoard(9)
        for index, move in enumerate(moves):
            if len(move) != 3 or str(move[0]) != ("B" if index % 2 == 0 else "W"):
                raise ValueError(f"Opening {seed} must alternate B-W-B-W")
            coord = int(move[1]), int(move[2])
            board.place_stone(str(move[0]), coord)
            if board.check_victory(coord, str(move[0])):
                raise ValueError(f"Opening {seed} is already terminal")
    if config.get("budgets_seconds_by_model") != {"A": 0.05, "B": 0.2, "C": 0.2, "D": 20.0}:
        raise ValueError("V4B budgets must remain A=50ms, B/C=200ms, D=20s")
    for key in ("ab_search", "d3_search"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"V4B is missing {key}")
    if int(config["ab_search"].get("max_depth", 0)) < 81:
        raise ValueError("B/C must allow iterative deepening through all remaining board plies")
    if float(config["d3_search"].get("time_limit_seconds", 0.0)) != 20.0:
        raise ValueError("D3 must use the V3 20-second per-move budget")
    if tuple(tuple(pair) for pair in config.get("matchups", ())) != _MATCHUPS:
        raise ValueError("Configured matchup order changed")
    return config


def _opening(config: dict[str, Any], seed: int) -> list[tuple[Player, tuple[int, int]]]:
    try:
        return [(str(player), (int(row), int(col))) for player, row, col in config["opening_catalog"][str(seed)]]  # type: ignore[list-item]
    except KeyError as exc:
        raise ValueError(f"Unregistered V4B opening seed: {seed}") from exc


def _game_id(pair: tuple[str, str], opening_seed: int, color_swap: bool) -> str:
    first, second = sorted(pair)
    return f"v4b-9x9-{first}-{second}-o{opening_seed}-s{int(color_swap)}"


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand all matchups into three paired openings and two color assignments."""

    frozen = load_config(config)
    schedule: list[dict[str, Any]] = []
    for pair in _MATCHUPS:
        canonical_pair = tuple(sorted(pair))
        for opening_index, opening_seed in enumerate(frozen["opening_seeds"]):
            for color_swap in (False, True):
                black_model, white_model = canonical_pair
                if color_swap:
                    black_model, white_model = white_model, black_model
                identifier = _game_id(canonical_pair, int(opening_seed), color_swap)
                schedule.append(
                    {
                        "experiment_id": frozen["experiment_id"],
                        "game_id": identifier,
                        "board_size": 9,
                        "matchup": canonical_pair,
                        "opening_index": opening_index,
                        "opening_seed": int(opening_seed),
                        "color_swap": color_swap,
                        "black_model": black_model,
                        "white_model": white_model,
                        "pair_id": f"v4b-9x9-{'-'.join(canonical_pair)}-o{opening_seed}",
                        "game_seed": derive_seed(int(frozen["base_seed"]), "v4b-game", identifier),
                        "opening": [
                            {"player": player, "row": coord[0], "col": coord[1]}
                            for player, coord in _opening(frozen, int(opening_seed))
                        ],
                        "budget_seconds_by_model": {
                            model: float(frozen["budgets_seconds_by_model"][model]) for model in canonical_pair
                        },
                    }
                )
    ids = [str(item["game_id"]) for item in schedule]
    if len(schedule) != 36 or len(ids) != len(set(ids)):
        raise ValueError("V4B schedule must contain 36 unique games")
    return schedule


def settings_for_model(config: dict[str, Any], model: str, occupied: int) -> SearchConfig:
    """Return V4 iterative-deepening settings or the exact V3 D3 settings."""

    model = str(model).upper()
    if model not in _MODELS:
        raise ValueError(f"Unknown V4B model: {model}")
    if occupied < 0 or occupied >= 81:
        raise ValueError("occupied must leave at least one legal cell")
    if model == "D":
        return SearchConfig(**config["d3_search"])
    search = dict(config["ab_search"])
    search["time_limit_seconds"] = float(config["budgets_seconds_by_model"][model])
    search["max_depth"] = 81 - occupied
    return SearchConfig(**search)


def _other(player: Player) -> Player:
    return "W" if player == "B" else "B"


def _legal_coordinate(board: GomokuBoard, move: object) -> bool:
    return (
        isinstance(move, (tuple, list))
        and len(move) == 2
        and type(move[0]) is int
        and type(move[1]) is int
        and board.in_bounds(move[0], move[1])
        and board.get((move[0], move[1])) is None
    )


def _opening_log(player: Player, coord: tuple[int, int], ply: int) -> dict[str, Any]:
    return {
        "ply": ply,
        "player": player,
        "model": None,
        "row": coord[0],
        "col": coord[1],
        "elapsed_seconds": 0.0,
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
    }


def _move_metrics(result: SearchResult, ply: int, player: Player, model: str, timed_out: bool) -> dict[str, Any]:
    return {
        "ply": ply,
        "player": player,
        "model": model,
        "elapsed_seconds": float(result.elapsed_seconds),
        "nodes": int(result.nodes),
        "completed_depth": int(result.completed_depth),
        "tree_depth": int(result.completed_depth),
        "cutoffs": int(result.cutoffs),
        "simulations": int(result.simulations),
        "rollout_plies": int(result.rollout_plies),
        "terminal_simulations": int(result.terminal_simulations),
        "truncated_simulations": int(result.truncated_simulations),
        "root_candidates": int(result.root_candidates),
        "root_min_visits": int(result.root_min_visits),
        "reused_nodes": int(result.reused_nodes),
        "timed_out": bool(timed_out),
        "opening": False,
    }


def run_game(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Play one complete game with a persistent D3 agent and move-level telemetry."""

    frozen = load_config(config)
    board = GomokuBoard(9)
    opening = _opening(frozen, int(spec["opening_seed"]))
    moves = []
    for ply, (player, coord) in enumerate(opening):
        board.place_stone(player, coord)
        moves.append(_opening_log(player, coord, ply))

    agents = {"D": MCTSAgent()}
    current_player: Player = "B"
    winner_color: Player | None = None
    winner_model: str | None = None
    termination = "full_board"
    for _ in range(81 - len(board.history)):
        model = spec["black_model"] if current_player == "B" else spec["white_model"]
        settings = settings_for_model(frozen, model, occupied=len(board.history))
        seed = derive_seed(int(spec["game_seed"]), "move", len(board.history), model, current_player)
        if model == "D":
            result = agents[model].choose_move(board, current_player, settings, seed)
        else:
            result = choose_move(board, current_player, model, settings, seed)
        elapsed = float(result.elapsed_seconds)
        budget = float(frozen["budgets_seconds_by_model"][model])
        timed_out = bool(result.timed_out or elapsed > budget)
        opponent_color = _other(current_player)
        opponent_model = spec["black_model"] if opponent_color == "B" else spec["white_model"]
        if not _legal_coordinate(board, result.move):
            bad_move = _move_metrics(result, len(moves), current_player, model, timed_out)
            bad_move.update(
                {
                    "attempted_move": list(result.move) if isinstance(result.move, (tuple, list)) else str(result.move),
                    "illegal": True,
                }
            )
            moves.append(bad_move)
            winner_color, winner_model, termination = opponent_color, opponent_model, "illegal_move_forfeit"
            break

        coord = int(result.move[0]), int(result.move[1])
        board.place_stone(current_player, coord)
        move_record = _move_metrics(result, len(moves), current_player, model, timed_out)
        move_record.update({"row": coord[0], "col": coord[1]})
        moves.append(move_record)
        if board.check_victory(coord, current_player):
            winner_color, winner_model, termination = current_player, model, "five_in_row"
            break
        if board.is_full():
            termination = "full_board"
            break
        current_player = opponent_color

    record = {
        **spec,
        "matchup": list(spec["matchup"]),
        "source_type": "new",
        "v4b_base_seed": int(frozen["base_seed"]),
        "implementation_commit": frozen.get("implementation_commit", "pending-commit"),
        "moves": moves,
        "move_count": len(moves),
        "winner_color": winner_color,
        "winner_model": winner_model,
        "outcome": "draw" if winner_color is None else f"{winner_color.lower()}_win",
        "termination": termination,
        "timeout_count": sum(bool(move.get("timed_out")) for move in moves),
        "search_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
        "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
        "rollout_plies_total": sum(int(move.get("rollout_plies", 0)) for move in moves),
        "terminal_simulations_total": sum(int(move.get("terminal_simulations", 0)) for move in moves),
        "truncated_simulations_total": sum(int(move.get("truncated_simulations", 0)) for move in moves),
        "tree_depth_max": max((int(move.get("tree_depth", 0)) for move in moves), default=0),
        "thinking_seconds_total": sum(float(move.get("elapsed_seconds", 0.0)) for move in moves),
    }
    return record


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _git_show(root: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"], cwd=root, check=True, capture_output=True
    )
    return result.stdout


def _model_alias(model: str | None) -> str | None:
    if model is None:
        return None
    return _V3_MODEL_MAP.get(model, model)


def _normalize_v3_record(
    source: dict[str, Any], source_line_hash: str, config: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    normalized = deepcopy(source)
    source_id = str(source["game_id"])
    normalized.update(
        {
            "experiment_id": config["experiment_id"],
            "game_id": spec["game_id"],
            "matchup": list(spec["matchup"]),
            "black_model": _model_alias(str(source["black_model"])),
            "white_model": _model_alias(str(source["white_model"])),
            "winner_model": _model_alias(source.get("winner_model")),
            "pair_id": spec["pair_id"],
            "opening_seed": int(spec["opening_seed"]),
            "v4b_game_seed": int(spec["game_seed"]),
            "v4b_base_seed": int(config["base_seed"]),
            "budget_seconds_by_model": {
                _model_alias(str(model)): float(budget)
                for model, budget in source["budget_seconds_by_model"].items()
            },
            "source_type": "reused_v3",
            "source_experiment_id": source.get("experiment_id"),
            "source_game_id": source_id,
            "source_game_record_sha256": source_line_hash,
            "source_code_commit": config["reuse_v3"]["source_code_commit"],
            "implementation_commit": config.get("implementation_commit", "pending-commit"),
        }
    )
    normalized["moves"] = [
        {**move, "model": _model_alias(move.get("model"))} for move in source.get("moves", ())
    ]
    normalized["opening"] = [dict(move) for move in spec["opening"]]
    normalized["color_swap"] = bool(source["color_swap"])
    normalized["game_seed"] = int(source["game_seed"])
    return normalized


def load_v3_reuse(config: dict[str, Any], project_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify V3 provenance, full V3 replay, exact D20/A pairing, then normalize six rows."""

    frozen = load_config(config)
    root = Path(project_root).resolve()
    reuse = frozen["reuse_v3"]
    commit = str(reuse["source_code_commit"])
    if not reuse.get("enabled"):
        raise ValueError("V3 reuse is disabled by config")
    v3_config_path = root / reuse["source_config"]
    source_path = root / reuse["source_games"]
    v3_config = json.loads(v3_config_path.read_text(encoding="utf-8"))
    if v3_config.get("experiment_id") != reuse["source_experiment_id"] or v3_config.get("code_commit") != commit:
        raise ValueError("V3 config does not match its registered experiment and source commit")
    if "unchanged V2 D2 MCTSAgent" not in str(v3_config.get("algorithm_statement", "")):
        raise ValueError("V3 D3 algorithm declaration does not match the registered pure-MCTS engine")
    mcts_path = root / "src/research/mcts_d2.py"
    current_mcts_hash = _sha256(mcts_path.read_bytes())
    historical_mcts_hash = _sha256(_git_show(root, commit, "src/research/mcts_d2.py"))
    expected_mcts_hash = str(reuse["mcts_d2_source_sha256"])
    if current_mcts_hash != expected_mcts_hash or historical_mcts_hash != expected_mcts_hash:
        raise ValueError("V3 MCTS source does not match the registered D3 implementation")
    historical_runner = _git_show(root, commit, "src/research/mcts_v2.py").decode("utf-8")
    if 'agents = {model: MCTSAgent() for model in models if model in {"D2", "D3"}}' not in historical_runner:
        raise ValueError("V3 runner did not use a persistent D3 MCTSAgent per game")
    all_v3_rows = [
        row
        for path in sorted((root / "research/v3/results").glob("*_games.jsonl"))
        for row in read_historical_games(path)
    ]
    from .mcts_v3_analysis import validate_v3_records

    validate_v3_records(all_v3_rows, v3_config, require_complete=True)

    expected_games = int(reuse["require_exactly_games"])
    raw_bytes = source_path.read_bytes()
    raw_lines = [line for line in raw_bytes.splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in raw_lines]
    if len(source_rows) != expected_games:
        raise ValueError(f"Expected {expected_games} D20-vs-A V3 records, found {len(source_rows)}")
    line_hashes = {
        str(row["game_id"]): _sha256(line)
        for row, line in zip(source_rows, raw_lines)
    }
    schedule = build_schedule(frozen)
    v4_schedule = {
        (int(item["opening_index"]), bool(item["color_swap"])): item
        for item in schedule
        if item["matchup"] == ("A", "D")
    }
    raw_catalog = v3_config["opening_catalog"]
    seen: set[tuple[int, bool]] = set()
    normalized: list[dict[str, Any]] = []
    for source in source_rows:
        if int(source.get("board_size", 0)) != 9 or tuple(source.get("matchup", ())) != ("A", "D3"):
            raise ValueError("V3 source game is not a 9x9 A-D3 matchup")
        budgets = source.get("budget_seconds_by_model", {})
        if float(budgets.get("A", -1)) != 0.05 or float(budgets.get("D3", -1)) != 20.0:
            raise ValueError("V3 source game does not use A=50ms and D3=20s")
        opening_index = int(source["opening_index"])
        color_swap = bool(source["color_swap"])
        key = (opening_index, color_swap)
        if key in seen or opening_index not in range(3):
            raise ValueError("V3 source opening/color assignment is duplicated or unregistered")
        seen.add(key)
        expected_seed = int(frozen["opening_seeds"][opening_index])
        expected_v3_opening = [
            {"player": str(player), "row": int(row), "col": int(col)}
            for player, row, col in raw_catalog[str(expected_seed)]
        ]
        if source.get("opening") != expected_v3_opening:
            raise ValueError(f"V3 opening {expected_seed} does not match the registered V4B opening")
        if int(source.get("opening_plies", 0)) != 4:
            raise ValueError("V3 source game does not use four registered opening plies")
        expected_black, expected_white = (("A", "D3") if not color_swap else ("D3", "A"))
        if (source.get("black_model"), source.get("white_model")) != (expected_black, expected_white):
            raise ValueError("V3 source color assignment is inconsistent with its pair ID")
        spec = v4_schedule[key]
        if not source.get("game_id") or source.get("winner_model") not in {None, "A", "D3"}:
            raise ValueError("V3 source outcome metadata is invalid")
        normalized.append(_normalize_v3_record(source, line_hashes[str(source["game_id"])], frozen, spec))
    if seen != {(index, swap) for index in range(3) for swap in (False, True)}:
        raise ValueError("V3 reuse source is missing an opening/color-swap partner")
    normalized.sort(key=lambda row: row["game_id"])
    audit = {
        "verified": True,
        "source_experiment_id": v3_config["experiment_id"],
        "source_config": str(reuse["source_config"]),
        "source_games": str(reuse["source_games"]),
        "source_games_sha256": _sha256(raw_bytes),
        "source_code_commit": commit,
        "current_mcts_d2_sha256": current_mcts_hash,
        "historical_mcts_d2_sha256": historical_mcts_hash,
        "full_v3_replay_validation": "passed",
        "opening_policy_validation": "passed: three seeds with both colors",
        "reused_games": len(normalized),
        "source_game_ids": sorted(str(row["game_id"]) for row in source_rows),
        "v4b_game_ids": sorted(str(row["game_id"]) for row in normalized),
        "verified_at_utc": _utc_now(),
    }
    return normalized, audit


def _load_unique_games(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records = load_game_records(path)
    ids = [str(record["game_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("V4B canonical games contain duplicate IDs")
    return records, set(ids)


def run_experiment(
    config_source: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    max_games: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume the fixed schedule and fsync every complete game before progress."""

    config = load_config(config_source)
    if max_games is not None and max_games < 0:
        raise ValueError("max_games cannot be negative")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing V4B output belongs to a different config")
    if not config_path.exists():
        write_json_atomic(config_path, config)
    config_hash = _sha256(_canonical_json(config).encode("utf-8"))
    records_path = root / "9x9_games.jsonl"
    records, completed = _load_unique_games(records_path)
    schedule = build_schedule(config)
    schedule_by_id = {str(spec["game_id"]): spec for spec in schedule}
    unexpected = completed - set(schedule_by_id)
    if unexpected:
        raise ValueError(f"Unscheduled game IDs in V4B output: {sorted(unexpected)[:3]}")

    audit_path = root / "reuse_audit.json"
    raw_source_path = root / "v3_reused_games.jsonl"
    if not audit_path.exists():
        if completed:
            raise ValueError("Existing partial V4B results lack the required reuse audit metadata")
        reuse_records: list[dict[str, Any]] = []
        if config["reuse_v3"].get("enabled"):
            try:
                reuse_records, reuse_audit = load_v3_reuse(
                    config, project_root or Path(__file__).resolve().parents[2]
                )
            except Exception as exc:
                reuse_audit = {
                    "verified": False,
                    "reused_games": 0,
                    "fallback": "rerun all six A-D games as new V4B games",
                    "verification_error": f"{type(exc).__name__}: {exc}",
                    "verified_at_utc": _utc_now(),
                }
            if reuse_audit.get("verified"):
                original_path = Path(project_root or Path(__file__).resolve().parents[2]) / config["reuse_v3"]["source_games"]
                raw_bytes = original_path.read_bytes()
                _write_bytes_atomic(raw_source_path, raw_bytes)
                for record in reuse_records:
                    append_game(records_path, record, completed)
                records.extend(reuse_records)
        else:
            reuse_audit = {"verified": False, "reused_games": 0, "reason": "reuse disabled by config"}
        write_json_atomic(audit_path, reuse_audit)
    else:
        reuse_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if reuse_audit.get("verified"):
            reused = [record for record in records if record.get("source_type") == "reused_v3"]
            if len(reused) != int(reuse_audit.get("reused_games", -1)) or not raw_source_path.is_file():
                raise ValueError("V4B reused records do not match their durable reuse audit")

    progress_path = root / "progress.json"
    previous_elapsed = 0.0
    started_utc = _utc_now()
    now_progress = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "implementation_commit": config.get("implementation_commit", "pending-commit"),
        "planned_games": len(schedule),
        "completed_games": len(completed),
        "completed_ids": sorted(completed),
        "new_games": sum(record.get("source_type") == "new" for record in records),
        "reused_games": sum(record.get("source_type") == "reused_v3" for record in records),
        "updated_at_utc": _utc_now(),
    }
    if progress_path.exists():
        prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            prior_progress.get("experiment_id") != config["experiment_id"]
            or prior_progress.get("config_sha256") != config_hash
            or not set(prior_progress.get("completed_ids", ())).issubset(completed)
        ):
            raise ValueError("V4B progress metadata does not match durable results")
        previous_elapsed = float(prior_progress.get("elapsed_seconds_total", 0.0))
        started_utc = str(prior_progress.get("first_started_at_utc", started_utc))
    now_progress["first_started_at_utc"] = started_utc
    now_progress["last_started_at_utc"] = _utc_now()
    now_progress["elapsed_seconds_total"] = previous_elapsed
    from .abcd_final_v4b_analysis import validate_records

    validate_records(records, config, require_complete=False)
    write_json_atomic(progress_path, now_progress)

    started = time.perf_counter()
    newly_completed = 0
    for spec in schedule:
        identifier = str(spec["game_id"])
        if identifier in completed:
            continue
        if max_games is not None and newly_completed >= max_games:
            break
        record = run_game(spec, config)
        if str(record.get("game_id")) != identifier:
            raise ValueError("run_game returned a mismatched unique game ID")
        from .abcd_final_v4b_analysis import validate_game_record

        validate_game_record(record, config, spec)
        append_game(records_path, record, completed)
        records.append(record)
        newly_completed += 1
        now_progress = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "config_sha256": config_hash,
            "implementation_commit": config.get("implementation_commit", "pending-commit"),
            "planned_games": len(schedule),
            "completed_games": len(completed),
            "completed_ids": sorted(completed),
            "new_games": sum(item.get("source_type") == "new" for item in records),
            "reused_games": sum(item.get("source_type") == "reused_v3" for item in records),
            "last_game_id": identifier,
            "first_started_at_utc": started_utc,
            "last_started_at_utc": now_progress["last_started_at_utc"],
            "elapsed_seconds_total": previous_elapsed + (time.perf_counter() - started),
            "updated_at_utc": _utc_now(),
        }
        write_json_atomic(progress_path, now_progress)
        print(
            json.dumps(
                {
                    "completed": len(completed),
                    "planned": len(schedule),
                    "last_game_id": identifier,
                    "outcome": record.get("outcome"),
                    "elapsed_seconds": record.get("thinking_seconds_total"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        {
            "remaining_games": len(schedule) - len(completed),
            "complete": len(completed) == len(schedule),
            "games_completed_this_run": newly_completed,
            "elapsed_seconds_this_run": elapsed,
            "elapsed_seconds_total": previous_elapsed + elapsed,
            "first_started_at_utc": started_utc,
            "last_completed_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
        }
    )
    write_json_atomic(progress_path, progress)
    return progress


def load_v4b_records(results_dir: str | Path) -> list[dict[str, Any]]:
    records, _ = _load_unique_games(Path(results_dir) / "9x9_games.jsonl")
    return records


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/abcd_final_v4b.json")
    parser.add_argument("--output-dir", default="research/v4b/results")
    parser.add_argument("--max-games", type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    started_utc = _utc_now()
    result = run_experiment(args.config, args.output_dir, max_games=args.max_games)
    execution = {
        "experiment_id": result["experiment_id"],
        "started_at_utc": started_utc,
        "completed_at_utc": _utc_now(),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "wall_seconds_accumulated": result.get("elapsed_seconds_total", 0.0),
        "games_completed_this_invocation": result.get("games_completed_this_run", 0),
        "games_completed_total": result.get("completed_games", 0),
        "remaining_games": result.get("remaining_games", 0),
    }
    write_json_atomic(Path(args.output_dir) / "execution.json", execution)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
