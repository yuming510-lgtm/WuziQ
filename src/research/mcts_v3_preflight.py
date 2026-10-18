"""Measured D3 performance, cloud resources, and V2-grounded runtime estimates."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import re
import shutil
import time
from typing import Any

from ..ai import SearchConfig
from ..gomoku import GomokuBoard, Player
from .checkpoint import load_game_records
from .mcts_d2 import MCTSAgent
from .mcts_v2 import _current_code_commit, derive_seed

_D3_BUDGETS = (10.0, 20.0, 50.0)
_OPPONENTS = ("A", "B", "C")


def code_commit_for_config(config: dict[str, Any]) -> str:
    """Return the config's full code SHA or the checked-out commit SHA."""

    commit = str(config.get("code_commit") or _current_code_commit())
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("preflight code_commit must be a full 40-character lowercase SHA")
    return commit


def representative_position() -> tuple[GomokuBoard, Player]:
    """Return the same nonterminal 12-ply 9×9 position for every budget."""

    sequence: tuple[tuple[Player, tuple[int, int]], ...] = (
        ("B", (4, 4)),
        ("W", (6, 3)),
        ("B", (7, 5)),
        ("W", (6, 1)),
        ("B", (2, 2)),
        ("W", (3, 5)),
        ("B", (1, 4)),
        ("W", (5, 6)),
        ("B", (3, 1)),
        ("W", (0, 4)),
        ("B", (6, 7)),
        ("W", (2, 6)),
    )
    board = GomokuBoard(9)
    for player, coordinate in sequence:
        board.place_stone(player, coordinate)
        if board.check_victory(coordinate, player):
            raise ValueError("registered preflight position unexpectedly contains a win")
    return board, "B"


def historical_turn_averages(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute V2 5-second D2/opponent move counts used only for planning."""

    output: dict[str, dict[str, float]] = {}
    for opponent in _OPPONENTS:
        games = [
            record
            for record in records
            if int(record.get("board_size", 0)) == 9
            and set(record.get("matchup", ())) == {"D2", opponent}
        ]
        if not games:
            raise ValueError(f"No V2 9×9 D2-vs-{opponent} calibration games were found")
        counts = {
            model: [
                sum(1 for move in record.get("moves", ()) if move.get("model") == model)
                for record in games
            ]
            for model in ("D2", opponent)
        }
        output[opponent] = {
            "calibration_games": len(games),
            "D2_moves_per_game": sum(counts["D2"]) / len(games),
            "opponent_moves_per_game": sum(counts[opponent]) / len(games),
            "calibration_track": "V2 9×9 fair 5-second games",
        }
    return output


def estimate_official_runtime(
    historical: dict[str, dict[str, float]],
    measured_seconds_by_budget: dict[float, float],
    *,
    games_per_condition: int = 6,
) -> dict[str, Any]:
    """Estimate 54-game wall time from measured think time and V2 turn counts."""

    expected_d3_moves: dict[str, float] = {}
    expected_d3_seconds = 0.0
    expected_opponent_seconds = 0.0
    for budget in _D3_BUDGETS:
        if budget not in measured_seconds_by_budget:
            raise ValueError(f"Missing measured benchmark for D3's {budget:g}-second budget")
        seconds_per_move = float(measured_seconds_by_budget[budget])
        moves = games_per_condition * sum(
            float(historical[opponent]["D2_moves_per_game"])
            for opponent in _OPPONENTS
        )
        expected_d3_moves[str(int(budget))] = moves
        expected_d3_seconds += moves * seconds_per_move
        expected_opponent_seconds += games_per_condition * sum(
            float(historical[opponent]["opponent_moves_per_game"])
            for opponent in _OPPONENTS
        ) * 0.05
    total = expected_d3_seconds + expected_opponent_seconds
    return {
        "official_games": len(_D3_BUDGETS) * len(_OPPONENTS) * games_per_condition,
        "condition_count": len(_D3_BUDGETS) * len(_OPPONENTS),
        "games_per_condition": games_per_condition,
        "historical_turn_basis": historical,
        "measured_seconds_per_d3_move_by_budget": {
            str(int(budget)): float(measured_seconds_by_budget[budget])
            for budget in _D3_BUDGETS
        },
        "estimated_d3_moves_by_budget": expected_d3_moves,
        "estimated_d3_compute_seconds": expected_d3_seconds,
        "estimated_opponent_compute_seconds": expected_opponent_seconds,
        "estimated_total_seconds": total,
        "estimated_total_minutes": total / 60.0,
        "estimated_total_hours": total / 3600.0,
        "estimate_note": (
            "Planning estimate only: uses V2 5-second average move counts and one measured "
            "representative D3 think time per budget; outcomes and turn counts can change."
        ),
    }


def resource_snapshot(path: str | Path) -> dict[str, Any]:
    """Report visible CPU, cgroup limits, memory, and free workspace storage."""

    cpu_model = "unknown"
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    meminfo: dict[str, int] = {}
    memory_path = Path("/proc/meminfo")
    if memory_path.exists():
        for line in memory_path.read_text(errors="replace").splitlines():
            key, _, rest = line.partition(":")
            if rest.strip().endswith("kB"):
                meminfo[key] = int(rest.strip().split()[0]) * 1024
    cpu_quota = None
    cpu_limit_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_limit_path.exists():
        quota, period = cpu_limit_path.read_text().split()[:2]
        cpu_quota = None if quota == "max" else int(quota) / int(period)
    memory_limit = None
    memory_limit_path = Path("/sys/fs/cgroup/memory.max")
    if memory_limit_path.exists():
        raw_limit = memory_limit_path.read_text().strip()
        memory_limit = None if raw_limit == "max" else int(raw_limit)
    disk = shutil.disk_usage(Path(path))
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "cgroup_cpu_quota_cores": cpu_quota,
        "load_average_1_5_15m": load_average,
        "memory_total_bytes": meminfo.get("MemTotal"),
        "memory_available_bytes": meminfo.get("MemAvailable"),
        "cgroup_memory_limit_bytes": memory_limit,
        "workspace_disk_total_bytes": disk.total,
        "workspace_disk_used_bytes": disk.used,
        "workspace_disk_free_bytes": disk.free,
    }


def run_preflight(
    config_path: str | Path,
    results_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Benchmark real D3 decisions at 10/20/50 seconds and write estimates."""

    started = time.perf_counter()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    board, player = representative_position()
    original_history = list(board.history)
    base_settings = SearchConfig(**config["search"])
    benchmark: list[dict[str, Any]] = []
    for budget in _D3_BUDGETS:
        settings = replace(base_settings, time_limit_seconds=budget)
        result = MCTSAgent().choose_move(
            board,
            player,
            settings,
            derive_seed(int(config["base_seed"]), "v3-preflight", budget, "12-ply"),
        )
        if result.simulations < 1:
            raise ValueError(f"D3 completed no simulation at the {budget:g}-second budget")
        if board.history != original_history or board.get(result.move) is not None:
            raise ValueError("D3 preflight changed the board or returned an illegal move")
        benchmark.append(
            {
                "budget_seconds": budget,
                "move": list(result.move),
                "simulations": result.simulations,
                "simulations_per_second": result.simulations / result.elapsed_seconds,
                "tree_depth": result.completed_depth,
                "elapsed_seconds": result.elapsed_seconds,
                "timed_out": result.timed_out,
                "root_candidates": result.root_candidates,
            }
        )

    calibration_path = root.parent.parent / "v2" / "results" / "9x9_fair-9x9-5s_games.jsonl"
    if not calibration_path.is_file():
        calibration_path = Path(config_path).resolve().parent.parent / "research/v2/results/9x9_fair-9x9-5s_games.jsonl"
    historical_records = load_game_records(calibration_path)
    historical = historical_turn_averages(historical_records)
    estimated = estimate_official_runtime(
        historical,
        {row["budget_seconds"]: row["elapsed_seconds"] for row in benchmark},
    )
    calibration_bytes_per_game = calibration_path.stat().st_size / max(1, len(historical_records))
    estimated["v2_calibration_average_bytes_per_game"] = calibration_bytes_per_game
    estimated["estimated_raw_game_storage_bytes"] = int(calibration_bytes_per_game * 54 * 1.2)
    result = {
        "experiment_id": config["experiment_id"],
        "benchmarked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": code_commit_for_config(config),
        "position": {
            "board_size": board.size,
            "to_move": player,
            "move_count": len(board.history),
            "stones": [
                {"player": side, "row": coord[0], "col": coord[1]}
                for side, coord in board.history
            ],
        },
        "resources": resource_snapshot(root),
        "benchmarks": benchmark,
        "estimate": estimated,
        "preflight_wall_seconds": time.perf_counter() - started,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
