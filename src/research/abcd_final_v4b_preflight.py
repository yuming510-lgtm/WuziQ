"""Real move benchmark, resource snapshot, and runtime estimate for V4B."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
from typing import Any

from ..ai import choose_move
from ..gomoku import GomokuBoard, Player
from .abcd_final_v4b import derive_seed, load_config, read_historical_games, settings_for_model
from .checkpoint import write_json_atomic
from .mcts_d2 import MCTSAgent


def _representative_position(config: dict[str, Any]) -> tuple[GomokuBoard, Player]:
    board = GomokuBoard(9)
    for player, row, col in config["opening_catalog"]["731091"]:
        board.place_stone(str(player), (int(row), int(col)))
    return board, "B"


def _resource_snapshot(directory: str | Path) -> dict[str, Any]:
    cpu_model = "unknown"
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    meminfo: dict[str, int] = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text(errors="replace").splitlines():
            key, _, rest = line.partition(":")
            if rest.strip().endswith("kB"):
                meminfo[key] = int(rest.strip().split()[0]) * 1024
    cpu_quota = None
    if Path("/sys/fs/cgroup/cpu.max").exists():
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        cpu_quota = None if quota == "max" else int(quota) / int(period)
    memory_limit = None
    if Path("/sys/fs/cgroup/memory.max").exists():
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        memory_limit = None if raw == "max" else int(raw)
    disk = shutil.disk_usage(Path(directory))
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": cpu_model,
        "cgroup_cpu_quota_cores": cpu_quota,
        "load_average_1_5_15m": load,
        "memory_total_bytes": meminfo.get("MemTotal"),
        "memory_available_bytes": meminfo.get("MemAvailable"),
        "cgroup_memory_limit_bytes": memory_limit,
        "workspace_disk_total_bytes": disk.total,
        "workspace_disk_free_bytes": disk.free,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def _historical_calibration(root: Path) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for opponent in ("B", "C"):
        path = root / "research/v3/results" / f"9x9_d3-20s-vs-{opponent.lower()}_games.jsonl"
        games = read_historical_games(path)
        if len(games) != 6:
            raise ValueError(f"V3 D20 vs {opponent} calibration needs six games; found {len(games)}")
        if any(
            int(game["board_size"]) != 9
            or float(game["budget_seconds_by_model"].get("D3", 0)) != 20.0
            or float(game["budget_seconds_by_model"].get(opponent, 0)) != 0.05
            for game in games
        ):
            raise ValueError(f"V3 D20 vs {opponent} calibration has different budgets")
        rows[opponent] = games
    totals: dict[str, dict[str, Any]] = {}
    for opponent, games in rows.items():
        d3_seconds = []
        d3_simulations = []
        opponent_moves = []
        d3_moves = []
        for game in games:
            d_moves = [move for move in game["moves"] if move.get("model") == "D3"]
            o_moves = [move for move in game["moves"] if move.get("model") == opponent]
            d3_seconds.append(sum(float(move.get("elapsed_seconds", 0.0)) for move in d_moves))
            d3_simulations.append(sum(int(move.get("simulations", 0)) for move in d_moves))
            opponent_moves.append(len(o_moves))
            d3_moves.append(len(d_moves))
        totals[opponent] = {
            "historical_games": len(games),
            "historical_file": f"research/v3/results/9x9_d3-20s-vs-{opponent.lower()}_games.jsonl",
            "D3_seconds_by_game": d3_seconds,
            "D3_total_seconds": sum(d3_seconds),
            "D3_simulations_total": sum(d3_simulations),
            "D3_moves_by_game": d3_moves,
            "opponent_moves_total": sum(opponent_moves),
            "wall_time_per_game_seconds": [float(game["thinking_seconds_total"]) for game in games],
        }
    return totals


def estimate_runtime(calibration: dict[str, Any], margin: float = 0.25) -> dict[str, Any]:
    """Use actual V3 D20 turn costs and conservative 200ms B/C upper bounds."""

    d3_seconds_by_matchup = {
        opponent: float(data["D3_total_seconds"]) for opponent, data in calibration.items()
    }
    d3_base = sum(d3_seconds_by_matchup.values())
    opponent_moves = sum(int(data["opponent_moves_total"]) for data in calibration.values())
    d3_opponent_search_seconds = opponent_moves * 0.2
    ab_game_upper_bound = {
        "A-B": 6 * 39 * 0.2,
        "A-C": 6 * 39 * 0.2,
        "B-C": 6 * 77 * 0.2,
    }
    ab_upper = sum(ab_game_upper_bound.values())
    d3_estimate = (d3_base + d3_opponent_search_seconds) * (1.0 + margin)
    formal_seconds = d3_estimate + ab_upper
    d3_fullboard_ceiling = 12 * 39 * 20.0
    return {
        "basis": "V3 actual D3-20s-vs-B/C games at 50ms for turn counts and D3 think time; B/C 200ms opponent work; A/B/C-only games bounded by all remaining moves.",
        "historical_D3_seconds_by_opponent_for_six_games": d3_seconds_by_matchup,
        "historical_D3_seconds_total": d3_base,
        "historical_opponent_move_count": opponent_moves,
        "estimated_200ms_opponent_think_seconds": d3_opponent_search_seconds,
        "D3_turn_cost_uncertainty_margin_fraction": margin,
        "estimated_D_games_wall_compute_seconds": d3_estimate,
        "estimated_ABC_games_seconds_upper_bound": ab_upper,
        "estimated_formal_run_seconds": formal_seconds,
        "estimated_formal_run_minutes": formal_seconds / 60,
        "estimated_formal_run_range_minutes": [formal_seconds / 60, formal_seconds / 60 * 1.25],
        "D20_fullboard_theoretical_ceiling_seconds_for_12_new_D_games": d3_fullboard_ceiling,
        "estimate_warning": "Turn counts can change at 200ms. The full-board ceiling is intentionally loose; this estimate is anchored to all twelve existing V3 D20-vs-B/C games and includes a 25% margin.",
    }


def run_preflight(
    config_source: str | Path | dict[str, Any],
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Measure A/B/C/D on one fixed registered opening and save a before-run estimate."""

    started = time.perf_counter()
    config = load_config(config_source)
    root = Path(project_root or Path(__file__).resolve().parents[2])
    output_root = Path(output_path).parent
    output_root.mkdir(parents=True, exist_ok=True)
    base_board, player = _representative_position(config)
    original = list(base_board.history)
    benchmarks = []
    for model in ("A", "B", "C", "D"):
        board = GomokuBoard(9)
        for side, coordinate in original:
            board.place_stone(side, coordinate)
        settings = settings_for_model(config, model, occupied=len(board.history))
        seed = derive_seed(int(config["base_seed"]), "v4b-preflight", 731091, model)
        move_started = time.perf_counter()
        if model == "D":
            result = MCTSAgent().choose_move(board, player, settings, seed)
        else:
            result = choose_move(board, player, model, settings, seed)
        wall_seconds = time.perf_counter() - move_started
        if board.history != original or board.get(result.move) is not None:
            raise ValueError(f"{model} preflight changed the position or returned an illegal move")
        if result.simulations == 0 and model == "D":
            raise ValueError("D3 preflight completed no MCTS simulation")
        benchmarks.append(
            {
                "model": model,
                "engine": "MCTSAgent/D3" if model == "D" else model,
                "budget_seconds": float(settings.time_limit_seconds),
                "move": list(result.move),
                "elapsed_seconds_reported": float(result.elapsed_seconds),
                "elapsed_seconds_wall": wall_seconds,
                "completed_depth": int(result.completed_depth),
                "nodes": int(result.nodes),
                "cutoffs": int(result.cutoffs),
                "simulations": int(result.simulations),
                "simulations_per_second": result.simulations / result.elapsed_seconds if result.elapsed_seconds else 0.0,
                "tree_depth": int(result.completed_depth),
                "timed_out": bool(result.timed_out or result.elapsed_seconds > settings.time_limit_seconds),
            }
        )
    calibration = _historical_calibration(root)
    estimate = estimate_runtime(calibration)
    source_file_sizes = []
    for path in sorted((root / "research/v3/results").glob("9x9_d3-20s-vs-*_games.jsonl")):
        source_file_sizes.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    source_bytes_per_record = [
        item["bytes"] / 6 for item in source_file_sizes
    ]
    expected_storage = int((max(source_bytes_per_record) if source_bytes_per_record else 0) * 36 * 1.5 + 2_000_000)
    result = {
        "experiment_id": config["experiment_id"],
        "preflight_only": True,
        "official_games_started": 0,
        "benchmarked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip(),
        "position": {
            "board_size": 9,
            "opening_seed": 731091,
            "opening_moves": [{"player": side, "row": coord[0], "col": coord[1]} for side, coord in original],
            "to_move": player,
        },
        "resources": _resource_snapshot(output_root),
        "benchmarks": benchmarks,
        "v3_D20_calibration": calibration,
        "estimate": estimate,
        "storage_estimate": {
            "historical_files": source_file_sizes,
            "estimated_final_recovery_bundle_bytes": expected_storage,
            "available_workspace_disk_bytes": _resource_snapshot(output_root)["workspace_disk_free_bytes"],
        },
        "preflight_elapsed_seconds": time.perf_counter() - started,
        "preflight_seed_policy": "fixed seed derived from V4B base seed, model, and opening 731091",
    }
    write_json_atomic(output_path, result)
    return result
