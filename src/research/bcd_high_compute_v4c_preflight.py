"""Measure B1s/C200ms/D50s and estimate V4C runtime/resources before play."""
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
from ..gomoku import GomokuBoard
from .bcd_high_compute_v4c import derive_seed, load_config, settings_for_model
from .bcd_high_compute_v4c import _sha256  # same stable digest helper used by the frozen config
from .checkpoint import write_json_atomic
from .mcts_d2 import MCTSAgent


def resource_snapshot(directory: str | Path) -> dict[str, Any]:
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
        "platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": os.cpu_count(), "cpu_model": cpu_model,
        "cgroup_cpu_quota_cores": cpu_quota, "load_average_1_5_15m": load,
        "memory_total_bytes": meminfo.get("MemTotal"), "memory_available_bytes": meminfo.get("MemAvailable"),
        "cgroup_memory_limit_bytes": memory_limit,
        "workspace_disk_total_bytes": disk.total, "workspace_disk_free_bytes": disk.free,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def _representative_board(config: dict[str, Any]) -> tuple[GomokuBoard, str]:
    board = GomokuBoard(9)
    for player, row, col in config["opening_catalog"]["731091"]:
        board.place_stone(str(player), (int(row), int(col)))
    return board, "B"


def _historical_estimate(project_root: Path, margin: float = 0.25) -> dict[str, Any]:
    path = project_root / "research/v4b/results/9x9_games.jsonl"
    games = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    games = [game for game in games if game.get("source_type") == "new" and set(game.get("matchup", ())) in ({"B", "D"}, {"C", "D"})]
    if len(games) != 12:
        raise ValueError(f"V4B runtime calibration expected 12 fresh B/C-D games, found {len(games)}")
    d_moves = [move for game in games for move in game["moves"] if move.get("model") == "D"]
    if not d_moves:
        raise ValueError("V4B calibration contains no D decisions")
    d20_seconds = sum(float(move["elapsed_seconds"]) for move in d_moves)
    d20_full_budget_count = sum(float(move["elapsed_seconds"]) >= 19.0 for move in d_moves)
    d20_quick_seconds = sum(float(move["elapsed_seconds"]) for move in d_moves if float(move["elapsed_seconds"]) < 19.0)
    d50_estimate = d20_full_budget_count * 50.0 + d20_quick_seconds
    d_vs_b = [game for game in games if set(game["matchup"]) == {"B", "D"}]
    d_vs_c = [game for game in games if set(game["matchup"]) == {"C", "D"}]
    b_in_bd = sum(move.get("model") == "B" for game in d_vs_b for move in game["moves"])
    c_in_cd = sum(move.get("model") == "C" for game in d_vs_c for move in game["moves"])
    bc_games = [game for game in path.read_text(encoding="utf-8").splitlines() if game.strip()]
    bc_games = [json.loads(line) for line in bc_games if set(json.loads(line).get("matchup", ())) == {"B", "C"} and json.loads(line).get("source_type") == "new"]
    if len(bc_games) != 6:
        raise ValueError(f"V4B B-C calibration expected six new games, found {len(bc_games)}")
    b_in_bc = sum(move.get("model") == "B" for game in bc_games for move in game["moves"])
    c_in_bc = sum(move.get("model") == "C" for game in bc_games for move in game["moves"])
    opponent_seconds = b_in_bd * 1.0 + c_in_cd * 0.2
    bc_seconds = b_in_bc * 1.0 + c_in_bc * 0.2
    core_estimate = d50_estimate + opponent_seconds + bc_seconds
    estimate_with_margin = core_estimate * (1.0 + margin)
    return {
        "calibration_source": "V4B research/v4b/results/9x9_games.jsonl, new B-D and C-D games only",
        "calibration_games": len(games), "calibration_D20_moves": len(d_moves),
        "V4B_D20_elapsed_seconds": d20_seconds,
        "V4B_D20_moves_at_or_above_19s": d20_full_budget_count,
        "V4C_D50_scaled_slow_move_seconds": d20_full_budget_count * 50.0,
        "V4B_quick_D_move_seconds_held_constant": d20_quick_seconds,
        "V4C_estimated_D50_seconds": d50_estimate,
        "estimated_B1s_opponent_moves_in_BD_games": b_in_bd,
        "estimated_C200ms_opponent_moves_in_CD_games": c_in_cd,
        "estimated_B1s_moves_in_BC_games": b_in_bc,
        "estimated_C200ms_moves_in_BC_games": c_in_bc,
        "estimated_opponent_search_seconds_in_D_games": opponent_seconds,
        "estimated_BC_game_search_seconds": bc_seconds,
        "runtime_uncertainty_margin_fraction": margin,
        "estimated_formal_run_seconds_with_margin": estimate_with_margin,
        "estimated_formal_run_minutes_with_margin": estimate_with_margin / 60.0,
        "estimated_formal_run_range_minutes": [estimate_with_margin / 60.0, estimate_with_margin * 1.3 / 60.0],
        "D50_full_board_theoretical_ceiling_seconds": 12 * 39 * 50,
        "long_game_risk": "If C-D reaches long draws, twelve D games could approach the 6.5-hour full-board D-turn ceiling; estimate uses V4B observed turn counts and marks a 30% high-side range.",
    }


def run_preflight(config_source: str | Path, output_path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_source)
    root = Path(project_root or Path(__file__).resolve().parents[2])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base, to_move = _representative_board(config)
    history = tuple(base.history)
    benchmarks = []
    for model in ("B", "C", "D"):
        board = GomokuBoard(9)
        for player, coord in history:
            board.place_stone(player, coord)
        settings = settings_for_model(config, model, len(board.history))
        seed = derive_seed(int(config["base_seed"]), "v4c-preflight", 731091, model)
        move_start = time.perf_counter()
        result = MCTSAgent().choose_move(board, to_move, settings, seed) if model == "D" else choose_move(board, to_move, model, settings, seed)
        wall_elapsed = time.perf_counter() - move_start
        if tuple(board.history) != history:
            raise ValueError(f"{model} preflight mutated the shared source position")
        if not board.in_bounds(*result.move) or board.get(result.move) is not None:
            raise ValueError(f"{model} preflight returned an illegal move")
        if model == "D" and result.simulations <= 0:
            raise ValueError("D50 preflight did not complete any MCTS simulations")
        benchmarks.append({
            "model": model, "budget_seconds": settings.time_limit_seconds,
            "move": list(result.move), "elapsed_seconds_reported": result.elapsed_seconds,
            "elapsed_seconds_wall": wall_elapsed, "completed_depth": result.completed_depth,
            "nodes": result.nodes, "cutoffs": result.cutoffs, "simulations": result.simulations,
            "simulations_per_second": result.simulations / result.elapsed_seconds if result.elapsed_seconds else None,
            "timed_out": bool(result.timed_out or result.elapsed_seconds > settings.time_limit_seconds),
        })
    estimate = _historical_estimate(root)
    result_doc = {
        "experiment_id": config["experiment_id"], "preflight_only": True,
        "official_games_started": 0, "benchmarked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip(),
        "position": {"board_size": 9, "opening_seed": 731091, "opening_moves": [
            {"player": player, "row": coord[0], "col": coord[1]} for player, coord in history
        ], "to_move": to_move},
        "benchmarks": benchmarks, "resources": resource_snapshot(output.parent),
        "estimate": estimate,
        "storage_estimate": {
            "historical_v4b_jsonl_bytes": (root / "research/v4b/results/9x9_games.jsonl").stat().st_size,
            "estimated_v4c_jsonl_bytes": int((root / "research/v4b/results/9x9_games.jsonl").stat().st_size * 18 / 36),
            "estimated_total_artifacts_bytes": 2_000_000,
            "available_workspace_disk_bytes": shutil.disk_usage(output.parent).free,
        },
        "preflight_elapsed_seconds": time.perf_counter() - started,
        "model_configuration_verification": {
            "B_C_search_engine_source_hashes_pinned": True,
            "D_mcts_source_sha256": config["engine_sha256"]["src/research/mcts_d2.py"],
            "D_algorithm_unchanged_except_budget": True,
        },
    }
    write_json_atomic(output, result_doc)
    return result_doc
