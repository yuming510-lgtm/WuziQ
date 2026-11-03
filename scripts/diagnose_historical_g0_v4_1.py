#!/usr/bin/env python3
"""Reuse every historical V3/V4 G0 checkpoint and G0 self-play log for diagnosis."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.research.alphazero_g.replay import _validate_game
from src.research.alphazero_g.training import load_g_checkpoint
from src.research.alphazero_g.v3_evaluation import build_v3_tactical_puzzles, evaluate_v3_tactics
from src.research.alphazero_g.v4_1_config import load_v4_1_config
from src.research.alphazero_g.v4_1_runner import _cpu_used, _record_cpu


def stream_historical_logs() -> dict:
    groups = {
        "V3 G0": sorted((ROOT / "research/g_v3/results/selfplay").glob("G0-*.jsonl")),
        "V4 G0": sorted((ROOT / "research/g_v4/results/selfplay").glob("*-G0-*.jsonl")),
    }
    result = {"format_version": 1, "logs": [], "total_games": 0, "total_states": 0}
    all_ids: set[str] = set()
    duplicate_ids = []
    for cohort, paths in groups.items():
        for path in paths:
            stats = {"path": path.relative_to(ROOT).as_posix(), "cohort": cohort, "games": 0,
                     "states": 0, "budgeted_states": 0, "winner_black": 0, "winner_white": 0, "draws": 0,
                     "total_plies": 0, "total_moves": 0, "entropy_sum": 0.0, "entropy_count": 0,
                     "root_value_abs_error": 0.0, "root_value_squared_error": 0.0,
                     "root_value_count": 0, "duplicate_ids": 0}
            local_ids: set[str] = set()
            log_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    try:
                        log_digest.update(raw_line)
                        game = json.loads(raw_line)
                        _validate_game(game)
                    except Exception as exc:
                        raise ValueError(f"invalid historical record {path}:{line_number}: {exc}") from exc
                    game_id = str(game["game_id"])
                    if game_id in local_ids or game_id in all_ids:
                        stats["duplicate_ids"] += 1
                        duplicate_ids.append({"path": stats["path"], "game_id": game_id})
                    local_ids.add(game_id)
                    all_ids.add(game_id)
                    stats["games"] += 1
                    steps = len(game["moves"]) - len(game["opening"])
                    stats["states"] += len(game["experience_rows"])
                    stats["budgeted_states"] += int(game.get("budgeted_environment_steps", len(game["experience_rows"])))
                    stats["total_plies"] += steps
                    stats["total_moves"] += len(game["moves"])
                    stats["winner_black"] += game["winner_color"] == "B"
                    stats["winner_white"] += game["winner_color"] == "W"
                    stats["draws"] += game["winner_color"] is None
                    for row in game["experience_rows"]:
                        entropy = row.get("policy_entropy")
                        if entropy is not None and math.isfinite(float(entropy)):
                            stats["entropy_sum"] += float(entropy)
                            stats["entropy_count"] += 1
                        value = row.get("root_value")
                        target = row.get("value_target")
                        if value is not None and target is not None:
                            residual = float(value) - float(target)
                            stats["root_value_abs_error"] += abs(residual)
                            stats["root_value_squared_error"] += residual * residual
                            stats["root_value_count"] += 1
            stats["unique_game_ids"] = len(local_ids)
            stats["sha256"] = log_digest.hexdigest()
            stats["mean_post_opening_plies"] = stats["total_plies"] / max(stats["games"], 1)
            stats["mean_game_moves"] = stats["total_moves"] / max(stats["games"], 1)
            stats["mean_policy_entropy"] = stats["entropy_sum"] / max(stats["entropy_count"], 1)
            stats["root_value_mae"] = stats["root_value_abs_error"] / max(stats["root_value_count"], 1)
            stats["root_value_mse"] = stats["root_value_squared_error"] / max(stats["root_value_count"], 1)
            for key in ("entropy_sum", "root_value_abs_error", "root_value_squared_error"):
                del stats[key]
            result["total_games"] += stats["games"]
            result["total_states"] += stats["states"]
            result.setdefault("total_budgeted_states", 0)
            result["total_budgeted_states"] += stats["budgeted_states"]
            result["logs"].append(stats)
    result["globally_unique_game_ids"] = len(all_ids)
    result["cross_log_duplicate_ids"] = duplicate_ids
    return result


def sweep_checkpoints() -> dict:
    paths = sorted(list((ROOT / "research/g_v3/checkpoints/G0").rglob("*.pt")) +
                   list((ROOT / "research/g_v4/checkpoints").rglob("*.pt")))
    paths = [path for path in paths if path.name != "training_state.pt"]
    aliases: dict[str, list[str]] = defaultdict(list)
    representatives: dict[str, Path] = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aliases[digest].append(path.relative_to(ROOT).as_posix())
        representatives.setdefault(digest, path)
    puzzles = build_v3_tactical_puzzles()
    rows = []
    for digest, path in sorted(representatives.items(), key=lambda item: sorted(aliases[item[0]])[0]):
        alias_paths = sorted(aliases[digest])
        stage_match = next((re.search(r"node_(\d+)\.pt$", name) for name in alias_paths
                            if re.search(r"node_(\d+)\.pt$", name)), None)
        stage = int(stage_match.group(1)) if stage_match else None
        if "B-G0-v3-champion-continue" in str(path):
            cohort_id = "V4 continuation"
        else:
            match = re.search(r"seed-(\d+)|s(540\d+)", str(path))
            cohort_id = ("V3 seed " + match.group(1)) if "research/g_v3/" in str(path) and match else (
                "V4 seed " + match.group(2) if match and match.group(2) else str(path).split("/")[-2]
            )
        model = load_g_checkpoint(path, device="cpu")
        for mode in ("network", "puct"):
            result = evaluate_v3_tactics(model, mode=mode, puzzles=puzzles, simulations=64)
            rows.append({
                "checkpoint_aliases": alias_paths,
                "checkpoint_sha256": digest,
                "checkpoint_id": path.relative_to(ROOT).as_posix(),
                "cohort_id": cohort_id,
                "training_states": stage,
                "mode": mode,
                "simulations_per_puct_move": 64 if mode == "puct" else 0,
                "total_correct": int(result["correct"]),
                "total_puzzles": int(result["puzzles"]),
                "accuracy": float(result["accuracy"]),
                "categories": result["categories"],
            })
    return {
        "format_version": 1,
        "data_scope": "all historical V3/V4 G0 policy-value checkpoints; duplicate byte-identical files evaluated once",
        "puzzle_source": "4 rule-generated positions with D4 symmetry, exact five-in-a-row labels",
        "historical_checkpoint_files": len(paths),
        "unique_checkpoint_weight_files": len(representatives),
        "puzzle_count_per_mode": len(puzzles),
        "evaluations": rows,
    }


def main() -> None:
    load_v4_1_config(ROOT)
    torch.set_num_threads(1)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    logs = stream_historical_logs()
    tactical = sweep_checkpoints()
    elapsed_cpu, elapsed_wall = time.process_time() - start_cpu, time.perf_counter() - start_wall
    ledger = ROOT / "research/g_v4_1/results/cpu_ledger.json"
    if _cpu_used(ROOT) + elapsed_cpu > 6 * 3600:
        raise RuntimeError("G V4.1 six CPU-hour hard cap reached during historical checkpoint sweep")
    _record_cpu(ROOT, f"diagnostics:all-historical-g0-checkpoints-and-logs:{time.time_ns()}",
                elapsed_cpu, elapsed_wall, "diagnostics")
    payload = {
        "experiment_id": "wuziq-alphazero-g-v4-1",
        "elapsed_cpu_seconds": elapsed_cpu,
        "elapsed_wall_seconds": elapsed_wall,
        "historical_selfplay_audit": logs,
        "historical_checkpoint_tactical_sweep": tactical,
    }
    destination = ROOT / "research/g_v4_1/results/diagnostics/historical_g0_checkpoint_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({
        "historical_logs": len(logs["logs"]), "historical_games": logs["total_games"],
        "historical_states": logs["total_states"], "unique_checkpoint_files": tactical["unique_checkpoint_weight_files"],
        "checkpoint_files": tactical["historical_checkpoint_files"],
        "tactical_evaluations": len(tactical["evaluations"]),
        "elapsed_cpu_seconds": elapsed_cpu,
        "elapsed_wall_seconds": elapsed_wall,
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
