"""V6.3 score, compute, and V6.2 direct-head-to-head summaries."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ..checkpoint import load_game_records
from ..v6.config import ROOT, canonical, load_config as load_v6_config, sha256
from ..v6_2.config import load_v6_2_config
from ..v6_2.schedule import build_schedule as build_v6_2_schedule
from ..v6_2.validation import validate_v6_2_records
from .config import load_v6_3_config
from .schedule import build_schedule
from .validation import validate_v6_3_records


def _wilson(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _move_stats(game: dict[str, Any], model: str) -> dict[str, float | int]:
    rows = [m for m in game["moves"] if not m.get("opening") and m.get("model") == model]
    return {
        "decisions": len(rows),
        "thinking_seconds": sum(float(m["elapsed_seconds"]) for m in rows),
        "nodes": sum(int(m.get("nodes", 0)) for m in rows),
        "cutoffs": sum(int(m.get("cutoffs", 0)) for m in rows),
        "simulations": sum(int(m.get("simulations", 0)) for m in rows),
        "max_depth": max((int(m.get("completed_depth", 0)) for m in rows), default=0),
        "timeouts": sum(bool(m.get("timed_out")) for m in rows),
        "over_budget": sum(float(m["elapsed_seconds"]) > 200.0 for m in rows),
    }


def _historical_v6_2() -> dict[str, Any]:
    config = load_v6_2_config()
    schedule = build_v6_2_schedule(config)
    path = ROOT / "research/v6_2/results/9x9_games.jsonl"
    records = load_game_records(path)
    checked = validate_v6_2_records(records, schedule, require_complete=True)
    direct_specs = [spec for spec in schedule if spec["pair"] == ["C", "E5"]]
    games = [record for record in records if record.get("pair") == ["C", "E5"]]
    if len(direct_specs) != 2 or len(games) != 2:
        raise ValueError("V6.2 C versus E5 reference must contain exactly two validated games")
    if config["engine_sha256"] != load_v6_config()["engine_sha256"]:
        raise ValueError("V6.2 and V6.3 decision-engine pins differ")
    standings = {model: {"wins": 0, "draws": 0, "losses": 0, "points": 0.0,
                         "thinking_seconds": 0.0, "decisions": 0, "nodes": 0,
                         "cutoffs": 0, "timeouts": 0, "over_budget": 0}
                 for model in ("C", "E5")}
    for game in games:
        winner = game.get("winner_model")
        for model in ("C", "E5"):
            row = standings[model]
            if winner is None:
                row["draws"] += 1
                row["points"] += 0.5
            elif winner == model:
                row["wins"] += 1
                row["points"] += 1.0
            else:
                row["losses"] += 1
            stats = _move_stats(game, model)
            for key in ("thinking_seconds", "decisions", "nodes", "cutoffs", "timeouts", "over_budget"):
                row[key] += stats[key]
    return {
        "experiment_id": config["experiment_id"],
        "source_commit": "f2af15017573383115246242e8b820d1a56c1e1e",
        "games": len(games),
        "opening_id": config["opening_id"],
        "results": [{
            "game_id": game["game_id"], "black_model": game["black_model"],
            "white_model": game["white_model"], "winner_model": game["winner_model"],
            "move_count": game["move_count"], "thinking_seconds_total": game["thinking_seconds_total"],
        } for game in sorted(games, key=lambda row: bool(row["color_swap"]))],
        "standings": standings,
        "full_validation_games": checked["games"],
    }


def summarize_v6_3(
    config_source: str | Path | dict[str, Any] | None,
    output_dir: str | Path,
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_v6_3_config(config_source)
    root = Path(output_dir)
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    records = load_game_records(root / "9x9_games.jsonl")
    checked = validate_v6_3_records(records, schedule, require_complete=True)
    old = _historical_v6_2()
    if config["engine_sha256"] != load_v6_config()["engine_sha256"]:
        raise ValueError("The registered C/E5 engines changed during analysis")

    c_score = {"win": 0, "draw": 0, "loss": 0}
    c_non_draw_wins = c_non_draw = 0
    opening_rows = []
    game_rows = []
    move_rows = []
    for record in records:
        winner = record.get("winner_model")
        if winner is None:
            result_c = "draw"
            c_score["draw"] += 1
        elif winner == "C":
            result_c = "C_win"
            c_score["win"] += 1
            c_non_draw_wins += 1
            c_non_draw += 1
        else:
            result_c = "E5_win"
            c_score["loss"] += 1
            c_non_draw += 1
        c_points = 0.5 if winner is None else (1.0 if winner == "C" else 0.0)
        c_stats, e5_stats = _move_stats(record, "C"), _move_stats(record, "E5")
        game_rows.append({
            "game_id": record["game_id"], "stage": record["stage"],
            "opening_id": str(record["opening_seed"]),
            "black_model": record["black_model"], "white_model": record["white_model"],
            "color_swap": record["color_swap"], "winner_model": winner,
            "result_from_C": result_c, "C_points": c_points,
            "move_count": record["move_count"], "termination": record["termination"],
            "thinking_C_seconds": c_stats["thinking_seconds"],
            "thinking_E5_seconds": e5_stats["thinking_seconds"],
            "decisions_C": c_stats["decisions"], "decisions_E5": e5_stats["decisions"],
            "nodes_C": c_stats["nodes"], "nodes_E5": e5_stats["nodes"],
            "cutoffs_C": c_stats["cutoffs"], "cutoffs_E5": e5_stats["cutoffs"],
            "simulations_C": c_stats["simulations"], "simulations_E5": e5_stats["simulations"],
            "max_depth_C": c_stats["max_depth"], "max_depth_E5": e5_stats["max_depth"],
            "timeouts_C": c_stats["timeouts"], "timeouts_E5": e5_stats["timeouts"],
            "over_budget_C": c_stats["over_budget"], "over_budget_E5": e5_stats["over_budget"],
        })
        opening_rows.append((str(record["opening_seed"]), record, c_points))
        for move in record["moves"]:
            if move.get("opening"):
                continue
            move_rows.append({
                "game_id": record["game_id"], "stage": record["stage"],
                "opening_id": str(record["opening_seed"]), "ply": move["ply"],
                "player": move["player"], "model": move["model"],
                "row": move["row"], "col": move["col"],
                "elapsed_seconds": move["elapsed_seconds"],
                "completed_depth": move.get("completed_depth", 0),
                "nodes": move.get("nodes", 0), "cutoffs": move.get("cutoffs", 0),
                "simulations": move.get("simulations", 0),
                "timed_out": move.get("timed_out", False),
                "over_200s": float(move["elapsed_seconds"]) > 200.0,
            })

    pair_blocks = []
    for opening_id in ("731301", "731302", "731303"):
        games = sorted((row for row in game_rows if row["opening_id"] == opening_id), key=lambda row: bool(row["color_swap"]))
        if not games:
            continue
        score = sum(float(row["C_points"]) for row in games)
        e5_score = len(games) - score
        opening_rows.append({
            "opening_id": opening_id,
            "games": len(games),
            "C_points": score,
            "E5_points": e5_score,
            "C_wins": sum(row["winner_model"] == "C" for row in games),
            "draws": sum(row["winner_model"] is None for row in games),
            "E5_wins": sum(row["winner_model"] == "E5" for row in games),
            "C_black_games": sum(row["black_model"] == "C" for row in games),
            "C_black_wins": sum(row["black_model"] == "C" and row["winner_model"] == "C" for row in games),
            "E5_black_games": sum(row["black_model"] == "E5" for row in games),
            "E5_black_wins": sum(row["black_model"] == "E5" and row["winner_model"] == "E5" for row in games),
            "sum_thinking_seconds": sum(float(row["thinking_C_seconds"]) + float(row["thinking_E5_seconds"]) for row in games),
            "block_result": "C_ahead" if score > e5_score else "E5_ahead" if score < e5_score else "tied",
        })
        pair_blocks.append({"opening_id": opening_id, "C_points": score, "E5_points": e5_score,
                            "result": "C_ahead" if score > e5_score else "E5_ahead" if score < e5_score else "tied"})

    decisive_blocks = [row for row in pair_blocks if row["result"] != "tied"]
    c_block_wins = sum(row["result"] == "C_ahead" for row in decisive_blocks)
    old_standings = old["standings"]
    combined = {}
    for model in ("C", "E5"):
        new = checked["standings"][model]
        historical = old_standings[model]
        combined[model] = {
            "games": historical["wins"] + historical["draws"] + historical["losses"] + new["games"],
            "wins": historical["wins"] + new["wins"],
            "draws": historical["draws"] + new["draws"],
            "losses": historical["losses"] + new["losses"],
            "points": historical["points"] + new["points"],
            "historical_v6_2_points": historical["points"],
            "v6_3_points": new["points"],
            "thinking_seconds": historical["thinking_seconds"] + new["thinking_seconds"],
        }
        if not math.isclose(combined[model]["points"], combined[model]["wins"] + combined[model]["draws"] / 2):
            raise ValueError(f"Combined historical + V6.3 points mismatch for {model}")

    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    preflight = preflight or {}
    model_summary = {}
    for model in ("C", "E5"):
        row = checked["standings"][model]
        model_moves = [
            move for game in records for move in game["moves"]
            if not move.get("opening") and move.get("model") == model
        ]
        model_summary[model] = {
            **row,
            "average_seconds_per_decision": row["thinking_seconds"] / row["decisions"] if row["decisions"] else 0,
            "average_nodes_per_decision": row["nodes"] / row["decisions"] if row["decisions"] else 0,
            "average_cutoffs_per_decision": row["cutoffs"] / row["decisions"] if row["decisions"] else 0,
            "average_completed_depth": sum(int(move.get("completed_depth", 0)) for move in model_moves) / len(model_moves) if model_moves else 0,
            "nodes_per_second": row["nodes"] / row["thinking_seconds"] if row["thinking_seconds"] else 0,
            "cutoffs_per_node": row["cutoffs"] / row["nodes"] if row["nodes"] else 0,
        }
    summary = {
        "experiment_id": config["experiment_id"],
        "baseline_commit": config["source_v6_2_commit"],
        "engine_sha256": config["engine_sha256"],
        "new_games": checked["games"],
        "initial_games": checked["initial_games"],
        "conditional_games": checked["conditional_games"],
        "tiebreak_status": checked["tiebreak_status"],
        "champions": checked["leaders"],
        "standings": checked["standings"],
        "model_compute": model_summary,
        "black_wins": checked["black_wins"],
        "white_wins": checked["white_wins"],
        "pair_blocks": pair_blocks,
        "paired_block_descriptive": {
            "opening_blocks": len(pair_blocks),
            "C_ahead": sum(row["result"] == "C_ahead" for row in pair_blocks),
            "tied": sum(row["result"] == "tied" for row in pair_blocks),
            "E5_ahead": sum(row["result"] == "E5_ahead" for row in pair_blocks),
            "C_ahead_wilson_95_among_decisive_blocks": _wilson(c_block_wins, len(decisive_blocks)),
            "note": "Exploratory descriptive interval only; paired design and very small sample do not support a significance claim.",
        },
        "combined_with_v6_2_direct": combined,
        "v6_2_direct_reference": old,
        "elapsed_wall_seconds": float(progress.get("elapsed_wall_seconds_total", 0.0)),
        "elapsed_compute_seconds": float(progress.get("elapsed_seconds_total", 0.0)),
        "preflight_seconds": float(preflight.get("preflight_elapsed_seconds", 0.0)),
        "total_measured_seconds_including_preflight": float(progress.get("elapsed_wall_seconds_total", 0.0)) + float(preflight.get("preflight_elapsed_seconds", 0.0)),
        "validation": {
            "valid": checked["valid"], "games": checked["games"],
            "unique_ids": checked["unique_ids"], "replayed_moves": checked["replayed_moves"],
            "complete": checked["complete"],
        },
        "implementation_commit": progress["implementation_commit"],
    }

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "validation.json").write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "combined_v6_2.json").write_text(json.dumps({"v6_2": old, "combined": combined}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(root / "games.csv", game_rows)
    _write_csv(root / "moves.csv", move_rows)
    _write_csv(root / "opening_summary.csv", [row for row in opening_rows if isinstance(row, dict)])
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

