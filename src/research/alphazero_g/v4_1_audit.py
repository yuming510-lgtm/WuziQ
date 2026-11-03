"""Cross-artifact integrity audit for the G V4.1 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluation import load_evaluation_log
from .replay import load_game_log
from .v4_1_analysis import summarize_phase
from .v4_1_config import ARM_IDS, load_v4_1_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_experiment(root: str | Path) -> dict[str, Any]:
    """Replay every stored game and reconcile configuration, checkpoint, and summary data."""
    root = Path(root).resolve()
    config = load_v4_1_config(root)
    if config["training"]["maximum_total_new_states"] != 40_000:
        raise ValueError("registered experiment state cap changed")
    if config["training"]["simulations_per_move"] != 64:
        raise ValueError("formal self-play must use PUCT-64")

    selfplay = []
    global_game_ids: set[str] = set()
    arm_reports = []
    expected_checkpoint_hashes = {
        "control_v4": "cd59c87c78fac1f0ec58e966b43895182bcc86e8e813e75a7012d131d35f2bf2",
        "state_uniform": "df30619ad4f7f592a262b0f507596e7b5ea8cbd8b5bcd392ac983a503f75894e",
        "winner_balanced": "01d463ac4ae06be5b2435eb4d3d206b77dd51278b4e6b9f253aca60cfcd2da9f",
        "low_lr": "4eb6b8389f25b73a4b25f974c9adbd82b6c0b3945fd9fe3cad29237c8befa11b",
    }
    for arm_id in ARM_IDS:
        path = root / "research/g_v4_1/results/selfplay" / f"{arm_id}.jsonl"
        games, ids = load_game_log(path)
        if len(ids) != len(games) or global_game_ids.intersection(ids):
            raise ValueError(f"duplicate self-play game ID in {arm_id}")
        global_game_ids.update(ids)
        budgeted_states = sum(int(game.get("budgeted_environment_steps", -1)) for game in games)
        if budgeted_states != 10_000:
            raise ValueError(f"{arm_id} accepted {budgeted_states} states instead of 10000")
        expected_log_arm = f"G0-v4-1-{arm_id}"
        if any(game.get("arm_id") != expected_log_arm or game.get("source") != "self_play" for game in games):
            raise ValueError(f"{arm_id} contains cross-arm or non-self-play training data")
        checkpoint = root / "research/g_v4_1/checkpoints" / arm_id / "node_10000.pt"
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != expected_checkpoint_hashes[arm_id]:
            raise ValueError(f"checkpoint digest mismatch for {arm_id}")
        arm_reports.append({
            "arm_id": arm_id,
            "complete_games": len(games),
            "unique_game_ids": len(ids),
            "budgeted_states": budgeted_states,
            "simulations": sum(int(game["stats"]["simulations"]) for game in games),
            "checkpoint_sha256": checkpoint_hash,
            "winner_black": sum(game.get("winner_color") == "B" for game in games),
            "winner_white": sum(game.get("winner_color") == "W" for game in games),
            "draws": sum(game.get("winner_color") is None for game in games),
        })
        selfplay.extend(games)

    historical_path = root / "research/g_v4_1/results/diagnostics/historical_g0_checkpoint_audit.json"
    historical = json.loads(historical_path.read_text())
    historical_logs = historical["historical_selfplay_audit"]
    historical_sweep = historical["historical_checkpoint_tactical_sweep"]
    if (historical_logs["total_games"] != 9_154 or historical_logs["total_budgeted_states"] != 250_000
            or historical_logs["globally_unique_game_ids"] != 9_154
            or historical_logs["cross_log_duplicate_ids"]):
        raise ValueError("historical G0 self-play logs failed the registered integrity audit")
    for row in historical_logs["logs"]:
        log_path = root / row["path"]
        if not log_path.is_file() or sha256_file(log_path) != row.get("sha256"):
            raise ValueError(f"historical G0 self-play log digest mismatch: {row['path']}")
    checkpoint_file_count = 0
    for row in historical_sweep["evaluations"]:
        for alias in row["checkpoint_aliases"]:
            alias_path = root / alias
            if not alias_path.is_file() or sha256_file(alias_path) != row["checkpoint_sha256"]:
                raise ValueError(f"historical checkpoint alias digest mismatch: {alias}")
            checkpoint_file_count += 1
    if (checkpoint_file_count != historical_sweep["historical_checkpoint_files"] * 2
            or historical_sweep["historical_checkpoint_files"] != 39
            or historical_sweep["unique_checkpoint_weight_files"] != 37
            or len(historical_sweep["evaluations"]) != 74):
        raise ValueError("historical G0 checkpoint sweep is incomplete")

    evaluation_reports = []
    evaluation_ids: set[str] = set()
    for phase, expected_games in (("development", 816), ("blind", 360)):
        path = root / f"research/g_v4_1/results/evaluation/{phase}.jsonl"
        matches, ids = load_evaluation_log(path)
        if len(matches) != expected_games:
            raise ValueError(f"{phase} evaluation has {len(matches)} records, expected {expected_games}")
        if evaluation_ids.intersection(ids):
            raise ValueError("development and blind evaluations share a game ID")
        evaluation_ids.update(ids)
        summary = summarize_phase(path, phase=phase)
        stored = json.loads((root / f"research/g_v4_1/results/evaluation/{phase}_summary.json").read_text())
        if summary != stored:
            raise ValueError(f"{phase} summary does not match its raw match records")
        evaluation_reports.append({
            "phase": phase,
            "games": len(matches),
            "unique_game_ids": len(ids),
            "legal_completed_matches": len(matches),
            "opening_groups_per_comparison": summary["paired_group_count"],
            "comparisons": summary["comparison_count"],
            "score_data_matches_raw_records": True,
        })

    selection = json.loads((root / "research/g_v4_1/results/evaluation/selection.json").read_text())
    if selection.get("blind_results_used") is not False:
        raise ValueError("blind results were used in model selection")
    if selection.get("selected_arm") != "winner_balanced" or selection.get("promotion_passed") is not False:
        raise ValueError("stored model-selection result changed unexpectedly")
    return {
        "experiment_id": config["experiment_id"],
        "branch": config["branch"],
        "base_commit": config["base_commit"],
        "config_sha256": config["config_sha256"],
        "total_budgeted_training_states": sum(row["budgeted_states"] for row in arm_reports),
        "total_training_games": len(selfplay),
        "unique_training_game_ids": len(global_game_ids),
        "training_arms": arm_reports,
        "evaluation": evaluation_reports,
        "total_evaluation_games": sum(row["games"] for row in evaluation_reports),
        "unique_evaluation_game_ids": len(evaluation_ids),
        "blind_used_for_selection": False,
        "historical_v3_v4_g0_games_replayed": historical_logs["total_games"],
        "historical_v3_v4_g0_budgeted_states": historical_logs["total_budgeted_states"],
        "historical_checkpoint_files": historical_sweep["historical_checkpoint_files"],
        "historical_unique_checkpoint_weights": historical_sweep["unique_checkpoint_weight_files"],
        "historical_checkpoint_hash_aliases_verified": checkpoint_file_count,
        "selected_arm": selection["selected_arm"],
        "promotion_passed": selection["promotion_passed"],
        "all_selfplay_and_evaluation_games_replayed_legal": True,
    }
