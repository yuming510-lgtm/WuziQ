from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.research.alphazero_g.v5_audit import audit_evaluation_game, audit_selfplay_game
from src.research.alphazero_g.v5_evaluation import opening_key, summarize_paired_games
from src.research.alphazero_g.v6_3_training import ARTIFACT, ARM, MILESTONES, validate_training_state


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_historical_training_coverage(prior: dict) -> dict:
    """Describe the verified replay history without equating checkpoint count with replay count."""
    checkpoint = prior.get("current_checkpoint", {})
    baseline = prior.get("baseline_v6_integrity", {})
    baseline_rows = int(baseline.get("selfplay_games_verified", 0))
    added_rows = int(prior.get("v6_2_selfplay_games_verified", 0))
    if (not prior.get("all_completed_games_legal_and_terminal")
            or not checkpoint.get("valid")
            or int(checkpoint.get("game_count", 0)) != 2500
            or int(checkpoint.get("environment_states", 0)) != 45284
            or int(checkpoint.get("optimizer_updates", 0)) != 2499
            or int(prior.get("v6_2_added_games", -1)) != added_rows
            or int(prior.get("v6_2_unique_selfplay_ids", -1)) != added_rows):
        raise ValueError("V6.2 G2500 checkpoint or previously audited replay history is invalid")
    verified_rows = baseline_rows + added_rows
    if verified_rows > 2500:
        raise ValueError("historical replay coverage exceeds the G2500 checkpoint counter")
    return {
        "checkpoint_games": 2500,
        "checkpoint_state_valid": True,
        "replay_games_previously_verified": verified_rows,
        "initial_legacy_games_without_replay_records": 2500 - verified_rows,
    }


def audit(root: Path = ROOT) -> dict:
    artifact = root / ARTIFACT
    state_path = artifact / "checkpoints" / ARM / "training_state.pt"
    state_validation = validate_training_state(state_path)
    games_completed = int(state_validation["game_count"])
    if not 2500 <= games_completed <= 5000:
        raise ValueError("V6.3 current trainer state is outside the registered range")

    raw_dir = artifact / "raw/selfplay" / ARM
    expected_ids = set(range(2500, games_completed))
    training_rows = []
    ids: set[int] = set()
    missing_samples = []
    for sample in raw_dir.glob("*.npz"):
        if not sample.with_suffix(".json").is_file():
            missing_samples.append(str(sample))
    if missing_samples:
        raise ValueError(f"orphan V6.3 training arrays found: {missing_samples[:3]}")
    for path in sorted(raw_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(record.get("game_id", ""))
        index = int(game_id.rsplit("-", 1)[1])
        if game_id != f"{ARM}-game-{index:06d}" or index in ids:
            raise ValueError("training game ID is malformed or duplicated")
        ids.add(index)
        training_rows.append(audit_selfplay_game(path))
    if ids != expected_ids:
        raise ValueError(f"V6.3 new training ID range incomplete: missing={len(expected_ids-ids)}, extra={len(ids-expected_ids)}")
    history_path = artifact / "results/training" / f"{ARM}.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    history_ids = [str(row["game_id"]) for row in history]
    if len(history_ids) != len(set(history_ids)) or set(history_ids) != {f"{ARM}-game-{i:06d}" for i in expected_ids}:
        raise ValueError("training metric history does not match the complete self-play ID range")
    total_new_states = sum(row["samples"] for row in training_rows)
    expected_states = 45284 + total_new_states
    if int(state_validation["environment_states"]) != expected_states:
        raise ValueError("environment-state counter differs from audited training game samples")
    if int(state_validation["optimizer_updates"]) != 2499 + len(expected_ids):
        raise ValueError("Adam update count does not match completed new games")

    milestone_checks = []
    for games in MILESTONES:
        if games_completed < games:
            continue
        full = artifact / "checkpoints" / ARM / f"training_state_games_{games}.pt"
        node = artifact / "checkpoints" / ARM / f"node_games_{games}.pt"
        if not full.is_file() or not node.is_file():
            raise ValueError(f"G{games} model/full-state checkpoint is missing")
        restored = validate_training_state(full, expected_games=games)
        model_payload = torch.load(node, map_location="cpu", weights_only=False)
        state_payload = torch.load(full, map_location="cpu", weights_only=False)
        if (int(model_payload["game_count"]) != games
                or not all(torch.equal(model_payload["model_state"][key], state_payload["model_state"][key])
                           for key in state_payload["model_state"])):
            raise ValueError(f"G{games} model checkpoint differs from full training state")
        milestone_checks.append({"games": games, "full_state_sha256": restored["sha256"],
                                 "model_sha256": _sha(node), **{key: restored[key] for key in
                                 ("model_restored", "adam_restored", "replay_restored", "python_rng_restored",
                                  "numpy_rng_restored", "torch_rng_restored")}})

    schedule_path = artifact / "config/opening_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    openings = schedule["openings"]
    keys = [opening_key(row["moves"], 9) for row in openings]
    if len(openings) != 40 or len(set(keys)) != 40:
        raise ValueError("final opening schedule is not 40 D4-distinct legal positions")
    matches_root = artifact / "results/matches"
    all_match_rows = []
    condition_results = {}
    seen_match_ids: set[str] = set()
    for model in ("G2500", "G5000"):
        for opponent in ("A", "B", "C"):
            condition = f"{model}_vs_{opponent}"
            directory = matches_root / condition
            paths = sorted(directory.glob("v63-*.json")) if directory.exists() else []
            rows = []
            for path in paths:
                audit_row = audit_evaluation_game(path)
                game = json.loads(path.read_text(encoding="utf-8"))
                if game["candidate"] != model or game["opponent"] != opponent:
                    raise ValueError("V6.3 match identity differs from its directory")
                if int(game.get("simulations_per_candidate_turn", 0)) != 400:
                    raise ValueError("V6.3 candidate PUCT-400 accounting differs")
                if audit_row["game_id"] in seen_match_ids:
                    raise ValueError("duplicate V6.3 challenge game ID")
                seen_match_ids.add(audit_row["game_id"])
                rows.append(game)
                all_match_rows.append(audit_row)
            if rows:
                by_opening = {}
                for row in rows:
                    by_opening.setdefault(str(row["opening_group_id"]), []).append(row)
                paired_rows = []
                for opening_id, pair in sorted(by_opening.items()):
                    colors = [str(row["candidate_color"]) for row in pair]
                    if len(colors) != len(set(colors)):
                        raise ValueError(f"challenge pair repeats a color: {opening_id}")
                    if len(pair) == 2 and set(colors) == {"B", "W"}:
                        paired_rows.extend(pair)
                summary = summarize_paired_games(
                    paired_rows, seed=int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16)
                )
                summary["raw_completed_games"] = len(rows)
                expected_games = 80
                summary["complete"] = len(rows) == expected_games and summary["paired_openings"] == 40
                if summary["complete"] and len({row["opening_group_id"] for row in rows}) != 40:
                    raise ValueError("complete condition is missing paired opening diversity")
                condition_results[condition] = summary
            elif games_completed == 5000:
                condition_results[condition] = {"games": 0, "complete": False}
    final_path = artifact / "results/final_challenge.json"
    challenge_complete = False
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        challenge_complete = bool(final.get("complete"))
        for name, summary in condition_results.items():
            if name in final.get("conditions", {}) and int(final["conditions"][name]["games"]) != int(summary["games"]):
                raise ValueError(f"aggregate summary does not match raw records: {name}")
    prior = json.loads((root / "research/g_v6_2/results/integrity_audit.json").read_text(encoding="utf-8"))
    historical_coverage = summarize_historical_training_coverage(prior)
    return {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
        "training_current_state": state_validation,
        "new_training_games_verified": len(training_rows),
        "historical_training_coverage": historical_coverage,
        "total_training_games_covered": 2500 + len(training_rows),
        "new_training_states_verified": total_new_states,
        "cumulative_training_states": expected_states,
        "training_metrics_rows_verified": len(history),
        "milestone_checkpoints": milestone_checks,
        "opening_schedule": {
            "groups": len(openings), "unique_d4_openings": len(set(keys)),
            "sha256": _sha(schedule_path), "legal_and_nonterminal": schedule["validation"]["all_openings_legal_and_nonterminal"],
        },
        "challenge_games_verified": len(all_match_rows),
        "challenge_conditions": condition_results,
        "challenge_complete": challenge_complete,
        "all_new_completed_games_legal_and_terminal": True,
        "all_training_game_ids_unique": True,
        "all_challenge_game_ids_unique": True,
    }


def main() -> None:
    started_cpu = time.process_time()
    result = audit(ROOT)
    result["audit_cpu_seconds"] = time.process_time() - started_cpu
    path = ROOT / ARTIFACT / "results/integrity_audit.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temp.replace(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
