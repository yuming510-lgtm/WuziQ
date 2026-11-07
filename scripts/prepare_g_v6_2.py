from __future__ import annotations

import json
import hashlib
from pathlib import Path
import statistics
import time

import torch

from src.research.alphazero_g.v6_2_registration import validate_schedule, write_frozen_registration
from src.research.alphazero_g.v6_2_training import ARTIFACT, prepare_v6_2, ROOT


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temp.replace(path)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))])


def main() -> None:
    torch.set_num_threads(1)
    registration_path = ROOT / ARTIFACT / "config/registration.json"
    if registration_path.is_file():
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        body = dict(registration)
        declared = body.pop("registration_sha256", None)
        actual = hashlib.sha256(json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if actual != declared:
            raise ValueError("frozen V6.2 registration hash is invalid")
        schedule = json.loads((ROOT / ARTIFACT / "config/opening_schedule.json").read_text(encoding="utf-8"))
        validate_schedule(schedule, root=ROOT)
        if registration.get("opening_schedule_sha256") != hashlib.sha256(
            (ROOT / ARTIFACT / "config/opening_schedule.json").read_bytes()
        ).hexdigest():
            raise ValueError("V6.2 opening schedule differs from its frozen registration")
    else:
        registration = write_frozen_registration(ROOT)
    resume = prepare_v6_2(ROOT)
    history_dir = ROOT / "research/g_v6/raw/selfplay/reference400_empty"
    games = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(history_dir.glob("*.json"))]
    if len(games) != 965:
        raise ValueError(f"V6 historical self-play count changed: {len(games)}")
    train_summary = json.loads((ROOT / "research/g_v6/results/training_summary.json").read_text(encoding="utf-8"))
    v6_incremental_training_cpu = float(train_summary["v6_training_cpu_seconds"])
    per_game_values = [float(row["runner_cpu_seconds"]) for row in games]
    tail = per_game_values[-200:]
    remaining_to_2k = 2_000 - 1_012
    remaining_to_3k = 3_000 - 1_012
    v6_cpu_per_game = v6_incremental_training_cpu / len(games)
    # Recent game lengths grew; use the full observed trainer cost as the optimistic
    # rate, then inflate for the recent tail and long-game variance.
    train_rates = {
        "optimistic_seconds_per_game": max(34.0, v6_cpu_per_game),
        "normal_seconds_per_game": max(40.0, v6_cpu_per_game * 1.25),
        "conservative_seconds_per_game": max(55.0, v6_cpu_per_game * 1.80),
    }
    v61_ledger_path = ROOT.parent / "WuziQ-g-v6-1/research/g_v6_1/results/cpu_ledger.json"
    v61_ledger = json.loads(v61_ledger_path.read_text(encoding="utf-8"))
    evaluation_seconds_per_game = float(v61_ledger["formal_cpu_seconds_accounted"]) / len(v61_ledger["accounted_game_ids"])
    projected_eval_games_full = 2_040
    projected_eval_games_at_2k = 1_560
    evaluation_estimates = {
        "optimistic_seconds_per_game": min(5.0, evaluation_seconds_per_game),
        "normal_seconds_per_game": evaluation_seconds_per_game,
        "conservative_seconds_per_game": max(9.0, evaluation_seconds_per_game * 1.45),
    }
    tactics_hours = {"optimistic": 0.15, "normal": 0.25, "conservative": 0.40}
    audit_and_report_hours = {"optimistic": 0.25, "normal": 0.40, "conservative": 0.60}
    estimates = {}
    for label, prefix in (("optimistic", "optimistic"), ("normal", "normal"), ("conservative", "conservative")):
        train_per_game = train_rates[f"{prefix}_seconds_per_game"]
        eval_per_game = evaluation_estimates[f"{prefix}_seconds_per_game"]
        upto_2k = (remaining_to_2k * train_per_game + projected_eval_games_at_2k * eval_per_game) / 3600
        upto_3k = (remaining_to_3k * train_per_game + projected_eval_games_full * eval_per_game) / 3600
        estimates[label] = {
            "to_2000_total_cpu_hours": upto_2k + tactics_hours[label] + audit_and_report_hours[label],
            "to_3000_total_cpu_hours_if_gate_passes": upto_3k + tactics_hours[label] + audit_and_report_hours[label],
            "training_hours_to_2000": remaining_to_2k * train_per_game / 3600,
            "training_hours_to_3000": remaining_to_3k * train_per_game / 3600,
            "evaluation_games_to_2000": projected_eval_games_at_2k,
            "evaluation_games_to_3000_including_blind": projected_eval_games_full,
            "evaluation_hours_to_3000_including_blind": projected_eval_games_full * eval_per_game / 3600,
            "tactics_hours": tactics_hours[label],
            "audit_report_hours": audit_and_report_hours[label],
        }
    preflight = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resource_snapshot": registration["resource_snapshot"],
        "environment": {
            "logical_cpus": 3,
            "cpu_quota_cores": 2,
            "memory_limit_bytes": 8_589_934_592,
            "disk_free_bytes": registration["resource_snapshot"]["disk_free_bytes_at_registration"],
            "torch": torch.__version__,
            "device": "CPU; CUDA unavailable",
            "torch_threads": 1,
        },
        "restore_validation": resume,
        "history_basis": {
            "v6_added_games": len(games),
            "v6_added_training_cpu_seconds": v6_incremental_training_cpu,
            "v6_full_incremental_cpu_seconds_per_game": v6_cpu_per_game,
            "last_200_game_runner_cpu_mean": statistics.mean(tail),
            "last_200_game_runner_cpu_median": statistics.median(tail),
            "last_200_game_runner_cpu_p90": _percentile(tail, 0.9),
            "v61_evaluation_cpu_seconds_per_game": evaluation_seconds_per_game,
            "v61_evaluation_games": len(v61_ledger["accounted_game_ids"]),
        },
        "estimate_method": "Use V6 measured self-play plus optimizer CPU/game, V6.1 actual paired-evaluation CPU/game, and explicit tactic/audit reserves; projected conditions are derived from this registered schedule.",
        "remaining_new_training_games_to_2000": remaining_to_2k,
        "remaining_new_training_games_to_3000": remaining_to_3k,
        "estimates_cpu_hours": estimates,
        "session_cpu_cap_hours": 10,
        "total_new_cpu_cap_hours": 30,
        "status": "Preflight complete. Begin training from the verified V6 full state; use game-boundary resumable sessions.",
    }
    preflight_path = ROOT / ARTIFACT / "results/preflight.json"
    if preflight_path.is_file():
        previous = json.loads(preflight_path.read_text(encoding="utf-8"))
        if previous.get("actual_benchmark"):
            preflight["actual_benchmark"] = previous["actual_benchmark"]
            preflight["status"] = previous.get("status", preflight["status"])
    _write_json(preflight_path, preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
