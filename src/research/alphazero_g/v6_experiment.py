"""Stage runner and development-only checkpoint selection for G V6."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from .v5_evaluation import evaluate_reference_tactics
from .v5_reference import ReferencePolicyValueNet
from .v6_evaluation import run_match_set
from .v6_registration import (
    BLIND_PAIRS,
    CPU_HARD_CAP_SECONDS,
    EVALUATION_BUDGETS,
    EXPERIMENT_ID,
    MILESTONES,
    validate_opening_schedule,
)
from .v6_training import train_v6


ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_OPPONENTS = [
    "G0-V3-510901-5K", "G0-V3-510901-10K", "G0-V3-510901-20K",
]
OPPONENTS = ["ref-v6-initial", *HISTORICAL_OPPONENTS, "A", "B-50ms"]
SELECTION_OPPONENT = "G0-V3-510901-10K"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _load_node(path: Path, board_size: int = 9) -> ReferencePolicyValueNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ReferencePolicyValueNet(board_size, seed=1)
    model.load_policy_param(payload["model_state"])
    model.network.eval()
    return model


def _all_opponent_paths(root: Path) -> dict[str, Path]:
    cfg = _load_json(root / "research/g_v5/config/experiment.json")
    result: dict[str, Path] = {}
    for row in cfg["nine_by_nine"]["historical_checkpoints"]:
        path = root / row["path"]
        if _sha(path) != row["sha256"]:
            raise ValueError(f"frozen historical G0 checkpoint hash mismatch: {row['id']}")
        result[row["id"]] = path
    return result


def run_development_stage(
    root: str | Path,
    milestone: int,
    *,
    schedule: dict[str, Any] | None = None,
    cpu_deadline: float | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    root = Path(root)
    if int(milestone) not in MILESTONES:
        raise ValueError("unregistered V6 training milestone")
    registration = _load_json(root / "research/g_v6/config/registration.json")
    schedule = schedule or _load_json(root / "research/g_v6/config/opening_schedule.json")
    validate_opening_schedule(schedule, root)
    node_path = root / f"research/g_v6/checkpoints/reference400_empty/node_{int(milestone)}.pt"
    if not node_path.is_file():
        raise FileNotFoundError(f"V6 milestone checkpoint is not available: {node_path}")
    checkpoint = torch.load(node_path, map_location="cpu", weights_only=False)
    actual_states = int(checkpoint["environment_states"])
    training_metric = None
    training_metrics_path = root / "research/g_v6/results/training/reference400_empty_metrics.jsonl"
    if training_metrics_path.is_file():
        for line in training_metrics_path.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                if int(item.get("milestone_states", -1)) == int(milestone):
                    training_metric = item
    if training_metric is None:
        raise ValueError(f"V6 training metric row is missing for milestone {milestone}")
    candidate = _load_node(node_path)
    initial_path = root / "research/g_v6/checkpoints/reference400_empty/node_0.pt"
    initial_model = _load_node(initial_path)
    history_paths = _all_opponent_paths(root)
    openings = schedule["development"]
    stage_path = root / f"research/g_v6/results/development/stage_{int(milestone)}.json"
    previous = _load_json(stage_path, {})
    if previous and (previous.get("checkpoint_sha256") != _sha(node_path)
                     or previous.get("development_seed") != schedule["development_seed"]):
        raise ValueError("existing V6 stage results do not match checkpoint/opening registration")
    result = previous or {
        "milestone": int(milestone), "actual_states": actual_states,
        "games_completed": int(checkpoint["game_count"]),
        "optimizer_updates": int(checkpoint["update_count"]),
        "training_metric": training_metric,
        "checkpoint_path": str(node_path.relative_to(root)),
        "checkpoint_sha256": _sha(node_path),
        "development_seed": schedule["development_seed"],
        "opening_group_ids": [row["opening_group_id"] for row in openings],
        "comparisons": {str(budget): {} for budget in EVALUATION_BUDGETS},
        "tactics": {},
    }

    for budget in EVALUATION_BUDGETS:
        budget_key = str(budget)
        result["comparisons"].setdefault(budget_key, {})
        for opponent in OPPONENTS:
            if cpu_deadline is not None and time.process_time() >= cpu_deadline:
                break
            existing = result["comparisons"][budget_key].get(opponent)
            if existing and existing.get("complete"):
                continue
            kwargs: dict[str, Any] = {
                "root": root,
                "phase": f"development-{milestone}",
                "candidate_name": f"ref-v6@{milestone}",
                "opponent_name": opponent,
                "openings": openings,
                "pairs": int(registration["development_pairs"]),
                "candidate_reference_model": candidate,
                "simulations": int(budget),
                "board_size": 9,
                "n_in_row": 5,
                "cpu_deadline": cpu_deadline,
            }
            if opponent == "ref-v6-initial":
                kwargs["opponent_reference_model"] = initial_model
            elif opponent in history_paths:
                kwargs["opponent_g0_checkpoint"] = history_paths[opponent]
            summary = run_match_set(**kwargs)
            result["comparisons"][budget_key][opponent] = summary
            _atomic_json(stage_path, result)
            if progress_callback:
                progress_callback({"phase": "development", "milestone": int(milestone),
                                   "budget": int(budget), "opponent": opponent})
            if not summary.get("complete"):
                break

        if cpu_deadline is not None and time.process_time() >= cpu_deadline:
            break

        tactics_path = root / f"research/g_v6/results/tactics/stage_{int(milestone)}_puct{int(budget)}.json"
        old_tactics = _load_json(tactics_path)
        if old_tactics:
            if old_tactics.get("checkpoint_sha256") != _sha(node_path):
                raise ValueError("V6 tactical stage result belongs to another checkpoint")
            result["tactics"][budget_key] = old_tactics
        else:
            puzzles = _load_json(root / "research/g_v6/config/tactics.json")
            tactical = evaluate_reference_tactics(
                candidate, puzzles, simulations=int(budget),
                seed=696110 + int(milestone) + int(budget),
            )
            tactical.update({
                "milestone": int(milestone), "actual_states": actual_states,
                "checkpoint_sha256": _sha(node_path),
                "budget_role": "search evaluation only; not a training target or model selection input",
            })
            _atomic_json(tactics_path, tactical)
            result["tactics"][budget_key] = tactical
        _atomic_json(stage_path, result)

    expected_opponents = set(OPPONENTS)
    result["complete"] = (
        all(set(result["comparisons"].get(str(budget), {})) == expected_opponents
            and all(row.get("complete") and row.get("games") == 2 * int(registration["development_pairs"])
                    for row in result["comparisons"][str(budget)].values())
            for budget in EVALUATION_BUDGETS)
        and all(str(budget) in result["tactics"] for budget in EVALUATION_BUDGETS)
    )
    _atomic_json(stage_path, result)
    return result


def select_development_candidate(stage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_milestones = set(MILESTONES)
    by_node = {int(row["milestone"]): row for row in stage_rows}
    if set(by_node) != required_milestones:
        raise ValueError("V6 development checkpoint selection requires every registered stage")
    candidates = []
    for milestone in MILESTONES:
        row = by_node[milestone]
        for budget in EVALUATION_BUDGETS:
            comparisons = row.get("comparisons", {}).get(str(budget), {})
            if set(comparisons) != set(OPPONENTS):
                raise ValueError(f"V6 development evaluation is incomplete at {milestone}")
            if any(not value.get("complete") or int(value.get("games", 0)) != 16
                   for value in comparisons.values()):
                raise ValueError(f"V6 development evaluation is incomplete at {milestone}")
        match = row["comparisons"]["64"][SELECTION_OPPONENT]
        candidates.append({
            "milestone": int(milestone),
            "actual_states": int(row["actual_states"]),
            "score_rate": float(match["score_rate"]),
            "paired_ci95": match["paired_opening_bootstrap_ci95"],
            "selection_budget": 64,
            "opponent": SELECTION_OPPONENT,
            "development_seed": int(row["development_seed"]),
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        })
    return max(candidates, key=lambda row: (row["score_rate"], -row["milestone"]))


def freeze_development_candidate(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    selection_path = root / "research/g_v6/results/development/frozen_candidate.json"
    existing = _load_json(selection_path)
    if existing:
        checkpoint = root / existing["checkpoint_path"]
        if _sha(checkpoint) != existing["checkpoint_sha256"]:
            raise ValueError("frozen V6 development candidate checkpoint changed")
        return existing
    stage_rows = []
    for milestone in MILESTONES:
        path = root / f"research/g_v6/results/development/stage_{milestone}.json"
        if not path.is_file():
            raise ValueError("cannot freeze V6 candidate before all development stages finish")
        stage_rows.append(json.loads(path.read_text(encoding="utf-8")))
    selected = select_development_candidate(stage_rows)
    checkpoint = root / selected["checkpoint_path"]
    if not checkpoint.is_file() or _sha(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected V6 development checkpoint is missing or changed")
    selection = {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "frozen_before_blind": True,
        "blind_results_used_for_selection": False,
        "selection_rule": "highest PUCT-64 paired score against frozen G0-V3-510901-10K on development set; ties choose the earlier milestone",
        "selected": selected,
    }
    _atomic_json(selection_path, selection)
    return selection


def run_final_blind(root: str | Path, *, schedule: dict[str, Any] | None = None,
                    cpu_deadline: float | None = None, progress_callback=None) -> dict[str, Any]:
    root = Path(root)
    registration = _load_json(root / "research/g_v6/config/registration.json")
    schedule = schedule or _load_json(root / "research/g_v6/config/opening_schedule.json")
    validate_opening_schedule(schedule, root)
    selection = freeze_development_candidate(root)
    selected = selection["selected"]
    model = _load_node(root / selected["checkpoint_path"])
    rows = _load_json(root / "research/g_v6/results/final_blind.json", {"selection": selection, "matches": {}})
    if rows.get("selection", {}).get("selected", {}).get("checkpoint_sha256") != selected["checkpoint_sha256"]:
        raise ValueError("existing V6 blind results do not match the frozen candidate")
    for budget in EVALUATION_BUDGETS:
        for opponent in ("A", "B-50ms"):
            if cpu_deadline is not None and time.process_time() >= cpu_deadline:
                break
            key = f"{opponent}|puct{budget}"
            existing = rows["matches"].get(key)
            if existing and existing.get("complete"):
                continue
            summary = run_match_set(
                root=root,
                phase="blind-final",
                candidate_name=f"ref-v6@{selected['milestone']}",
                opponent_name=opponent,
                openings=schedule["blind"],
                pairs=BLIND_PAIRS,
                candidate_reference_model=model,
                simulations=int(budget),
                board_size=9,
                n_in_row=5,
                cpu_deadline=cpu_deadline,
            )
            rows["matches"][key] = summary
            rows.update({
                "blind_seed": int(schedule["blind_seed"]),
                "blind_pairs": int(registration["blind_pairs_per_opponent_and_budget"]),
                "games_per_opponent_and_budget": 2 * BLIND_PAIRS,
                "blind_openings_are_new": True,
            })
            _atomic_json(root / "research/g_v6/results/final_blind.json", rows)
            if progress_callback:
                progress_callback({"phase": "blind", "budget": int(budget), "opponent": opponent})
            if not summary.get("complete"):
                break
        if cpu_deadline is not None and time.process_time() >= cpu_deadline:
            break
    rows["complete"] = all(
        rows["matches"].get(f"{opponent}|puct{budget}", {}).get("complete")
        and rows["matches"][f"{opponent}|puct{budget}"].get("games") == 40
        for budget in EVALUATION_BUDGETS for opponent in ("A", "B-50ms")
    )
    _atomic_json(root / "research/g_v6/results/final_blind.json", rows)
    return rows


def _read_progress(path: Path) -> dict[str, Any]:
    return _load_json(path, {"format_version": 1, "experiment_id": EXPERIMENT_ID,
                             "v6_cumulative_cpu_seconds": 0.0, "phase": "registered"})


def run_to_stage(
    root: str | Path,
    through_stage: int,
    *,
    session_cpu_seconds: float | None = None,
) -> dict[str, Any]:
    """Advance one cumulative milestone and its development panel under a global cap."""
    root = Path(root)
    if int(through_stage) not in MILESTONES:
        raise ValueError("--through-stage must be a registered V6 milestone")
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != "research/alphazero-g-v6":
        raise ValueError("V6 runner must execute on research/alphazero-g-v6")
    registration = _load_json(root / "research/g_v6/config/registration.json")
    schedule = _load_json(root / "research/g_v6/config/opening_schedule.json")
    validate_opening_schedule(schedule, root)
    progress_path = root / "research/g_v6/results/progress.json"
    progress = _read_progress(progress_path)
    cpu_before = float(progress.get("v6_cumulative_cpu_seconds", 0.0))
    remaining = CPU_HARD_CAP_SECONDS - cpu_before
    if remaining <= 0:
        return {**progress, "cpu_cap_reached": True, "complete": False}
    session_start = time.process_time()
    allowed = remaining if session_cpu_seconds is None else min(remaining, max(0.0, float(session_cpu_seconds)))
    cpu_deadline = session_start + allowed

    def save_progress(**updates: Any) -> None:
        current_used = cpu_before + (time.process_time() - session_start)
        progress.update({
            "format_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "v6_cumulative_cpu_seconds": current_used,
            "cpu_cap_seconds": CPU_HARD_CAP_SECONDS,
            "active_stage": int(through_stage),
            **updates,
        })
        _atomic_json(progress_path, progress)

    save_progress(phase="training", complete=False)
    training = train_v6(
        root, source_root=root, target_states=int(through_stage), stop_at_cpu_seconds=cpu_deadline,
    )
    save_progress(
        phase="training_complete" if training["completed"] else "training_partial",
        actual_states=int(training["environment_states"]),
        games_completed=int(training["game_count"]),
        optimizer_updates=int(training["update_count"]),
        latest_training_cpu_seconds=float(training["v6_training_cpu_seconds"]),
    )
    if not training["completed"]:
        result = {"milestone": int(through_stage), "training": training,
                  "development": None, "complete": False, "cpu_cap_reached": cpu_before + time.process_time() - session_start >= CPU_HARD_CAP_SECONDS}
        save_progress(phase="training_partial", complete=False)
        return result
    development = run_development_stage(
        root, int(through_stage), schedule=schedule, cpu_deadline=cpu_deadline,
        progress_callback=lambda info: save_progress(**info),
    )
    complete = bool(development.get("complete"))
    result = {"milestone": int(through_stage), "training": training,
              "development": development, "complete": complete,
              "cpu_cap_reached": cpu_before + time.process_time() - session_start >= CPU_HARD_CAP_SECONDS}
    save_progress(
        phase="stage_complete" if complete else "development_partial",
        milestone_complete=complete,
        v6_cumulative_cpu_seconds=cpu_before + time.process_time() - session_start,
    )
    return result


def run_blind_session(root: str | Path, *, session_cpu_seconds: float | None = None) -> dict[str, Any]:
    root = Path(root)
    progress_path = root / "research/g_v6/results/progress.json"
    progress = _read_progress(progress_path)
    cpu_before = float(progress.get("v6_cumulative_cpu_seconds", 0.0))
    remaining = CPU_HARD_CAP_SECONDS - cpu_before
    if remaining <= 0:
        return {"complete": False, "cpu_cap_reached": True}
    session_start = time.process_time()
    allowed = remaining if session_cpu_seconds is None else min(remaining, max(0.0, float(session_cpu_seconds)))
    deadline = session_start + allowed
    result = run_final_blind(
        root, cpu_deadline=deadline,
        progress_callback=lambda info: _atomic_json(progress_path, {
            **progress,
            "phase": "blind", "active_stage": 20_000,
            "v6_cumulative_cpu_seconds": cpu_before + time.process_time() - session_start,
            **info,
        }),
    )
    progress.update({
        "phase": "complete" if result.get("complete") else "blind_partial",
        "complete": bool(result.get("complete")),
        "v6_cumulative_cpu_seconds": cpu_before + time.process_time() - session_start,
        "cpu_cap_seconds": CPU_HARD_CAP_SECONDS,
    })
    _atomic_json(progress_path, progress)
    return result

