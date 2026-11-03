"""Deterministic, resumable development and one-shot blind evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from .evaluation import append_evaluation_match, load_evaluation_log, summarize_matches
from .training import load_g_checkpoint
from .v3_evaluation import play_v3_match
from .v4_1_config import ARM_IDS, load_v4_1_config
from .v4_1_runner import _cpu_used, _record_cpu


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seed(config_sha: str, game_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{config_sha}:{game_id}".encode()).digest()[:4], "big")


def development_tasks() -> list[dict[str, str]]:
    """Freeze the complete 816-game V4.1 development matrix."""
    tasks: list[dict[str, str]] = []

    def add(candidate: str, opponent: str, mode: str, group: int, color: str) -> None:
        tasks.append({"candidate": candidate, "opponent": opponent, "mode": mode,
                      "group_index": str(group), "agent_color": color})

    # Starting checkpoint versus A and B, both without search and PUCT-64.
    for opponent in ("A", "B"):
        for mode in ("network", "puct"):
            for group in range(12):
                for color in ("B", "W"):
                    add("G0-V3-20K", opponent, mode, group, color)

    # Each trained candidate versus the frozen start, A, and B.
    for candidate in ARM_IDS:
        for opponent in ("G0-V3-20K", "A", "B"):
            for mode in ("network", "puct"):
                for group in range(12):
                    for color in ("B", "W"):
                        add(candidate, opponent, mode, group, color)

    # One PUCT-64 paired series per distinct pair of training arms.
    for left_index, candidate in enumerate(ARM_IDS):
        for opponent in ARM_IDS[left_index + 1:]:
            for group in range(12):
                for color in ("B", "W"):
                    add(candidate, opponent, "puct", group, color)
    return tasks


def blind_tasks(finalist: str) -> list[dict[str, str]]:
    if finalist not in ARM_IDS:
        raise ValueError("blind finalist must be one of the four registered G0 arms")
    tasks: list[dict[str, str]] = []
    for candidate in ("G0-V3-20K", finalist):
        for opponent in ("A", "B"):
            for mode in ("network", "puct"):
                for group in range(20):
                    for color in ("B", "W"):
                        tasks.append({"candidate": candidate, "opponent": opponent, "mode": mode,
                                      "group_index": str(group), "agent_color": color})
    for group in range(20):
        for color in ("B", "W"):
            tasks.append({"candidate": finalist, "opponent": "G0-V3-20K", "mode": "puct",
                          "group_index": str(group), "agent_color": color})
    return tasks


def _checkpoint_path(root: Path, candidate: str, config: dict[str, Any]) -> Path:
    if candidate == "G0-V3-20K":
        return root / str(config["initial_checkpoint"])
    path = root / "research/g_v4_1/checkpoints" / candidate / "node_10000.pt"
    if not path.is_file():
        raise FileNotFoundError(f"candidate checkpoint is missing: {candidate}")
    return path


def _run_tasks(root: Path, phase: str, tasks: Iterable[dict[str, str]], *, finalist: str | None = None,
               max_matches: int | None = None) -> dict[str, Any]:
    config = load_v4_1_config(root)
    schedule = json.loads((root / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    opening_groups = schedule["development_openings"] if phase == "development" else schedule["blind_openings"]
    path = root / f"research/g_v4_1/results/evaluation/{phase}.jsonl"
    progress_path = root / f"research/g_v4_1/results/evaluation/{phase}_progress.json"
    matches, seen = load_evaluation_log(path)
    match_map = {str(row["game_id"]): row for row in matches}
    ledger_path = root / "research/g_v4_1/results/cpu_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"events": []}
    ledger_ids = {str(row["event_id"]) for row in ledger.get("events", [])}
    for match in matches:
        game_id = str(match["game_id"])
        event_id = f"eval:{phase}:{game_id}"
        if event_id not in ledger_ids:
            _record_cpu(root, event_id, float(match["stats"]["total_cpu_seconds"]),
                        float(match["stats"]["total_wall_seconds"]), "evaluation")
    models: dict[str, torch.nn.Module] = {}

    def get_model(candidate: str) -> torch.nn.Module:
        if candidate not in models:
            models[candidate] = load_g_checkpoint(_checkpoint_path(root, candidate, config), device="cpu")
            models[candidate].eval()
        return models[candidate]

    selected_tasks = list(tasks)
    completed_this_call = 0
    for task_index, task in enumerate(selected_tasks):
        group_index = int(task["group_index"])
        group = opening_groups[group_index]
        candidate = task["candidate"]
        opponent = task["opponent"]
        mode = task["mode"]
        color = task["agent_color"]
        comparison_id = f"{phase}:{candidate}:{opponent}:{mode}:{group['opening_group_id']}:{color}"
        game_id = "g-v4-1-" + hashlib.sha256(comparison_id.encode()).hexdigest()[:32]
        if game_id in seen:
            continue
        cap = float(config["compute"]["cpu_hour_cap"]) * 3600
        if _cpu_used(root) + 60.0 >= cap:
            _atomic_json(progress_path, {
                "phase": phase.upper(), "completed": False, "stop_reason": "six_cpu_hour_cap",
                "finalist": finalist, "planned_matches": len(selected_tasks),
                "completed_matches": len(match_map), "config_sha256": config["config_sha256"],
            })
            return {"phase": phase.upper(), "completed": False,
                    "stop_reason": "six_cpu_hour_cap", "completed_matches": len(match_map),
                    "planned_matches": len(selected_tasks)}
        candidate_model = get_model(candidate)
        opponent_model = get_model(opponent) if opponent.startswith("G0-") or opponent in ARM_IDS else None
        opponent_family = "G0" if opponent_model is not None else opponent
        result = play_v3_match(
            candidate_model, opponent_model,
            game_id=game_id, opening_group=group,
            agent_color=color, mode=mode, opponent=opponent_family,
            seed=_seed(str(config["config_sha256"]), game_id), simulations=64,
            candidate_checkpoint_id=candidate,
            opponent_checkpoint_id=opponent if opponent_model is not None else None,
            phase=phase,
        )
        result.update({
            "v4_1_comparison_id": comparison_id,
            "registered_candidate": candidate,
            "registered_opponent": opponent,
            "registered_mode": mode,
            "registered_phase": phase,
            "registered_group_index": group_index,
            "registered_color": color,
            "registration_config_sha256": config["config_sha256"],
            "blind_finalist": finalist if phase == "blind" else None,
        })
        if not append_evaluation_match(path, result, seen):
            raise ValueError(f"duplicate G V4.1 evaluation result: {game_id}")
        _record_cpu(root, f"eval:{phase}:{game_id}",
                    float(result["stats"]["total_cpu_seconds"]),
                    float(result["stats"]["total_wall_seconds"]), "evaluation")
        match_map[game_id] = result
        completed_this_call += 1
        _atomic_json(progress_path, {
            "phase": phase.upper(), "completed": False,
            "finalist": finalist, "planned_matches": len(selected_tasks),
            "completed_matches": len(match_map), "last_game_id": game_id,
            "config_sha256": config["config_sha256"],
        })
        if max_matches is not None and completed_this_call >= max_matches:
            break

    final_matches, _ = load_evaluation_log(path)
    summary = summarize_matches(final_matches) if final_matches else None
    progress = {
        "phase": phase.upper(),
        "completed": len(final_matches) == len(selected_tasks),
        "finalist": finalist,
        "planned_matches": len(selected_tasks),
        "completed_matches": len(final_matches),
        "summary": summary,
        "config_sha256": config["config_sha256"],
    }
    _atomic_json(progress_path, progress)
    return {**progress, "matches_added_this_call": completed_this_call}


def run_development(root: str | Path, *, max_matches: int | None = None) -> dict[str, Any]:
    return _run_tasks(Path(root).resolve(), "development", development_tasks(), max_matches=max_matches)


def select_development_finalist(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v4_1_config(root_path)
    path = root_path / "research/g_v4_1/results/evaluation/development.jsonl"
    matches, _ = load_evaluation_log(path)
    if not matches:
        raise ValueError("development evaluation has not started")
    candidates = []
    for arm_id in ARM_IDS:
        subset = [row for row in matches if row.get("registered_candidate") == arm_id
                  and row.get("registered_opponent") == "G0-V3-20K"
                  and row.get("registered_mode") == "puct"]
        if len(subset) != 24:
            raise ValueError(f"development series for {arm_id} is incomplete ({len(subset)}/24)")
        summary = summarize_matches(subset)
        ci = summary["paired_opening_bootstrap_ci95"]
        promotion = bool(summary["score_rate"] >= 0.625 and ci and ci[0] > 0.5
                         and summary["by_color"]["B"]["score_rate"] >= 0.5
                         and summary["by_color"]["W"]["score_rate"] >= 0.5)
        candidates.append({"arm_id": arm_id, "summary": summary, "promoted": promotion})
    # Deterministic development-only selection. Promotion takes precedence;
    # among the same class choose paired PUCT score, then lower CPU use.
    selected = sorted(candidates, key=lambda row: (
        bool(row["promoted"]), float(row["summary"]["score_rate"]),
        -sum(float(match["stats"]["agent_seconds"]) for match in matches
             if match.get("registered_candidate") == row["arm_id"]
             and match.get("registered_opponent") == "G0-V3-20K"
             and match.get("registered_mode") == "puct"),
        row["arm_id"],
    ), reverse=True)[0]
    result = {
        "selection_rule": "pre-registered developer-only promotion gate then score then CPU",
        "selected_arm": selected["arm_id"],
        "promotion_passed": selected["promoted"],
        "candidate_summaries": candidates,
        "blind_results_used": False,
        "config_sha256": config["config_sha256"],
    }
    _atomic_json(root_path / "research/g_v4_1/results/evaluation/selection.json", result)
    return result


def run_blind(root: str | Path, finalist: str | None = None,
              *, max_matches: int | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    selection_path = root_path / "research/g_v4_1/results/evaluation/selection.json"
    if finalist is None:
        if not selection_path.is_file():
            raise ValueError("development selection must be frozen before blind evaluation")
        finalist = str(json.loads(selection_path.read_text())["selected_arm"])
    return _run_tasks(root_path, "blind", blind_tasks(finalist), finalist=finalist,
                      max_matches=max_matches)
