"""Registered V5 experiment runner and all paired evaluation routes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch

from ..imitation_f.model import load_f_checkpoint
from .v5_evaluation import (
    evaluate_reference_tactics,
    generate_six_by_six_tactics,
    play_match,
    summarize_paired_games,
)
from .v5_reference import ReferencePolicyValueNet
from .v5_training import train_reference_arm


ROOT = Path(__file__).resolve().parents[3]
REF_REPO = "third_party/AlphaZero_Gomoku"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "big")


def _load_reference_node(path: Path, board_size: int) -> ReferencePolicyValueNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ReferencePolicyValueNet(board_size, seed=1)
    model.load_policy_param(checkpoint["model_state"])
    model.network.eval()
    return model


def _append_match_set(
    *,
    root: Path,
    phase: str,
    candidate_name: str,
    opponent_name: str,
    openings: list[dict[str, Any]],
    pairs: int,
    candidate_reference_model: ReferencePolicyValueNet | None = None,
    opponent_reference_model: ReferencePolicyValueNet | None = None,
    candidate_g0_checkpoint: Path | None = None,
    opponent_g0_checkpoint: Path | None = None,
    simulations: int = 64,
    board_size: int = 9,
    n_in_row: int = 5,
    cpu_deadline: float | None = None,
) -> dict[str, Any]:
    opening_rows = openings[:int(pairs)]
    rows = []
    base = root / "research/g_v5/results/matches" / _slug(phase) / f"{_slug(candidate_name)}_vs_{_slug(opponent_name)}"
    for opening in opening_rows:
        # Always finish a paired color swap; then yield at an idempotent point.
        if cpu_deadline is not None and time.process_time() >= cpu_deadline:
            break
        for color in ("B", "W"):
            game_id = "v5m-" + _stable_seed(phase, candidate_name, opponent_name,
                                            str(opening["opening_group_id"]), color).__format__("08x")
            path = base / f"{game_id}.json"
            if path.exists():
                game = json.loads(path.read_text(encoding="utf-8"))
                if game.get("game_id") != game_id:
                    raise ValueError("V5 game ID collision")
            else:
                game = play_match(
                    game_id=game_id,
                    candidate_name=candidate_name,
                    opponent_name=opponent_name,
                    opening=opening,
                    candidate_color=color,
                    seed=_stable_seed(game_id, str(opening.get("seed", 0))),
                    candidate_reference_model=candidate_reference_model,
                    opponent_reference_model=opponent_reference_model,
                    candidate_g0_checkpoint=candidate_g0_checkpoint,
                    opponent_g0_checkpoint=opponent_g0_checkpoint,
                    simulations=int(simulations),
                    board_size=int(board_size),
                    n_in_row=int(n_in_row),
                )
                _atomic_json(path, game)
            rows.append(game)
        _write_progress(root, phase=phase, last_completed_match_pair=opening["opening_group_id"],
                        total_match_games=sum(1 for _ in (root / "research/g_v5/results/matches").rglob("v5m-*.json")))
    summary = summarize_paired_games(rows, seed=_stable_seed(phase, candidate_name, opponent_name))
    summary.update({"phase": phase, "candidate": candidate_name, "opponent": opponent_name,
                    "simulations_per_reference_turn": int(simulations),
                    "opening_group_ids": [row["opening_group_id"] for row in opening_rows]})
    summary_path = base / "summary.json"
    _atomic_json(summary_path, summary)
    return summary


def _write_progress(root: Path, **fields: Any) -> None:
    path = root / "research/g_v5/results/progress.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(fields)
    _atomic_json(path, existing)


def _config(root: Path) -> dict[str, Any]:
    config = json.loads((root / "research/g_v5/config/experiment.json").read_text(encoding="utf-8"))
    expected = config.get("config_sha256")
    canonical = dict(config)
    canonical.pop("config_sha256", None)
    actual = _hash_bytes(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    if not expected or expected != actual:
        raise ValueError("V5 config is not registered or its SHA-256 changed")
    return config


def _record_tactics(root: Path, arm: str, node: int, model: ReferencePolicyValueNet,
                    puzzles: list[dict[str, Any]], simulations: int) -> dict[str, Any]:
    summary = evaluate_reference_tactics(model, puzzles, simulations=simulations,
                                          seed=_stable_seed(arm, str(node), str(simulations)))
    path = root / "research/g_v5/results/tactics" / f"{_slug(arm)}_node_{node}_puct{simulations}.json"
    _atomic_json(path, summary)
    return summary


def _run_6x6(root: Path, config: dict[str, Any], schedule: dict[str, Any],
             puzzles: list[dict[str, Any]], cpu_stop: float) -> dict[str, Any]:
    settings = config["six_by_six"]
    arm = "reference6x6-400-empty"
    model = ReferencePolicyValueNet(6, seed=int(settings["network_seed"]))
    init_path = root / "research/g_v5/checkpoints" / arm / "node_0.pt"
    if init_path.exists():
        initial = _load_reference_node(init_path, 6)
    else:
        initial = model
    train_result = train_reference_arm(
        root, arm=arm, board_size=6, n_in_row=4,
        simulations=int(settings["simulations_per_move"]),
        target_states=10_000, milestones=[m for m in settings["milestones"] if m <= 10_000],
        seed=int(settings["training_seed"]), config=config,
        stop_at_cpu_seconds=cpu_stop,
    )
    model_dir = root / "research/g_v5/checkpoints" / arm
    openings = schedule["six_by_six"]["development"]
    stage_rows = []
    for milestone in (2000, 5000, 10000):
        node_path = model_dir / f"node_{milestone}.pt"
        if not node_path.exists():
            continue
        candidate = _load_reference_node(node_path, 6)
        pairs = 20 if milestone in (5000, 10000) else int(settings["development_pairs_at_intermediate_nodes"])
        match = _append_match_set(
            root=root, phase=f"6x6-dev-node-{milestone}",
            candidate_name=f"ref6x6@{milestone}", opponent_name="ref6x6@0",
            openings=openings, pairs=pairs,
            candidate_reference_model=candidate, opponent_reference_model=initial,
            simulations=64, board_size=6, n_in_row=4, cpu_deadline=cpu_stop,
        )
        puzzle = _record_tactics(root, arm, milestone, candidate, puzzles, 64)
        stage_rows.append({"node": milestone, "match": match,
                           "network_tactics": puzzle["network_accuracy"],
                           "puct64_tactics": puzzle["puct_accuracy"]})
    gates = [row for row in stage_rows if row["node"] in (5000, 10000)]
    gate_passed = len(gates) == 2 and all(
        row["match"]["score_rate"] >= float(settings["continue_to_9x9_gate"]["candidate_min_score_rate_vs_frozen_random_initial"])
        and row["match"]["paired_opening_bootstrap_ci95"][0] > float(settings["continue_to_9x9_gate"]["paired_ci95_lower_above"])
        for row in gates
    )
    if not gate_passed and train_result.get("completed") and time.process_time() < cpu_stop:
        train_result = train_reference_arm(
            root, arm=arm, board_size=6, n_in_row=4,
            simulations=int(settings["simulations_per_move"]),
            target_states=int(settings["maximum_environment_states"]),
            milestones=[m for m in settings["milestones"] if m > 10_000],
            seed=int(settings["training_seed"]), config=config,
            stop_at_cpu_seconds=cpu_stop,
        )
        for milestone in (15000, 20000):
            node_path = model_dir / f"node_{milestone}.pt"
            if not node_path.exists():
                continue
            candidate = _load_reference_node(node_path, 6)
            match = _append_match_set(
                root=root, phase=f"6x6-dev-node-{milestone}",
                candidate_name=f"ref6x6@{milestone}", opponent_name="ref6x6@0",
                openings=openings, pairs=20,
                candidate_reference_model=candidate, opponent_reference_model=initial,
                simulations=64, board_size=6, n_in_row=4, cpu_deadline=cpu_stop,
            )
            puzzle = _record_tactics(root, arm, milestone, candidate, puzzles, 64)
            stage_rows.append({"node": milestone, "match": match,
                               "network_tactics": puzzle["network_accuracy"],
                               "puct64_tactics": puzzle["puct_accuracy"]})
    if (not gate_passed
            and int(train_result.get("environment_states", 0)) < int(settings["maximum_environment_states"])):
        result = {
            "arm": train_result, "stages": stage_rows,
            "continue_to_9x9_gate_passed": False, "gate_nodes": [],
            "dev_selected_node": None, "blind_result": None,
            "partial": True,
        }
        _atomic_json(root / "research/g_v5/results/six_by_six_summary.json", result)
        return result
    if not train_result.get("completed"):
        result = {
            "arm": train_result, "stages": stage_rows,
            "continue_to_9x9_gate_passed": False, "gate_nodes": [],
            "dev_selected_node": None, "blind_result": None,
            "partial": True,
        }
        _atomic_json(root / "research/g_v5/results/six_by_six_summary.json", result)
        return result
    late = [row for row in stage_rows if row["node"] in (10000, 15000, 20000)]
    gate_nodes = late[-2:]
    gate_passed = len(gate_nodes) == 2 and all(
        row["match"]["score_rate"] >= float(settings["continue_to_9x9_gate"]["candidate_min_score_rate_vs_frozen_random_initial"])
        and row["match"]["paired_opening_bootstrap_ci95"][0] > float(settings["continue_to_9x9_gate"]["paired_ci95_lower_above"])
        for row in gate_nodes
    )
    selection = max(stage_rows, key=lambda row: (row["match"]["score_rate"], -row["node"])) if stage_rows else None
    frozen_node = int(selection["node"]) if selection else 0
    if frozen_node > 0:
        frozen_model = _load_reference_node(model_dir / f"node_{frozen_node}.pt", 6)
        blind = _append_match_set(
            root=root, phase="6x6-blind-final", candidate_name=f"ref6x6@{frozen_node}",
            opponent_name="ref6x6@0", openings=schedule["six_by_six"]["blind"],
            pairs=int(settings["blind_pairs"]), candidate_reference_model=frozen_model,
            opponent_reference_model=initial, simulations=64, board_size=6, n_in_row=4,
            cpu_deadline=cpu_stop,
        )
    else:
        blind = None
    result = {
        "arm": train_result,
        "stages": stage_rows,
        "continue_to_9x9_gate_passed": gate_passed,
        "gate_nodes": [int(row["node"]) for row in gate_nodes],
        "dev_selected_node": frozen_node,
        "blind_result": blind,
    }
    _atomic_json(root / "research/g_v5/results/six_by_six_summary.json", result)
    return result


def _historical_9x9(root: Path, config: dict[str, Any], schedule: dict[str, Any],
                    cpu_stop: float) -> dict[str, Any]:
    rows = []
    checkpoints = config["nine_by_nine"]["historical_checkpoints"]
    opening_groups = schedule["nine_by_nine"]["historical"]
    for row in checkpoints:
        path = root / row["path"]
        if _hash_file(path) != row["sha256"]:
            raise ValueError(f"frozen historical G0 checkpoint hash mismatch: {row['id']}")
        for opponent in ("A", "B"):
            if time.process_time() >= cpu_stop:
                break
            summary = _append_match_set(
                root=root, phase="historical-dev", candidate_name=row["id"],
                opponent_name=opponent, openings=opening_groups,
                pairs=int(config["nine_by_nine"]["historical_pairs"]),
                candidate_g0_checkpoint=path, simulations=64, cpu_deadline=cpu_stop,
            )
            rows.append(summary)
    direct = []
    by_id = {row["id"]: root / row["path"] for row in checkpoints}
    names = config["nine_by_nine"]["historical_direct_comparison"]
    direct.append(_append_match_set(
        root=root, phase="historical-direct-10k-vs-20k",
        candidate_name=names[0], opponent_name=names[1], openings=opening_groups,
        pairs=int(config["nine_by_nine"]["historical_pairs"]),
        candidate_g0_checkpoint=by_id[names[0]], opponent_g0_checkpoint=by_id[names[1]],
        simulations=64, cpu_deadline=cpu_stop,
    ))
    complete = all(row.get("games", 0) == 2 * int(config["nine_by_nine"]["historical_pairs"]) for row in rows)
    complete = complete and all(row.get("games", 0) == 2 * int(config["nine_by_nine"]["historical_pairs"]) for row in direct)
    result = {"comparisons_vs_A_B": rows, "direct_checkpoint_matches": direct,
              "historical_checkpoint_ids": [row["id"] for row in checkpoints],
              "evaluation_simulations": 64, "c_puct": 5.0, "complete": complete}
    _atomic_json(root / "research/g_v5/results/historical_9x9.json", result)
    return result


def _run_9x9(root: Path, config: dict[str, Any], schedule: dict[str, Any],
             cpu_stop: float) -> dict[str, Any]:
    nine = config["nine_by_nine"]
    histories = {row["id"]: root / row["path"] for row in nine["historical_checkpoints"]}
    champion_name = "G0-V3-510901-20K"
    champion_path = histories[champion_name]
    dev_openings = schedule["nine_by_nine"]["development"]
    arm_specs = nine["arms"]
    shared_seed = int(nine["network_seed"])
    random_model = ReferencePolicyValueNet(9, seed=shared_seed)
    initial_state = {key: value.detach().cpu().clone() for key, value in random_model.get_policy_param().items()}
    midgame_pool = json.loads((root / "research/g_v5/config/historical_midgame_prefixes.json").read_text(encoding="utf-8"))
    results = []
    arm_order = ["reference64_empty", "reference200_empty", "reference64_historical_midgame", "reference400_empty"]
    for arm in arm_order:
        if time.process_time() >= cpu_stop:
            break
        spec = arm_specs[arm]
        provider = None
        if arm == "reference64_historical_midgame":
            def provider(game_index: int, game_seed: int, pool=midgame_pool):
                choice = int(np.random.default_rng(game_seed).integers(0, len(pool)))
                return list(pool[choice]["moves"])
        settings = train_reference_arm(
            root, arm=arm, board_size=9, n_in_row=5,
            simulations=int(spec["simulations_per_move"]),
            target_states=int(spec["maximum_environment_states"]),
            milestones=list(map(int, spec["milestones"])),
            seed=int(nine["training_seed"]), config=config,
            opening_provider=provider, initial_model_state=initial_state,
            stop_at_cpu_seconds=cpu_stop,
        )
        arm_dir = root / "research/g_v5/checkpoints" / arm
        stage_results = []
        for milestone in spec["milestones"]:
            node_path = arm_dir / f"node_{milestone}.pt"
            if not node_path.exists():
                continue
            candidate = _load_reference_node(node_path, 9)
            node_name = f"{arm}@{milestone}"
            opponent_models = [("ref9x9@0", _load_reference_node(arm_dir / "node_0.pt", 9), None),
                               (champion_name, None, champion_path), ("A", None, None), ("B", None, None)]
            for opponent_name, opponent_ref, opponent_g0 in opponent_models:
                summary = _append_match_set(
                    root=root, phase=f"9x9-dev-{arm}-node-{milestone}",
                    candidate_name=node_name, opponent_name=opponent_name,
                    openings=dev_openings, pairs=int(nine["development_pairs"]),
                    candidate_reference_model=candidate,
                    opponent_reference_model=opponent_ref,
                    opponent_g0_checkpoint=opponent_g0,
                    simulations=64, cpu_deadline=cpu_stop,
                )
                stage_results.append(summary)
            stage_results.append({"node": int(milestone), "actual_states": settings["environment_states"]})
        result = {"arm": arm, "training": settings, "stages": stage_results,
                  "simulations_per_move": spec["simulations_per_move"],
                  "opening_distribution": spec["opening_distribution"]}
        results.append(result)
        _atomic_json(root / "research/g_v5/results/9x9_arms.json", results)
    complete_count = sum(bool(row["training"].get("completed")) for row in results)
    return {"arms": results, "complete_arm_count": complete_count,
            "planned_arm_count": len(arm_order)}


def _all_dev_candidates(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    arms = config["nine_by_nine"]["arms"]
    for arm, spec in arms.items():
        arm_dir = root / "research/g_v5/checkpoints" / arm
        for milestone in spec["milestones"]:
            path = arm_dir / f"node_{milestone}.pt"
            if not path.exists():
                continue
            candidate_id = f"{arm}@{milestone}"
            summary_path = root / "research/g_v5/results/matches" / _slug(f"9x9-dev-{arm}-node-{milestone}") / f"{_slug(candidate_id)}_vs_{_slug('G0-V3-510901-20K')}" / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(summary.get("games", 0)) != 2 * int(config["nine_by_nine"]["development_pairs"]):
                continue
            candidates.append({"candidate_id": candidate_id, "arm": arm,
                               "milestone": int(milestone), "checkpoint_path": str(path.relative_to(root)),
                               "checkpoint_sha256": _hash_file(path),
                               "paired_score_rate_vs_v3_champion": float(summary["score_rate"]),
                               "paired_ci95": summary["paired_opening_bootstrap_ci95"],
                               "actual_states": int(torch.load(path, map_location="cpu", weights_only=False)["environment_states"])})
    return candidates


def _final_9x9(root: Path, config: dict[str, Any], schedule: dict[str, Any],
               cpu_stop: float, arm_results: dict[str, Any]) -> dict[str, Any]:
    candidates = _all_dev_candidates(root, config)
    if not candidates:
        return {"skipped": True, "reason": "no 9x9 candidate has the registered development comparison"}
    # Selection is development-only; prefer score vs the frozen V3 champion,
    # then a smaller training budget when development scores tie.
    selected = max(candidates, key=lambda row: (row["paired_score_rate_vs_v3_champion"], -row["actual_states"]))
    path = root / selected["checkpoint_path"]
    model = _load_reference_node(path, 9)
    blind_openings = schedule["nine_by_nine"]["blind"]
    frozen = []
    checkpoint_by_id = {row["id"]: root / row["path"] for row in config["nine_by_nine"]["historical_checkpoints"]}
    for opponent in ("A", "B", "G0-V3-510901-5K", "G0-V3-510901-10K",
                     "G0-V3-510901-20K", "G0-V4.2-history-pool-10K"):
        if time.process_time() >= cpu_stop:
            break
        opponent_path = checkpoint_by_id.get(opponent)
        summary = _append_match_set(
            root=root, phase="9x9-blind-final", candidate_name=selected["candidate_id"],
            opponent_name=opponent, openings=blind_openings,
            pairs=int(config["nine_by_nine"]["blind_pairs"]),
            candidate_reference_model=model, opponent_g0_checkpoint=opponent_path,
            simulations=64, cpu_deadline=cpu_stop,
        )
        frozen.append(summary)
    historical_blind = []
    # Same untouched blind openings also re-check whether V3 5K/10K/20K and the
    # V4.2 10K history-pool model rank differently than on the development set.
    for candidate in ("G0-V3-510901-5K", "G0-V3-510901-10K", "G0-V3-510901-20K"):
        for opponent in ("A", "B"):
            if time.process_time() >= cpu_stop:
                break
            historical_blind.append(_append_match_set(
                root=root, phase="9x9-blind-historical",
                candidate_name=candidate, opponent_name=opponent,
                openings=blind_openings, pairs=int(config["nine_by_nine"]["blind_pairs"]),
                candidate_g0_checkpoint=checkpoint_by_id[candidate], simulations=64,
                cpu_deadline=cpu_stop,
            ))
    # Same trained checkpoint, same development openers, only search budget
    # varies. This isolates inference/search gain from the training arms.
    search_budget_rows = []
    for budget in (64, 200, 400):
        if time.process_time() >= cpu_stop:
            break
        search_budget_rows.append(_append_match_set(
            root=root, phase=f"9x9-search-budget-{budget}",
            candidate_name=selected["candidate_id"], opponent_name="A",
            openings=schedule["nine_by_nine"]["development"],
            pairs=int(config["nine_by_nine"]["development_pairs"]),
            candidate_reference_model=model, simulations=budget, cpu_deadline=cpu_stop,
        ))
    puzzle_path = root / "research/g_v4_2/config/tactical_puzzles.json"
    puzzle_data = json.loads(puzzle_path.read_text(encoding="utf-8"))
    puzzle_budgets = []
    for budget in (64, 200, 400):
        if time.process_time() >= cpu_stop:
            break
        puzzle_budgets.append(evaluate_reference_tactics(
            model, puzzle_data, simulations=budget,
            seed=_stable_seed(selected["candidate_id"], "nine-puzzle", str(budget)),
        ))
    result = {
        "development_candidates": candidates,
        "selection_rule": "highest development PUCT-64 score vs frozen G0-V3-510901-20K; tie favors fewer environment states",
        "selected_candidate": selected,
        "blind_results": frozen,
        "historical_g0_blind_results": historical_blind,
        "same_checkpoint_search_budget_vs_A_dev": search_budget_rows,
        "same_checkpoint_tactical_search_budget": puzzle_budgets,
        "blind_untouched_until_candidate_freeze": True,
        "arms_complete": arm_results.get("complete_arm_count"),
        "complete": (
            len(frozen) == 6 and all(row.get("games", 0) == 2 * int(config["nine_by_nine"]["blind_pairs"]) for row in frozen)
            and len(historical_blind) == 6 and all(row.get("games", 0) == 2 * int(config["nine_by_nine"]["blind_pairs"]) for row in historical_blind)
            and len(search_budget_rows) == 3 and all(row.get("games", 0) == 2 * int(config["nine_by_nine"]["development_pairs"]) for row in search_budget_rows)
        ),
    }
    _atomic_json(root / "research/g_v5/results/final_9x9.json", result)
    return result


def run_v5(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT).resolve()
    config = _config(root)
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    cpu_started = time.process_time()
    progress_path = root / "research/g_v5/results/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    if progress.get("config_sha256") not in (None, config["config_sha256"]):
        raise ValueError("existing V5 progress belongs to a different registered config")
    training_cpu = 0.0
    checkpoints = root / "research/g_v5/checkpoints"
    if checkpoints.exists():
        for state_path in checkpoints.glob("*/training_state.pt"):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            training_cpu += float(state.get("cumulative_cpu_seconds", 0.0))
    match_cpu = 0.0
    match_root = root / "research/g_v5/results/matches"
    if match_root.exists():
        for game_path in match_root.rglob("v5m-*.json"):
            match_cpu += float(json.loads(game_path.read_text(encoding="utf-8")).get("total_cpu_seconds", 0.0))
    preflight_path = root / "research/g_v5/results/preflight.json"
    preflight_cpu = float(json.loads(preflight_path.read_text(encoding="utf-8")).get("preflight_cpu_seconds", 0.0)) if preflight_path.exists() else 0.0
    spent_cpu = training_cpu + match_cpu + preflight_cpu
    remaining = max(0.0, float(config["compute"]["cpu_hour_hard_cap"]) * 3600.0
                    - float(config["compute"]["cpu_hour_reserve_for_final_audit"]) * 3600.0 - spent_cpu)
    session_limit = float(os.environ.get("WUZIQ_V5_SESSION_CPU_SECONDS", "inf"))
    cpu_stop = cpu_started + min(remaining, session_limit)
    schedule = json.loads((root / "research/g_v5/config/opening_schedule.json").read_text(encoding="utf-8"))
    puzzles6 = json.loads((root / "research/g_v5/config/six_by_six_tactics.json").read_text(encoding="utf-8"))
    _write_progress(root, experiment_id=config["experiment_id"], config_sha256=config["config_sha256"],
                    phase="starting", completed=False, cpu_hour_cap=config["compute"]["cpu_hour_hard_cap"],
                    spent_cpu_seconds_before_run=spent_cpu, cpu_remaining_seconds=remaining,
                    started_cpu_seconds=cpu_started)
    six_result = _run_6x6(root, config, schedule, puzzles6, cpu_stop)
    _write_progress(root, phase="six_by_six_complete", six_by_six_gate=six_result["continue_to_9x9_gate_passed"],
                    cumulative_cpu_seconds=spent_cpu + time.process_time() - cpu_started)

    if six_result.get("partial"):
        summary = {"format_version": 1, "experiment_id": config["experiment_id"],
                   "config_sha256": config["config_sha256"], "partial": True,
                   "cpu_seconds_this_session": time.process_time() - cpu_started,
                   "six_by_six": six_result}
        _atomic_json(root / "research/g_v5/results/experiment_summary.json", summary)
        return summary

    historical = _historical_9x9(root, config, schedule, cpu_stop)
    _write_progress(root, phase="historical_g0_remeasurement_complete", cumulative_cpu_seconds=spent_cpu + time.process_time() - cpu_started)
    if not historical["complete"]:
        summary = {"format_version": 1, "experiment_id": config["experiment_id"],
                   "config_sha256": config["config_sha256"], "partial": True,
                   "cpu_seconds_this_session": time.process_time() - cpu_started,
                   "six_by_six": six_result, "historical_g0_remeasurement": historical}
        _atomic_json(root / "research/g_v5/results/experiment_summary.json", summary)
        return summary
    if six_result["continue_to_9x9_gate_passed"]:
        arms = _run_9x9(root, config, schedule, cpu_stop)
        _write_progress(root, phase="nine_by_nine_training_complete", nine_by_nine_arms=arms,
                        cumulative_cpu_seconds=spent_cpu + time.process_time() - cpu_started)
        if arms.get("complete_arm_count") != arms.get("planned_arm_count"):
            summary = {"format_version": 1, "experiment_id": config["experiment_id"],
                       "config_sha256": config["config_sha256"], "partial": True,
                       "cpu_seconds_this_session": time.process_time() - cpu_started,
                       "six_by_six": six_result, "historical_g0_remeasurement": historical,
                       "nine_by_nine_arms": arms}
            _atomic_json(root / "research/g_v5/results/experiment_summary.json", summary)
            return summary
        final = _final_9x9(root, config, schedule, cpu_stop, arms)
    else:
        arms = {"skipped": True, "reason": "6x6 learning gate did not pass"}
        final = {"skipped": True, "reason": "6x6 learning gate did not pass"}
    complete = (bool(historical["complete"])
                and (not six_result["continue_to_9x9_gate_passed"] or
                     (not isinstance(arms, dict) or arms.get("complete_arm_count") == arms.get("planned_arm_count")))
                and (not six_result["continue_to_9x9_gate_passed"] or
                     (not final.get("skipped") and bool(final.get("complete")))))
    _write_progress(root, phase="completed" if complete else "budget_exhausted",
                    completed=complete,
                    cumulative_cpu_seconds=spent_cpu + time.process_time() - cpu_started,
                    six_by_six_gate=six_result["continue_to_9x9_gate_passed"],
                    nine_by_nine_training=arms, final_9x9=final)
    summary = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config["config_sha256"],
        "cpu_seconds": spent_cpu + time.process_time() - cpu_started,
        "cpu_seconds_this_session": time.process_time() - cpu_started,
        "preflight_cpu_seconds": preflight_cpu,
        "training_cpu_seconds": training_cpu,
        "evaluation_cpu_seconds": match_cpu,
        "completed": complete,
        "six_by_six": six_result,
        "historical_g0_remeasurement": historical,
        "nine_by_nine_arms": arms,
        "final_9x9": final,
    }
    _atomic_json(root / "research/g_v5/results/experiment_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_v5(), ensure_ascii=False, indent=2))
