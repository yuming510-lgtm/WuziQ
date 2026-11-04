"""Registration, preflight and staged execution for V6.1 (evaluation only)."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .v5_evaluation import opening_key
from .v6_1_tactics import _Grid, evaluate_tactics, generate_tactical_suite, validate_tactical_suite
from .v6_1_tournament import (
    BRANCH,
    CHECKPOINTS,
    CHALLENGE_PAIRS,
    CPU_LIMIT_SECONDS,
    DIRECT_PAIRS,
    EXPERIMENT_ID,
    ROOT,
    SIMULATION_BUDGETS,
    TACTIC_COUNT,
    _atomic_json,
    _sha,
    _stable_id,
    audit_v61_games,
    build_openings,
    load_frozen_checkpoints,
    make_selection,
    play_v61_match,
    run_match_set,
)


def _checkpoint_file_manifest(root: Path, name: str) -> tuple[str, int]:
    path = root / CHECKPOINTS[name][0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _sha(path), int(payload["environment_states"])


def collect_historical_tactical_keys(root: str | Path) -> set[str]:
    """Collect D4-canonical prior training/evaluation positions to exclude."""
    root = Path(root).resolve()
    keys: set[str] = set()
    def add_moves(moves: Any, to_move: str | None = None) -> None:
        if not isinstance(moves, list) or not moves:
            return
        grid = _Grid()
        expected = "B"
        for item in moves:
            if not isinstance(item, dict) or not {"player", "row", "col"} <= item.keys():
                return
            color = str(item["player"])
            if color != expected:
                return
            try:
                grid.place(color, (int(item["row"]), int(item["col"])))
            except (ValueError, KeyError, TypeError):
                return
            expected = "W" if expected == "B" else "B"
            if not grid.has_any_win():
                try:
                    keys.add(grid.key(expected))
                except (ValueError, RuntimeError):
                    pass
            else:
                break
    for path in (root / "research").rglob("*.json"):
        if "g_v6_1" in path.parts or "checkpoints" in path.parts:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            size = value.get("board_size", value.get("size", 9))
            if int(size or 9) == 9:
                if isinstance(value.get("moves"), list):
                    add_moves(value["moves"])
                if isinstance(value.get("opening"), list):
                    add_moves(value["opening"])
                if isinstance(value.get("position"), list):
                    add_moves(value["position"])
            # Registered puzzle collections often put positions in lists.
            for key in ("puzzles", "tactics", "positions", "items"):
                rows = value.get(key)
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict) and int(row.get("board_size", 9)) == 9:
                            add_moves(row.get("moves", row.get("position", [])))
        elif isinstance(value, list):
            for row in value:
                if isinstance(row, dict) and int(row.get("board_size", 9)) == 9:
                    add_moves(row.get("moves", row.get("position", [])))
    return keys


def register_v61(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                      capture_output=True, text=True).stdout.strip() != BRANCH:
        raise ValueError(f"registration must run on {BRANCH}")
    base = root / "research/g_v6_1"
    config_dir, result_dir = base / "config", base / "results"
    config_dir.mkdir(parents=True, exist_ok=True)
    existing = config_dir / "registration.json"
    if existing.is_file():
        return json.loads(existing.read_text(encoding="utf-8"))

    frozen = load_frozen_checkpoints(root)
    openings = build_openings(root)
    historical_keys = collect_historical_tactical_keys(root)
    tactics = generate_tactical_suite(seed=610612, per_category=32,
                                      historical_keys=historical_keys)
    tactic_audit = validate_tactical_suite(tactics, expected_count=TACTIC_COUNT)
    old_tactics_path = root / "research/g_v6/config/tactics.json"
    old_tactics = json.loads(old_tactics_path.read_text(encoding="utf-8"))
    if len(old_tactics) != 64:
        raise ValueError("historical V6 tactic baseline is not the expected 64 positions")
    # Exact D4-position exclusion against the historical tactic set.
    if historical_keys.intersection(item["position_key"] for item in tactics):
        raise ValueError("new V6.1 tactic set overlaps historical training/evaluation positions")

    _atomic_json(config_dir / "openings.json", openings)
    _atomic_json(config_dir / "tactics_new.json", tactics)
    # Persist exact historical 64-set as a read-only reference and pin its hash.
    _atomic_json(config_dir / "tactics_historical_v6_64.json", old_tactics)
    _atomic_json(config_dir / "tactic_generation_audit.json", {
        **tactic_audit,
        "historical_position_keys_excluded": len(historical_keys),
        "new_tactic_positions_intersect_historical": 0,
        "seed": 610612,
    })

    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip()
    resource = {
        "cpu_count_visible": os.cpu_count(),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").is_file() else None,
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").is_file() else None,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "CPU (checkpoint inference only)",
    }
    registration = {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "branch": BRANCH,
        "baseline_v6_commit": "0405b288d44e0cc35cfefb08a17f0714a1b3683e",
        "registration_code_commit": current_commit,
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board_size": 9,
        "n_in_row": 5,
        "rules": "freestyle Gomoku; five in a row wins; same V6 reference engine, independent grid oracle",
        "models": frozen["metadata"],
        "algorithm_state": "frozen V6 checkpoints; no training, weight update, hyperparameter tuning, or model modification",
        "network_architecture": json.loads((root / "research/g_v5/config/experiment.json").read_text())["reference"]["native_architecture"],
        "search": {"c_puct": 5.0, "root_noise": False, "temperature": 1e-3,
                   "actual_root_action_visits": "N; upstream configured playouts = N+1 due initial root expansion",
                   "budgets": list(SIMULATION_BUDGETS), "tree_reuse_between_moves": False},
        "opponent_aliases": {
            "A-Minimax": {"repository_engine": "B", "time_limit_seconds": 0.05,
                           "source": "src/ai.py choose_move; native repository A is a separate rule selector"},
            "B-AB-50ms": {"repository_engine": "C", "time_limit_seconds": 0.05,
                           "source": "src/ai.py choose_move; Alpha-Beta implementation"},
        },
        "opening_schedule_sha256": _sha(config_dir / "openings.json"),
        "historical_v6_tactics_sha256": _sha(config_dir / "tactics_historical_v6_64.json"),
        "new_tactics_sha256": _sha(config_dir / "tactics_new.json"),
        "tactic_generation_audit_sha256": _sha(config_dir / "tactic_generation_audit.json"),
        "new_tactics": {"count": len(tactics), "per_category": 32,
                        "categories": ["direct_win", "unique_block", "forced_threat_3ply", "false_threat"],
                        "seed": 610612, "historical_canonical_positions_excluded": len(historical_keys)},
        "direct_match_pairs_per_pairing_and_budget": DIRECT_PAIRS,
        "challenge_match_pairs_per_model_opponent_and_budget": CHALLENGE_PAIRS,
        "blind_match_pairs_per_selected_model_opponent_and_budget": 40,
        "opening_plies": 6,
        "opening_schedule_seeds": {"development": openings["development_seed"], "blind": openings["blind_seed"]},
        "selection_rule_id": "h2h_then_A_then_B_then_tactics",
        "selection_rule": [
            "For each search budget separately, the primary metric is equal-weight mean paired score against the other two frozen checkpoints.",
            "Only exact primary point-estimate ties are resolved lexicographically by score versus A-Minimax, then versus B-AB-50ms, then PUCT tactic accuracy; A/B results are kept separate.",
            "Every metric and paired 95% CI is retained; point-estimate selection does not imply a statistically established rank.",
            "Freeze one selected candidate per budget; if the selected identity is identical across budgets, run it once at both budgets.",
            "Blind results are not available to or used by the selector.",
        ],
        "selection_is_development_only": True,
        "cpu_hard_cap_seconds": CPU_LIMIT_SECONDS,
        "resource_snapshot": resource,
        "preflight_required_before_formal_runs": True,
        "training_performed": False,
    }
    canonical = json.dumps(registration, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    registration["registration_sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json(existing, registration)
    return registration


def _preflight_opening(root: Path) -> dict[str, Any]:
    return {
        "opening_group_id": "v61-preflight-only-610609",
        "seed": 610609,
        "moves": [
            {"player": "B", "row": 4, "col": 4},
            {"player": "W", "row": 4, "col": 5},
            {"player": "B", "row": 3, "col": 4},
            {"player": "W", "row": 5, "col": 5},
            {"player": "B", "row": 3, "col": 5},
            {"player": "W", "row": 5, "col": 4},
        ],
    }


def run_preflight(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    registration = json.loads((root / "research/g_v6_1/config/registration.json").read_text())
    if registration.get("training_performed") is not False:
        raise ValueError("V6.1 registration must be inference-only")
    existing = root / "research/g_v6_1/results/preflight.json"
    if existing.is_file():
        result = json.loads(existing.read_text())
        # Convert the short complete-game samples into per-neural-decision CPU
        # costs, then scale by 30/50/70 future plies. The 70-ply case is a
        # conservative long-game scenario and may exceed the 10 CPU-hour cap.
        per_decision: dict[str, float] = {}
        for budget in SIMULATION_BUDGETS:
            rates = []
            for row in result["preflight_games"]:
                if int(row["simulations"]) != budget:
                    continue
                candidate_decisions = max(1, (int(row["game_moves"]) - 6 + 1) // 2)
                rates.append(float(row["candidate_cpu_seconds"]) / candidate_decisions)
            per_decision[str(budget)] = max(rates)
        scenarios = {}
        for scenario, total_plies in (("optimistic", 30), ("normal", 50), ("conservative", 70)):
            p0 = 360 * total_plies * sum(per_decision.values())
            p1 = 3 * (TACTIC_COUNT + 64) * sum(per_decision.values())
            # One winner per budget by the frozen selection rule; the two
            # budget winners may be different checkpoints. This is 80*T
            # candidate decisions at each budget after color pairing.
            p2 = 80 * total_plies * sum(per_decision.values())
            scenarios[scenario] = {
                "assumed_total_game_plies_excluding_opening": total_plies,
                "p0_direct_and_A_B_development_cpu_hours": p0 / 3600,
                "p1_tactics_cpu_hours": p1 / 3600,
                "p2_blind_cpu_hours_one_candidate_per_budget": p2 / 3600,
                "preflight_cpu_hours": float(result["preflight_cpu_seconds"]) / 3600,
                "total_projected_cpu_hours": (p0 + p1 + p2 + float(result["preflight_cpu_seconds"])) / 3600,
                "fits_10_cpu_hour_cap": p0 + p1 + p2 + float(result["preflight_cpu_seconds"]) <= CPU_LIMIT_SECONDS,
            }
        result["max_observed_model_cpu_seconds_per_search_decision"] = per_decision
        result["updated_total_cpu_estimates"] = scenarios
        result["estimate_method"] = (
            "Per-model search CPU measured in six complete preflight games, divided by candidate MCTS decisions; "
            "extrapolated over expected turns for each phase, with separate 30/50/70-ply scenarios."
        )
        result["projected_worst_case_formal_cpu_seconds"] = scenarios["conservative"]["total_projected_cpu_hours"] * 3600
        result["projected_worst_case_formal_cpu_hours"] = scenarios["conservative"]["total_projected_cpu_hours"]
        result["cpu_cap_feasible_by_preflight"] = scenarios["normal"]["fits_10_cpu_hour_cap"]
        _atomic_json(existing, result)
        return result
    torch.set_num_threads(1)
    frozen = load_frozen_checkpoints(root)
    results, started_cpu, started_wall = [], time.process_time(), time.perf_counter()
    opening = _preflight_opening(root)
    for model_name, model in frozen["models"].items():
        for simulations in SIMULATION_BUDGETS:
            raw = _stable_id("v61-preflight", model_name, simulations, opening["seed"])
            name = "preflight-" + raw[:12]
            game = play_v61_match(
                game_id=name, candidate_name=model_name, opponent_name="A-Minimax",
                candidate_color="B", opening=opening, seed=int(raw[12:20], 16),
                simulations=simulations, candidate_model=model, opponent_model=None,
                model_shas=frozen["sha256"],
            )
            game["phase"] = "preflight_only_excluded_from_scores"
            game["opening_seed"] = opening["seed"]
            game["configured_root_action_visits"] = simulations
            _atomic_json(root / "research/g_v6_1/results/preflight" / f"{name}.json", game)
            results.append({
                "model": model_name,
                "simulations": simulations,
                "game_cpu_seconds": game["total_cpu_seconds"],
                "candidate_cpu_seconds": game["candidate_cpu_seconds"],
                "candidate_simulations_observed": sum(int(m.get("simulations", 0))
                    for m in game["moves"] if m.get("actor") == model_name),
                "game_moves": game["move_count"],
                "termination": game["termination"],
            })
    # Conservative projection: for model-vs-model games both neural players
    # consume roughly a full game's candidate CPU; development/blind matches
    # include one neural side. Use the largest observed model cost per budget.
    max_cost = {budget: max(row["game_cpu_seconds"] for row in results if row["simulations"] == budget)
                for budget in SIMULATION_BUDGETS}
    direct_games_per_budget = 3 * DIRECT_PAIRS * 2
    development_games_per_budget = 3 * 2 * CHALLENGE_PAIRS * 2
    # Include up to three blind candidates for conservative resource planning.
    blind_games_per_budget = 3 * 2 * 40 * 2
    tactic_positions = 3 * 2 * (TACTIC_COUNT + 64)
    tactic_cpu_per_pos = {budget: max_cost[budget] / max(1, 40) for budget in SIMULATION_BUDGETS}
    projected = {
        budget: (direct_games_per_budget * max_cost[budget] * 2
                 + (development_games_per_budget + blind_games_per_budget) * max_cost[budget]
                 + tactic_positions * tactic_cpu_per_pos[budget])
        for budget in SIMULATION_BUDGETS
    }
    total_estimate_seconds = sum(projected.values())
    preflight_cpu = time.process_time() - started_cpu
    result = {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "preflight_only": True,
        "formal_match_results_included": False,
        "environment": registration["resource_snapshot"],
        "preflight_games": results,
        "preflight_cpu_seconds": preflight_cpu,
        "preflight_wall_seconds": time.perf_counter() - started_wall,
        "completed_formal_games": 0,
        "projected_worst_case_formal_cpu_seconds": total_estimate_seconds,
        "projected_worst_case_formal_cpu_hours": total_estimate_seconds / 3600,
        "formula": {
            "direct": "3 pairings x 40 paired openings x 2 colors x 2 neural sides x max observed game CPU, per budget",
            "A_B_development": "3 checkpoints x 2 distinct opponents x 20 paired openings x 2 colors x max observed game CPU, per budget",
            "blind": "conservative 3 candidates x 2 opponents x 40 paired openings x 2 colors x max observed game CPU, per budget",
            "tactics": "3 checkpoints x 2 budgets x (128 new + 64 historical) positions, using observed average per-puzzle proxy",
        },
        "estimate_is_conservative_proxy": True,
        "cpu_cap_seconds": CPU_LIMIT_SECONDS,
        "cpu_cap_feasible_by_preflight": total_estimate_seconds + preflight_cpu < CPU_LIMIT_SECONDS,
    }
    _atomic_json(existing, result)
    return result


def _read_summary(root: Path, phase: str, candidate: str, opponent: str, budget: int) -> dict[str, Any] | None:
    safe = lambda s: "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
    path = root / "research/g_v6_1/results/summaries" / phase / (
        f"{safe(candidate)}_vs_{safe(opponent)}_puct{budget}.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stratified_h2h_rate(root: Path, model: str, budget: int) -> tuple[float, list[float], list[float | None]]:
    scores = []
    distributions = []
    for opponent in (name for name in CHECKPOINTS if name != model):
        summary = _read_summary(root, "direct_h2h", model, opponent, budget)
        reverse = False
        if summary is None:
            summary = _read_summary(root, "direct_h2h", opponent, model, budget)
            reverse = True
        if summary is None:
            return 0.0, []
        rate = float(summary["score_rate"])
        scores.append(1.0 - rate if reverse else rate)
        paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for game in summary.get("game_rows", []):
            paired[str(game["opening_group_id"])].append(game)
        distribution = []
        for rows in paired.values():
            if len(rows) != 2:
                continue
            value = float(np.mean([row["points"] for row in rows]))
            distribution.append(1.0 - value if reverse else value)
        distributions.append(distribution)
    if all(distributions) and len(distributions) == 2:
        rng = np.random.default_rng(int(_stable_id(model, budget, "h2h-ci")[:8], 16))
        samples = 10_000
        boots = []
        for distribution in distributions:
            values = np.asarray(distribution, dtype=np.float64)
            idx = rng.integers(0, len(values), size=(samples, len(values)))
            boots.append(values[idx].mean(axis=1))
        mean_boot = (boots[0] + boots[1]) / 2
        ci = [float(np.quantile(mean_boot, 0.025)), float(np.quantile(mean_boot, 0.975))]
    else:
        ci = [None, None]
    return float(np.mean(scores)), scores, ci


def _build_dev_metrics(root: Path, tactics_by_model_budget: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for budget in SIMULATION_BUDGETS:
        rows = {}
        for model in CHECKPOINTS:
            h2h_rate, component_rates, h2h_ci = _stratified_h2h_rate(root, model, budget)
            vs_a = _read_summary(root, "development", model, "A-Minimax", budget)
            vs_b = _read_summary(root, "development", model, "B-AB-50ms", budget)
            if not vs_a or not vs_b:
                raise ValueError("P0 challenge summaries are missing before checkpoint selection")
            tactic_eval = tactics_by_model_budget[model][str(budget)]["new"]
            rows[model] = {
                "h2h_rate": h2h_rate,
                "h2h_component_score_rates": component_rates,
                "h2h_ci95": h2h_ci,
                "vs_A_minimax_rate": float(vs_a["score_rate"]),
                "vs_A_minimax_ci95": vs_a["paired_bootstrap_ci95"],
                "vs_B_ab50_rate": float(vs_b["score_rate"]),
                "vs_B_ab50_ci95": vs_b["paired_bootstrap_ci95"],
                "tactic_puct_accuracy": float(tactic_eval["puct_accuracy"]),
                "tactic_network_top1_accuracy": float(tactic_eval["network_top1_accuracy"]),
                "tactic_target_visit_mass": float(tactic_eval["puct_optimal_visit_mass"]),
            }
        metrics[str(budget)] = rows
    return metrics


def run_v61(root: str | Path = ROOT, *, cpu_cap_seconds: int = CPU_LIMIT_SECONDS) -> dict[str, Any]:
    root = Path(root).resolve()
    registration_path = root / "research/g_v6_1/config/registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    openings = json.loads((root / "research/g_v6_1/config/openings.json").read_text(encoding="utf-8"))
    tactics = json.loads((root / "research/g_v6_1/config/tactics_new.json").read_text(encoding="utf-8"))
    old_tactics = json.loads((root / "research/g_v6_1/config/tactics_historical_v6_64.json").read_text(encoding="utf-8"))
    frozen = load_frozen_checkpoints(root)
    if frozen["sha256"] != {name: registration["models"][name]["sha256"] for name in CHECKPOINTS}:
        raise ValueError("checkpoint SHA-256 changed after registration")
    torch.set_num_threads(1)
    preflight = json.loads((root / "research/g_v6_1/results/preflight.json").read_text(encoding="utf-8"))
    preflight_cpu = float(preflight["preflight_cpu_seconds"])
    ledger_path = root / "research/g_v6_1/results/cpu_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {
        "formal_cpu_seconds_accounted": 0.0, "accounted_game_ids": []}
    accounted = set(ledger.get("accounted_game_ids", []))
    # If the process stopped between match-set commits, count persisted complete
    # games absent from the ledger. This makes a restart CPU-cap aware.
    all_game_paths = sorted((root / "research/g_v6_1/results/matches").rglob("v61m-*.json"))
    persisted_unaccounted_cpu = 0.0
    for path in all_game_paths:
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
            if game["game_id"] not in accounted:
                persisted_unaccounted_cpu += float(game.get("total_cpu_seconds", 0.0))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    prior_formal_cpu = float(ledger.get("formal_cpu_seconds_accounted", 0.0)) + persisted_unaccounted_cpu
    _atomic_json(ledger_path, {
        "formal_cpu_seconds_accounted": prior_formal_cpu,
        "accounted_game_ids": sorted(accounted | {
            json.loads(path.read_text())["game_id"] for path in all_game_paths
            if path.is_file() and path.stat().st_size > 0
        }),
        "recovered_unaccounted_game_cpu_seconds": persisted_unaccounted_cpu,
        "preflight_cpu_seconds": preflight_cpu,
        "cpu_cap_seconds": cpu_cap_seconds,
    })
    remaining_cpu = max(0.0, float(cpu_cap_seconds) - preflight_cpu - prior_formal_cpu)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    deadline = started_cpu + remaining_cpu

    def save_ledger() -> None:
        completed_ids = set()
        for path in (root / "research/g_v6_1/results/matches").rglob("v61m-*.json"):
            try:
                completed_ids.add(json.loads(path.read_text(encoding="utf-8"))["game_id"])
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        _atomic_json(ledger_path, {
            "formal_cpu_seconds_accounted": prior_formal_cpu + time.process_time() - started_cpu,
            "accounted_game_ids": sorted(completed_ids),
            "preflight_cpu_seconds": preflight_cpu,
            "cpu_cap_seconds": cpu_cap_seconds,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    total_phase_summaries = {}
    direct = (("G10K", "G15K"), ("G10K", "G20K"), ("G15K", "G20K"))
    p0_complete = True
    for budget in SIMULATION_BUDGETS:
        for candidate, opponent in direct:
            summary = run_match_set(
                root=root, phase="direct_h2h", candidate_name=candidate,
                opponent_name=opponent, openings=openings["development"],
                pairs=DIRECT_PAIRS, simulations=budget, models=frozen["models"],
                model_shas=frozen["sha256"], cpu_deadline=deadline,
            )
            total_phase_summaries[f"direct:{candidate}:{opponent}:{budget}"] = summary
            p0_complete &= bool(summary["complete"])
            save_ledger()
    for budget in SIMULATION_BUDGETS:
        for candidate in CHECKPOINTS:
            for opponent in ("A-Minimax", "B-AB-50ms"):
                summary = run_match_set(
                    root=root, phase="development", candidate_name=candidate,
                    opponent_name=opponent, openings=openings["development"][:CHALLENGE_PAIRS],
                    pairs=CHALLENGE_PAIRS, simulations=budget, models=frozen["models"],
                    model_shas=frozen["sha256"], cpu_deadline=deadline,
                )
                total_phase_summaries[f"challenge:{candidate}:{opponent}:{budget}"] = summary
                p0_complete &= bool(summary["complete"])
                save_ledger()
    _atomic_json(root / "research/g_v6_1/results/p0_status.json", {
        "complete": p0_complete,
        "summaries": total_phase_summaries,
        "formal_cpu_seconds_accounted": prior_formal_cpu + time.process_time() - started_cpu,
        "cpu_used_including_preflight_seconds": preflight_cpu + prior_formal_cpu + time.process_time() - started_cpu,
        "cpu_cap_seconds": cpu_cap_seconds,
    })
    save_ledger()
    if not p0_complete or time.process_time() >= deadline:
        return {"p0_complete": p0_complete, "p1_complete": False, "p2_complete": False,
                "reason": "CPU cap reached or P0 incomplete", "p0_status": str(root / "research/g_v6_1/results/p0_status.json")}

    tactic_summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    for model_name, model in frozen["models"].items():
        for budget in SIMULATION_BUDGETS:
            for label, suite in (("new", tactics), ("historical_v6_64", old_tactics)):
                path = root / "research/g_v6_1/results/tactics" / model_name / str(budget) / f"{label}.json"
                value = evaluate_tactics(
                    model, suite, simulations=budget,
                    seed=int(_stable_id(model_name, budget, label)[:8], 16),
                    output_path=path, cpu_deadline=deadline,
                )
                tactic_summaries[model_name].setdefault(str(budget), {})[label] = value
                save_ledger()
    p1_complete = all(
        tactic_summaries[name][str(budget)].get(label, {}).get("puzzles") == len(suite)
        and tactic_summaries[name][str(budget)].get(label, {}).get("complete") is True
        for name in CHECKPOINTS for budget in SIMULATION_BUDGETS
        for label, suite in (("new", tactics), ("historical_v6_64", old_tactics))
    )
    _atomic_json(root / "research/g_v6_1/results/tactics_summary.json", tactic_summaries)
    if not p1_complete or time.process_time() >= deadline:
        save_ledger()
        return {"p0_complete": True, "p1_complete": p1_complete, "p2_complete": False,
                "reason": "CPU cap reached or P1 incomplete"}

    dev_metrics = _build_dev_metrics(root, tactic_summaries)
    selection_path = root / "research/g_v6_1/results/frozen_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("blind_results_used_for_selection"):
            raise ValueError("existing selection illegally used blind results")
        if selection.get("development_metrics") != dev_metrics:
            raise ValueError("frozen selection registration differs from P0/P1 development data")
    else:
        selection = make_selection(dev_metrics, frozen_checkpoint_sha256=frozen["sha256"])
        _atomic_json(selection_path, selection)
    p2_complete = True
    for budget in SIMULATION_BUDGETS:
        selected = selection["selected_by_budget"][str(budget)]
        selected_rows = selected if isinstance(selected, list) else [selected]
        for candidate in selected_rows:
            for opponent in ("A-Minimax", "B-AB-50ms"):
                summary = run_match_set(
                    root=root, phase="final_blind", candidate_name=candidate,
                    opponent_name=opponent, openings=openings["blind"], pairs=BLIND_PAIRS,
                    simulations=budget, models=frozen["models"], model_shas=frozen["sha256"],
                    cpu_deadline=deadline,
                )
                p2_complete &= bool(summary["complete"])
                save_ledger()
    game_audit = audit_v61_games(root)
    total_cpu = preflight_cpu + prior_formal_cpu + time.process_time() - started_cpu
    final = {
        "experiment_id": EXPERIMENT_ID,
        "p0_complete": p0_complete,
        "p1_complete": p1_complete,
        "p2_complete": p2_complete,
        "selection": selection,
        "development_metrics": dev_metrics,
        "tactics": tactic_summaries,
        "game_audit": game_audit,
        "preflight_cpu_seconds": preflight_cpu,
        "formal_cpu_seconds": prior_formal_cpu + time.process_time() - started_cpu,
        "total_cpu_seconds_including_preflight": total_cpu,
        "cpu_cap_seconds": cpu_cap_seconds,
        "runner_wall_seconds": time.perf_counter() - started_wall,
        "all_requested_phases_complete": p0_complete and p1_complete and p2_complete,
    }
    _atomic_json(root / "research/g_v6_1/results/final.json", final)
    save_ledger()
    return final
