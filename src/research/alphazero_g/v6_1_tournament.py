"""Frozen-checkpoint tournament and tactical audit for AlphaZero G V6.1.

No training path is imported here. All checkpoint reads are CPU-only and the
only neural-network operations are inference calls from the V6 reference code.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard
from .v5_audit import audit_evaluation_game
from .v5_evaluation import (
    _IndependentRuleBoard,
    _canonical_position,
    generate_opening_pool,
    make_board,
    opening_key,
    paired_bootstrap_ci,
)
from .v5_reference import ReferencePolicyValueNet, instrumented_player
from .v6_experiment import _load_node


EXPERIMENT_ID = "wuziq-alphazero-g-v6-1-checkpoint-tournament"
BRANCH = "research/alphazero-g-v6-1-tournament"
CHECKPOINTS = {
    "G10K": ("research/g_v6/checkpoints/reference400_empty/node_10000.pt", 10004),
    "G15K": ("research/g_v6/checkpoints/reference400_empty/node_15000.pt", 15006),
    "G20K": ("research/g_v6/checkpoints/reference400_empty/node_20000.pt", 20001),
}
SIMULATION_BUDGETS = (64, 400)
DIRECT_PAIRS = 40
CHALLENGE_PAIRS = 20
BLIND_PAIRS = 40
TACTIC_COUNT = 128
CPU_LIMIT_SECONDS = 10 * 3600
ROOT = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_frozen_checkpoints(root: str | Path) -> dict[str, Any]:
    """Verify node metadata, SHA-256, architecture and inference loading."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    models: dict[str, ReferencePolicyValueNet] = {}
    shas: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    reference_config = json.loads((root / "research/g_v5/config/experiment.json").read_text())
    expected_architecture = reference_config.get("reference", {}).get("native_architecture")
    shapes: dict[str, tuple[int, ...]] | None = None
    for name, (relative, expected_states) in CHECKPOINTS.items():
        path = root / relative
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (int(payload.get("board_size", -1)) != 9 or int(payload.get("n_in_row", -1)) != 5
                or int(payload.get("simulations_per_move", -1)) != 400
                or int(payload.get("environment_states", -1)) != expected_states):
            raise ValueError(f"frozen checkpoint metadata mismatch: {name}")
        model = _load_node(path)
        model.network.eval()
        current_shapes = {key: tuple(value.shape) for key, value in model.get_policy_param().items()}
        if shapes is None:
            shapes = current_shapes
        elif current_shapes != shapes:
            raise ValueError("frozen checkpoints have different network architectures")
        # A forward pass verifies that the checkpoint is actually usable.
        board = make_board(9, 5, 0)
        priors, value = model.policy_value_fn(board)
        priors = list(priors)
        legal_probability_mass = sum(float(prob) for _, prob in priors)
        if (not priors or not np.isfinite([float(prob) for _, prob in priors]).all()
                or legal_probability_mass <= 0 or legal_probability_mass > 1.0002
                or not np.isfinite(float(value))):
            raise ValueError(f"frozen checkpoint failed inference validation: {name}")
        models[name] = model
        shas[name] = _sha(path)
        metadata[name] = {
            "path": relative,
            "sha256": shas[name],
            "board_size": int(payload["board_size"]),
            "n_in_row": int(payload["n_in_row"]),
            "simulations_per_move": int(payload["simulations_per_move"]),
            "environment_states": int(payload["environment_states"]),
            "games_completed": int(payload.get("games_completed", -1)),
            "config_sha256": payload.get("config_sha256"),
            "network_state_shapes": current_shapes,
            "reference_architecture_config": expected_architecture,
            "inference_legal_moves": len(priors),
            "inference_legal_probability_mass": legal_probability_mass,
            "inference_value": float(value),
        }
    return {"models": models, "sha256": shas, "metadata": metadata}


def _historical_opening_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in root.glob("research/**/config/*opening*schedule*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("moves"), list):
                    moves = value["moves"]
                    if moves and all({"player", "row", "col"} <= item.keys() for item in moves):
                        try:
                            keys.add(opening_key(moves, 9))
                        except (KeyError, TypeError, ValueError):
                            pass
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(data)
    # Also include opening prefixes embedded in completed V6 games.
    for path in (root / "research/g_v6/results/matches").rglob("v6m-*.json"):
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
            moves = game.get("opening", [])
            if moves:
                keys.add(opening_key(moves, 9))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return keys


def build_openings(root: str | Path, *, development_seed: int = 610610,
                   blind_seed: int = 610611) -> dict[str, Any]:
    root = Path(root).resolve()
    historical = _historical_opening_keys(root)
    for attempt in range(500):
        development = generate_opening_pool(size=9, n_in_row=5, groups=40, plies=6,
                                            seed=development_seed + attempt * 1000)
        blind = generate_opening_pool(size=9, n_in_row=5, groups=40, plies=6,
                                      seed=blind_seed + attempt * 1000)
        seen: set[str] = set()
        collision = False
        for phase, rows in (("development", development), ("blind", blind)):
            for index, row in enumerate(rows):
                key = opening_key(row["moves"], 9)
                if key in historical or key in seen:
                    collision = True
                    break
                seen.add(key)
                row["opening_group_id"] = f"v61-{phase}-{row['seed']}-{index:03d}"
            if collision:
                break
        if not collision:
            schedule = {
                "format_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "development_seed": development_seed + attempt * 1000,
                "blind_seed": blind_seed + attempt * 1000,
                "plies": 6,
                "development_pairs": 40,
                "challenge_pairs": 20,
                "blind_pairs": 40,
                "development": development,
                "blind": blind,
                "excluded_historical_D4_openings": len(historical),
                "D4_distinct_between_sets": True,
            }
            validate_opening_sets(schedule, root=root)
            return schedule
    raise RuntimeError("could not create V6.1 openings absent from historical schedules")


def validate_opening_sets(schedule: dict[str, Any], *, root: str | Path | None = None) -> None:
    root = Path(root or ROOT).resolve()
    if len(schedule.get("development", [])) != 40 or len(schedule.get("blind", [])) != 40:
        raise ValueError("V6.1 requires 40 development and 40 blind opening groups")
    old = _historical_opening_keys(root)
    seen: set[str] = set()
    for phase in ("development", "blind"):
        for opening in schedule[phase]:
            moves = opening.get("moves", [])
            if len(moves) != 6:
                raise ValueError("V6.1 openings must contain exactly six plies")
            board = make_board(9, 5, 0)
            expected = "B"
            for move in moves:
                row, col = int(move["row"]), int(move["col"])
                if move["player"] != expected:
                    raise ValueError("V6.1 opening does not alternate colors")
                action = row * 9 + col
                if action not in board.availables:
                    raise ValueError("V6.1 opening has an illegal/repeated move")
                board.do_move(action)
                if board.game_end()[0]:
                    raise ValueError("V6.1 opening is terminal")
                expected = "W" if expected == "B" else "B"
            key = opening_key(moves, 9)
            if key in seen or key in old:
                raise ValueError("V6.1 opening is repeated or was used in an earlier registered schedule")
            seen.add(key)


def make_selection(development_metrics: dict[str, Any], *,
                   frozen_checkpoint_sha256: dict[str, str]) -> dict[str, Any]:
    """Apply the preregistered per-budget lexicographic selector.

    Primary is equal-weight H2H score against the other two checkpoints;
    exact point-estimate ties use A, then B, then tactical accuracy. The two
    opponent scores remain separate in the output. Confidence intervals are
    always reported and are not misrepresented as proof of a strength gap.
    """
    result: dict[str, Any] = {
        "selection_rule_id": "h2h_then_A_then_B_then_tactics",
        "frozen_before_blind": True,
        "blind_results_used_for_selection": False,
        "development_metrics": development_metrics,
        "selected_by_budget": {},
        "checkpoint_sha256": frozen_checkpoint_sha256,
        "rule": [
            "For each budget, choose the highest equal-weight mean paired score against the other two frozen checkpoints.",
            "Only exact H2H point-estimate ties are resolved by score against A-Minimax, then B-AlphaBeta-50ms, then exact tactical accuracy.",
            "A/B remain separate development metrics; they are never added into a single opponent win rate.",
            "If every registered metric is exactly tied, retain all tied models for blind evaluation.",
            "The 95% paired bootstrap intervals determine whether a point-estimate ordering is statistically unresolved.",
        ],
    }
    for budget in map(str, SIMULATION_BUDGETS):
        rows = development_metrics[budget]
        models = list(CHECKPOINTS)
        columns = ("h2h_rate", "vs_A_minimax_rate", "vs_B_ab50_rate", "tactic_puct_accuracy")
        remaining = models
        for column in columns:
            best = max(float(rows[name][column]) for name in remaining)
            remaining = [name for name in remaining if float(rows[name][column]) == best]
            if len(remaining) == 1:
                break
        result["selected_by_budget"][budget] = remaining[0] if len(remaining) == 1 else remaining
    selected_names = set()
    for value in result["selected_by_budget"].values():
        selected_names.update(value if isinstance(value, list) else [value])
    result["blind_candidates"] = sorted(selected_names)
    return result


def summarize_tournament_games(games: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    by_opening: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_opening[str(game["opening_group_id"])].append(game)
    if any(len(rows) != 2 or {row["candidate_color"] for row in rows} != {"B", "W"}
           for rows in by_opening.values()):
        raise ValueError("each paired opening requires one black and one white candidate game (color-swapped)")
    wins = draws = losses = 0
    pair_scores, black, white = [], [], []
    for _, rows in sorted(by_opening.items()):
        pair_scores.append(float(np.mean([float(row["points"]) for row in rows])))
        for row in rows:
            score = float(row["points"])
            (black if row["candidate_color"] == "B" else white).append(score)
            wins += int(score == 1.0)
            draws += int(score == 0.5)
            losses += int(score == 0.0)
    lo, hi = paired_bootstrap_ci(pair_scores, seed=int(seed))
    total = wins + draws + losses
    return {
        "games": total, "paired_openings": len(by_opening),
        "wins": wins, "draws": draws, "losses": losses,
        "points": wins + draws / 2,
        "score_rate": (wins + draws / 2) / total if total else None,
        "paired_opening_score_rate": float(np.mean(pair_scores)) if pair_scores else None,
        "paired_bootstrap_ci95": [lo, hi],
        "black_score_rate": float(np.mean(black)) if black else None,
        "white_score_rate": float(np.mean(white)) if white else None,
        "mean_moves": float(np.mean([row["move_count"] for row in games])) if games else None,
        "mean_candidate_cpu_seconds": float(np.mean([row.get("candidate_cpu_seconds", 0.0) for row in games])) if games else None,
        "mean_opponent_cpu_seconds": float(np.mean([row.get("opponent_cpu_seconds", 0.0) for row in games])) if games else None,
        "mean_total_cpu_seconds": float(np.mean([row.get("total_cpu_seconds", 0.0) for row in games])) if games else None,
    }


def _project_board(board) -> GomokuBoard:
    result = GomokuBoard(9)
    for action, player in board.states.items():
        result.place_stone("B" if int(player) == 1 else "W", divmod(int(action), 9))
    return result


def _move(agent: dict[str, Any], board, color: str, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
    started = time.process_time()
    if agent["kind"] == "reference":
        action, _probs = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(action), 9)
        search = dict(agent["player"].last_search_stats)
        actual = int(search.get("root_visits", 0))
        return (row, col), {
            "simulations": actual,
            "search_depth": int(search.get("tree_depth", 0)),
            "search_nodes": int(search.get("tree_nodes", 0)),
            "root_action_visits": actual,
            "root_children": int(search.get("root_children", 0)),
            "cpu_seconds": time.process_time() - started,
            "search_cpu_seconds": float(search.get("search_cpu_seconds", 0.0)),
            "timed_out": False,
        }
    project = _project_board(board)
    # V6.1 semantic aliases are frozen: A-Minimax calls repository engine B;
    # B-AlphaBeta-50ms calls repository engine C. The code's native label A is
    # a separate rule selector and is intentionally not mistaken for minimax.
    engine = agent["engine"]
    limit = 0.05
    outcome = choose_move(project, color, engine, SearchConfig(time_limit_seconds=limit), seed)
    return outcome.move, {
        "simulations": int(outcome.simulations),
        "search_depth": int(outcome.completed_depth),
        "search_nodes": int(outcome.nodes),
        "cutoffs": int(outcome.cutoffs),
        "root_action_visits": 0,
        "cpu_seconds": time.process_time() - started,
        "search_cpu_seconds": float(outcome.elapsed_seconds),
        "timed_out": bool(outcome.timed_out),
        "configured_time_budget_seconds": limit,
    }


def play_v61_match(*, game_id: str, candidate_name: str, opponent_name: str,
                   candidate_color: str, opening: dict[str, Any], seed: int,
                   simulations: int, candidate_model: ReferencePolicyValueNet | None,
                   opponent_model: ReferencePolicyValueNet | None,
                   model_shas: dict[str, str]) -> dict[str, Any]:
    if candidate_color not in {"B", "W"}:
        raise ValueError("candidate color must be B or W")
    np.random.seed(int(seed) & 0xFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    if candidate_name in CHECKPOINTS:
        if candidate_model is None:
            raise ValueError("candidate checkpoint model is required")
        candidate = {"kind": "reference", "player": instrumented_player(
            candidate_model.policy_value_fn, c_puct=5.0, n_playout=simulations + 1,
            is_selfplay=False)}
    else:
        candidate = {"kind": "search", "engine": "B" if candidate_name == "A-Minimax" else "C"}
    if opponent_name in CHECKPOINTS:
        if opponent_model is None:
            raise ValueError("opponent checkpoint model is required")
        opponent = {"kind": "reference", "player": instrumented_player(
            opponent_model.policy_value_fn, c_puct=5.0, n_playout=simulations + 1,
            is_selfplay=False)}
    else:
        opponent = {"kind": "search", "engine": "B" if opponent_name == "A-Minimax" else "C"}

    board = make_board(9, 5, 0)
    oracle = _IndependentRuleBoard(9, 5)
    history: list[dict[str, Any]] = []
    expected = "B"
    for item in opening.get("moves", []):
        color = str(item["player"])
        row, col = int(item["row"]), int(item["col"])
        if color != expected or row * 9 + col not in board.availables:
            raise ValueError("registered opening has an illegal color or move")
        board.do_move(row * 9 + col)
        oracle.place_stone(color, (row, col))
        if board.game_end()[0] or oracle.check_victory((row, col), color):
            raise ValueError("registered opening is terminal")
        history.append({"player": color, "row": row, "col": col, "opening": True})
        expected = "W" if expected == "B" else "B"

    started_wall, started_cpu = time.perf_counter(), time.process_time()
    candidate_cpu = opponent_cpu = 0.0
    winner: str | None = None
    termination = ""
    while True:
        color = "B" if board.current_player == 1 else "W"
        is_candidate = color == candidate_color
        actor = candidate if is_candidate else opponent
        actor_name = candidate_name if is_candidate else opponent_name
        move_seed = int(rng.integers(0, 2**31 - 1))
        elapsed_start = time.perf_counter()
        coord, metrics = _move(actor, board, color, move_seed)
        elapsed = time.perf_counter() - elapsed_start
        if is_candidate:
            candidate_cpu += float(metrics["cpu_seconds"])
        else:
            opponent_cpu += float(metrics["cpu_seconds"])
        row, col = map(int, coord)
        action = row * 9 + col
        if action not in board.availables or oracle.get((row, col)) is not None:
            raise ValueError(f"{actor_name} returned illegal move {(row, col)}")
        board.do_move(action)
        oracle.place_stone(color, (row, col))
        history.append({"player": color, "row": row, "col": col, "opening": False,
                        "actor": actor_name, "elapsed_seconds": elapsed, **metrics})
        engine_end, engine_winner = board.game_end()
        oracle_win = oracle.check_victory((row, col), color)
        if bool(engine_end) != bool(oracle_win or oracle.is_full()):
            raise AssertionError("game engine and independent win oracle disagree")
        if oracle_win:
            if engine_winner != (1 if color == "B" else 2):
                raise AssertionError("winner disagrees with independent oracle")
            winner, termination = color, "five_in_row"
            break
        if oracle.is_full():
            termination = "full_board"
            break
    points = 0.5 if winner is None else (1.0 if winner == candidate_color else 0.0)
    return {
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "game_id": game_id,
        "phase": "",
        "candidate": candidate_name,
        "opponent": opponent_name,
        "board_size": 9,
        "n_in_row": 5,
        "candidate_color": candidate_color,
        "winner_color": winner,
        "points": points,
        "termination": termination,
        "opening_group_id": opening["opening_group_id"],
        "opening": opening["moves"],
        "seed": int(seed),
        "simulations_per_reference_turn": int(simulations),
        "reference_configured_playouts": int(simulations + 1),
        "model_sha256": {
            candidate_name: model_shas.get(candidate_name),
            opponent_name: model_shas.get(opponent_name),
        },
        "moves": history,
        "move_count": len(history),
        "candidate_cpu_seconds": candidate_cpu,
        "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }


def _game_path(root: Path, phase: str, candidate: str, opponent: str,
               budget: int, game_id: str) -> Path:
    safe = lambda s: "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
    return root / "research/g_v6_1/results/matches" / safe(phase) / (
        f"{safe(candidate)}_vs_{safe(opponent)}_puct{budget}") / f"{game_id}.json"


def run_match_set(*, root: str | Path, phase: str, candidate_name: str,
                  opponent_name: str, openings: list[dict[str, Any]], pairs: int,
                  simulations: int, models: dict[str, ReferencePolicyValueNet],
                  model_shas: dict[str, str], cpu_deadline: float | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    if pairs < 1 or len(openings) < pairs or simulations not in SIMULATION_BUDGETS:
        raise ValueError("invalid V6.1 match-set registration")
    completed: list[dict[str, Any]] = []
    for opening in openings[:pairs]:
        rows: list[dict[str, Any]] = []
        stopped = False
        for color in ("B", "W"):
            raw = _stable_id(EXPERIMENT_ID, phase, candidate_name, opponent_name,
                             simulations, opening["opening_group_id"], color)
            game_id = "v61m-" + raw[:16]
            path = _game_path(root, phase, candidate_name, opponent_name, simulations, game_id)
            if path.is_file():
                game = json.loads(path.read_text(encoding="utf-8"))
                if (game.get("game_id") != game_id or game.get("candidate") != candidate_name
                        or game.get("opponent") != opponent_name
                        or game.get("candidate_color") != color
                        or game.get("opening_group_id") != opening["opening_group_id"]
                        or int(game.get("simulations_per_reference_turn", -1)) != simulations):
                    raise ValueError("V6.1 game ID collision or configuration mismatch")
                audit_evaluation_game(path)
            else:
                if cpu_deadline is not None and time.process_time() >= cpu_deadline:
                    stopped = True
                    break
                seed = int(raw[16:24], 16)
                game = play_v61_match(
                    game_id=game_id, candidate_name=candidate_name, opponent_name=opponent_name,
                    candidate_color=color, opening=opening, seed=seed, simulations=simulations,
                    candidate_model=models.get(candidate_name), opponent_model=models.get(opponent_name),
                    model_shas=model_shas,
                )
                game["phase"] = phase
                game["opening_seed"] = int(opening.get("seed", 0))
                game["configured_root_action_visits"] = simulations
                game["engine_mapping"] = {
                    "A-Minimax": "src.ai.choose_move('B', time_limit_seconds=0.05)",
                    "B-AB-50ms": "src.ai.choose_move('C', time_limit_seconds=0.05)",
                }
                _atomic_json(path, game)
                audit_evaluation_game(path)
            rows.append(game)
        if len(rows) == 2:
            completed.extend(rows)
            _atomic_json(root / "research/g_v6_1/results/progress.json", {
                "experiment_id": EXPERIMENT_ID,
                "phase": phase,
                "candidate": candidate_name,
                "opponent": opponent_name,
                "simulations": simulations,
                "completed_pairs_in_set": len(completed) // 2,
                "planned_pairs": pairs,
                "completed_game_ids": [row["game_id"] for row in completed],
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        elif stopped:
            break
    summary = summarize_tournament_games(completed, seed=int(_stable_id(
        phase, candidate_name, opponent_name, simulations, "bootstrap")[:8], 16)) if completed else {
        "games": 0, "paired_openings": 0, "wins": 0, "draws": 0, "losses": 0,
        "points": 0, "score_rate": None, "paired_bootstrap_ci95": [None, None],
    }
    summary.update({
        "phase": phase, "candidate": candidate_name, "opponent": opponent_name,
        "simulations_per_reference_turn": simulations,
        "planned_pairs": pairs, "planned_games": pairs * 2,
        "complete": len(completed) == pairs * 2,
        "game_rows": [{key: game.get(key) for key in (
            "game_id", "opening_group_id", "candidate_color", "winner_color", "points",
            "termination", "move_count", "candidate_cpu_seconds", "opponent_cpu_seconds",
            "total_cpu_seconds", "total_wall_seconds")}
            for game in completed],
    })
    _atomic_json(root / "research/g_v6_1/results/summaries" / phase / (
        f"{candidate_name}_vs_{opponent_name}_puct{simulations}.json"), summary)
    return summary


def audit_v61_games(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = sorted((root / "research/g_v6_1/results/matches").rglob("v61m-*.json"))
    ids: set[str] = set()
    pair_colors: dict[tuple[str, str, str, int, str], set[str]] = defaultdict(set)
    reports = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["game_id"] in ids:
            raise ValueError(f"duplicate game ID: {record['game_id']}")
        ids.add(record["game_id"])
        if int(record["simulations_per_reference_turn"]) not in SIMULATION_BUDGETS:
            raise ValueError("unregistered simulation budget in result")
        key = (record["phase"], record["candidate"], record["opponent"],
               int(record["simulations_per_reference_turn"]), record["opening_group_id"])
        color_set = pair_colors[key]
        if record["candidate_color"] in color_set:
            raise ValueError("duplicate color in one paired opening")
        color_set.add(record["candidate_color"])
        reports.append(audit_evaluation_game(path))
    return {
        "games_verified": len(reports),
        "unique_game_ids": len(ids),
        "complete_color_swapped_pairs": sum(value == {"B", "W"} for value in pair_colors.values()),
        "incomplete_pairs": sum(value != {"B", "W"} for value in pair_colors.values()),
        "all_completed_games_legal_terminal_and_engine_consistent": True,
        "game_sha256": {row["game_id"]: row["game_sha256"] for row in reports},
    }


def _score_against(summary: dict[str, Any], model: str, opponent: str) -> float:
    if summary["candidate"] == model and summary["opponent"] == opponent:
        return float(summary["score_rate"])
    if summary["candidate"] == opponent and summary["opponent"] == model:
        return 1.0 - float(summary["score_rate"])
    raise ValueError("summary does not include requested model comparison")

