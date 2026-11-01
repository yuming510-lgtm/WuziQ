"""Paired development/blind evaluation and rule-derived tactical probes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ...gomoku import GomokuBoard, Player
from .evaluation import (
    F2_SHA256,
    append_evaluation_match,
    load_evaluation_log,
    load_frozen_opponent,
    play_evaluation_match,
    summarize_matches,
)
from .puct import PUCTSearch
from .training import load_g_checkpoint
from .v3_budget import ensure_v3_cpu_room, record_v3_cpu_event
from .v3_runner import ARMS, load_v3_config
from .v3_schedule import validate_v3_schedule


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


def _transform(row: int, col: int, turns: int, reflect: bool) -> tuple[int, int]:
    if reflect:
        col = 8 - col
    for _ in range(turns):
        row, col = col, 8 - row
    return row, col


def _opening_board(position: list[dict[str, Any]], to_move: Player) -> GomokuBoard:
    board = GomokuBoard(9)
    expected: Player = "B"
    for move in position:
        if move.get("player") != expected:
            raise ValueError("tactical puzzle positions must alternate from black")
        coordinate = (int(move["row"]), int(move["col"]))
        board.place_stone(expected, coordinate)
        if board.check_victory(coordinate, expected) or board.is_full():
            raise ValueError("tactical puzzle must be a nonterminal legal position")
        expected = "W" if expected == "B" else "B"
    if expected != to_move:
        raise ValueError("puzzle side-to-move does not match its legal history")
    return board


def _winning_actions(board: GomokuBoard, player: Player) -> list[int]:
    result = []
    for row in range(9):
        for col in range(9):
            if board.get((row, col)) is not None:
                continue
            # GomokuBoard deliberately exposes mutable placement but not a copy API;
            # replay its public history to keep each tactical candidate isolated.
            clone = GomokuBoard(9)
            for existing_player, existing_coordinate in board.history:
                clone.place_stone(existing_player, existing_coordinate)
            clone.place_stone(player, (row, col))
            if clone.check_victory((row, col), player):
                result.append(row * 9 + col)
    return result


def _exact_actions(puzzle: dict[str, Any]) -> list[int]:
    player: Player = str(puzzle["to_move"])  # type: ignore[assignment]
    if player not in {"B", "W"}:
        raise ValueError("tactical puzzle to_move must be B or W")
    board = _opening_board(puzzle["position"], player)
    if puzzle["category"] in {"direct_win", "edge"}:
        wins = _winning_actions(board, player)
        if not wins:
            raise ValueError("direct-win/edge puzzle has no immediate winning action")
        return wins
    opponent: Player = "W" if player == "B" else "B"
    opponent_wins = _winning_actions(board, opponent)
    if not opponent_wins:
        raise ValueError("blocking puzzle has no immediate opponent win to answer")
    safe: list[int] = []
    for row in range(9):
        for col in range(9):
            if board.get((row, col)) is not None:
                continue
            clone = GomokuBoard(9)
            for existing_player, existing_coordinate in board.history:
                clone.place_stone(existing_player, existing_coordinate)
            clone.place_stone(player, (row, col))
            if clone.check_victory((row, col), player) or not _winning_actions(clone, opponent):
                safe.append(row * 9 + col)
    if puzzle["category"] == "unique_block" and len(safe) != 1:
        raise ValueError("unique-block puzzle does not have exactly one rule-valid defense")
    if puzzle["category"] == "false_threat" and len(safe) != 1:
        raise ValueError("false-threat puzzle must have one exact defense despite an attacking pattern")
    return safe


def build_v3_tactical_puzzles() -> list[dict[str, Any]]:
    """Construct D4-expanded immediate-win, unique-block, edge and false-threat puzzles.

    Targets are computed from legal five-in-a-row checks; no network or search
    teacher contributes labels. The false-threat case includes an apparent
    three-stone attack but requires the only defense to a different open four.
    """
    bases = [
        ("direct_win", [
            ("B", 4, 1), ("W", 4, 0), ("B", 4, 2), ("W", 0, 0),
            ("B", 4, 3), ("W", 0, 2), ("B", 4, 4), ("W", 0, 4),
        ]),
        ("unique_block", [
            ("B", 4, 5), ("W", 4, 0), ("B", 0, 0), ("W", 4, 1),
            ("B", 0, 2), ("W", 4, 2), ("B", 0, 4), ("W", 4, 3),
        ]),
        ("edge", [
            ("B", 8, 0), ("W", 0, 0), ("B", 8, 1), ("W", 0, 1),
            ("B", 8, 2), ("W", 0, 2), ("B", 8, 3), ("W", 0, 3),
        ]),
        ("false_threat", [
            ("B", 4, 2), ("W", 0, 0), ("B", 4, 3), ("W", 0, 1),
            ("B", 4, 4), ("W", 0, 2), ("B", 2, 0), ("W", 0, 3),
        ]),
    ]
    puzzles = []
    for category, position in bases:
        for reflected in (False, True):
            for turns in range(4):
                transformed = [
                    {"player": player, "row": _transform(row, col, turns, reflected)[0],
                     "col": _transform(row, col, turns, reflected)[1]}
                    for player, row, col in position
                ]
                puzzle = {
                    "puzzle_id": f"g-v3-tactic-{category}-r{turns}-f{int(reflected)}",
                    "category": category,
                    "to_move": "B",
                    "position": transformed,
                    "label_source": "exact_legal_five_in_a_row_rules",
                }
                puzzle["optimal_actions"] = _exact_actions(puzzle)
                puzzles.append(puzzle)
    ids = [puzzle["puzzle_id"] for puzzle in puzzles]
    if len(ids) != len(set(ids)):
        raise ValueError("tactical puzzle IDs are not unique")
    return puzzles


def evaluate_v3_tactics(
    model: torch.nn.Module,
    *,
    mode: Literal["network", "puct"],
    puzzles: list[dict[str, Any]],
    simulations: int = 64,
) -> dict[str, Any]:
    """Measure exact tactical action accuracy without altering candidate moves."""
    if mode not in {"network", "puct"}:
        raise ValueError("mode must be network or puct")
    if mode == "puct" and int(simulations) != 64:
        raise ValueError("registered tactical PUCT evaluation uses exactly 64 simulations")
    if not puzzles:
        raise ValueError("tactical puzzle set must not be empty")
    from .evaluation import _network_move

    model.to("cpu")
    model.eval()
    search = PUCTSearch(model, device="cpu") if mode == "puct" else None
    records = []
    for puzzle in puzzles:
        board = _opening_board(puzzle["position"], str(puzzle["to_move"]))
        exact = _exact_actions(puzzle)
        registered = list(map(int, puzzle.get("optimal_actions", exact)))
        if sorted(registered) != sorted(exact):
            raise ValueError("stored tactical target differs from exact five-in-a-row rules")
        if mode == "puct":
            assert search is not None
            cpu_started = time.process_time()
            result = search.search(board, str(puzzle["to_move"]), 64, add_root_noise=False,
                                   temperature=0.0, seed=int(hashlib.sha256(puzzle["puzzle_id"].encode()).hexdigest()[:8], 16))
            cpu_seconds = time.process_time() - cpu_started
            action = result.move[0] * 9 + result.move[1]
            search_stats = {
                "simulations": int(result.simulations), "nodes": int(result.tree_nodes),
                "depth": int(result.max_depth), "cpu_seconds": float(cpu_seconds),
            }
        else:
            cpu_started = time.process_time()
            coordinate, extra = _network_move(model, board, str(puzzle["to_move"]), torch.device("cpu"))
            cpu_seconds = time.process_time() - cpu_started
            action = coordinate[0] * 9 + coordinate[1]
            search_stats = {"simulations": 0, "nodes": 0, "depth": 0,
                            "cpu_seconds": float(cpu_seconds),
                            "value_prediction": float(extra["value"]),
                            "policy_entropy": float(extra["policy_entropy"])}
        records.append({
            "puzzle_id": puzzle["puzzle_id"], "category": puzzle["category"],
            "action": action, "action_coordinate": list(divmod(action, 9)),
            "optimal_actions": exact, "correct": action in exact,
            "mode": mode, **search_stats,
        })
    categories = {}
    for category in sorted({str(record["category"]) for record in records}):
        subset = [record for record in records if record["category"] == category]
        categories[category] = {
            "puzzles": len(subset),
            "correct": sum(bool(record["correct"]) for record in subset),
            "accuracy": float(np.mean([record["correct"] for record in subset])),
        }
    return {
        "mode": mode,
        "simulations_per_move": 64 if mode == "puct" else 0,
        "puzzles": len(records),
        "correct": sum(bool(record["correct"]) for record in records),
        "accuracy": float(np.mean([record["correct"] for record in records])),
        "categories": categories,
        "records": records,
    }


def play_v3_match(
    model: torch.nn.Module,
    opponent_model: torch.nn.Module | None,
    *,
    game_id: str,
    opening_group: dict[str, Any],
    agent_color: Literal["B", "W"],
    mode: Literal["network", "puct"],
    opponent: Literal["A", "B", "F2", "G0", "G-F2"],
    seed: int,
    simulations: int = 64,
    opponent_checkpoint_id: str | None = None,
    candidate_checkpoint_id: str | None = None,
    phase: str = "development",
    device: str = "cpu",
) -> dict[str, Any]:
    if mode == "puct" and simulations != 64:
        raise ValueError("G V3 comparisons use fixed PUCT-64")
    if opponent in {"G0", "G-F2"} and opponent_model is None:
        raise ValueError("a G-family opponent needs its frozen checkpoint model")
    engine_opponent = "MODEL" if opponent in {"G0", "G-F2"} else opponent
    model_opponent = opponent_model
    if opponent == "F2" and model_opponent is None:
        model_opponent = load_frozen_opponent("F2", device=device)
    result = play_evaluation_match(
        model, game_id=game_id,
        opening_group_id=str(opening_group["opening_group_id"]),
        arm_id=str(candidate_checkpoint_id or "G-V3"),
        mode=mode, opponent=engine_opponent,
        opening=[dict(move) for move in opening_group["opening"]],
        agent_color=agent_color, seed=int(seed), simulations=simulations,
        opponent_mode=mode if engine_opponent in {"F1", "F2", "MODEL"} else "network",
        opponent_simulations=simulations,
        opponent_model=model_opponent,
        device=device,
    )
    result.update({
        "phase": phase,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "opponent_checkpoint_id": opponent_checkpoint_id,
        "opponent_family": opponent,
        "opening_kind": opening_group.get("kind", "random4"),
    })
    return result


def _group_seed(row: dict[str, Any]) -> int:
    if row.get("opening_group_id") is not None:
        token = ":".join(str(row.get(key, "")) for key in (
            "seed", "opening_group_id", "arm", "candidate_checkpoint_id", "opponent", "mode",
        ))
    else:
        token = f"{row.get('seed', 0)}:{row.get('game_id', '')}"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def summarize_v3_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Score paired outcomes and compute the measured search/runtime costs."""
    # If summaries pool independent model seeds, keep each seed-opening pair as
    # the bootstrap cluster even though every seed shares the same held-out
    # opening coordinates.
    paired_rows = []
    for match in matches:
        row = dict(match)
        if "seed" in row:
            row["opening_group_id"] = f"seed-{int(row['seed'])}:{row['opening_group_id']}"
        paired_rows.append(row)
    summary = summarize_matches(paired_rows)
    stats = [match.get("stats", {}) for match in matches]
    summary.update({
        "total_cpu_seconds": float(sum(float(row.get("total_cpu_seconds", 0.0)) for row in stats)),
        "mean_cpu_seconds_per_game": float(np.mean([float(row.get("total_cpu_seconds", 0.0)) for row in stats])),
        "mean_agent_cpu_seconds_per_game": float(np.mean([float(row.get("agent_seconds", 0.0)) for row in stats])),
        "mean_opponent_cpu_seconds_per_game": float(np.mean([float(row.get("opponent_seconds", 0.0)) for row in stats])),
        "agent_simulations": int(sum(int(row.get("agent_simulations", 0)) for row in stats)),
        "mean_agent_nodes_per_game": float(np.mean([float(row.get("agent_nodes", 0)) for row in stats])),
        "mean_agent_search_depth": float(np.mean([float(row.get("agent_max_search_depth", 0)) for row in stats])),
        "mean_opponent_search_depth": float(np.mean([float(row.get("opponent_max_search_depth", 0)) for row in stats])),
        "candidate_moves": int(sum(
            sum(move.get("actor") == "agent" for move in match.get("moves", []))
            for match in matches
        )),
    })
    return summary


def v3_value_and_entropy_diagnostics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Use terminal game outcomes to audit candidate value calibration by mode."""
    grouped: dict[str, dict[str, list[float]]] = {
        mode: {"prediction": [], "target": [], "entropy": []} for mode in ("network", "puct")
    }
    for match in matches:
        mode = str(match.get("mode"))
        if mode not in grouped:
            continue
        winner = match.get("winner_color")
        for move in match.get("moves", []):
            if move.get("actor") != "agent":
                continue
            prediction = move.get("value") if mode == "network" else move.get("root_value")
            entropy = move.get("policy_entropy") if mode == "network" else move.get("search_entropy")
            if prediction is None:
                continue
            target = 0.0 if winner is None else 1.0 if move["player"] == winner else -1.0
            grouped[mode]["prediction"].append(float(prediction))
            grouped[mode]["target"].append(target)
            if entropy is not None and math.isfinite(float(entropy)):
                grouped[mode]["entropy"].append(float(entropy))
    result = {}
    for mode, values in grouped.items():
        predictions = np.asarray(values["prediction"], dtype=np.float64)
        targets = np.asarray(values["target"], dtype=np.float64)
        result[mode] = {
            "candidate_move_values": int(len(predictions)),
            "value_mse_vs_terminal_outcome": float(np.mean((predictions - targets) ** 2)) if len(predictions) else None,
            "value_mae_vs_terminal_outcome": float(np.mean(np.abs(predictions - targets))) if len(predictions) else None,
            "mean_candidate_policy_entropy": float(np.mean(values["entropy"])) if values["entropy"] else None,
        }
    return result


def promotion_decision(
    matches: list[dict[str, Any]],
    *,
    candidate_id: str,
    incumbent_id: str,
    min_score_rate: float = 0.625,
    lower_ci_minimum: float = 0.5,
    min_color_score_rate: float = 0.5,
    required_groups: int = 12,
    seed: int = 731091,
) -> dict[str, Any]:
    """Apply the fixed challenger gate to paired opening/color records."""
    if not candidate_id or not incumbent_id:
        raise ValueError("promotion IDs must not be empty")
    grouped: dict[str, dict[str, float]] = {}
    for row in matches:
        group = str(row.get("opening_group_id", ""))
        color = str(row.get("agent_color", ""))
        if not group or color not in {"B", "W"} or color in grouped.get(group, {}):
            raise ValueError("promotion games need unique paired B/W rows per opening group")
        grouped.setdefault(group, {})[color] = float(row["score_for_agent"])
    if len(grouped) != int(required_groups) or any(set(row) != {"B", "W"} for row in grouped.values()):
        raise ValueError(f"promotion requires exactly {required_groups} complete color-paired opening groups")
    per_group = np.asarray([np.mean([row["B"], row["W"]]) for row in grouped.values()], dtype=np.float64)
    overall = float(per_group.mean())
    rng = np.random.default_rng(int(seed))
    sample = rng.choice(per_group, size=(10_000, len(per_group)), replace=True).mean(axis=1)
    ci = [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]
    by_color = {
        color: {
            "games": sum(color in row for row in grouped.values()),
            "score_rate": float(np.mean([row[color] for row in grouped.values()])),
        } for color in ("B", "W")
    }
    promote = (
        overall >= float(min_score_rate)
        and ci[0] > float(lower_ci_minimum)
        and all(row["score_rate"] >= float(min_color_score_rate) for row in by_color.values())
    )
    return {
        "candidate_id": candidate_id,
        "incumbent_id": incumbent_id,
        "promote": promote,
        "games": len(matches),
        "paired_opening_groups": len(grouped),
        "points": float(per_group.sum()),
        "score_rate": overall,
        "paired_opening_bootstrap_ci95": ci,
        "by_color": by_color,
        "thresholds": {
            "min_score_rate": float(min_score_rate),
            "lower_ci_minimum_exclusive": float(lower_ci_minimum),
            "min_color_score_rate": float(min_color_score_rate),
            "required_groups": int(required_groups),
        },
    }


def build_v3_blind_plan(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """Predeclare each registered route/seed final checkpoint blind matchup."""
    validate_v3_schedule(schedule)
    plan = []
    for seed in list(map(int, schedule["seeds"])):
        for arm in ARMS:
            opponents = ["A", "B", "F2", "other_route"]
            if arm == "G0":
                opponents.insert(3, "own_initial")
            for opponent in opponents:
                for mode in ("network", "puct"):
                    for group in schedule["blind_openings"]:
                        for color in ("B", "W"):
                            game_id = (
                                f"g-v3-blind-{arm}-s{seed}-vs-{opponent}-{mode}-"
                                f"{group['opening_group_id']}-{color}"
                            )
                            plan.append({
                                "game_id": game_id,
                                "arm": arm,
                                "seed": seed,
                                "opponent": opponent,
                                "mode": mode,
                                "opening_group_id": group["opening_group_id"],
                                "opening": [dict(move) for move in group["opening"]],
                                "agent_color": color,
                                "opening_seed": int(group["opening_seed"]),
                                "phase": "blind",
                            })
    ids = [str(row["game_id"]) for row in plan]
    if len(ids) != len(set(ids)):
        raise ValueError("G V3 blind plan contains duplicate game IDs")
    return plan


def _load_checkpoint(root: Path, relative: str) -> tuple[torch.nn.Module, str]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing registered G V3 checkpoint {relative}")
    model = load_g_checkpoint(path, device="cpu")
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


def _append_match(
    root: Path,
    log_path: Path,
    match: dict[str, Any],
    seen: set[str],
    config: dict[str, Any],
) -> None:
    if not append_evaluation_match(log_path, match, seen):
        raise ValueError(f"duplicate V3 evaluation game ID {match['game_id']}")
    record_v3_cpu_event(
        root, f"evaluation:{match['game_id']}",
        float(match.get("stats", {}).get("total_cpu_seconds", 0.0)),
        category="evaluation",
    )


def _selection_state(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / "research/g_v3/results/selection/state.json"
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config["config_sha256"]:
            raise ValueError("selection state does not match the registered config")
        return state
    champions = {}
    for seed in config["seeds"]:
        for arm in ARMS:
            rel = f"research/g_v3/checkpoints/{arm}/seed-{int(seed)}/node_0.pt"
            model, sha = _load_checkpoint(root, rel)
            del model
            champions[f"{arm}:{int(seed)}"] = {
                "checkpoint_id": f"{arm}-seed-{int(seed)}@0",
                "checkpoint": rel,
                "checkpoint_sha256": sha,
                "last_selection_node": 0,
            }
    state = {"format_version": 1, "config_sha256": config["config_sha256"], "champions": champions}
    _atomic_json(path, state)
    return state


def finalize_v3_champions(root: str | Path) -> dict[str, Any]:
    """Alias each route's last promoted challenger as its terminal model."""
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    selection_path = root_path / "research/g_v3/results/selection/state.json"
    if not selection_path.is_file():
        raise ValueError("G V3 selection state is missing; development promotions must finish first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("config_sha256") != config["config_sha256"]:
        raise ValueError("G V3 selection state has a mismatched config hash")
    final_node = int(config["stage_nodes"][-1])
    decision_ids = {str(row.get("candidate_id")) for row in selection.get("decisions", [])}
    models: dict[str, Any] = {}
    for seed in map(int, config["seeds"]):
        for arm in ARMS:
            key = f"{arm}:{seed}"
            champion = selection["champions"][key]
            if f"{arm}-seed-{seed}@{final_node}" not in decision_ids:
                raise ValueError(f"{arm} seed {seed} has no terminal champion decision")
            source = root_path / champion["checkpoint"]
            raw = source.read_bytes()
            digest = _sha(raw)
            if digest != champion["checkpoint_sha256"]:
                raise ValueError(f"selected champion checkpoint checksum mismatch: {source}")
            target = root_path / f"research/g_v3/checkpoints/{arm}/seed-{seed}/final.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".pt.tmp")
            with temporary.open("wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            if _sha(target.read_bytes()) != digest:
                raise ValueError("final champion alias failed its checksum verification")
            models[key] = {
                "checkpoint_id": champion["checkpoint_id"],
                "checkpoint": target.relative_to(root_path).as_posix(),
                "selected_source": champion["checkpoint"],
                "sha256": digest,
                "selected_node": int(champion["last_selection_node"]),
            }
    payload = {"format_version": 1, "config_sha256": config["config_sha256"], "models": models}
    _atomic_json(root_path / "research/g_v3/results/selection/final_models.json", payload)
    return payload


def run_v3_tactical_stage(root: str | Path, *, node: int, seed: int) -> list[dict[str, Any]]:
    """Measure exact-rule tactics on each unselected current checkpoint."""
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    valid_nodes = [0, *map(int, config["stage_nodes"])]
    if int(seed) not in list(map(int, config["seeds"])) or int(node) not in valid_nodes:
        raise ValueError("tactical evaluation node or seed is not registered")
    puzzles = build_v3_tactical_puzzles()
    path = root_path / "research/g_v3/results/evaluation/tactics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if path.is_file():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = {str(row["run_id"]) for row in records}
    new_rows = []
    for arm in ARMS:
        checkpoint = f"research/g_v3/checkpoints/{arm}/seed-{int(seed)}/node_{int(node)}.pt"
        model, checkpoint_sha = _load_checkpoint(root_path, checkpoint)
        for mode in ("network", "puct"):
            run_id = f"tactic-{arm}-s{int(seed)}-n{int(node)}-{mode}"
            if run_id in ids:
                continue
            ensure_v3_cpu_room(root_path, float(config["hard_cpu_hour_cap"]) * 3600)
            wall_started = time.perf_counter()
            result = evaluate_v3_tactics(model, mode=mode, puzzles=puzzles, simulations=64)
            wall_seconds = time.perf_counter() - wall_started
            cpu_seconds = float(sum(item["cpu_seconds"] for item in result["records"]))
            row = {
                "run_id": run_id,
                "arm": arm,
                "seed": int(seed),
                "node": int(node),
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config["config_sha256"],
                "cpu_seconds": cpu_seconds,
                "wall_seconds": wall_seconds,
                "stats": {"total_cpu_seconds": cpu_seconds, "total_wall_seconds": wall_seconds},
                **result,
            }
            payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode()
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    count = os.write(descriptor, payload[offset:])
                    if count <= 0:
                        raise OSError("failed to append a complete tactical record")
                    offset += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            ids.add(run_id)
            records.append(row)
            new_rows.append(row)
            record_v3_cpu_event(root_path, f"tactics:{run_id}", cpu_seconds, category="tactics")
    return new_rows


def run_v3_development(root: str | Path, *, node: int, seed: int) -> list[dict[str, Any]]:
    """Run the fixed development grid and one 12-group incumbent challenge."""
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    if int(seed) not in list(map(int, config["seeds"])) or int(node) not in [0, *map(int, config["stage_nodes"])]:
        raise ValueError("development node or seed is not registered")
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    path = root_path / "research/g_v3/results/evaluation/development.jsonl"
    records, seen = load_evaluation_log(path)
    by_id = {str(row["game_id"]): row for row in records}
    current_models: dict[str, torch.nn.Module] = {}
    current_sha: dict[str, str] = {}
    for arm in ARMS:
        relative = f"research/g_v3/checkpoints/{arm}/seed-{int(seed)}/node_{int(node)}.pt"
        current_models[arm], current_sha[arm] = _load_checkpoint(root_path, relative)
    initial_paths = {
        arm: f"research/g_v3/checkpoints/{arm}/seed-{int(seed)}/node_0.pt" for arm in ARMS
    }
    initial_models: dict[str, torch.nn.Module] = {}
    initial_sha: dict[str, str] = {}
    for arm, relative in initial_paths.items():
        initial_models[arm], initial_sha[arm] = _load_checkpoint(root_path, relative)
    f2_model = load_frozen_opponent("F2", device="cpu")
    f2_model.checkpoint_metadata = {"checkpoint_sha256": F2_SHA256}
    new_rows = []

    for arm in ARMS:
        # G-F2's own initial network is byte-identical to frozen F2. Keep one
        # measured matchup and identify the alias in the report/config instead
        # of counting the same game twice.
        opponents = ["A", "B", "F2", "other_route"]
        if arm == "G0":
            opponents.insert(3, "own_initial")
        for opponent in opponents:
            if node == 0 and opponent == "own_initial":
                continue
            if opponent == "own_initial":
                opponent_model = initial_models[arm]
                opponent_id = f"{arm}-seed-{int(seed)}@0"
                opponent_family = arm
            elif opponent == "other_route":
                other = "G-F2" if arm == "G0" else "G0"
                opponent_model = current_models[other]
                opponent_id = f"{other}-seed-{int(seed)}@{int(node)}"
                opponent_family = other
            elif opponent == "F2":
                opponent_model = f2_model
                opponent_id = "F2@frozen"
                opponent_family = "F2"
            else:
                opponent_model = None
                opponent_id = opponent
                opponent_family = opponent
            for mode in ("network", "puct"):
                for group in schedule["development_openings"][:4]:
                    for color in ("B", "W"):
                        game_id = f"g-v3-dev-n{node}-{arm}-s{seed}-vs-{opponent}-{mode}-{group['opening_group_id']}-{color}"
                        if game_id in by_id:
                            continue
                        ensure_v3_cpu_room(root_path, float(config["hard_cpu_hour_cap"]) * 3600)
                        match = play_v3_match(
                            current_models[arm], opponent_model,
                            game_id=game_id, opening_group=group, agent_color=color,
                            mode=mode, opponent=opponent_family, seed=_group_seed({
                                "seed": seed, "opening_group_id": group["opening_group_id"],
                                "arm": arm, "candidate_checkpoint_id": f"{arm}@{node}",
                                "opponent": opponent, "mode": mode, "game_id": game_id,
                            }),
                            simulations=64, opponent_checkpoint_id=opponent_id,
                            candidate_checkpoint_id=f"{arm}-seed-{int(seed)}@{int(node)}", phase="development",
                        )
                        match["config_sha256"] = config["config_sha256"]
                        match["opening_seed"] = int(group["opening_seed"])
                        match["candidate_checkpoint_sha256"] = current_sha[arm]
                        match["opponent_checkpoint_sha256"] = (
                            initial_sha[arm] if opponent == "own_initial" else
                            current_sha["G-F2" if arm == "G0" else "G0"] if opponent == "other_route" else
                            F2_SHA256 if opponent == "F2" else None
                        )
                        _append_match(root_path, path, match, seen, config)
                        by_id[game_id] = match
                        new_rows.append(match)

    if int(node) == 0:
        return new_rows

    selection = _selection_state(root_path, config)
    selection_key = f"{arm}:{int(seed)}"
    state_path = root_path / "research/g_v3/results/selection/state.json"
    decisions_path = root_path / "research/g_v3/results/selection/stages.jsonl"
    decisions = []
    if decisions_path.is_file():
        decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
    logged = {str(row["candidate_id"]) for row in decisions}
    for saved_decision in selection.get("decisions", []):
        if str(saved_decision["candidate_id"]) not in logged:
            decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with decisions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(saved_decision, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            decisions.append(saved_decision)
            logged.add(str(saved_decision["candidate_id"]))
    decided = {str(row["candidate_id"]) for row in selection.get("decisions", [])}
    for arm in ARMS:
        selection_key = f"{arm}:{int(seed)}"
        candidate_id = f"{arm}-seed-{int(seed)}@{int(node)}"
        if candidate_id in decided:
            continue
        champion = selection["champions"][selection_key]
        if int(champion["last_selection_node"]) >= int(node):
            continue
        incumbent, incumbent_sha = _load_checkpoint(root_path, champion["checkpoint"])
        for group in schedule["development_openings"][:int(config["promotion"]["opening_groups"])]:
            for color in ("B", "W"):
                game_id = f"g-v3-promo-{candidate_id}-vs-{champion['checkpoint_id']}-{group['opening_group_id']}-{color}"
                if game_id in by_id:
                    continue
                ensure_v3_cpu_room(root_path, float(config["hard_cpu_hour_cap"]) * 3600)
                match = play_v3_match(
                    current_models[arm], incumbent,
                    game_id=game_id, opening_group=group, agent_color=color,
                    mode="puct", opponent=arm, seed=_group_seed({
                        "seed": seed, "opening_group_id": group["opening_group_id"],
                        "arm": arm, "candidate_checkpoint_id": candidate_id,
                        "opponent": champion["checkpoint_id"], "mode": "puct", "game_id": game_id,
                    }),
                    simulations=64, opponent_checkpoint_id=champion["checkpoint_id"],
                    candidate_checkpoint_id=candidate_id, phase="promotion",
                )
                match["config_sha256"] = config["config_sha256"]
                match["candidate_checkpoint_sha256"] = current_sha[arm]
                match["opponent_checkpoint_sha256"] = incumbent_sha
                _append_match(root_path, path, match, seen, config)
                by_id[game_id] = match
                new_rows.append(match)
        promotion_matches = [row for row in by_id.values()
                             if row.get("phase") == "promotion"
                             and row.get("candidate_checkpoint_id") == candidate_id
                             and row.get("opponent_checkpoint_id") == champion["checkpoint_id"]]
        gate = config["promotion"]
        decision = promotion_decision(
            promotion_matches,
            candidate_id=candidate_id,
            incumbent_id=champion["checkpoint_id"],
            min_score_rate=float(gate["min_score_rate"]),
            lower_ci_minimum=float(gate["paired_bootstrap_lower_95_gt"]),
            min_color_score_rate=float(gate["min_color_score_rate"]),
            required_groups=int(gate["opening_groups"]),
            seed=seed + node,
        )
        decision.update({
            "seed": int(seed), "arm": arm, "node": int(node),
            "candidate_checkpoint_sha256": current_sha[arm],
            "incumbent_checkpoint_sha256": incumbent_sha,
            "config_sha256": config["config_sha256"],
        })
        if decision["promote"]:
            selection["champions"][selection_key] = {
                "checkpoint_id": candidate_id,
                "checkpoint": f"research/g_v3/checkpoints/{arm}/seed-{int(seed)}/node_{int(node)}.pt",
                "checkpoint_sha256": current_sha[arm],
                "last_selection_node": int(node),
            }
        else:
            champion["last_selection_node"] = int(node)
        selection.setdefault("decisions", []).append(decision)
        _atomic_json(state_path, selection)
        if candidate_id not in logged:
            decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with decisions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            decisions.append(decision)
            logged.add(candidate_id)
    return new_rows


def run_v3_blind(root: str | Path) -> list[dict[str, Any]]:
    """Execute the frozen final-model blind plan once; outcomes never feed selection."""
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    final_manifest = finalize_v3_champions(root_path)
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    plan = build_v3_blind_plan(schedule)
    path = root_path / "research/g_v3/results/evaluation/blind.jsonl"
    records, seen = load_evaluation_log(path)
    by_id = {str(row["game_id"]): row for row in records}
    final_models: dict[tuple[str, int], torch.nn.Module] = {}
    initial_models: dict[tuple[str, int], torch.nn.Module] = {}
    final_checkpoint_sha: dict[tuple[str, int], str] = {}
    initial_checkpoint_sha: dict[tuple[str, int], str] = {}
    for seed in list(map(int, config["seeds"])):
        for arm in ARMS:
            final_models[(arm, seed)], final_checkpoint_sha[(arm, seed)] = _load_checkpoint(
                root_path, f"research/g_v3/checkpoints/{arm}/seed-{seed}/final.pt",
            )
            if final_checkpoint_sha[(arm, seed)] != final_manifest["models"][f"{arm}:{seed}"]["sha256"]:
                raise ValueError("blind model does not match the selected route champion")
            initial_models[(arm, seed)], initial_checkpoint_sha[(arm, seed)] = _load_checkpoint(
                root_path, f"research/g_v3/checkpoints/{arm}/seed-{seed}/node_0.pt",
            )
    f2_model = load_frozen_opponent("F2", device="cpu")
    f2_model.checkpoint_metadata = {"checkpoint_sha256": F2_SHA256}
    results = []
    for row in plan:
        if row["game_id"] in by_id:
            continue
        arm, seed, opponent = row["arm"], int(row["seed"]), str(row["opponent"])
        if opponent in {"A", "B"}:
            opponent_model, opponent_family, opponent_id = None, opponent, opponent
        elif opponent == "F2":
            opponent_model, opponent_family, opponent_id = f2_model, "F2", "F2@frozen"
        elif opponent == "own_initial":
            opponent_model, opponent_family = initial_models[(arm, seed)], arm
            opponent_id = f"{arm}-seed-{seed}@0"
        else:
            other = "G-F2" if arm == "G0" else "G0"
            opponent_model, opponent_family = final_models[(other, seed)], other
            opponent_id = final_manifest["models"][f"{other}:{seed}"]["checkpoint_id"]
        ensure_v3_cpu_room(root_path, float(config["hard_cpu_hour_cap"]) * 3600)
        match = play_v3_match(
            final_models[(arm, seed)], opponent_model,
            game_id=row["game_id"], opening_group=row, agent_color=row["agent_color"],
            mode=row["mode"], opponent=opponent_family,
            seed=_group_seed(row), simulations=64,
            candidate_checkpoint_id=final_manifest["models"][f"{arm}:{seed}"]["checkpoint_id"],
            opponent_checkpoint_id=opponent_id, phase="blind",
        )
        match["config_sha256"] = config["config_sha256"]
        match["opening_seed"] = row["opening_seed"]
        match["candidate_checkpoint_sha256"] = final_checkpoint_sha[(arm, seed)]
        match["opponent_checkpoint_sha256"] = (
            F2_SHA256 if opponent == "F2" else
            initial_checkpoint_sha[(arm, seed)] if opponent == "own_initial" else
            final_checkpoint_sha[("G-F2" if arm == "G0" else "G0", seed)] if opponent == "other_route" else
            None
        )
        _append_match(root_path, path, match, seen, config)
        by_id[row["game_id"]] = match
        results.append(match)
    return results
