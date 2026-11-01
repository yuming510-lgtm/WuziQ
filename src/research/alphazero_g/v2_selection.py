"""Development-only evaluation and champion selection for WuziQ G V2."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ...gomoku import GomokuBoard, Player
from ..imitation_f.model import load_f_checkpoint, select_device
from .evaluation import (
    F2_SHA256,
    append_evaluation_match,
    load_evaluation_log,
    load_frozen_opponent,
    play_evaluation_match,
    summarize_matches,
    validate_match_record,
)
from .puct import PUCTSearch
from .training import F2_SHA256 as TRAINING_F2_SHA256, load_g_checkpoint
from .v2_budget import ensure_cpu_room
from .v2_runner import run_v2_candidate
from .v2_schedule import validate_v2_schedule

LR_ORDER = {"lr_3e5": 0, "lr_1e4": 1, "lr_1e3": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(bytes(value.numpy().tobytes()))
    return digest.hexdigest()


def _load_checkpoint(root: Path, path: Path, checkpoint_id: str, device: str) -> tuple[torch.nn.Module, str]:
    resolved = path if path.is_absolute() else root / path
    if checkpoint_id.startswith("F2"):
        if _sha256(resolved) != TRAINING_F2_SHA256:
            raise ValueError("frozen F2 checkpoint hash differs from the registered baseline")
        model = load_frozen_opponent("F2", device=device)
    else:
        model = load_g_checkpoint(resolved, device=device)
    model.eval()
    return model, _weights_sha256(model)


def _game_seed(seed: int, game_id: str) -> int:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[:8], "big")
    return value % (2**63 - 1)


def _pair_group(records: list[dict[str, Any]], *, expected_games: int = 16) -> dict[str, Any]:
    if len(records) != expected_games:
        raise ValueError(f"development pair needs exactly {expected_games} complete games")
    if any(row.get("evaluation_phase") != "development" for row in records):
        raise ValueError("development-only selection rejects non-development records")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["opening_group_id"]), []).append(record)
    if len(grouped) != expected_games // 2 or any(
        len(rows) != 2 or {row.get("agent_color") for row in rows} != {"B", "W"} for rows in grouped.values()
    ):
        raise ValueError("development records must contain paired black/white games per opening")
    return summarize_matches(records)


def run_v2_development(
    root: str | Path,
    checkpoint: str | Path,
    candidate_id: str,
    node: int,
    schedule: dict[str, Any],
    seed: int,
    *,
    opponents: tuple[str, ...] = ("F2", "A", "B"),
    modes: tuple[str, ...] = ("network", "puct"),
    device: str = "cpu",
    log_path: str | Path | None = None,
    cpu_cap_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Run idempotent fixed-opening development matches for one frozen checkpoint."""
    validate_v2_schedule(schedule)
    if not candidate_id or int(node) < 0:
        raise ValueError("development evaluation needs a candidate identifier and nonnegative node")
    if not opponents or not set(opponents).issubset({"F2", "A", "B"}):
        raise ValueError("development opponents must be a nonempty subset of F2, A and B")
    if not modes or not set(modes).issubset({"network", "puct"}):
        raise ValueError("development modes must be network or puct")
    if candidate_id == "F2":
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(root).resolve() / checkpoint_path
        if _sha256(checkpoint_path) != F2_SHA256:
            raise ValueError("node-zero baseline is not the frozen V2 F2 checkpoint")
        model = load_frozen_opponent("F2", device=device)
        checkpoint_sha = F2_SHA256
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(root).resolve() / checkpoint_path
        model = load_g_checkpoint(checkpoint_path, device=device)
        checkpoint_sha = _sha256(checkpoint_path)
    candidate_checkpoint_id = "F2@0" if candidate_id == "F2" else f"{candidate_id}@{int(node)}"
    target = Path(log_path) if log_path else Path(root) / "research/g_v2/results/evaluation/development.jsonl"
    matches, seen = load_evaluation_log(target)
    existing = {str(item["game_id"]): item for item in matches}
    result_rows: list[dict[str, Any]] = []
    for mode in modes:
        for opponent in opponents:
            for group in schedule["development_openings"]:
                for color in ("B", "W"):
                    game_id = (
                        f"g-v2-dev-{candidate_checkpoint_id}-{checkpoint_sha[:10]}-"
                        f"{mode}-{opponent}-{group['opening_group_id']}-{color}"
                    )
                    record = existing.get(game_id)
                    if record is None:
                        if cpu_cap_seconds is not None:
                            ensure_cpu_room(root, cpu_cap_seconds)
                        record = play_evaluation_match(
                            model,
                            game_id=game_id,
                            opening_group_id=group["opening_group_id"],
                            arm_id=candidate_id,
                            mode=mode,
                            opponent=opponent,
                            opening=group["opening"],
                            agent_color=color,
                            seed=_game_seed(int(seed), game_id),
                            simulations=64,
                            device=device,
                        )
                        record.update({
                            "evaluation_phase": "development",
                            "candidate_id": candidate_id,
                            "candidate_checkpoint_id": candidate_checkpoint_id,
                            "candidate_checkpoint_sha256": checkpoint_sha,
                            "opponent_checkpoint_id": "F2@0" if opponent == "F2" else opponent,
                            "evaluation_node": int(node),
                            "opponent_time_limit_seconds": 0.05 if opponent == "B" else None,
                        })
                        append_evaluation_match(target, record, seen)
                        existing[game_id] = record
                    if record.get("candidate_checkpoint_sha256") != checkpoint_sha:
                        raise ValueError("existing development game ID belongs to another checkpoint hash")
                    result_rows.append(record)
    return result_rows


def _play_checkpoint_game(
    challenger: torch.nn.Module,
    incumbent: torch.nn.Module,
    *,
    game_id: str,
    opening_group: dict[str, Any],
    challenger_id: str,
    incumbent_id: str,
    challenger_sha: str,
    incumbent_sha: str,
    challenger_weights_sha: str,
    incumbent_weights_sha: str,
    agent_color: Player,
    seed: int,
    device: str,
) -> dict[str, Any]:
    board = GomokuBoard(9)
    moves: list[dict[str, Any]] = []
    expected: Player = "B"
    for item in opening_group["opening"]:
        if item["player"] != expected:
            raise ValueError("champion opening must alternate from black")
        coord = (int(item["row"]), int(item["col"]))
        board.place_stone(expected, coord)
        if board.check_victory(coord, expected) or board.is_full():
            raise ValueError("champion opening must be nonterminal")
        moves.append({"player": expected, "row": coord[0], "col": coord[1], "opening": True})
        expected = "W" if expected == "B" else "B"
    to_move = expected
    challenger_search = PUCTSearch(challenger, device=device)
    incumbent_search = PUCTSearch(incumbent, device=device)
    rng = np.random.default_rng(int(seed))
    started = time.perf_counter()
    cpu_started = time.process_time()
    challenger_seconds = incumbent_seconds = 0.0
    challenger_sims = incumbent_sims = 0
    challenger_nodes = incumbent_nodes = 0
    challenger_depth = incumbent_depth = 0
    winner: Player | None = None
    termination = ""
    while True:
        is_challenger = to_move == agent_color
        move_started = time.perf_counter()
        search = challenger_search if is_challenger else incumbent_search
        result = search.search(
            board,
            to_move,
            64,
            add_root_noise=False,
            temperature=0.0,
            seed=int(rng.integers(0, 2**63 - 1)),
        )
        coord = result.move
        elapsed = time.perf_counter() - move_started
        actor = "challenger" if is_challenger else "incumbent"
        if is_challenger:
            challenger_seconds += elapsed
            challenger_sims += int(result.simulations)
            challenger_nodes += int(result.tree_nodes)
            challenger_depth = max(challenger_depth, int(result.max_depth))
        else:
            incumbent_seconds += elapsed
            incumbent_sims += int(result.simulations)
            incumbent_nodes += int(result.tree_nodes)
            incumbent_depth = max(incumbent_depth, int(result.max_depth))
        moves.append({
            "player": to_move, "row": coord[0], "col": coord[1], "opening": False,
            "actor": actor, "elapsed_seconds": elapsed, "simulations": int(result.simulations),
            "nodes": int(result.tree_nodes), "network_evaluations": int(result.network_evaluations),
            "search_depth": int(result.max_depth), "root_value": float(result.root_value),
            "search_entropy": float(result.visit_entropy),
        })
        board.place_stone(to_move, coord)
        if board.check_victory(coord, to_move):
            winner, termination = to_move, "five_in_row"
            break
        if board.is_full():
            winner, termination = None, "full_board"
            break
        to_move = "W" if to_move == "B" else "B"
    score = 0.5 if winner is None else 1.0 if winner == agent_color else 0.0
    return {
        "format_version": 1,
        "source": "evaluation",
        "evaluation_phase": "development",
        "game_id": game_id,
        "opening_group_id": opening_group["opening_group_id"],
        "arm_id": challenger_id.split("@", 1)[0],
        "candidate_id": challenger_id.split("@", 1)[0],
        "candidate_checkpoint_id": challenger_id,
        "candidate_checkpoint_sha256": challenger_sha,
        "opponent": incumbent_id,
        "opponent_checkpoint_id": incumbent_id,
        "opponent_checkpoint_sha256": incumbent_sha,
        "challenger_weights_sha256": challenger_weights_sha,
        "incumbent_weights_sha256": incumbent_weights_sha,
        "mode": "puct",
        "opening": [dict(move) for move in opening_group["opening"]],
        "agent_color": agent_color,
        "seed": int(seed),
        "simulations_per_agent_move": 64,
        "completed": True,
        "termination": termination,
        "winner_color": winner,
        "outcome_for_agent": "draw" if winner is None else "win" if winner == agent_color else "loss",
        "score_for_agent": score,
        "move_count": len(moves),
        "moves": moves,
        "stats": {
            "agent_seconds": challenger_seconds,
            "opponent_seconds": incumbent_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "total_cpu_seconds": time.process_time() - cpu_started,
            "agent_simulations": challenger_sims,
            "opponent_simulations": incumbent_sims,
            "agent_nodes": challenger_nodes,
            "opponent_nodes": incumbent_nodes,
            "agent_max_search_depth": challenger_depth,
            "opponent_max_search_depth": incumbent_depth,
        },
    }


def run_v2_champion_matches(
    root: str | Path,
    challenger_checkpoint: str | Path,
    challenger_id: str,
    incumbent_checkpoint: str | Path,
    incumbent_id: str,
    schedule: dict[str, Any],
    seed: int,
    *,
    device: str = "cpu",
    log_path: str | Path | None = None,
    cpu_cap_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Play one paired, color-swapped PUCT64 match per development opening."""
    validate_v2_schedule(schedule)
    root_path = Path(root).resolve()
    challenger_path = Path(challenger_checkpoint)
    incumbent_path = Path(incumbent_checkpoint)
    if not challenger_path.is_absolute():
        challenger_path = root_path / challenger_path
    if not incumbent_path.is_absolute():
        incumbent_path = root_path / incumbent_path
    challenger, challenger_weights_sha = _load_checkpoint(root_path, challenger_path, challenger_id, device)
    incumbent, incumbent_weights_sha = _load_checkpoint(root_path, incumbent_path, incumbent_id, device)
    if challenger_weights_sha == incumbent_weights_sha:
        raise ValueError("champion challenger checkpoints have identical weights")
    challenger_sha, incumbent_sha = _sha256(challenger_path), _sha256(incumbent_path)
    target = Path(log_path) if log_path else root_path / "research/g_v2/results/evaluation/champion.jsonl"
    existing_rows, seen = load_evaluation_log(target)
    existing = {str(item["game_id"]): item for item in existing_rows}
    result_rows = []
    for group in schedule["development_openings"]:
        for color in ("B", "W"):
            game_id = f"g-v2-champion-{challenger_id}-{challenger_sha[:8]}-vs-{incumbent_id}-{incumbent_sha[:8]}-{group['opening_group_id']}-{color}"
            record = existing.get(game_id)
            if record is None:
                if cpu_cap_seconds is not None:
                    ensure_cpu_room(root_path, cpu_cap_seconds)
                record = _play_checkpoint_game(
                    challenger, incumbent, game_id=game_id, opening_group=group,
                    challenger_id=challenger_id, incumbent_id=incumbent_id,
                    challenger_sha=challenger_sha, incumbent_sha=incumbent_sha,
                    challenger_weights_sha=challenger_weights_sha,
                    incumbent_weights_sha=incumbent_weights_sha,
                    agent_color=color, seed=_game_seed(seed, game_id), device=device,
                )
                append_evaluation_match(target, record, seen)
                existing[game_id] = record
            if record.get("challenger_weights_sha256") != challenger_weights_sha or record.get("incumbent_weights_sha256") != incumbent_weights_sha:
                raise ValueError("existing champion match ID belongs to different checkpoint weights")
            result_rows.append(record)
    return result_rows


def _summarize_opponent(matches: list[dict[str, Any]], candidate_id: str, mode: str, opponent: str) -> dict[str, Any]:
    selected = [
        row for row in matches
        if row.get("candidate_id") == candidate_id and row.get("mode") == mode and row.get("opponent") == opponent
    ]
    return _pair_group(selected)


def select_v2_champion(
    candidates: list[dict[str, Any]],
    incumbent: dict[str, Any],
    f2_baseline_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered development-only challenge and regression rules."""
    if not candidates:
        raise ValueError("champion selection needs all registered candidate results")
    if {str(item.get("candidate_id")) for item in candidates} != set(LR_ORDER):
        raise ValueError("champion selection needs exactly the three learning-rate candidates")
    all_records = list(f2_baseline_matches)
    for item in candidates:
        all_records.extend(item.get("challenge_matches", []))
        all_records.extend(item.get("development_matches", []))
    if any(record.get("evaluation_phase") != "development" for record in all_records):
        raise ValueError("champion selection is development-only; blind or preflight records are forbidden")
    for record in f2_baseline_matches:
        if record.get("candidate_id") != "F2" or record.get("mode") not in {"network", "puct"}:
            raise ValueError("regression baseline must be frozen F2 development results")
    baseline_a = _summarize_opponent(f2_baseline_matches, "F2", "puct", "A")
    baseline_b = _summarize_opponent(f2_baseline_matches, "F2", "puct", "B")
    decisions: dict[str, Any] = {}
    qualified: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in candidates:
        candidate_id = str(item["candidate_id"])
        comparator_id = str(item.get("comparison_id", ""))
        if comparator_id != str(incumbent.get("checkpoint_id", "")):
            raise ValueError("all candidate checkpoints at one node must challenge the same frozen incumbent")
        challenge = item.get("challenge_matches", [])
        expected_candidate_sha = str(item.get("candidate_checkpoint_sha256", ""))
        expected_candidate_checkpoint_id = str(item.get("candidate_checkpoint_id", ""))
        expected_incumbent_sha = str(incumbent.get("checkpoint_sha256", ""))
        if any(row.get("candidate_id") != candidate_id or row.get("mode") != "puct"
               or row.get("candidate_checkpoint_id") != expected_candidate_checkpoint_id
               or row.get("candidate_checkpoint_sha256") != expected_candidate_sha
               or row.get("opponent_checkpoint_id") != comparator_id
               or row.get("opponent_checkpoint_sha256") != expected_incumbent_sha
               or row.get("challenger_weights_sha256") == row.get("incumbent_weights_sha256")
               for row in challenge):
            raise ValueError("candidate challenge records do not match their frozen checkpoint IDs")
        challenge_summary = _pair_group(challenge)
        dev_rows = item.get("development_matches", [])
        if any(row.get("candidate_id") != candidate_id
               or row.get("candidate_checkpoint_id") != expected_candidate_checkpoint_id
               or row.get("candidate_checkpoint_sha256") != expected_candidate_sha
               for row in dev_rows):
            raise ValueError("candidate development records are attributed to the wrong candidate")
        score_a = _summarize_opponent(dev_rows, candidate_id, "puct", "A")
        score_b = _summarize_opponent(dev_rows, candidate_id, "puct", "B")
        color = challenge_summary["by_color"]
        ci = challenge_summary["paired_opening_bootstrap_ci95"]
        criteria = {
            "challenge_score": challenge_summary["score_rate"] >= 0.625,
            "paired_ci_lower": ci is not None and ci[0] > 0.5,
            "black_score": float(color["B"]["score_rate"] or 0.0) >= 0.5,
            "white_score": float(color["W"]["score_rate"] or 0.0) >= 0.5,
            "no_obvious_regression": (
                score_a["score_rate"] >= baseline_a["score_rate"] - 0.125
                and score_b["score_rate"] >= baseline_b["score_rate"] - 0.125
            ),
        }
        eligible = all(criteria.values())
        decisions[candidate_id] = {
            "comparison_id": comparator_id,
            "challenge": challenge_summary,
            "puct_vs_a": score_a,
            "puct_vs_b": score_b,
            "f2_puct_baseline_a": baseline_a,
            "f2_puct_baseline_b": baseline_b,
            "criteria": criteria,
            "eligible": eligible,
            "checkpoint": str(item.get("candidate_checkpoint", "")),
            "candidate_checkpoint_id": str(item.get("candidate_checkpoint_id", "")),
            "candidate_checkpoint_sha256": str(item.get("candidate_checkpoint_sha256", "")),
        }
        if eligible:
            qualified.append((item, challenge_summary))
    if qualified:
        winner, summary = sorted(
            qualified,
            key=lambda pair: (
                -float(pair[1]["score_rate"]),
                -float(pair[1]["paired_opening_bootstrap_ci95"][0]),
                LR_ORDER[str(pair[0]["candidate_id"])],
            ),
        )[0]
        champion = {
            "candidate_id": str(winner["candidate_id"]),
            "checkpoint_id": str(winner["candidate_checkpoint_id"]),
            "checkpoint": str(winner["candidate_checkpoint"]),
            "checkpoint_sha256": str(winner["candidate_checkpoint_sha256"]),
            "challenge_score": float(summary["score_rate"]),
        }
        return {"promoted": True, "champion": champion, "candidates": decisions,
                "f2_puct_baseline_a": baseline_a, "f2_puct_baseline_b": baseline_b}
    return {"promoted": False, "champion": dict(incumbent), "candidates": decisions,
            "f2_puct_baseline_a": baseline_a, "f2_puct_baseline_b": baseline_b}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_stage_once(path: Path, stage: dict[str, Any]) -> bool:
    """Append exactly one deterministic stage result, tolerating a lost state write."""
    canonical = json.dumps(stage, sort_keys=True, separators=(",", ":"), allow_nan=False)
    existing_by_node: dict[int, str] = {}
    if path.exists():
        raw = path.read_bytes()
        complete_end = raw.rfind(b"\n") + 1
        if complete_end != len(raw):
            with path.open("r+b") as handle:
                handle.truncate(complete_end)
                handle.flush()
                os.fsync(handle.fileno())
            raw = raw[:complete_end]
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                row = json.loads(line)
                node = int(row["node"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid stage history at line {line_number}") from exc
            if node in existing_by_node:
                raise ValueError(f"duplicate stage node {node} in history")
            existing_by_node[node] = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    node = int(stage["node"])
    prior = existing_by_node.get(node)
    if prior is not None:
        if prior != canonical:
            raise ValueError(f"conflicting stage history for node {node}")
        return False
    _append_jsonl(path, stage)
    return True


def run_v2_growth(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resume staged training, development-only challenges and incumbent selection."""
    root_path = Path(root).resolve()
    if config.get("registered") is not True:
        raise ValueError("G V2 growth runner requires the registered configuration")
    schedule_path = root_path / config["opening_schedule_path"]
    schedule = json.loads(schedule_path.read_text())
    validate_v2_schedule(schedule)
    if schedule["schedule_sha256"] != config["opening_schedule_sha256"]:
        raise ValueError("G V2 growth schedule hash does not match the registered config")
    f2_path = root_path / "research/f_v2/checkpoints/best_f2.pt"
    state_path = root_path / "research/g_v2/results/selection/state.json"
    history_path = root_path / "research/g_v2/results/selection/stages.jsonl"
    dev_log = root_path / "research/g_v2/results/evaluation/development.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config_sha256") != hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest():
            raise ValueError("resume state belongs to a different registered G V2 config")
    else:
        state = {
            "format_version": 1,
            "last_completed_node": 0,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "schedule_sha256": schedule["schedule_sha256"],
            "champion": {
                "candidate_id": "F2", "checkpoint_id": "F2@0",
                "checkpoint": "research/f_v2/checkpoints/best_f2.pt",
                "checkpoint_sha256": _sha256(f2_path),
            },
        }
        _atomic_json(state_path, state)
    baseline_matches = run_v2_development(
        root_path, f2_path, "F2", 0, schedule, int(config["seed"]),
        opponents=("A", "B"), modes=("network", "puct"),
        cpu_cap_seconds=float(config.get("hard_cpu_hour_cap", 2.0)) * 3600.0,
    )
    for node in map(int, config["stage_nodes"]):
        if node <= int(state["last_completed_node"]):
            continue
        frozen_champion = dict(state["champion"])
        candidates: list[dict[str, Any]] = []
        for lr_id in sorted(config["candidate_learning_rates"], key=lambda value: LR_ORDER[value]):
            training = run_v2_candidate(root_path, config, lr_id, until_node=node)
            checkpoint = Path(training["checkpoint"])
            candidate_sha = _sha256(checkpoint)
            checkpoint_id = f"{lr_id}@{node}"
            development = run_v2_development(
                root_path, checkpoint, lr_id, node, schedule, int(config["seed"]),
                opponents=("F2", "A", "B"), modes=("network", "puct"),
                cpu_cap_seconds=float(config.get("hard_cpu_hour_cap", 2.0)) * 3600.0,
            )
            challenge = run_v2_champion_matches(
                root_path, checkpoint, checkpoint_id,
                frozen_champion["checkpoint"], frozen_champion["checkpoint_id"],
                schedule, int(config["seed"]) + node,
                cpu_cap_seconds=float(config.get("hard_cpu_hour_cap", 2.0)) * 3600.0,
            )
            candidates.append({
                "candidate_id": lr_id,
                "candidate_checkpoint_id": checkpoint_id,
                "candidate_checkpoint": str(checkpoint.relative_to(root_path)),
                "candidate_checkpoint_sha256": candidate_sha,
                "comparison_id": frozen_champion["checkpoint_id"],
                "challenge_matches": challenge,
                "development_matches": development,
                "budgeted_environment_steps": int(training["budgeted_environment_steps"]),
                "optimizer_updates": int(training["optimizer_updates"]),
                "active_cpu_seconds": float(training["active_cpu_seconds"]),
            })
        selected = select_v2_champion(candidates, frozen_champion, baseline_matches)
        if selected["promoted"]:
            next_champion = dict(selected["champion"])
            next_champion["checkpoint"] = str(Path(next_champion["checkpoint"]).relative_to(root_path))
        else:
            next_champion = frozen_champion
        best_item = sorted(
            selected["candidates"].items(),
            key=lambda pair: (
                -float(pair[1]["challenge"]["score_rate"]),
                -float(pair[1]["challenge"]["paired_opening_bootstrap_ci95"][0]),
                LR_ORDER[pair[0]],
            ),
        )[0]
        best_for_blind = {
            "candidate_id": best_item[0],
            "checkpoint_id": best_item[1]["candidate_checkpoint_id"],
            "checkpoint": best_item[1]["checkpoint"],
            "checkpoint_sha256": best_item[1]["candidate_checkpoint_sha256"],
            "selection_node": node,
            "selection_rule": "development champion score, then paired CI lower bound, then LR order",
            "promoted_to_champion": bool(selected["promoted"] and selected["champion"]["candidate_id"] == best_item[0]),
        }
        stage = {
            "node": node,
            "frozen_incumbent": frozen_champion,
            "selected": selected,
            "candidate_training": [
                {key: item[key] for key in ("candidate_id", "candidate_checkpoint_id", "budgeted_environment_steps",
                 "optimizer_updates", "active_cpu_seconds")} for item in candidates
            ],
            "result_count": sum(len(item["challenge_matches"]) + len(item["development_matches"]) for item in candidates),
        }
        _append_stage_once(history_path, stage)
        state = {
            **state,
            "last_completed_node": node,
            "champion": next_champion,
            "best_candidate_for_blind": best_for_blind,
            "last_stage": stage,
        }
        _atomic_json(state_path, state)
    return state
