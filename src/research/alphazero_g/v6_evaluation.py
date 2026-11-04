"""Paired, idempotent V6 evaluations stored outside the frozen V5 results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from .v5_evaluation import play_match, summarize_paired_games


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "big")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _validate_reused_game(
    game: dict[str, Any], *, game_id: str, phase: str, candidate_name: str,
    opponent_name: str, opening: dict[str, Any], color: str, simulations: int,
) -> None:
    expected = {
        "game_id": game_id,
        "phase": phase,
        "candidate": candidate_name,
        "opponent": opponent_name,
        "opening_group_id": opening["opening_group_id"],
        "candidate_color": color,
        "simulations_per_reference_turn": int(simulations),
    }
    for key, value in expected.items():
        if game.get(key) != value:
            raise ValueError(f"V6 match ID collision or changed registration field: {key}")
    if not game.get("moves") or len(game.get("moves", [])) != int(game.get("move_count", -1)):
        raise ValueError("stored V6 evaluation is incomplete")


def run_match_set(
    *,
    root: str | Path,
    phase: str,
    candidate_name: str,
    opponent_name: str,
    openings: list[dict[str, Any]],
    pairs: int,
    candidate_reference_model=None,
    opponent_reference_model=None,
    candidate_g0_checkpoint: str | Path | None = None,
    opponent_g0_checkpoint: str | Path | None = None,
    simulations: int = 64,
    board_size: int = 9,
    n_in_row: int = 5,
    cpu_deadline: float | None = None,
) -> dict[str, Any]:
    """Run or resume paired color-swapped games, persisting each full game atomically."""
    root = Path(root)
    if int(pairs) < 1 or len(openings) < int(pairs):
        raise ValueError("V6 opening schedule does not cover the requested paired games")
    if int(simulations) not in {2, 64, 400}:
        raise ValueError("V6 registered evaluation budget is not supported")
    base = (root / "research/g_v6/results/matches" / _slug(phase)
            / f"{_slug(candidate_name)}_vs_{_slug(opponent_name)}_puct{int(simulations)}")
    complete_rows: list[dict[str, Any]] = []
    completed_pairs = 0
    for opening in openings[:int(pairs)]:
        pair_rows = []
        if cpu_deadline is not None and time.process_time() >= float(cpu_deadline):
            break
        for color in ("B", "W"):
            game_id = "v6m-" + _stable_seed(
                phase, candidate_name, opponent_name, str(simulations),
                str(opening["opening_group_id"]), color,
            ).__format__("08x")
            path = base / f"{game_id}.json"
            if path.exists():
                game = json.loads(path.read_text(encoding="utf-8"))
                _validate_reused_game(
                    game, game_id=game_id, phase=phase, candidate_name=candidate_name,
                    opponent_name=opponent_name, opening=opening, color=color,
                    simulations=simulations,
                )
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
                game["phase"] = phase
                game["configured_search_budget_root_visits"] = int(simulations)
                game["opening_seed"] = int(opening.get("seed", 0))
                _atomic_json(path, game)
            pair_rows.append(game)
        if len(pair_rows) == 2:
            complete_rows.extend(pair_rows)
            completed_pairs += 1
            _atomic_json(base / "progress.json", {
                "phase": phase, "candidate": candidate_name, "opponent": opponent_name,
                "simulations": int(simulations), "completed_pairs": completed_pairs,
                "expected_pairs": int(pairs),
                "completed_game_ids": [row["game_id"] for row in complete_rows],
            })
    summary = summarize_paired_games(
        complete_rows,
        seed=_stable_seed(phase, candidate_name, opponent_name, str(simulations), "bootstrap"),
    )
    summary.update({
        "phase": phase,
        "candidate": candidate_name,
        "opponent": opponent_name,
        "simulations_per_reference_turn": int(simulations),
        "configured_root_action_visits": int(simulations),
        "reference_configured_playouts_per_decision": int(simulations) + 1,
        "planned_pairs": int(pairs),
        "planned_games": 2 * int(pairs),
        "completed_pairs": completed_pairs,
        "complete": completed_pairs == int(pairs),
        "game_rows": [
            {key: row.get(key) for key in (
                "game_id", "opening_group_id", "candidate_color", "winner_color", "points",
                "termination", "move_count", "candidate_cpu_seconds", "opponent_cpu_seconds",
                "total_cpu_seconds", "total_wall_seconds",
            )}
            for row in complete_rows
        ],
    })
    _atomic_json(base / "summary.json", summary)
    return summary

