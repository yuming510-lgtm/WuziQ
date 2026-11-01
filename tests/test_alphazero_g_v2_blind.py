from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.research.alphazero_g.evaluation import load_evaluation_log, validate_match_record
from src.research.alphazero_g.training import initialize_arm, save_g_checkpoint
from src.research.alphazero_g.v2_schedule import build_v2_schedule
from src.research.alphazero_g.v2_blind import (
    build_v2_blind_plan,
    run_v2_blind,
    summarize_v2_results,
)


def _record(comparison, group, color, score):
    result = "win" if score == 1 else "loss" if score == 0 else "draw"
    opening = [
        {"player": "B", "row": 4, "col": 0},
        {"player": "W", "row": 0, "col": 0},
        {"player": "B", "row": 4, "col": 1},
        {"player": "W", "row": 0, "col": 1},
    ]
    winner = color if score == 1 else "W" if color == "B" else "B"
    continuation = (
        [("B", 4, 2), ("W", 0, 2), ("B", 4, 3), ("W", 0, 3), ("B", 4, 4)]
        if winner == "B" else
        [("B", 4, 2), ("W", 0, 2), ("B", 4, 3), ("W", 0, 3), ("B", 5, 0), ("W", 0, 4)]
    )
    moves = [dict(move, opening=True) for move in opening]
    moves.extend({"player": player, "row": row, "col": col, "opening": False}
                 for player, row, col in continuation)
    return {
        "format_version": 1,
        "source": "evaluation", "evaluation_phase": "blind",
        "game_id": f"{comparison}-{group}-{color}", "opening_group_id": group,
        "comparison_id": comparison, "agent_color": color,
        "outcome_for_agent": result, "score_for_agent": score,
        "arm_id": "test", "mode": "network", "opponent": "test",
        "opening": opening, "seed": 1, "simulations_per_agent_move": 0,
        "completed": True, "termination": "five_in_row", "winner_color": winner,
        "move_count": len(moves), "moves": moves,
        "stats": {"agent_seconds": 1.0, "opponent_seconds": 0.5,
                   "agent_simulations": 64, "agent_nodes": 50,
                   "agent_max_search_depth": 4, "total_cpu_seconds": 1.5},
    }


def test_full_blind_plan_has_560_unique_paired_games_and_fixed_models():
    schedule = build_v2_schedule(1301, train_groups=1, development_groups=1, blind_groups=20)
    plan = build_v2_blind_plan(schedule)

    assert len(plan) == 560
    assert len({row["game_id"] for row in plan}) == 560
    assert {row["agent_color"] for row in plan} == {"B", "W"}
    comparisons = {row["comparison_id"] for row in plan}
    assert "F2-PUCT64__vs__G-F2-PUCT64" in comparisons
    assert "G-F2-network__vs__A" in comparisons
    assert "G-F2-PUCT64__vs__B" in comparisons
    for comparison in comparisons:
        selected = [row for row in plan if row["comparison_id"] == comparison]
        assert len(selected) == 40
        by_opening = {}
        for row in selected:
            by_opening.setdefault(row["opening_group_id"], set()).add(row["agent_color"])
        assert len(by_opening) == 20
        assert all(colors == {"B", "W"} for colors in by_opening.values())


def test_blind_summary_reports_wdl_color_ci_compute_and_learning_delta():
    records = []
    for group in range(20):
        records.append(_record("F2-PUCT64__vs__G-F2-PUCT64", f"o{group}", "B", 1.0))
        records.append(_record("F2-PUCT64__vs__G-F2-PUCT64", f"o{group}", "W", 0.0))
        records.append(_record("F2-network__vs__F2-PUCT64", f"n{group}", "B", 0.0))
        records.append(_record("F2-network__vs__F2-PUCT64", f"n{group}", "W", 0.0))

    summary = summarize_v2_results(records)

    f2_vs_g = summary["comparisons"]["F2-PUCT64__vs__G-F2-PUCT64"]
    assert f2_vs_g["games"] == 40
    assert f2_vs_g["points"] == f2_vs_g["wins"] + f2_vs_g["draws"] / 2
    assert f2_vs_g["by_color"]["B"]["score_rate"] == 1.0
    assert f2_vs_g["by_color"]["W"]["score_rate"] == 0.0
    assert f2_vs_g["paired_opening_bootstrap_ci95"] == [0.5, 0.5]
    assert summary["learning"]["F2-PUCT64_to_G-F2-PUCT64"]["g_score_rate"] == 0.5
    assert summary["search"]["F2_network_to_PUCT64"]["puct_score_rate"] == 1.0


def test_blind_summary_rejects_missing_color_pair_and_invalid_record():
    with pytest.raises(ValueError, match="paired color games"):
        summarize_v2_results([_record("F2-network__vs__A", "o0", "B", 1.0)])
    invalid = _record("F2-network__vs__A", "o0", "B", 1.0)
    invalid["source"] = "selfplay"
    with pytest.raises(ValueError, match="completed evaluation"):
        validate_match_record(invalid)


def test_one_group_blind_run_is_idempotent_and_replay_validated(tmp_path, monkeypatch):
    schedule = build_v2_schedule(1302, train_groups=1, development_groups=1, blind_groups=1)
    config = json.loads(Path("research/g_v2/config/experiment.json").read_text())
    config = copy.deepcopy(config)
    config["evaluation"]["blind_groups"] = 1
    config["evaluation"]["blind_games"] = 28
    f2_path = tmp_path / "research/f_v2/checkpoints/best_f2.pt"
    f2_path.parent.mkdir(parents=True)
    f2_path.write_bytes(Path("research/f_v2/checkpoints/best_f2.pt").read_bytes())
    candidate = initialize_arm("G-F2", seed=1302)
    g_path = tmp_path / "candidate.pt"
    save_g_checkpoint(g_path, candidate, {"candidate_id": "lr_3e5@1000"})

    class FastPUCT:
        def __init__(self, model, **kwargs):
            pass

        def search(self, board, player, simulations, **kwargs):
            action = next(
                row * 9 + col for row in range(9) for col in range(9)
                if board.get((row, col)) is None
            )
            visits = np.zeros(81)
            visits[action] = 64
            return SimpleNamespace(
                move=divmod(action, 9), simulations=64, tree_nodes=64,
                network_evaluations=1, max_depth=3, root_value=0.0,
                visit_entropy=0.0,
            )

    monkeypatch.setattr("src.research.alphazero_g.v2_blind.PUCTSearch", FastPUCT)
    first = run_v2_blind(
        tmp_path, config, schedule, {"F2": f2_path, "G": g_path}, opponents=("A",)
    )
    second = run_v2_blind(
        tmp_path, config, schedule, {"F2": f2_path, "G": g_path}, opponents=("A",)
    )

    assert len(first) == len(second) == 20
    assert len({row["game_id"] for row in first}) == 20
    log, ids = load_evaluation_log(tmp_path / "research/g_v2/results/evaluation/blind.jsonl")
    assert len(log) == len(ids) == 20
    assert all(row["evaluation_phase"] == "blind" for row in log)
    for row in log:
        validate_match_record(row)
