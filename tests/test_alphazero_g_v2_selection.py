import json
import hashlib
from pathlib import Path

import pytest
import torch

from src.research.alphazero_g.evaluation import load_evaluation_log
from src.research.alphazero_g.training import initialize_arm, save_g_checkpoint
from src.research.alphazero_g.v2_schedule import build_v2_schedule
from src.research.alphazero_g.v2_selection import (
    _append_stage_once,
    run_v2_champion_matches,
    run_v2_development,
    select_v2_champion,
)


def _match(score, group, color, *, candidate="lr_1e4", opponent="F2@0", mode="puct", phase="development"):
    result = "win" if score == 1 else "loss" if score == 0 else "draw"
    return {
        "source": "evaluation", "evaluation_phase": phase,
        "game_id": f"{candidate}-{opponent}-{mode}-{group}-{color}",
        "opening_group_id": group, "candidate_id": candidate,
        "candidate_checkpoint_id": "F2@0" if candidate == "F2" else f"{candidate}@1000",
        "candidate_checkpoint_sha256": hashlib.sha256(("F2@0" if candidate == "F2" else f"{candidate}@1000").encode()).hexdigest(),
        "opponent": opponent,
        "opponent_checkpoint_id": opponent, "mode": mode, "agent_color": color,
        "opponent_checkpoint_sha256": hashlib.sha256(opponent.encode()).hexdigest(),
        "challenger_weights_sha256": hashlib.sha256((candidate + " weights").encode()).hexdigest(),
        "incumbent_weights_sha256": hashlib.sha256((opponent + " weights").encode()).hexdigest(),
        "outcome_for_agent": result, "score_for_agent": score,
        "evaluation_node": 1000,
    }


def _paired_scores(score, candidate, opponent):
    return [
        _match(score, f"dev-{group}", color, candidate=candidate, opponent=opponent)
        for group in range(8) for color in ("B", "W")
    ]


def _candidate(candidate, score, *, a_score=0.75, b_score=0.75, comparator="F2@0"):
    return {
        "candidate_id": candidate,
        "candidate_checkpoint_id": f"{candidate}@1000",
        "candidate_checkpoint": f"checkpoints/{candidate}/node_1000.pt",
        "candidate_checkpoint_sha256": hashlib.sha256(f"{candidate}@1000".encode()).hexdigest(),
        "comparison_id": comparator,
        "challenge_matches": _paired_scores(score, candidate, comparator),
        "development_matches": [
            *_paired_scores(a_score, candidate, "A"),
            *_paired_scores(b_score, candidate, "B"),
        ],
    }


def test_select_v2_champion_uses_paired_ci_color_and_regression_rules():
    incumbent = {"checkpoint_id": "F2@0", "checkpoint": "f2.pt",
                 "checkpoint_sha256": hashlib.sha256(b"F2@0").hexdigest()}
    baseline = [
        *_paired_scores(0.75, "F2", "A"),
        *_paired_scores(0.75, "F2", "B"),
    ]
    candidates = [
        _candidate("lr_1e4", 1.0),
        _candidate("lr_3e5", 1.0),
        _candidate("lr_1e3", 0.5),
    ]

    result = select_v2_champion(candidates, incumbent, baseline)

    assert result["promoted"] is True
    assert result["champion"]["candidate_id"] == "lr_3e5"
    assert result["candidates"]["lr_1e3"]["criteria"]["challenge_score"] is False
    assert result["candidates"]["lr_1e4"]["challenge"]["paired_opening_bootstrap_ci95"][0] > 0.5


def test_select_v2_champion_rejects_blind_records_and_f2_regression():
    incumbent = {"checkpoint_id": "F2@0", "checkpoint": "f2.pt",
                 "checkpoint_sha256": hashlib.sha256(b"F2@0").hexdigest()}
    baseline = [
        *_paired_scores(0.75, "F2", "A"),
        *_paired_scores(0.75, "F2", "B"),
    ]
    regressing = _candidate("lr_1e4", 1.0, a_score=0.5)
    result = select_v2_champion(
        [regressing, _candidate("lr_3e5", 0.5), _candidate("lr_1e3", 0.5)], incumbent, baseline
    )
    assert result["promoted"] is False
    assert result["candidates"]["lr_1e4"]["criteria"]["no_obvious_regression"] is False

    contaminated = _candidate("lr_3e5", 1.0)
    contaminated["challenge_matches"][0]["evaluation_phase"] = "blind"
    with pytest.raises(ValueError, match="development-only"):
        select_v2_champion(
            [contaminated, _candidate("lr_1e4", 0.5), _candidate("lr_1e3", 0.5)], incumbent, baseline
        )


def test_development_matcher_returns_eight_paired_color_games_per_condition(tmp_path):
    schedule = build_v2_schedule(910, train_groups=1, development_groups=8, blind_groups=20)
    checkpoint = Path("research/f_v2/checkpoints/best_f2.pt")

    frozen = tmp_path / checkpoint
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(Path(__file__).resolve().parents[1].joinpath(checkpoint).read_bytes())
    matches = run_v2_development(
        tmp_path, checkpoint, "F2", 0, schedule, 911,
        opponents=("A",), modes=("network",),
    )

    assert len(matches) == 16
    assert {row["agent_color"] for row in matches} == {"B", "W"}
    assert {row["evaluation_phase"] for row in matches} == {"development"}
    loaded, ids = load_evaluation_log(tmp_path / "research/g_v2/results/evaluation/development.jsonl")
    assert len(loaded) == len(ids) == 16


def test_champion_matcher_rejects_identical_checkpoint_weights(tmp_path):
    model = initialize_arm("G-F2", seed=912)
    same = tmp_path / "same.pt"
    save_g_checkpoint(same, model, {"candidate_id": "copy"})
    schedule = build_v2_schedule(912, train_groups=1, development_groups=8, blind_groups=20)

    with pytest.raises(ValueError, match="identical weights"):
        run_v2_champion_matches(
            tmp_path, same, "lr_1e4@1000", same, "lr_3e5@1000",
            schedule, 913,
        )


def test_stage_history_is_idempotent_and_detects_divergent_resume(tmp_path):
    history = tmp_path / "stages.jsonl"
    stage = {"node": 1000, "champion": "F2@0", "promoted": False}

    assert _append_stage_once(history, stage) is True
    assert _append_stage_once(history, stage) is False
    assert history.read_text().count('"node":1000') == 1

    with pytest.raises(ValueError, match="conflicting stage history"):
        _append_stage_once(history, {**stage, "promoted": True})
