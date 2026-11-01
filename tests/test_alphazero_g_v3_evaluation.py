import numpy as np
import torch

from src.research.alphazero_g.training import initialize_arm
from src.research.alphazero_g.v3_evaluation import (
    build_v3_blind_plan,
    build_v3_tactical_puzzles,
    evaluate_v3_tactics,
    promotion_decision,
    summarize_v3_matches,
    v3_value_and_entropy_diagnostics,
)
from src.research.alphazero_g.v3_schedule import build_v3_schedule


class HighestLegalPolicy(torch.nn.Module):
    def __init__(self, favored=80):
        super().__init__()
        self.favored = favored

    def forward(self, planes):
        logits = torch.zeros((planes.shape[0], 81), device=planes.device)
        logits[:, self.favored] = 10.0
        return logits, torch.zeros((planes.shape[0],), device=planes.device)


def _paired_scores(score_by_color):
    rows = []
    for group in range(12):
        for color in ("B", "W"):
            score = score_by_color[color]
            rows.append({
                "opening_group_id": f"dev-{group}",
                "agent_color": color,
                "score_for_agent": score,
                "outcome_for_agent": "win" if score == 1 else "draw" if score == 0.5 else "loss",
            })
    return rows


def test_tactical_labels_are_derived_from_legal_five_in_a_row_analysis():
    puzzles = build_v3_tactical_puzzles()
    by_category = {}
    for puzzle in puzzles:
        by_category.setdefault(puzzle["category"], []).append(puzzle)
    assert set(by_category) == {"direct_win", "unique_block", "edge", "false_threat"}
    for puzzle in puzzles:
        assert puzzle["optimal_actions"]
        if puzzle["category"] in {"unique_block", "false_threat"}:
            assert len(puzzle["optimal_actions"]) == 1
    assert all(any(
        divmod(action, 9)[0] in {0, 8} or divmod(action, 9)[1] in {0, 8}
        for action in puzzle["optimal_actions"]
    )
               for puzzle in by_category["edge"])


def test_network_tactics_evaluator_reports_policy_action_without_rule_override():
    puzzle = next(item for item in build_v3_tactical_puzzles() if item["category"] == "unique_block")
    model = HighestLegalPolicy(favored=80)
    result = evaluate_v3_tactics(model, mode="network", puzzles=[puzzle])
    row = result["records"][0]
    assert row["action"] == 80
    assert row["action"] not in puzzle["optimal_actions"]
    assert result["categories"]["unique_block"]["correct"] == 0


def test_promotion_requires_12_paired_groups_both_colors_and_configured_score_gate():
    result = promotion_decision(
        _paired_scores({"B": 1.0, "W": 1.0}), candidate_id="candidate@1k", incumbent_id="F2@0",
    )
    assert result["promote"] is True
    assert result["score_rate"] == 1.0
    assert result["paired_opening_groups"] == 12

    color_swing = promotion_decision(
        _paired_scores({"B": 1.0, "W": 0.0}), candidate_id="candidate@2k", incumbent_id="F2@0",
    )
    assert color_swing["promote"] is False
    assert color_swing["by_color"]["B"]["score_rate"] == 1.0
    assert color_swing["by_color"]["W"]["score_rate"] == 0.0


def test_blind_plan_uses_only_registered_blind_groups_and_has_color_pairs():
    schedule = build_v3_schedule(
        [510921, 510922], train_groups_per_seed=8, development_groups=12, blind_groups=20,
        excluded_openings=[],
    )
    plan = build_v3_blind_plan(schedule)
    assert plan
    assert all(row["opening_group_id"].startswith("g-v3-blind-") for row in plan)
    assert all(row["mode"] in {"network", "puct"} for row in plan)
    assert {row["agent_color"] for row in plan} == {"B", "W"}
    route = [row for row in plan if row["arm"] == "G0" and row["seed"] == 510921
             and row["opponent"] == "A" and row["mode"] == "puct"]
    assert len(route) == 40
    assert len({row["opening_group_id"] for row in route}) == 20
    assert len({row["game_id"] for row in plan}) == len(plan)


def test_v3_match_summary_recomputes_scores_and_keeps_paired_opening_uncertainty():
    rows = []
    for group, values in enumerate(((1.0, 0.0), (0.5, 0.5), (1.0, 1.0))):
        for color, score in zip(("B", "W"), values, strict=True):
            rows.append({
                "opening_group_id": f"o{group}",
                "agent_color": color,
                "score_for_agent": score,
                "outcome_for_agent": "win" if score == 1 else "draw" if score == 0.5 else "loss",
                "stats": {"total_cpu_seconds": 1.0, "agent_simulations": 64, "agent_max_search_depth": 3},
            })
    summary = summarize_v3_matches(rows)
    assert summary["wins"] == 3 and summary["draws"] == 2 and summary["losses"] == 1
    assert summary["points"] == 4.0
    assert summary["score_rate"] == 4.0 / 6.0
    assert summary["paired_opening_bootstrap_ci95"] is not None
    assert summary["mean_cpu_seconds_per_game"] == 1.0


def test_value_diagnostics_use_terminal_outcome_from_candidate_move_perspective():
    result = v3_value_and_entropy_diagnostics([{
        "mode": "network", "winner_color": "B",
        "moves": [
            {"actor": "agent", "player": "B", "value": 0.5, "policy_entropy": 1.0},
            {"actor": "MODEL", "player": "W", "value": -0.5, "policy_entropy": 1.0},
        ],
    }, {
        "mode": "puct", "winner_color": "W",
        "moves": [
            {"actor": "agent", "player": "B", "root_value": 0.2, "search_entropy": 0.3},
            {"actor": "agent", "player": "W", "root_value": -0.2, "search_entropy": 0.4},
        ],
    }])
    assert result["network"]["candidate_move_values"] == 1
    assert result["network"]["value_mse_vs_terminal_outcome"] == 0.25
    assert result["puct"]["candidate_move_values"] == 2
    assert result["puct"]["value_mse_vs_terminal_outcome"] == 1.44

