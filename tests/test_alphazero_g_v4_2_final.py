from __future__ import annotations

import pytest

from src.research.alphazero_g.v4_2_final import _paired_difference


def _rows(scores):
    rows = []
    for index, (black, white) in enumerate(scores):
        for color, score in (("B", black), ("W", white)):
            rows.append({"opening_group_id": f"open-{index}", "agent_color": color,
                         "score_for_agent": score})
    return rows


def test_paired_score_difference_bootstraps_opening_groups_not_individual_games():
    candidate = _rows([(1.0, 0.0), (1.0, 1.0), (0.5, 0.5)])
    reference = _rows([(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)])
    result = _paired_difference(candidate, reference)
    assert result["paired_opening_groups"] == 3
    assert result["mean_score_rate_difference"] == pytest.approx(1 / 3)
    assert result["paired_bootstrap_ci95"][0] <= result["mean_score_rate_difference"]
    assert result["paired_bootstrap_ci95"][1] >= result["mean_score_rate_difference"]


def test_paired_score_difference_rejects_mismatched_blind_protocols():
    with pytest.raises(ValueError, match="identical groups and colors"):
        _paired_difference(_rows([(1.0, 0.0)]), _rows([(0.0, 0.0), (1.0, 1.0)]))
