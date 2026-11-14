from __future__ import annotations

from scripts.report_g_v6_4 import build_score_chart


def test_score_chart_shows_all_ladder_levels_and_the_fifty_percent_reference() -> None:
    conditions = {
        name: {"score_rate": score, "paired_opening_bootstrap_ci95": ci}
        for name, score, ci in (
            ("A", 0.4, [0.2, 0.6]),
            ("C@200ms", 0.2, [0.0, 0.5]),
            ("C@1s", 0.3, [0.0, 1.0]),
            ("C@5s", 0.0, [0.0, 0.0]),
        )
    }

    chart = build_score_chart(conditions)

    for label in ("A", "C@200ms", "C@1s", "C@5s", "50%"):
        assert label in chart
