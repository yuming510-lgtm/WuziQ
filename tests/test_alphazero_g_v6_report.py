from __future__ import annotations

import json

from scripts import report_g_v6


def test_v5_comparison_rows_include_game_count_and_match_table_width(tmp_path, monkeypatch) -> None:
    base = tmp_path / "research/g_v5/results"
    base.mkdir(parents=True)
    (base / "9x9_arms.json").write_text(json.dumps([
        {
            "arm": "reference400_empty",
            "stages": [{
                "candidate": "reference400_empty@1500",
                "opponent": "A",
                "games": 16,
                "wins": 0,
                "draws": 1,
                "losses": 15,
                "score_rate": 0.03125,
                "paired_opening_bootstrap_ci95": [0.0, 0.09375],
            }],
        }
    ]), encoding="utf-8")
    (base / "final_9x9.json").write_text(json.dumps({
        "selected_candidate": {"candidate_id": "reference200_empty@2000"},
        "blind_results": [{
            "opponent": "A",
            "games": 40,
            "wins": 0,
            "draws": 0,
            "losses": 40,
            "score_rate": 0.0,
            "paired_opening_bootstrap_ci95": [0.0, 0.0],
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(report_g_v6, "ROOT", tmp_path)

    rows = report_g_v6._v5_comparison()

    assert len(rows) == 2
    assert [row.split("|")[4].strip() for row in rows] == ["16", "40"]
    assert all(len(row.split("|")) == 9 for row in rows)
