from src.research.v6.report import _model_usage, model_catalog, summarize_pair


def test_report_catalog_has_one_abcd_and_five_e_families_at_all_budgets():
    catalog = model_catalog()
    assert len(catalog) == 57
    assert "A@natural" in catalog
    assert "B@50ms" in catalog and "C@50s" in catalog and "D@1s" in catalog
    assert "E1@50ms" in catalog and "E5@50s" in catalog


def test_pair_summary_scores_draw_half_and_splits_by_color():
    group = {
        "pair": ["B", "E2"],
        "model_versions": {"B": "200ms", "E2": "1s"},
        "group_id": "g1",
    }
    games = [
        {"winner_model": "B", "black_model": "B", "white_model": "E2", "thinking_seconds_total": 4.0},
        {"winner_model": None, "black_model": "E2", "white_model": "B", "thinking_seconds_total": 6.0},
    ]
    summary = summarize_pair(group, games)
    assert summary["model_a"] == "B@200ms"
    assert summary["model_b"] == "E2@1s"
    assert summary["wins"]["B@200ms"] == 1
    assert summary["draws"] == 1
    assert summary["losses"]["B@200ms"] == 0
    assert summary["score_rate"]["B@200ms"] == 0.75
    assert summary["color"]["B@200ms"]["black"]["score_rate"] == 1.0
    assert summary["color"]["B@200ms"]["white"]["score_rate"] == 0.5


def test_model_usage_excludes_preset_opening_plies_without_an_ai_model():
    group = {
        "group_id": "g1",
        "model_versions": {"B": "200ms", "E2": "1s"},
    }
    rows = _model_usage(
        [{"group_id": "g1", "moves": [
            {"model": None, "elapsed_seconds": 0.0},
            {"model": "B", "elapsed_seconds": 0.2, "nodes": 4, "completed_depth": 2},
        ]}],
        {"g1": group},
    )
    by_model = {row["model"]: row for row in rows}
    assert by_model["B@200ms"]["decisions"] == 1
    assert by_model["B@200ms"]["mean_elapsed_seconds"] == 0.2
