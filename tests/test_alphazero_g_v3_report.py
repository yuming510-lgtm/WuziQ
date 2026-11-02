from src.research.alphazero_g import v3_report


def test_report_matches_checkpoint_ids_to_their_algorithm_family():
    assert v3_report._row_belongs_to_arm(
        {"arm_id": "G0-seed-510901@20000"}, "G0",
    )
    assert v3_report._row_belongs_to_arm(
        {"arm_id": "G-F2-seed-510902@0"}, "G-F2",
    )
    assert not v3_report._row_belongs_to_arm(
        {"arm_id": "G-F2-seed-510902@0"}, "G0",
    )


def test_final_blind_summary_uses_selected_checkpoint_and_recorded_opponent_family():
    final_models = {"models": {
        "G0:510901": {"checkpoint_id": "G0-seed-510901@20000"},
        "G-F2:510901": {"checkpoint_id": "G-F2-seed-510901@0"},
    }}
    rows = [
        {"arm_id": "G0-seed-510901@20000", "opponent_family": "G0",
         "mode": "puct", "score_rate": 0.7},
        {"arm_id": "G0-seed-510901@20000", "opponent_family": "G-F2",
         "mode": "puct", "score_rate": 0.6},
        {"arm_id": "G-F2-seed-510901@0", "opponent_family": "F2",
         "mode": "puct", "score_rate": 0.5},
        {"arm_id": "G-F2-seed-510901@0", "opponent_family": "G0",
         "mode": "puct", "score_rate": 0.4},
    ]

    result = v3_report._final_blind_comparisons(rows, final_models)

    assert result["G0:510901"]["own_initial"]["puct"]["score_rate"] == 0.7
    assert result["G0:510901"]["other_route"]["puct"]["score_rate"] == 0.6
    assert result["G-F2:510901"]["own_initial"]["puct"]["score_rate"] == 0.5
    assert result["G-F2:510901"]["other_route"]["puct"]["score_rate"] == 0.4
