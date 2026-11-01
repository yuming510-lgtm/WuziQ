from __future__ import annotations

from pathlib import Path

from src.research.alphazero_g.v2_report import render_v2_report


def test_renderer_writes_chinese_report_csv_and_three_reproducible_curves(tmp_path):
    summary = {
        "metadata": {"experiment_id": "test", "base_commit": "base", "f2_sha256": "f2",
                     "schedule_sha256": "schedule", "preflight_wall_seconds": 12.0,
                     "preflight_cpu_seconds": 11.0, "hard_cpu_hour_cap": 2.0,
                     "optimistic_cpu_seconds": 100.0, "central_cpu_seconds": 110.0,
                     "conservative_cpu_seconds": 120.0, "resources": {"cpu_count": 3, "device": "cpu"},
                     "unexecuted_nodes": [20000], "formal_selfplay_game_records": 24,
                     "development_games": 10, "challenge_games": 5, "blind_games": 40,
                     "total_actual_cpu_seconds": 200.0},
        "training": {"candidates": {
            "lr_1e4": {"learning_rate": 0.0001, "games": 8, "states": 1000, "completed": True, "updates": 62,
                       "optimizer_updates": 62, "training_cpu_seconds": 90.0,
                           "milestones": [{"node": 1000, "games": 8, "states": 1000, "training_cpu_seconds": 90.0,
                                       "puct_vs_f2": {"score_rate": 0.625,
                                                       "paired_opening_bootstrap_ci95": [0.5, 0.75]},
                                       "network_vs_f2": {"score_rate": 0.5,
                                                         "paired_opening_bootstrap_ci95": [0.375, 0.625]},
                                       "executed": True, "optimizer_updates": 62,
                                       "policy_entropy": 2.2, "value_mse": 0.4,
                                       "value_metrics": {"network": {"outcome_value_mse": 0.4}},
                                       "selfplay_simulations": 64000,
                                       "policy_loss": 1.2, "value_loss": 0.3}]},
            "lr_3e5": {"learning_rate": 0.00003, "games": 8, "states": 1000, "completed": True, "updates": 62,
                       "optimizer_updates": 62, "training_cpu_seconds": 95.0,
                       "milestones": []},
            "lr_1e3": {"learning_rate": 0.001, "games": 8, "states": 1000, "completed": True, "updates": 62,
                       "optimizer_updates": 62, "training_cpu_seconds": 85.0,
                       "milestones": []},
        }},
        "selection": {"champion": {"checkpoint_id": "F2@0"},
                      "best_candidate_for_blind": {"checkpoint_id": "lr_1e4@1000"},
                      "exact_tactics": {}, "outcome_value_metrics": {}},
        "blind": {"comparisons": {}, "learning": {}, "search": {}},
        "drive_status": {"status": "tested"},
        "integrity": {"all_games_valid": True, "duplicate_game_ids": 0},
    }

    outputs = render_v2_report(tmp_path, summary)

    report = Path(outputs["report"])
    assert report.is_file()
    content = report.read_text()
    assert "20,000" in content and "未执行" in content
    assert "同等搜索预算" in content
    assert len(outputs["figures"]) == 3
    assert all((tmp_path / item).is_file() and (tmp_path / item).stat().st_size > 1000 for item in outputs["figures"])
    assert (tmp_path / "research/g_v2/results/growth.csv").is_file()
