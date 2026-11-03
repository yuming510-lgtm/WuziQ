from src.research.alphazero_g.v4_2_report import (
    _frozen_search_comparison_lines,
    _training_tactical_lines,
)


def test_frozen_search_table_reports_direct_match_results_and_compute_cost():
    diagnostics = {
        "top3_history_models_selected_on_development_games": ["G0-test"],
        "frozen_search_match_summary": {
            "G0-test": {
                "value_neutral": {
                    "games": 16,
                    "wins": 11,
                    "draws": 0,
                    "losses": 5,
                    "score_rate": 0.6875,
                    "paired_opening_bootstrap_ci95": [0.4375, 0.875],
                    "mean_agent_cpu_seconds_per_game": 1.004,
                    "mean_agent_search_depth": 1.94,
                    "mean_agent_nodes_per_game": 56881,
                }
            }
        },
    }

    lines = _frozen_search_comparison_lines(diagnostics)

    assert any("G0-test" in line and "value_neutral" in line for line in lines)
    assert any("11-0-5" in line and "68.8%" in line for line in lines)
    assert any("[43.8%, 87.5%]" in line and "1.004" in line for line in lines)


def test_training_tactics_table_separates_network_and_search_accuracy():
    training = {
        "active_arms": ["history_pool"],
        "stage_summaries": {
            "history_pool": [
                {
                    "training_states": 1000,
                    "tactical_search_puct64": {
                        "categories": {
                            "direct_win": {"accuracy": 0.375, "network_accuracy": 0.0},
                            "unique_block": {"accuracy": 0.125},
                            "false_threat": {"accuracy": 0.0},
                            "edge_win": {"accuracy": 0.375},
                        },
                        "immediate_win_network_value_mae_vs_exact_win": 1.908,
                    },
                }
            ]
        },
    }

    lines = _training_tactical_lines(training)

    assert len(lines) == 3
    assert "history_pool" in lines[2] and "1,000" in lines[2]
    assert "0.0%" in lines[2] and "37.5%" in lines[2]
    assert "1.908" in lines[2]
