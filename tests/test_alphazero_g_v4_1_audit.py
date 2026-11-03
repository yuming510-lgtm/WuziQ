from pathlib import Path

from src.research.alphazero_g.v4_1_audit import audit_experiment


def test_v4_1_full_artifact_audit_replays_all_stored_games():
    root = Path(__file__).resolve().parents[1]
    result = audit_experiment(root)
    assert result["total_budgeted_training_states"] == 40_000
    assert result["total_training_games"] == 1_369
    assert result["total_evaluation_games"] == 1_176
    assert result["unique_evaluation_game_ids"] == 1_176
    assert result["all_selfplay_and_evaluation_games_replayed_legal"] is True
    assert result["blind_used_for_selection"] is False
    assert result["promotion_passed"] is False
