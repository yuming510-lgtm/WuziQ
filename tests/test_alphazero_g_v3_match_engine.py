from src.research.alphazero_g.evaluation import play_evaluation_match, validate_match_record
from src.research.alphazero_g.training import initialize_arm


def test_model_opponents_can_share_puct_budget_and_record_both_searches():
    candidate = initialize_arm("G0", seed=90231)
    opponent = initialize_arm("G0", seed=90232)
    opening = [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 3, "col": 5},
        {"player": "W", "row": 5, "col": 3},
    ]
    match = play_evaluation_match(
        candidate,
        game_id="v3-model-puct-pair-test",
        opening_group_id="v3-test-opening",
        arm_id="G0-test",
        mode="puct",
        opponent="MODEL",
        opponent_model=opponent,
        opponent_mode="puct",
        opening=opening,
        agent_color="B",
        seed=90233,
        simulations=1,
        opponent_simulations=1,
    )

    validate_match_record(match)
    assert match["opponent_mode"] == "puct"
    assert match["stats"]["agent_simulations"] > 0
    assert match["stats"]["opponent_simulations"] > 0
    assert all(move.get("simulations", 0) == 1 for move in match["moves"] if move.get("actor") == "agent")
    assert all(move.get("simulations", 0) == 1 for move in match["moves"] if move.get("actor") == "MODEL")
