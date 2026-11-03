from src.research.alphazero_g.v4_1_config import ARM_IDS
from src.research.alphazero_g.v4_1_evaluation import blind_tasks, development_tasks
from src.research.alphazero_g.v4_1_analysis import expected_development_comparisons, wilson_interval


def test_v4_1_development_matrix_has_all_preregistered_pairs_once() -> None:
    tasks = development_tasks()
    assert len(tasks) == 816
    identities = [tuple(sorted(row.items())) for row in tasks]
    assert len(identities) == len(set(identities))
    assert all(row["agent_color"] in {"B", "W"} for row in tasks)
    assert sum(row["candidate"] == "G0-V3-20K" for row in tasks) == 96
    assert sum(row["candidate"] in ARM_IDS for row in tasks) == 720


def test_v4_1_blind_matrix_is_frozen_40_pair_games_plus_baseline() -> None:
    tasks = blind_tasks("control_v4")
    assert len(tasks) == 360
    assert sum(row["candidate"] == "G0-V3-20K" for row in tasks) == 160
    assert sum(row["candidate"] == "control_v4" for row in tasks) == 200
    assert len({tuple(sorted(row.items())) for row in tasks}) == len(tasks)


def test_v4_1_confidence_interval_does_not_call_zero_wins_certain() -> None:
    interval = wilson_interval(0, 40)
    assert interval is not None
    assert interval[0] == 0.0
    assert 0.08 < interval[1] < 0.10
    assert wilson_interval(0, 0) is None


def test_v4_1_development_declares_all_34_model_mode_comparisons() -> None:
    assert len(expected_development_comparisons()) == 34
