from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.ai import SearchConfig
from src.research.alphazero_g import v6_3_evaluation as evaluation
from src.research.alphazero_g.v6_3_training import (
    ARTIFACT, BASE_GAMES, BASE_STATES, BASE_UPDATES, ROOT, build_opening_schedule,
    prepare_experiment, validate_training_state,
)


def test_frozen_g2500_state_restores_all_training_components():
    prepared = prepare_experiment(ROOT)
    validation = validate_training_state(
        ROOT / ARTIFACT / "checkpoints/reference400_empty/training_state_games_2500.pt",
        expected_games=BASE_GAMES,
    )
    assert validation["environment_states"] == BASE_STATES
    assert validation["optimizer_updates"] == BASE_UPDATES
    assert all(validation[key] for key in (
        "model_restored", "adam_restored", "replay_restored", "python_rng_restored",
        "numpy_rng_restored", "torch_rng_restored",
    ))
    assert prepared["openings"]["historical_d4_overlap"] == 0


def test_opening_schedule_has_40_legal_fresh_paired_positions():
    schedule = build_opening_schedule(ROOT, seed=630603, groups=40)
    assert len(schedule["openings"]) == 40
    assert schedule["validation"] == {
        "all_openings_legal_and_nonterminal": True,
        "unique_d4_openings": 40,
        "historical_d4_overlap": 0,
        "same_openings_for_all_models_and_opponents": True,
    }


@pytest.mark.parametrize(("identity", "budget"), [("A", 1e9), ("B", 0.05), ("C", 0.05)])
def test_original_a_minimax_b_and_alpha_beta_c_keep_their_engine_identity(identity, budget):
    from src.gomoku import GomokuBoard

    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    fake_result = SimpleNamespace(
        move=(4, 5), simulations=0, completed_depth=2, nodes=17,
        cutoffs=3, elapsed_seconds=0.05, timed_out=True,
    )
    with patch.object(evaluation, "choose_move", return_value=fake_result) as choose:
        move, metrics = evaluation._agent_choice({"kind": identity},
                                                  evaluation.make_board(9, 5, 0),
                                                  "W", 17)
    assert move == (4, 5)
    assert choose.call_args.args[2] == identity
    config = choose.call_args.args[3]
    assert isinstance(config, SearchConfig)
    assert config.time_limit_seconds == budget
    assert metrics["search_depth"] == 2
    assert metrics["cutoffs"] == 3
    assert metrics["timed_out"] is True


@pytest.mark.parametrize("identity", ["A", "B", "C"])
def test_original_opponents_return_legal_moves_with_registered_metrics(identity):
    board = evaluation.make_board(9, 5, 0)
    board.do_move(40)
    move, metrics = evaluation._agent_choice({"kind": identity}, board, "W", 20261111)
    assert 0 <= move[0] < 9 and 0 <= move[1] < 9
    assert move != (4, 4)
    assert metrics["search_depth"] >= 0
    assert metrics["search_nodes"] >= 0
    assert metrics["cpu_seconds"] >= 0
    if identity in {"B", "C"}:
        assert metrics["elapsed_seconds"] >= 0
        assert isinstance(metrics["timed_out"], bool)


def test_duplicate_challenge_game_requires_same_registered_pair():
    opening = {"opening_group_id": "open-1", "seed": 77}
    game = {
        "game_id": "v63-example", "candidate": "G2500", "opponent": "C",
        "opening_group_id": "open-1", "candidate_color": "B",
        "simulations_per_candidate_turn": 400, "moves": [{"row": 0, "col": 0}],
        "move_count": 1, "termination": "five_in_row",
    }
    evaluation._validate_existing(game, game_id="v63-example", candidate="G2500",
                                  opponent="C", opening=opening, color="B")
    with pytest.raises(ValueError, match="collision"):
        evaluation._validate_existing(game, game_id="v63-example", candidate="G2500",
                                      opponent="B", opening=opening, color="B")


def test_incomplete_color_pair_is_saved_but_excluded_from_paired_statistics():
    complete = [
        {"opening_group_id": "open-1", "candidate_color": "B", "points": 1.0},
        {"opening_group_id": "open-1", "candidate_color": "W", "points": 0.0},
    ]
    rows, count = evaluation._complete_pair_rows(complete + [
        {"opening_group_id": "open-2", "candidate_color": "B", "points": 1.0}
    ])
    assert count == 1
    assert len(rows) == 2
    assert {row["opening_group_id"] for row in rows} == {"open-1"}


def test_g2500_and_g5000_use_shared_fixed_rng_seed_per_paired_game():
    seed_2500 = evaluation._paired_game_seed("A", "opening-7", "B")
    seed_5000 = evaluation._paired_game_seed("A", "opening-7", "B")
    assert seed_2500 == seed_5000
    assert seed_2500 != evaluation._paired_game_seed("A", "opening-7", "W")
    assert seed_2500 != evaluation._paired_game_seed("B", "opening-7", "B")


def test_report_computes_same_opening_g5000_minus_g2500_delta(tmp_path):
    import json
    from scripts.report_g_v6_3 import _model_differences

    for model, points in (("G2500", (1.0, 0.0)), ("G5000", (1.0, 1.0))):
        directory = tmp_path / "research/g_v6_3/results/matches" / f"{model}_vs_A"
        directory.mkdir(parents=True)
        for color, score in zip(("B", "W"), points):
            (directory / f"v63-{model}-{color}.json").write_text(json.dumps({
                "opening_group_id": "opening-1", "candidate_color": color, "points": score,
            }), encoding="utf-8")
    result = _model_differences(tmp_path)
    assert result["A"]["paired_openings"] == 1
    assert result["A"]["score_rate_delta"] == pytest.approx(0.5)
    assert result["A"]["ci95"] == pytest.approx([0.5, 0.5])


def test_snapshot_registry_requires_download_hash_and_restore(tmp_path, monkeypatch):
    import json
    from scripts import register_g_v6_3_snapshot

    artifact = tmp_path / "research/g_v6_3"
    (artifact / "results").mkdir(parents=True)
    (artifact / "results/snapshot_status.json").write_text(json.dumps({
        "drive_folder_id": "drive-folder", "verified_snapshots": []
    }), encoding="utf-8")
    monkeypatch.setattr(register_g_v6_3_snapshot, "ARTIFACT", artifact)
    with pytest.raises(ValueError, match="only be registered"):
        register_g_v6_3_snapshot.register(
            sequence=1, games=2500, training_cpu_seconds=0,
            file_name="s.zip", drive_file_id="file", drive_url="url",
            sha256="a", size_bytes=1, code_commit="commit", package_cpu_seconds=0,
            download_sha256="a", empty_restore_verified=False,
        )
    row = register_g_v6_3_snapshot.register(
        sequence=1, games=2500, training_cpu_seconds=0,
        file_name="s.zip", drive_file_id="file", drive_url="url",
        sha256="a", size_bytes=1, code_commit="commit", package_cpu_seconds=0,
        download_sha256="a", empty_restore_verified=True,
    )
    assert row["empty_directory_restore_verified"] is True
    assert json.loads((artifact / "results/snapshot_status.json").read_text())["latest_verified_sequence"] == 1
