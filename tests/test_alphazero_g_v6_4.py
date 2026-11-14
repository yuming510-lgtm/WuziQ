from __future__ import annotations

from src.research.alphazero_g.v6_4_ladder import (
    build_schedule,
    collect_historical_openings,
    estimate_ladder_cost,
    complete_opening_pairs,
    search_config_for_opponent,
    schedule_games,
    summarize_condition,
    validate_match_record,
    validate_schedule,
)


def test_schedule_has_three_shared_openings_and_twenty_four_unique_games() -> None:
    schedule = build_schedule(seed=640614, historical_openings=[])
    games = schedule_games(schedule)

    assert [row["name"] for row in schedule["opponents"]] == [
        "A", "C@200ms", "C@1s", "C@5s"
    ]
    assert len(schedule["openings"]) == 3
    assert len(games) == 24
    assert len({row["game_id"] for row in games}) == 24
    for opponent in ("A", "C@200ms", "C@1s", "C@5s"):
        condition = [row for row in games if row["opponent"] == opponent]
        assert len(condition) == 6
        for opening in schedule["openings"]:
            pair = [row for row in condition if row["opening_group_id"] == opening["opening_group_id"]]
            assert len(pair) == 2
            assert {row["candidate_color"] for row in pair} == {"B", "W"}
            assert pair[0]["opening"] == pair[1]["opening"]


def test_schedule_avoids_d4_equivalent_historical_openings() -> None:
    first = build_schedule(seed=640614, historical_openings=[])["openings"][0]["moves"]
    schedule = build_schedule(seed=640614, historical_openings=[first])

    assert schedule["opening_seed"] > 640614
    assert schedule["validation"]["historical_d4_overlap"] == 0
    assert schedule["validation"]["unique_d4_openings"] == 3


def test_schedule_integrity_rejects_any_post_registration_edit() -> None:
    schedule = build_schedule(seed=640614, historical_openings=[])
    schedule["openings"][0]["moves"][0]["row"] = 8

    try:
        validate_schedule(schedule)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("the schedule integrity check accepted a changed opening")


def test_repository_opening_scan_includes_v6_3_frozen_schedule(tmp_path) -> None:
    schedule_dir = tmp_path / "research/g_v6_3/config"
    schedule_dir.mkdir(parents=True)
    moves = [
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 4, "col": 5},
        {"player": "B", "row": 5, "col": 4},
        {"player": "W", "row": 3, "col": 4},
        {"player": "B", "row": 5, "col": 5},
        {"player": "W", "row": 3, "col": 5},
    ]
    (schedule_dir / "opening_schedule.json").write_text(
        '{"openings":[{"opening_group_id":"old","moves":'
        + __import__("json").dumps(moves)
        + '}]}',
        encoding="utf-8",
    )

    historical = collect_historical_openings(tmp_path)

    assert moves in historical


def test_c_budgets_share_v4_search_settings_and_change_only_deadline() -> None:
    from dataclasses import replace

    configs = [search_config_for_opponent(name) for name in ("C@200ms", "C@1s", "C@5s")]

    assert [row.time_limit_seconds for row in configs] == [0.2, 1.0, 5.0]
    assert all(row.max_depth == 69 and row.candidate_radius == 2 for row in configs)
    assert replace(configs[0], time_limit_seconds=5.0) == configs[2]


def test_a_uses_natural_runtime_without_an_artificial_wait() -> None:
    config = search_config_for_opponent("A")

    assert config.time_limit_seconds > 1e6
    assert config.max_depth == 69


def test_cost_estimate_uses_measured_turn_cost_and_keeps_all_conditions() -> None:
    estimate = estimate_ladder_cost(
        candidate_cpu_per_move=0.5,
        opponent_cpu_per_move={"A": 0.01, "C@200ms": 0.2, "C@1s": 1.0, "C@5s": 5.0},
        side_turns=(10, 20, 30),
    )

    assert estimate["side_turns_per_game"] == [10, 20, 30]
    assert estimate["games"] == 24
    assert estimate["candidate_cpu_seconds"]["typical"] == 240.0
    assert estimate["opponent_cpu_seconds"]["typical"] == 745.2


def _synthetic_finished_game():
    opening = [
        {"player": "B", "row": 0, "col": 0},
        {"player": "W", "row": 8, "col": 8},
        {"player": "B", "row": 1, "col": 0},
        {"player": "W", "row": 8, "col": 7},
        {"player": "B", "row": 2, "col": 0},
        {"player": "W", "row": 8, "col": 6},
    ]
    continuation = [
        ("B", 4, 0), ("W", 7, 0), ("B", 4, 1), ("W", 7, 2),
        ("B", 4, 2), ("W", 7, 4), ("B", 4, 3), ("W", 7, 6),
        ("B", 4, 4),
    ]
    moves = [{**move, "opening": True} for move in opening]
    for player, row, col in continuation:
        moves.append({"player": player, "row": row, "col": col, "opening": False,
                      "actor": "G5000" if player == "B" else "C@200ms",
                      "elapsed_seconds": 0.001, "cpu_seconds": 0.001,
                      "search_depth": 1, "search_nodes": 1, "simulations": 400})
    return {
        "format_version": 1, "experiment_id": "wuziq-g5000-ladder-v6-4",
        "game_id": "test-game", "candidate": "G5000", "opponent": "C@200ms",
        "board_size": 9, "n_in_row": 5, "candidate_color": "B", "winner_color": "B",
        "points": 1.0, "termination": "five_in_row", "opening_group_id": "test-opening",
        "opening_seed": 1, "opening": opening, "seed": 2,
        "candidate_search": "reference PUCT-400", "opponent_config": {"name": "C@200ms"},
        "simulations_per_candidate_turn": 400, "candidate_configured_playouts_per_turn": 401,
        "moves": moves, "move_count": len(moves),
    }


def test_completed_game_replays_and_matches_terminal_result() -> None:
    record = _synthetic_finished_game()

    audit = validate_match_record(record)

    assert audit["legal"] is True
    assert audit["winner_color"] == "B"
    assert audit["replayed_plies"] == 15


def test_completed_game_audit_rejects_moves_after_a_win() -> None:
    record = _synthetic_finished_game()
    record["moves"].append({"player": "W", "row": 6, "col": 0, "opening": False})
    record["move_count"] += 1

    try:
        validate_match_record(record)
    except ValueError as exc:
        assert "after terminal" in str(exc)
    else:
        raise AssertionError("the audit accepted a move after the game had ended")


def test_partial_results_only_count_complete_color_swapped_pairs() -> None:
    black = _synthetic_finished_game()
    white = dict(black, game_id="test-game-white", candidate_color="W", points=0.0)
    unpaired = dict(black, game_id="unpaired", opening_group_id="other")

    paired, discarded = complete_opening_pairs([black, white, unpaired])

    assert len(paired) == 2
    assert discarded == 1


def test_partial_condition_summary_is_strict_json_without_fake_scores() -> None:
    rows = [_synthetic_finished_game()]

    summary = summarize_condition(rows, seed=1)

    import json
    json.dumps(summary, allow_nan=False)
    assert summary["raw_completed_games"] == 1
    assert summary["paired_games"] == 0
    assert summary["score_rate"] is None
    assert summary["complete"] is False
