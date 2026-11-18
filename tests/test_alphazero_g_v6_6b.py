from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import ai
from src.ai import SearchConfig
from src.gomoku import GomokuBoard
from src.research.alphazero_g import v6_6b_evaluation as v66b


@pytest.mark.parametrize(
    ("name", "budget"),
    [("C@200ms", 0.2), ("C@1s", 1.0), ("C@5s", 5.0)],
)
def test_c_protocol_restores_v64_search_limits(name: str, budget: float) -> None:
    config = v66b.search_config_for_opponent(name)
    assert config.max_depth == 69
    assert config.candidate_radius == 2
    assert config.time_limit_seconds == budget


def test_c_agent_passes_corrected_limits_to_unchanged_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.alphazero_g.v5_reference import make_board

    observed: dict[str, object] = {}

    def fake_choose_move(board, color, model, config, seed):
        observed.update(model=model, config=config, seed=seed)
        return SimpleNamespace(
            move=(4, 4), value=0.0, completed_depth=3, nodes=17,
            cutoffs=5, simulations=0, elapsed_seconds=0.2,
            timed_out=True, best_moves=((4, 4),),
        )

    monkeypatch.setattr(v66b, "choose_move", fake_choose_move)
    board = make_board(9, 5, 0)
    move, metrics = v66b._agent_choice(
        {"kind": "C", "name": "C@200ms", "time_limit": 0.2}, board, "B", 123
    )
    assert move == (4, 4)
    assert observed["model"] == "C"
    config = observed["config"]
    assert isinstance(config, SearchConfig)
    assert (config.max_depth, config.candidate_radius, config.time_limit_seconds) == (69, 2, 0.2)
    assert metrics["search_depth"] == 3
    assert metrics["timed_out"] is True


def test_alpha_beta_timeout_keeps_last_complete_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def controlled_root(board, player, depth, root_moves, config, deadline, use_alpha_beta, counters):
        calls.append(depth)
        if depth == 1:
            counters[0] += 1
            return 123.0, (root_moves[0],)
        raise ai._SearchTimeout

    monkeypatch.setattr(ai, "_search_root", controlled_root)
    result = ai.choose_move(
        GomokuBoard(size=9), "B", "C",
        SearchConfig(max_depth=69, time_limit_seconds=1.0, candidate_radius=2), seed=7,
    )
    assert calls == [1, 2]
    assert result.completed_depth == 1
    assert result.root_value == 123.0
    assert result.timed_out is True
    assert GomokuBoard(size=9).in_bounds(*result.move)


def test_preregistered_schedule_is_200_unique_games_and_color_paired(tmp_path) -> None:
    schedule = v66b.build_opening_schedule(tmp_path, seed=660701)
    assert v66b.validate_opening_schedule(schedule)["opening_groups"] == 20
    plan = v66b.build_match_plan(schedule["openings"])
    assert len(plan) == 200
    assert len({row["game_id"] for row in plan}) == 200
    for model in ("G5000", "G8000"):
        for opponent, expected in (("C@200ms", 40), ("C@1s", 40), ("C@5s", 20)):
            rows = [row for row in plan if row["candidate"] == model and row["opponent"] == opponent]
            assert len(rows) == expected
            assert {row["candidate_color"] for row in rows} == {"B", "W"}
            assert len({row["opening_group_id"] for row in rows}) == expected // 2
    assert not any(row["opponent"] in {"A", "B@50ms", "C@50ms"} for row in plan)
    for opponent in ("C@200ms", "C@1s", "C@5s"):
        left = {(row["opening_group_id"], row["candidate_color"]): (row["seed"], row["opening"])
                for row in plan if row["candidate"] == "G5000" and row["opponent"] == opponent}
        right = {(row["opening_group_id"], row["candidate_color"]): (row["seed"], row["opening"])
                 for row in plan if row["candidate"] == "G8000" and row["opponent"] == opponent}
        assert left == right


def test_v66_frozen_weights_match_registered_sha256() -> None:
    root = v66b.ROOT
    for model, expected in v66b.MODEL_SHA256.items():
        path = root / "research/g_v6_6/checkpoints" / f"{model}.pt"
        assert v66b.sha_file(path) == expected
        v66b.load_frozen_model(path, model)
