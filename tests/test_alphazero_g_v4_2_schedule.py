from __future__ import annotations


def test_v4_2_schedule_excludes_historical_and_cross_split_openings() -> None:
    from src.research.alphazero_g.v3_schedule import _opening_digest
    from src.research.alphazero_g.v4_2_schedule import build_v4_2_schedule

    excluded = [{"player": "B", "row": 0, "col": 0}, {"player": "W", "row": 0, "col": 1},
                {"player": "B", "row": 1, "col": 0}, {"player": "W", "row": 1, "col": 1}]
    schedule = build_v4_2_schedule(
        seed=742_201, train_groups_per_arm=8, development_groups=5, blind_groups=7,
        excluded_openings=[excluded],
    )

    train = schedule["train_openings"]
    development = schedule["development_openings"]
    blind = schedule["blind_openings"]
    all_nonempty = [g["opening"] for g in train + development + blind if g["opening"]]
    digests = [_opening_digest(opening) for opening in all_nonempty]
    assert len(digests) == len(set(digests))
    assert _opening_digest(excluded) not in set(digests)
    assert len(train) == 8
    assert len(development) == 5
    assert len(blind) == 7
