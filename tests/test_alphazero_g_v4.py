from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.research.alphazero_g.v3_schedule import _opening_digest, validate_v3_schedule
from src.research.alphazero_g.evaluation import load_evaluation_log
from src.research.alphazero_g.v4_budget import (
    cpu_seconds_committed, next_snapshot_threshold, record_cpu_event,
)
from src.research.alphazero_g.v4_config import (
    V3_CHAMPION_ID, V3_CHAMPION_REL, audit_v3_continuation, build_v4_opening_schedule,
    load_v4_config,
)
from src.research.alphazero_g.v4_report import validate_v4_results
from src.research.alphazero_g.v4_snapshot import create_v4_snapshot, restore_v4_snapshot
from src.research.alphazero_g.v4_runner import (
    phase_a_job, phase_b_job, phase_c_job, choose_final_development_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v4_snapshot_cadence_uses_cumulative_runtime():
    assert [next_snapshot_threshold(value) for value in (0, 3599, 3600, 10799, 10800, 25200)] == [
        3600, 3600, 10800, 10800, 25200, 46800,
    ]
    assert next_snapshot_threshold(82800) == 140400
    assert next_snapshot_threshold(140400) == 234000
    with pytest.raises(ValueError):
        next_snapshot_threshold(-1)


def test_v4_report_formats_missing_peak_rss_without_crashing():
    from src.research.alphazero_g.v4_report import _format_optional_bytes

    assert _format_optional_bytes(None) == "未测得"
    assert _format_optional_bytes(341946368) == "341,946,368 B"


def test_v4_report_normalizes_legacy_tactical_checkpoint_field():
    from src.research.alphazero_g.v4_report import _normalize_tactic_record

    assert _normalize_tactic_record({"candidate_checkpoint_id": "old@10k"})["checkpoint_id"] == "old@10k"
    assert _normalize_tactic_record({"checkpoint_id": "new@10k"})["checkpoint_id"] == "new@10k"


def test_v4_report_total_cpu_includes_event_ledger_and_registered_preflight():
    from src.research.alphazero_g.v4_report import _total_cpu_seconds

    events = [{"cpu_seconds": 100.0}, {"cpu_seconds": 20.0}]
    assert _total_cpu_seconds(events, 8.0) == 128.0


def test_v4_report_csv_uses_clean_lf_line_endings(tmp_path):
    from src.research.alphazero_g.v4_report import _write_csv

    path = tmp_path / "summary.csv"
    _write_csv(path, [{"name": "row", "value": 1}])
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_v4_cpu_ledger_is_durable_and_deduplicated(tmp_path):
    assert record_cpu_event(tmp_path, "game:1", 2.0, wall_seconds=2.1, category="selfplay")
    assert not record_cpu_event(tmp_path, "game:1", 2.0, wall_seconds=2.1, category="selfplay")
    assert cpu_seconds_committed(tmp_path) == 2.0


def test_v4_budget_routes_and_v3_continuation_are_explicit():
    audit = audit_v3_continuation(ROOT)
    assert audit["checkpoint_verified"]
    assert not audit["node_20000_training_state_available"]
    assert audit["v4_continuation_mode"] == "weight_inheritance_optimizer_reset_replay_rebuilt_from_first_20k_v3_states"
    assert [phase_a_job(seed)["new_states"] for seed in (540401, 540402, 540403)] == [30_000] * 3
    assert phase_b_job()["new_states"] == 100_000
    assert phase_c_job(540401, "candidate.pt", "A-G0-s540401@30000")["new_states"] == 110_000
    assert 3 * 30_000 + 100_000 + 110_000 == 300_000


def test_promotion_match_selection_disambiguates_duplicate_checkpoint_roles():
    from src.research.alphazero_g.v3_evaluation import promotion_decision
    from src.research.alphazero_g.v4_runner import _candidate_matches

    candidate_id = "B@10000"
    incumbent_id = V3_CHAMPION_ID
    rows = []
    for group_index in range(12):
        for color in ("B", "W"):
            common = {
                "candidate_checkpoint_id": candidate_id,
                "opponent_checkpoint_id": incumbent_id,
                "mode": "puct",
                "opening_group_id": f"opening-{group_index:02d}",
                "agent_color": color,
            }
            rows.append({**common, "v4_opponent_role": "stage_champion", "score_for_agent": 1.0})
            rows.append({**common, "v4_opponent_role": "G0-V3-20K", "score_for_agent": 0.0})

    matches = _candidate_matches(
        rows, candidate_id, incumbent_id, opponent_role="stage_champion",
    )
    assert len(matches) == 24
    assert {row["v4_opponent_role"] for row in matches} == {"stage_champion"}
    decision = promotion_decision(
        matches, candidate_id=candidate_id, incumbent_id=incumbent_id,
        seed=123,
    )
    assert decision["score_rate"] == 1.0
    assert decision["by_color"]["B"]["games"] == 12


def test_v4_schedule_excludes_all_nonempty_v3_openings():
    schedule = build_v4_opening_schedule(ROOT)
    validate_v3_schedule(schedule)
    old = json.loads((ROOT / "research/g_v3/config/openings.json").read_text())
    old_digests = {
        _opening_digest(group["opening"])
        for groups in old["train_openings_by_seed"].values() for group in groups if group["opening"]
    }
    old_digests.update(_opening_digest(group["opening"]) for key in ("development_openings", "blind_openings")
                       for group in old[key] if group["opening"])
    assert old_digests <= set(schedule["excluded_opening_digests"])
    assert len(schedule["development_openings"]) == 12
    assert len(schedule["blind_openings"]) == 20


def test_final_checkpoint_selection_only_uses_qualified_or_promoted_champions(tmp_path, monkeypatch):
    from src.research.alphazero_g import v4_runner

    root = tmp_path
    candidate_id = "A-G0-s540401@10000"
    checkpoint = root / "research/g_v4/checkpoints/A-G0-s540401/node_10000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"qualified candidate")
    config = {
        "config_sha256": "a" * 64,
        "stages": {"A": {"new_random_seeds": [540401, 540402, 540403]}},
    }
    phase_a_gate = {"per_seed": [
        {"champion_checkpoint_id": candidate_id,
         "promotion_vs_v3_champion": {"promote": True}},
        {"champion_checkpoint_id": "A-G0-s540402@30000",
         "promotion_vs_v3_champion": {"promote": False}},
    ]}
    monkeypatch.setattr(v4_runner, "_candidate_score_against_v3", lambda _root, _id: (0.75, [0.55, 0.9]))
    state = {"config_sha256": config["config_sha256"], "phase_a_global_gate": phase_a_gate}
    selected = choose_final_development_candidate(root, config, state)
    assert selected["checkpoint_id"] == candidate_id
    assert selected["checkpoint_sha256"]
    assert selected["checkpoint_id"] != "A-G0-s540402@30000"


def test_empty_v4_result_audit_replays_registered_metadata(tmp_path):
    from src.research.alphazero_g.v4_config import _atomic_json

    config = load_v4_config(ROOT)
    for relative in (
        "research/g_v4/config/experiment.json", "research/g_v4/config/openings.json",
        "research/g_v4/results/preflight/preflight.json",
        "research/g_v4/results/preflight/v3_continuation_audit.json",
        V3_CHAMPION_REL, "research/f_v2/checkpoints/best_f2.pt",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    result = validate_v4_results(tmp_path)
    assert result["budgeted_training_states"] == 0
    assert result["all_selfplay_and_evaluation_games_replayed"]
    assert result["training_states_by_job"][phase_b_job()["job_id"]] == 0


def test_v4_snapshot_restores_registered_state_to_an_empty_directory(tmp_path):
    expected_evaluation_games = sum(
        len(load_evaluation_log(ROOT / f"research/g_v4/results/evaluation/{name}")[0])
        for name in ("development.jsonl", "blind.jsonl")
    )
    archive = tmp_path / "g_v4_recovery.zip"
    created = create_v4_snapshot(ROOT, archive)
    destination = tmp_path / "empty_restore"
    restored = restore_v4_snapshot(archive, destination)
    assert created["sha256"] == restored["archive_sha256"]
    assert restored["empty_restore"]
    assert restored["selfplay_game_records"] == created["selfplay_game_records"]
    assert restored["evaluation_game_records"] == expected_evaluation_games
    assert (destination / V3_CHAMPION_REL).is_file()
    assert (destination / "research/g_v4/checkpoints/B-G0-v3-champion-continue/training_state.pt").is_file()

