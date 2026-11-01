import copy

import pytest

from src.research.alphazero_g.v2_preflight import register_v2_budget
from src.research.alphazero_g.v2_preflight import register_budget_in_config
from src.research.alphazero_g.v2_schedule import build_v2_schedule

LR_IDS = {"lr_1e4", "lr_3e5", "lr_1e3"}


def _inputs(states_per_second=20.0):
    schedule = build_v2_schedule(510701, 0, 8, 20)
    preflight = {
        "format_version": 1,
        "simulation_benchmarks": [
            {"position_id": position, "simulations": sims, "mean_seconds": sims / 1000}
            for sims in (16, 32, 64)
            for position in ("opening", "middle", "tactical")
        ],
        "selfplay_games": [
            {"lr_id": lr, "completed": True, "budgeted_states": 20,
             "cpu_seconds": 20 / states_per_second, "simulations": 1280}
            for lr in sorted(LR_IDS)
        ],
        "training_updates": [
            {"lr_id": lr, "completed": True, "updates": 20, "cpu_seconds": 0.2,
             "batch_size": 16}
            for lr in sorted(LR_IDS)
        ],
        "evaluation_games": [
            {"mode": mode, "opponent": opponent, "completed": True,
             "total_cpu_seconds": 0.25 if mode == "network" else 0.6,
             "agent_seconds": 0.08 if mode == "network" else 0.4,
             "agent_moves": 4, "post_opening_moves": 8}
            for mode in ("network", "puct")
            for opponent in ("A", "B", "F2")
        ],
        "resources": {"affinity_cpus": 3, "max_rss_bytes": 1000000,
                      "disk_free_bytes": 1000000000, "device": "cpu"},
        "torch_threads": 1,
    }
    proposed = {
        "states_per_candidate": 20000,
        "candidate_learning_rates": {"lr_1e4": 1e-4, "lr_3e5": 3e-5, "lr_1e3": 1e-3},
        "batch_size": 16,
        "update_every_budgeted_states": 16,
        "replay_capacity": 20000,
        "hard_cpu_hour_cap": 2.0,
        "puct_simulations": 64,
    }
    return preflight, proposed, schedule


def test_budget_requires_each_simulation_level_on_opening_middle_and_tactical_positions():
    preflight, proposed, schedule = _inputs()
    preflight["simulation_benchmarks"] = [
        row for row in preflight["simulation_benchmarks"]
        if not (row["simulations"] == 64 and row["position_id"] == "tactical")
    ]

    with pytest.raises(ValueError, match="simulation measurements"):
        register_v2_budget(preflight, proposed, schedule)


def test_budget_registers_equal_quotas_and_all_reachable_milestones():
    preflight, proposed, schedule = _inputs()
    result = register_v2_budget(preflight, proposed, schedule)

    assert result["registered"] is True
    assert result["states_per_candidate"] == 20000
    assert set(result["candidate_learning_rates"]) == LR_IDS
    assert result["stage_nodes"] == [1000, 2000, 5000, 10000, 20000]
    assert result["stage_environment_steps"] == [1000, 1000, 3000, 5000, 10000]
    assert result["updates_per_candidate"] == 1250
    assert result["simulation_budget"] == 64
    assert result["evaluation"]["development_matches_per_candidate_node"] == 112
    assert result["estimates"]["development_champion_cpu_seconds"] > 0
    assert result["estimates"]["conservative_cpu_seconds"] <= 7200


def test_budget_scales_all_candidates_to_same_registered_cpu_feasible_milestone():
    preflight, proposed, schedule = _inputs(states_per_second=1.3)
    result = register_v2_budget(preflight, proposed, schedule)

    assert result["states_per_candidate"] == 2000
    assert result["stage_nodes"] == [1000, 2000]
    assert all(item["states_per_candidate"] == 2000 for item in result["candidates"].values())
    assert result["estimates"]["conservative_cpu_seconds"] <= 7200


def test_budget_rejects_missing_learning_rate_measurement():
    preflight, proposed, schedule = _inputs()
    preflight["training_updates"] = [row for row in preflight["training_updates"] if row["lr_id"] != "lr_3e5"]

    with pytest.raises(ValueError, match="learning rate|LR|candidate"):
        register_v2_budget(preflight, proposed, schedule)


def test_config_registration_preserves_model_metadata_and_binds_preflight_hash(tmp_path):
    import hashlib
    import json

    preflight, proposed, schedule = _inputs()
    preflight["schedule_sha256"] = schedule["schedule_sha256"]
    proposed["network"] = {"board_size": 9, "channels": 32, "blocks": 2}
    proposed["opening_schedule_path"] = "research/g_v2/config/openings.json"
    paths = {
        "preflight": tmp_path / "preflight.json",
        "proposed": tmp_path / "experiment.json",
        "schedule": tmp_path / "openings.json",
    }
    paths["preflight"].write_text(json.dumps(preflight))
    paths["proposed"].write_text(json.dumps(proposed))
    paths["schedule"].write_text(json.dumps(schedule))
    output = tmp_path / "registered.json"

    registered = register_budget_in_config(paths["preflight"], paths["proposed"], paths["schedule"], output)

    assert registered["network"] == {"board_size": 9, "channels": 32, "blocks": 2}
    assert registered["preflight_sha256"] == hashlib.sha256(paths["preflight"].read_bytes()).hexdigest()
    assert registered["opening_schedule_sha256"] == schedule["schedule_sha256"]
    assert registered["preflight_path"] == "research/g_v2/results/preflight/preflight.json"
    assert json.loads(output.read_text())["registered"] is True
