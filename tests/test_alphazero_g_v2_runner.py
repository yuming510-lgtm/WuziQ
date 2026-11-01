import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.research.alphazero_g.replay import load_game_log
from src.research.alphazero_g.training import F2_SHA256
from scripts.run_g_v2 import main as run_g_v2_cli
import src.research.alphazero_g.v2_runner as v2_runner_module
from src.research.alphazero_g.v2_runner import (
    budgeted_game_prefix,
    initialize_v2_candidate,
    run_v2_candidate,
)
from src.research.alphazero_g.v2_schedule import build_v2_schedule


def _tiny_experiment(tmp_path):
    schedule = build_v2_schedule(901, train_groups=4, development_groups=8, blind_groups=20)
    schedule_path = tmp_path / "research/g_v2/config/openings.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule, sort_keys=True))
    config = {
        "experiment_id": "v2-runner-test",
        "seed": 902,
        "f2_sha256": F2_SHA256,
        "candidate_learning_rates": {"lr_1e4": 1e-4, "lr_3e5": 3e-5, "lr_1e3": 1e-3},
        "states_per_candidate": 16,
        "updates_per_candidate": 1,
        "batch_size": 16,
        "update_every_budgeted_states": 16,
        "stage_nodes": [8, 16],
        "stage_environment_steps": [8, 8],
        "stage_updates": [0, 1],
        "replay_capacity": 64,
        "simulation_budget": 64,
        "hard_cpu_hour_cap": 2.0,
        "opening_schedule_path": str(schedule_path),
        "opening_schedule_sha256": schedule["schedule_sha256"],
    }
    return config, schedule


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for one, two in zip(left, right, strict=True):
            _assert_nested_equal(one, two)
    else:
        assert left == right


def test_budget_prefix_keeps_full_terminal_record_but_only_budget_rows():
    log = Path("research/g_v2/results/preflight/selfplay.jsonl")
    games, _ = load_game_log(log)
    source = copy.deepcopy(games[0])
    accepted = min(3, len(source["experience_rows"]))

    prefix = budgeted_game_prefix(source, accepted)

    assert prefix["completed"] is True
    assert len(prefix["moves"]) == source["move_count"]
    assert len(prefix["experience_rows"]) == accepted
    assert prefix["replay_prefix"] is True
    assert prefix["budgeted_environment_steps"] == accepted
    assert prefix["completion_overrun_steps"] == len(source["moves"]) - len(source["opening"]) - accepted


def test_candidate_initializers_share_f2_values_but_not_optimizer_or_replay(tmp_path):
    config, _ = _tiny_experiment(tmp_path)
    first = initialize_v2_candidate(tmp_path, config, "lr_1e4")
    second = initialize_v2_candidate(tmp_path, config, "lr_3e5")

    assert first["model"].checkpoint_metadata["initialization"] == "frozen_f2"
    assert second["model"].checkpoint_metadata["checkpoint_sha256"] == F2_SHA256
    for left, right in zip(first["model"].state_dict().values(), second["model"].state_dict().values(), strict=True):
        assert torch.equal(left, right)
    assert first["trainer"].optimizer is not second["trainer"].optimizer
    assert first["replay"] is not second["replay"]
    assert first["trainer"].optimizer.param_groups[0]["lr"] == 1e-4
    assert second["trainer"].optimizer.param_groups[0]["lr"] == 3e-5


def test_interrupted_candidate_resume_matches_uninterrupted_training(tmp_path):
    config, _ = _tiny_experiment(tmp_path)
    full_root = tmp_path / "full"
    resume_root = tmp_path / "resume"
    full = run_v2_candidate(full_root, config, "lr_3e5", until_node=16)
    paused = run_v2_candidate(resume_root, config, "lr_3e5", until_node=16, stop_after_games=1)
    resumed = run_v2_candidate(resume_root, config, "lr_3e5", until_node=16)

    assert full["budgeted_environment_steps"] == resumed["budgeted_environment_steps"] == 16
    assert full["optimizer_updates"] == resumed["optimizer_updates"] == 1
    full_state = torch.load(full_root / "research/g_v2/checkpoints/lr_3e5/training_state.pt", map_location="cpu", weights_only=False)
    resume_state = torch.load(resume_root / "research/g_v2/checkpoints/lr_3e5/training_state.pt", map_location="cpu", weights_only=False)
    _assert_nested_equal(full_state["trainer"], resume_state["trainer"])
    _assert_nested_equal(full_state["replay"]["rng_state"], resume_state["replay"]["rng_state"])
    full_games, full_ids = load_game_log(full_root / "research/g_v2/results/selfplay/lr_3e5.jsonl")
    resume_games, resume_ids = load_game_log(resume_root / "research/g_v2/results/selfplay/lr_3e5.jsonl")
    assert full_ids == resume_ids
    assert len(resume_games) == len(resume_ids)
    assert paused["completed"] is (paused["budgeted_environment_steps"] == 16)


def test_resume_reconciles_game_durable_before_its_training_checkpoint(tmp_path, monkeypatch):
    config, _ = _tiny_experiment(tmp_path)
    full_root = tmp_path / "crash-full"
    crash_root = tmp_path / "crash-resume"
    run_v2_candidate(full_root, config, "lr_1e4", until_node=16)
    original_save = v2_runner_module.save_training_state
    crashed = {"done": False}

    def interrupt_after_durable_game(path, trainer, replay, *, progress):
        if int(progress.get("optimizer_updates", 0)) >= 1 and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("simulated interruption after game append")
        return original_save(path, trainer, replay, progress=progress)

    monkeypatch.setattr(v2_runner_module, "save_training_state", interrupt_after_durable_game)
    import pytest
    with pytest.raises(RuntimeError, match="after game append"):
        run_v2_candidate(crash_root, config, "lr_1e4", until_node=16)
    monkeypatch.setattr(v2_runner_module, "save_training_state", original_save)
    resumed = run_v2_candidate(crash_root, config, "lr_1e4", until_node=16)

    assert resumed["budgeted_environment_steps"] == 16
    assert any(row.get("reason") == "reconcile_durable_games_after_interruption"
               for row in resumed["training_history"])
    expected = torch.load(full_root / "research/g_v2/checkpoints/lr_1e4/training_state.pt",
                          map_location="cpu", weights_only=False)
    actual = torch.load(crash_root / "research/g_v2/checkpoints/lr_1e4/training_state.pt",
                        map_location="cpu", weights_only=False)
    _assert_nested_equal(expected["trainer"], actual["trainer"])
    full_games, full_ids = load_game_log(full_root / "research/g_v2/results/selfplay/lr_1e4.jsonl")
    resumed_games, resumed_ids = load_game_log(crash_root / "research/g_v2/results/selfplay/lr_1e4.jsonl")
    assert full_ids == resumed_ids
    assert len(resumed_games) == len(resumed_ids)


def test_runner_cli_rejects_unregistered_config_before_creating_candidate_artifacts(tmp_path, capsys):
    config, schedule = _tiny_experiment(tmp_path)
    config["registered"] = False
    root = tmp_path / "cli-root"
    schedule_path = root / "research/g_v2/config/openings.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule, sort_keys=True))
    config_path = root / "research/g_v2/config/experiment.json"
    config_path.write_text(json.dumps(config))

    import pytest
    with pytest.raises(ValueError, match="registered experiment config"):
        run_g_v2_cli(["--root", str(root), "--config", str(config_path), "--candidate", "lr_1e4"])
    assert not (root / "research/g_v2/checkpoints").exists()
