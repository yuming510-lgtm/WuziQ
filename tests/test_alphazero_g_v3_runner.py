import hashlib
import json
import subprocess
from pathlib import Path

import torch

from src.research.alphazero_g.v3_schedule import build_v3_schedule
from src.research.alphazero_g.v3_runner import (
    completed_v3_node_progress,
    initialize_v3_arm,
    initialize_v3_starts,
    load_v3_state,
    run_v3_arm,
    save_v3_state,
)
from src.research.alphazero_g.v3_bn import capture_bn_buffers
from src.research.alphazero_g.training import F2_SHA256


ROOT = Path(__file__).resolve().parents[1]


def _state_hash(model):
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _test_root(path: Path, seed: int | list[int]) -> Path:
    path.mkdir()
    seeds = [int(seed)] if isinstance(seed, int) else list(map(int, seed))
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "V3 test"], cwd=path, check=True)
    schedule = build_v3_schedule(
        seeds, train_groups_per_seed=4, development_groups=12, blind_groups=20,
        excluded_openings=[],
    )
    schedule_path = path / "research/g_v3/config/openings.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(json.dumps(schedule, sort_keys=True))
    config = json.loads((ROOT / "research/g_v3/config/experiment.json").read_text())
    config["seeds"] = seeds
    config["opening_schedule_sha256"] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    config["config_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    (path / "research/g_v3/config/experiment.json").write_text(json.dumps(config, sort_keys=True))
    for relative in (
        "research/g_v3/results/preflight/preflight.json",
        "research/g_v3/results/bn_probe/probe.json",
    ):
        source = ROOT / relative
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(source, target)
    f2 = path / "research/f_v2/checkpoints/best_f2.pt"
    f2.parent.mkdir(parents=True)
    import shutil
    shutil.copyfile(ROOT / "research/f_v2/checkpoints/best_f2.pt", f2)
    return path


def test_initialization_has_independent_seeded_g0_and_identical_f2_lineage():
    g0_a = initialize_v3_arm(ROOT, arm="G0", seed=510901)
    g0_b = initialize_v3_arm(ROOT, arm="G0", seed=510902)
    f2_a = initialize_v3_arm(ROOT, arm="G-F2", seed=510901)
    f2_b = initialize_v3_arm(ROOT, arm="G-F2", seed=510902)

    assert _state_hash(g0_a[0]) != _state_hash(g0_b[0])
    assert _state_hash(f2_a[0]) == _state_hash(f2_b[0])
    assert f2_a[0].checkpoint_metadata["checkpoint_sha256"] == F2_SHA256
    assert g0_a[1] is not g0_b[1] and g0_a[1].optimizer is not g0_b[1].optimizer
    assert g0_a[2] is not g0_b[2]
    assert g0_a[1].bn_mode == "freeze"


def test_v3_state_restores_rng_optimizer_and_replay_identity(tmp_path):
    model, trainer, replay = initialize_v3_arm(ROOT, arm="G0", seed=510901)
    progress = {"config_sha256": json.loads((ROOT / "research/g_v3/config/experiment.json").read_text())["config_sha256"],
                "game_count": 0, "budgeted_environment_steps": 0, "completed_game_ids": []}
    initial_bn = capture_bn_buffers(model)
    checkpoint = tmp_path / "state.pt"
    save_v3_state(
        checkpoint, trainer, replay, arm="G0", seed=510901,
        progress=progress, initial_bn_buffers=initial_bn,
    )
    expected_state = {key: value.clone() for key, value in trainer.model.state_dict().items()}

    restored_model, restored_trainer, restored_replay = initialize_v3_arm(ROOT, arm="G0", seed=510901)
    loaded = load_v3_state(
        checkpoint, restored_trainer, restored_replay, arm="G0", seed=510901,
        expected_config_sha256=progress["config_sha256"],
    )
    assert loaded == progress
    assert all(torch.equal(restored_model.state_dict()[key], value) for key, value in expected_state.items())
    assert restored_trainer.bn_mode == "freeze"


def test_interrupted_and_uninterrupted_game_batches_resume_to_identical_weights(tmp_path):
    seed = 510911
    split_root = _test_root(tmp_path / "split", seed)
    whole_root = _test_root(tmp_path / "whole", seed)
    first = run_v3_arm(split_root, arm="G0", seed=seed, until_node=1000, stop_after_games=1)
    assert not first["completed"]
    resumed = run_v3_arm(split_root, arm="G0", seed=seed, until_node=1000, stop_after_games=1)
    whole = run_v3_arm(whole_root, arm="G0", seed=seed, until_node=1000, stop_after_games=2)

    assert resumed["game_count"] == whole["game_count"] == 2
    assert resumed["budgeted_environment_steps"] == whole["budgeted_environment_steps"]
    split_state = torch.load(split_root / "research/g_v3/checkpoints/G0/seed-510911/training_state.pt",
                             map_location="cpu", weights_only=False)
    whole_state = torch.load(whole_root / "research/g_v3/checkpoints/G0/seed-510911/training_state.pt",
                             map_location="cpu", weights_only=False)
    assert split_state["trainer"]["update_count"] == whole_state["trainer"]["update_count"]
    assert all(torch.equal(split_state["trainer"]["model_state"][key], value)
               for key, value in whole_state["trainer"]["model_state"].items())
    assert split_state["replay"]["rng_state"] == whole_state["replay"]["rng_state"]
    split_games = (split_root / "research/g_v3/results/selfplay/G0-seed-510911.jsonl").read_text().splitlines()
    whole_games = (whole_root / "research/g_v3/results/selfplay/G0-seed-510911.jsonl").read_text().splitlines()
    split_ids = [json.loads(row)["game_id"] for row in split_games]
    whole_ids = [json.loads(row)["game_id"] for row in whole_games]
    assert split_ids == whole_ids


def test_all_route_seed_initial_checkpoints_exist_before_first_promotion_state(tmp_path):
    seeds = [510921, 510922]
    root = _test_root(tmp_path / "multi_seed", seeds)
    for arm in ("G0", "G-F2"):
        result = run_v3_arm(root, arm=arm, seed=seeds[0], until_node=1000, stop_after_games=0)
        assert result["budgeted_environment_steps"] == 0

    from src.research.alphazero_g.v3_evaluation import _selection_state

    import pytest
    with pytest.raises(FileNotFoundError, match="seed-510922/node_0.pt"):
        _selection_state(root, json.loads((root / "research/g_v3/config/experiment.json").read_text()))

    initialized = initialize_v3_starts(root)
    assert initialized == [(seed, arm) for seed in seeds for arm in ("G0", "G-F2")]
    selection = _selection_state(root, json.loads((root / "research/g_v3/config/experiment.json").read_text()))
    assert set(selection["champions"]) == {
        f"{arm}:{seed}" for seed in seeds for arm in ("G0", "G-F2")
    }
    assert not list((root / "research/g_v3/results/selfplay").glob("*.jsonl"))


def test_start_initializer_accepts_routes_already_past_first_node(tmp_path, monkeypatch):
    root = _test_root(tmp_path / "already_advanced", [510931])
    import src.research.alphazero_g.v3_runner as runner

    calls = []

    def fake_run(root_path, *, arm, seed, until_node, stop_after_games):
        calls.append((seed, arm, until_node, stop_after_games))
        assert until_node == 30_000
        checkpoint = root_path / f"research/g_v3/checkpoints/{arm}/seed-{seed}/node_0.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"initial")
        return {"budgeted_environment_steps": 2_000}

    monkeypatch.setattr(runner, "run_v3_arm", fake_run)
    assert runner.initialize_v3_starts(root) == [(510931, "G0"), (510931, "G-F2")]
    assert len(calls) == 2


def test_completed_node_lookup_skips_older_milestones_and_checks_checkpoint(tmp_path):
    import hashlib
    from src.research.alphazero_g.training import save_g_checkpoint

    seed = 510941
    root = _test_root(tmp_path / "milestone_lookup", seed)
    config = json.loads((root / "research/g_v3/config/experiment.json").read_text())
    model, _, _ = initialize_v3_arm(root, arm="G0", seed=seed)
    relative = f"research/g_v3/checkpoints/G0/seed-{seed}/node_1000.pt"
    checkpoint = root / relative
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    digest = save_g_checkpoint(checkpoint, model, {
        "experiment": "g_v3", "arm": "G0", "seed": seed, "node": 1000,
        "config_sha256": config["config_sha256"],
    })
    progress = {"arm": "G0", "seed": seed, "config_sha256": config["config_sha256"],
                "budgeted_environment_steps": 2000, "game_count": 3,
                "optimizer_updates": 125, "last_checkpoint": relative}
    (checkpoint.parent / "progress.json").write_text(json.dumps(progress))
    stage = {"checkpoint_id": f"G0-seed-{seed}@1000", "arm": "G0", "seed": seed,
             "node": 1000, "checkpoint": relative, "checkpoint_sha256": digest,
             "config_sha256": config["config_sha256"]}
    stages = root / "research/g_v3/results/training/stages.jsonl"
    stages.parent.mkdir(parents=True, exist_ok=True)
    stages.write_text(json.dumps(stage) + "\n")

    completed = completed_v3_node_progress(root, arm="G0", seed=seed, node=1000)
    assert completed is not None
    assert completed["completed"] is True
    assert completed["target_node"] == 1000
    assert completed["budgeted_environment_steps"] == 2000
    assert completed_v3_node_progress(root, arm="G0", seed=seed, node=5000) is None

