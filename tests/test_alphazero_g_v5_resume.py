import json
from pathlib import Path

import torch

from src.research.alphazero_g.v5_training import train_reference_arm
from src.research.alphazero_g.v5_audit import audit_selfplay_game


def test_reference_runner_saves_full_state_and_resume_is_idempotent(tmp_path: Path):
    torch.set_num_threads(1)
    config = {
        "experiment_id": "wuziq-alphazero-g-v5-reference-and-closed-loop",
        "config_sha256": "test-config-sha",
        "training": {
            "buffer_size": 128,
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.002,
            "kl_target": 0.02,
        },
    }
    first = train_reference_arm(
        tmp_path, arm="resume_smoke", board_size=6, n_in_row=4, simulations=2,
        target_states=1, milestones=[1], seed=77, config=config,
        opening_provider=lambda _i, _s: [
            {"player": "B", "row": 2, "col": 2},
            {"player": "W", "row": 3, "col": 3},
        ],
    )
    checkpoint_path = Path(first["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["game_count"] == 1
    assert checkpoint["environment_states"] >= 1
    assert checkpoint["optimizer_state"]
    assert checkpoint["replay_buffer"]
    assert checkpoint["python_rng_state"]
    raw = list((tmp_path / "research/g_v5/raw/selfplay/resume_smoke").glob("*.json"))
    assert len(raw) == 1
    record = json.loads(raw[0].read_text())
    assert record["completed"] if "completed" in record else record["termination"] in {"five_in_row", "full_board"}
    audit = audit_selfplay_game(raw[0])
    assert audit["game_id"] == record["game_id"]
    assert audit["samples"] == record["training_state_count"]

    second = train_reference_arm(
        tmp_path, arm="resume_smoke", board_size=6, n_in_row=4, simulations=2,
        target_states=1, milestones=[1], seed=77, config=config,
        opening_provider=lambda _i, _s: [],
    )
    checkpoint2 = torch.load(Path(second["checkpoint"]), map_location="cpu", weights_only=False)
    assert checkpoint2["game_count"] == checkpoint["game_count"]
    assert checkpoint2["environment_states"] == checkpoint["environment_states"]
    assert len(list((tmp_path / "research/g_v5/raw/selfplay/resume_smoke").glob("*.json"))) == 1
