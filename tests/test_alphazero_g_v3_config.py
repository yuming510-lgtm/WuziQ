import json
from pathlib import Path
import shutil

import pytest

from src.research.alphazero_g.v3_config import evaluation_game_counts, register_v3_budget
from src.research.alphazero_g.v3_schedule import build_v3_schedule


def _write_schedule(root: Path, seeds: list[int]) -> None:
    config_dir = root / "research/g_v3/config"
    config_dir.mkdir(parents=True)
    schedule = build_v3_schedule(
        seeds, train_groups_per_seed=4, development_groups=4, blind_groups=4,
        excluded_openings=[],
    )
    (config_dir / "openings.json").write_text(json.dumps(schedule, sort_keys=True))


def _copy_f2(root: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "research/f_v2/checkpoints/best_f2.pt"
    target = root / "research/f_v2/checkpoints/best_f2.pt"
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)


def _write_evidence(root: Path) -> None:
    preflight_dir = root / "research/g_v3/results/preflight"
    probe_dir = root / "research/g_v3/results/bn_probe"
    preflight_dir.mkdir(parents=True)
    probe_dir.mkdir(parents=True)
    preflight = {
        "resources": {"affinity_cpus": 3},
        "simulation_benchmarks": [{"simulations": count} for count in (16, 32, 64)],
        "selfplay_games": [
            {"arm_id": arm, "sample_positions": 10, "cpu_seconds": 1.0}
            for arm in ("G0", "G-F2")
        ],
        "training_updates": [
            {"arm_id": arm, "updates": 20, "cpu_seconds": 0.1}
            for arm in ("G0", "G-F2")
        ],
        "evaluation_games": [{"completed": True, "cpu_seconds": 0.1}],
    }
    probe = {"formal_training": False, "selection_rule": {"selected_common_mode": "freeze"}}
    (preflight_dir / "preflight.json").write_text(json.dumps(preflight))
    (probe_dir / "probe.json").write_text(json.dumps(probe))


def test_budget_registration_is_hash_locked_and_idempotent(tmp_path):
    _write_schedule(tmp_path, [510601, 510602])
    _copy_f2(tmp_path)
    _write_evidence(tmp_path)

    config = register_v3_budget(
        tmp_path, seeds=[510601, 510602], states_per_seed=10000,
        cpu_hour_cap=3.0, bn_policy="freeze",
    )
    config_path = tmp_path / "research/g_v3/config/experiment.json"
    stored = json.loads(config_path.read_text())

    assert config == stored
    assert config["base_commit"] == "c95eb5aa919c769af11ea6b7fc22df2594ae1c28"
    assert config["f2_sha256"] == "50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257"
    assert config["simulation_budget"] == 64
    assert config["bn_policy"] == "freeze"
    assert config["states_per_seed"] == 10000
    assert config["runtime_estimate"]["evaluation_games"] == 4000
    assert config["evaluation"]["opponent_aliases"]["G-F2:own_initial"] == "G-F2:F2@frozen"
    assert register_v3_budget(
        tmp_path, seeds=[510601, 510602], states_per_seed=10000,
        cpu_hour_cap=3.0, bn_policy="freeze",
    ) == config

    with pytest.raises(ValueError, match="already registered|different"):
        register_v3_budget(
            tmp_path, seeds=[510601, 510602], states_per_seed=30000,
            cpu_hour_cap=8.0, bn_policy="freeze",
        )


def test_registered_evaluation_count_includes_promotions_and_deduplicates_f2_initial():
    assert evaluation_game_counts(
        seed_count=2, stage_count=6, development_groups=4,
        promotion_groups=12, blind_groups=20,
    ) == {"development": 1984, "promotion": 576, "blind": 1440, "total": 4000}


def test_budget_registration_rejects_unverified_initialization_and_bn_mode(tmp_path):
    _write_schedule(tmp_path, [7])
    _copy_f2(tmp_path)
    _write_evidence(tmp_path)
    f2 = tmp_path / "research/f_v2/checkpoints/best_f2.pt"
    f2.write_bytes(f2.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="F2.*hash|checksum"):
        register_v3_budget(
            tmp_path, seeds=[7], states_per_seed=10000,
            cpu_hour_cap=3.0, bn_policy="freeze",
        )

    f2.write_bytes(Path(__file__).resolve().parents[1].joinpath(
        "research/f_v2/checkpoints/best_f2.pt").read_bytes())
    with pytest.raises(ValueError, match="bn_policy"):
        register_v3_budget(
            tmp_path, seeds=[7], states_per_seed=10000,
            cpu_hour_cap=3.0, bn_policy="restore",
        )
