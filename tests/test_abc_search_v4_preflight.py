"""Resource and runtime estimate checks for the V4 preflight."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs" / "abc_search_compute_v4.json"


def _module():
    assert importlib.util.find_spec("src.research.abc_search_v4_preflight") is not None
    from src.research import abc_search_v4_preflight

    return abc_search_v4_preflight


def test_representative_positions_cover_all_position_categories() -> None:
    import src.research.abc_search_v4 as runner

    preflight = _module()
    config = runner.load_config(CONFIG_PATH)

    positions = preflight.representative_positions(config)

    assert len(positions) == 5
    assert {position["category"] for position in positions} == {
        "opening",
        "midgame",
        "late_game",
        "attack",
        "defense",
    }


def test_category_weighted_estimate_covers_all_252_runs_and_storage() -> None:
    import src.research.abc_search_v4 as runner

    preflight = _module()
    config = runner.load_config(CONFIG_PATH)
    samples = [
        {
            "category": position["category"],
            "model": model,
            "budget_ms": budget,
            "elapsed_seconds": 1.0,
            "nodes": 10,
        }
        for position in preflight.representative_positions(config)
        for model in ("A", "B", "C")
        for budget in config["budgets_ms"]
    ]

    estimate = preflight.estimate_full_runtime(config, samples)

    assert estimate["preflight_sample_runs"] == 105
    assert estimate["official_runs"] == len(runner.expected_run_ids(config)) == 252
    assert estimate["estimated_official_compute_seconds"] == 252.0
    assert estimate["estimated_result_storage_bytes"] > 0


def test_resource_snapshot_reports_cpu_memory_and_disk() -> None:
    preflight = _module()

    result = preflight.resource_snapshot(ROOT)

    assert result["logical_cpu_count"] and result["logical_cpu_count"] > 0
    assert result["workspace_disk_free_bytes"] > 0
    assert "memory_available_bytes" in result


def test_preflight_writes_105_sample_rows_and_category_weighted_estimate(
    tmp_path: Path, monkeypatch
) -> None:
    import src.research.abc_search_v4_preflight as preflight

    def fake_run(config, config_hash, code_commit, position, model, budget_ms):
        return {
            "position_id": position["position_id"],
            "category": position["category"],
            "model": model,
            "budget_ms": budget_ms,
            "elapsed_seconds": 0.001,
            "nodes": 1,
        }

    monkeypatch.setattr(preflight, "_run_one", fake_run)
    path = tmp_path / "preflight.json"

    result = preflight.run_preflight(CONFIG_PATH, path, resource_path=tmp_path)

    assert path.is_file()
    assert len(result["sample_runs"]) == 105
    assert len({row["preflight_id"] for row in result["sample_runs"]}) == 105
    assert result["estimate"]["official_runs"] == 252
    assert result["estimate"]["estimated_official_compute_seconds"] == pytest.approx(0.252)
