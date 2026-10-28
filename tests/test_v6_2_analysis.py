from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.research.checkpoint import load_game_records
from src.research.v6_2.analysis import aggregate_compute, build_report, render_champion_svg
from src.research.v6_2.config import load_v6_2_config
from src.research.v6_2.runner import run_championship
from src.research.v6_2.schedule import build_schedule
from src.research.v6_2.validation import validate_v6_2_records


def test_compute_aggregation_uses_real_decision_rows_and_excludes_opening_moves() -> None:
    records = [{
        "game_id": "fixture",
        "moves": [
            {"model": None, "opening": True, "elapsed_seconds": 0, "nodes": 0},
            {"model": "C", "opening": False, "elapsed_seconds": 2.0, "nodes": 100,
             "cutoffs": 20, "simulations": 0, "completed_depth": 5,
             "tree_depth": 5, "timed_out": True},
            {"model": "D", "opening": False, "elapsed_seconds": 200.0, "nodes": 900,
             "cutoffs": 0, "simulations": 50_000, "completed_depth": 4,
             "tree_depth": 4, "timed_out": True},
        ],
    }]

    result = aggregate_compute(records)

    assert result["C"]["moves"] == 1
    assert result["C"]["thinking_seconds"] == 2.0
    assert result["C"]["nodes"] == 100
    assert result["C"]["average_seconds_per_move"] == 2.0
    assert result["D"]["simulations"] == 50_000
    assert result["D"]["timeouts"] == 1
    assert result["D"]["max_tree_depth"] == 4


def test_champion_svg_renders_score_and_compute_comparisons() -> None:
    standings = {
        "C": {"points": 3.0},
        "D": {"points": 2.0},
        "E5": {"points": 1.0},
    }
    compute = {
        "C": {"thinking_seconds": 100.0},
        "D": {"thinking_seconds": 400.0},
        "E5": {"thinking_seconds": 200.0},
    }

    svg = render_champion_svg(standings, compute)

    assert svg.startswith("<svg")
    assert all(model in svg for model in ("C", "D", "E5"))
    assert "积分" in svg and "计算秒数" in svg
    assert "viewBox" in svg


def test_report_builder_writes_consistent_results_from_validated_games(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("v6_2_runner_test_fixtures", root / "tests/test_v6_2_runner.py")
    assert spec is not None and spec.loader is not None
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    output = tmp_path / "results"
    config = load_v6_2_config()
    run_championship(
        config, output, game_runner=fixture_module._fake_game,
        implementation_commit="test-commit",
    )
    records = load_game_records(output / "9x9_games.jsonl")
    validation = validate_v6_2_records(records, build_schedule(config), require_complete=True)
    preflight = {
        "base_seed": config["base_seed"],
        "position_id": "test fixture",
        "estimate": {"scenarios_hours": {"optimistic": 1.0, "central": 2.0, "conservative": 3.0}},
        "benchmark_rows": [],
    }
    environment = {"cpu_logical": 3, "cpu_physical": 3, "memory_total_bytes": 1000, "disk_free_bytes": 1000}

    result = build_report(
        records, validation, json.loads((output / "progress.json").read_text()),
        preflight, environment, tmp_path / "report",
    )

    report = (tmp_path / "report/REPORT.md").read_text(encoding="utf-8")
    assert result["champions"] == ["C", "D", "E5"]
    assert "v6.2-9x9-c-d-o731201-s0" in report
    assert "探索性" in report
    assert (tmp_path / "report/results/standings.csv").is_file()
    assert (tmp_path / "report/figures/champion_comparison.svg").is_file()
