"""Statistical consistency and summary metric tests for V4."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gomoku import GomokuBoard

CONFIG_PATH = ROOT / "configs" / "abc_search_compute_v4.json"


def _module():
    assert importlib.util.find_spec("src.research.abc_search_v4_analysis") is not None
    from src.research import abc_search_v4_analysis

    return abc_search_v4_analysis


def _pair(config, *, c_score: int = 100):
    position_id = config["positions"][0]["position_id"]
    rows = []
    for model, root2, nodes, cutoffs in [
        ("B", 100, 1200, 0),
        ("C", c_score, 400, 36),
    ]:
        rows.append(
            {
                "run_id": f"{position_id}__{model}__00050ms",
                "position_id": position_id,
                "category": "opening",
                "board_size": 9,
                "plies": 4,
                "model": model,
                "budget_ms": 50,
                "completed_depth": 2,
                "max_depth_limit": 77,
                "root_value": root2,
                "move": [0, 0],
                "nodes": nodes,
                "completed_depth_nodes": nodes,
                "partial_iteration_nodes": 0,
                "cutoffs": cutoffs,
                "completed_depth_cutoffs": cutoffs,
                "partial_iteration_cutoffs": 0,
                "elapsed_seconds": 0.05,
                "timed_out": True,
                "depth_iterations": [
                    {"depth": 1, "root_value": 10, "move": [0, 0], "nodes": 20, "cutoffs": 0},
                    {"depth": 2, "root_value": root2, "move": [0, 0], "nodes": nodes - 20, "cutoffs": cutoffs},
                ],
            }
        )
    return rows


def test_validator_compares_only_matching_complete_depths() -> None:
    import src.research.abc_search_v4 as runner

    analysis = _module()
    config = runner.load_config(CONFIG_PATH)
    result = analysis.validate_runs(_pair(config), config, require_complete=False)

    assert result["checked_runs"] == 2
    assert result["score_mismatch_count"] == 0
    assert result["shared_depth_comparisons"] == 2
    assert result["missing_runs"] == 250


def test_validator_rejects_b_c_score_disagreement_at_common_depth() -> None:
    import src.research.abc_search_v4 as runner

    analysis = _module()
    config = runner.load_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="root score mismatch"):
        analysis.validate_runs(_pair(config, c_score=99), config, require_complete=False)


def test_summary_and_node_savings_use_complete_shared_depths() -> None:
    import src.research.abc_search_v4 as runner

    analysis = _module()
    config = runner.load_config(CONFIG_PATH)
    pair = _pair(config)
    summary = analysis.summarize_runs(pair, config)

    b = summary["by_model_budget"]["B"]["50"]
    c = summary["by_model_budget"]["C"]["50"]
    savings = summary["alpha_beta_node_savings"]["50"]
    assert b["run_count"] == c["run_count"] == 1
    assert b["mean_nodes"] == 1200
    assert c["mean_cutoffs"] == 36
    assert savings["shared_position_depth_pairs"] == 2
    assert savings["mean_node_reduction_ratio"] > 0


def test_full_validation_rejects_missing_matrix_rows() -> None:
    import src.research.abc_search_v4 as runner

    analysis = _module()
    config = runner.load_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="missing run IDs"):
        analysis.validate_runs(_pair(config), config, require_complete=True)


def test_complete_report_writes_summary_and_all_trend_charts(tmp_path: Path) -> None:
    import src.research.abc_search_v4 as runner

    analysis = _module()
    config = runner.load_config(CONFIG_PATH)
    records = []
    for position in config["positions"]:
        board = GomokuBoard(9)
        for move in position["moves"]:
            board.place_stone(move["player"], (move["row"], move["col"]))
        move = next(
            [row, col]
            for row in range(9)
            for col in range(9)
            if board.get((row, col)) is None
        )
        for model in ("A", "B", "C"):
            for budget in config["budgets_ms"]:
                expected = position.get("expected_best_moves")
                records.append(
                    {
                        "run_id": runner.make_run_id(position["position_id"], model, budget),
                        "position_id": position["position_id"],
                        "category": position["category"],
                        "board_size": 9,
                        "plies": len(position["moves"]),
                        "model": model,
                        "budget_ms": budget,
                        "max_depth_limit": 81 - len(position["moves"]),
                        "elapsed_seconds": 0.001,
                        "move": move,
                        "root_value": 0,
                        "completed_depth": 0,
                        "depth_iterations": [],
                        "nodes": 0,
                        "completed_depth_nodes": 0,
                        "partial_iteration_nodes": 0,
                        "cutoffs": 0,
                        "completed_depth_cutoffs": 0,
                        "tactical_expected_moves": expected,
                        "tactical_move_matches_expected": (
                            move in expected if expected is not None else None
                        ),
                    }
                )

    report = tmp_path / "report_zh.md"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analysis.write_report(config, records, report, tmp_path / "figures")

    assert result["validation"]["complete"]
    assert result["summary"]["actual_run_count"] == 252
    assert len(result["figures"]) == 5
    assert all(Path(path).is_file() and Path(path).stat().st_size > 1000 for path in result["figures"])
    assert (tmp_path / "summary.json").is_file()
    assert "直接攻防着法检查" in report.read_text(encoding="utf-8")
    assert not [warning for warning in caught if issubclass(warning.category, UserWarning)]
