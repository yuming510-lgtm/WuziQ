"""Tests for the reproducible V4 ABC search benchmark."""
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
    assert importlib.util.find_spec("src.research.abc_search_v4") is not None
    from src.research import abc_search_v4

    return abc_search_v4


def test_frozen_matrix_has_twelve_legal_nonterminal_positions_and_252_run_ids() -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)

    assert len(config["positions"]) >= 12
    assert len(config["budgets_ms"]) == 7
    assert len(runner.expected_run_ids(config)) == 252
    assert {row["category"] for row in config["positions"]} >= {
        "opening",
        "midgame",
        "late_game",
        "attack",
        "defense",
    }


def test_direct_tactical_positions_freeze_expected_win_and_forced_block() -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    positions = {row["position_id"]: row for row in config["positions"]}

    assert positions["attack_open_four"]["expected_best_moves"] == [[4, 1], [4, 6]]
    assert positions["defense_block_four"]["expected_best_moves"] == [[4, 4]]


@pytest.mark.parametrize("position_id", ["attack_open_four", "defense_block_four"])
@pytest.mark.parametrize("model", ["B", "C"])
def test_minimax_models_take_the_frozen_immediate_win_or_forced_block(
    position_id: str, model: str
) -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    position = next(row for row in config["positions"] if row["position_id"] == position_id)

    result = runner._run_one(config, "test-config", "test-code", position, model, 50)

    assert result["tactical_move_matches_expected"] is True


def test_matrix_rejects_terminal_position(tmp_path: Path) -> None:
    runner = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["positions"][0]["moves"] = [
        {"player": "B", "row": 4, "col": 0},
        {"player": "W", "row": 0, "col": 0},
        {"player": "B", "row": 4, "col": 1},
        {"player": "W", "row": 0, "col": 1},
        {"player": "B", "row": 4, "col": 2},
        {"player": "W", "row": 0, "col": 2},
        {"player": "B", "row": 4, "col": 3},
        {"player": "W", "row": 0, "col": 3},
        {"player": "B", "row": 4, "col": 4},
        {"player": "W", "row": 0, "col": 4},
    ]
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal"):
        runner.load_config(invalid_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("occupied", "already occupied"),
        ("out_of_bounds", "out of bounds"),
        ("nonalternating", "nonalternating"),
        ("wrong_to_move", "incorrect to_move"),
    ],
)
def test_matrix_rejects_invalid_move_histories(
    tmp_path: Path, mutation: str, message: str
) -> None:
    runner = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    position = config["positions"][0]
    if mutation == "occupied":
        position["moves"][1]["row"] = position["moves"][0]["row"]
        position["moves"][1]["col"] = position["moves"][0]["col"]
    elif mutation == "out_of_bounds":
        position["moves"][0]["row"] = 9
    elif mutation == "nonalternating":
        position["moves"][1]["player"] = "B"
    else:
        position["to_move"] = "W"
    invalid_path = tmp_path / f"{mutation}.json"
    invalid_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.load_config(invalid_path)


def test_runner_persists_unique_runs_and_resumes_without_recounting(tmp_path: Path) -> None:
    runner = _module()

    first = runner.run_matrix(CONFIG_PATH, tmp_path, max_new_runs=1)
    records = runner.load_run_records(tmp_path / "search_runs.jsonl")
    assert first["new_runs"] == 1
    assert len(records) == 1
    assert records[0]["model"] == "A"
    assert records[0]["budget_ms"] == 50
    assert records[0]["elapsed_seconds"] >= 0

    second = runner.run_matrix(CONFIG_PATH, tmp_path, max_new_runs=1)
    records = runner.load_run_records(tmp_path / "search_runs.jsonl")
    assert second["new_runs"] == 1
    assert second["skipped_runs"] == 1
    assert len(records) == 2
    assert len({row["run_id"] for row in records}) == 2

    with pytest.raises(ValueError, match="Duplicate run_id"):
        runner.append_run_record(
            tmp_path / "search_runs.jsonl", records[0], {row["run_id"] for row in records}
        )


def test_rule_model_uses_actual_decision_time_without_budget_wait(tmp_path: Path) -> None:
    runner = _module()

    result = runner.run_matrix(CONFIG_PATH, tmp_path, max_new_runs=7)
    records = runner.load_run_records(tmp_path / "search_runs.jsonl")

    assert result["new_runs"] == 7
    assert [row["budget_ms"] for row in records] == [50, 200, 1000, 5000, 10000, 20000, 50000]
    assert all(row["model"] == "A" for row in records)
    assert all(0 <= row["elapsed_seconds"] < 0.5 for row in records)
    assert len({tuple(row["move"]) for row in records}) == 1
    assert all(row["nodes"] == 0 and row["completed_depth"] == 0 for row in records)


def test_runner_rejects_off_schedule_run_id(tmp_path: Path) -> None:
    runner = _module()
    (tmp_path / "search_runs.jsonl").write_text(
        json.dumps({"run_id": "not-in-schedule"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="outside the configured matrix"):
        runner.run_matrix(CONFIG_PATH, tmp_path)


def test_runner_truncates_only_corrupt_final_jsonl_tail(tmp_path: Path) -> None:
    runner = _module()
    runner.run_matrix(CONFIG_PATH, tmp_path, max_new_runs=1)
    records_path = tmp_path / "search_runs.jsonl"
    records_path.write_bytes(records_path.read_bytes() + b'{"broken":')

    records = runner.load_run_records(records_path)

    assert len(records) == 1
    assert records_path.read_bytes().endswith(b"\n")
