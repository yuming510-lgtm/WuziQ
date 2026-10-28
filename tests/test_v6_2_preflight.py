from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_script_imports_its_frozen_engine_adapter() -> None:
    source = ROOT / "scripts/preflight_v6_2.py"
    spec = importlib.util.spec_from_file_location("v6_2_preflight", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module._decision)
    assert callable(module._estimates)


def test_preflight_csv_writer_emits_only_registered_fields(tmp_path: Path) -> None:
    source = ROOT / "scripts/preflight_v6_2.py"
    spec = importlib.util.spec_from_file_location("v6_2_preflight_csv", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "benchmark_moves.csv"
    row = {
        "model": "D", "position_id": "fixture", "budget_seconds": 200.0,
        "move": [4, 4], "legal": True, "elapsed_seconds": 200.0,
        "wall_seconds": 200.1, "nodes": 1000, "completed_depth": 4,
        "tree_depth": 4, "cutoffs": 0, "simulations": 50_000,
        "timed_out": True, "over_budget_seconds": 0.0,
        "stats": {"extra": "ignored by the flat CSV"}, "board_unchanged": True,
    }

    module._write_benchmark_csv(output, [row])

    import csv
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["model"] == "D"
    assert rows[0]["move"] == "[4, 4]"
    assert rows[0]["simulations"] == "50000"
    assert "stats" not in rows[0]
