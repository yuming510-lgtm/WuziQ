#!/usr/bin/env python3
"""Run a fixed-position, fixed-depth E5 component ablation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.v6.e5 import run_ablation
from src.research.v6.puzzles import board_for_position, load_tactical_positions
from src.gomoku import GomokuBoard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/v6/validation")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--budget", type=float, default=2.0)
    args = parser.parse_args()
    fixtures = {row["position_id"]: row for row in load_tactical_positions()}
    selected = ("opening_731101", "blocked_four_pseudo_win", "empty_edge_no_proof")
    positions = [
        (position_id, board_for_position(fixtures[position_id]), str(fixtures[position_id]["to_move"]))
        for position_id in selected
    ]
    rows = run_ablation(
        positions, depth=args.depth, time_limit_seconds=args.budget, seed=6042026,
    )
    target = ROOT / args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "e5_ablation.json"
    csv_path = target / "e5_ablation.csv"
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "positions": list(selected),
        "depth_limit": args.depth,
        "budget_seconds": args.budget,
        "variants": sorted({row["variant"] for row in rows}),
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row if not isinstance(row.get(key), (dict, list))})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    completed = [row for row in rows if row["completed_depth"] == args.depth]
    print(json.dumps({
        "rows": len(rows),
        "completed_at_depth_limit": len(completed),
        "root_values_by_position_and_depth": {
            f"{row['position_id']}:{row['completed_depth']}": row["root_value"]
            for row in rows
        },
        "json": str(json_path.relative_to(ROOT)),
        "csv": str(csv_path.relative_to(ROOT)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
