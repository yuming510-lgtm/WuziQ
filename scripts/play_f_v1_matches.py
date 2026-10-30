#!/usr/bin/env python3
"""Run six exploratory F-vs-A/B/C games on three frozen unseen openings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gomoku import GomokuBoard
from src.research.imitation_f.dataset import canonicalize_position, validate_game_record
from src.research.imitation_f.evaluation import play_match
from src.research.imitation_f.generation import build_evaluation_openings
from src.research.imitation_f.model import load_f_checkpoint


def _atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "research/f_v1/data"
    results_dir = root / "research/f_v1/results"
    opening_path = results_dir / "evaluation_openings.json"
    matches_path = results_dir / "matches.jsonl"
    checkpoint = args.checkpoint or root / "research/f_v1/checkpoints/best.pt"
    train_rows = [json.loads(line) for line in (data_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    training_groups = {group for row in train_rows for group in row.get("opening_group_ids", [])}
    training_states = {row["state_key"] for row in train_rows}
    if opening_path.exists():
        openings = json.loads(opening_path.read_text(encoding="utf-8"))
    else:
        openings = build_evaluation_openings(731901, 3, tuple(training_groups))
        _atomic_json(opening_path, openings)
    if len(openings) != 3 or any(row["opening_group_id"] in training_groups for row in openings):
        raise ValueError("evaluation openings must be three frozen groups absent from training")

    model = load_f_checkpoint(checkpoint, device="auto")
    existing = {}
    if matches_path.exists():
        for line in matches_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            validate_game_record(record)
            if record["game_id"] in existing:
                raise ValueError(f"duplicate persisted match id: {record['game_id']}")
            existing[record["game_id"]] = record
    for index, opponent in enumerate(("A", "B", "C")):
        opening_spec = openings[index]
        for f_color in ("B", "W"):
            game_id = f"f-v1-match-9x9-f-vs-{opponent.lower()}-s{int(opening_spec['opening_seed'])}-f{f_color.lower()}"
            if game_id in existing:
                continue
            record = play_match(
                model, opponent, opening_spec["opening"], f_color,
                int(opening_spec["opening_seed"]), opponent_budget_seconds=0.05,
            )
            validate_game_record(record)
            existing[record["game_id"]] = record
            _atomic_jsonl(matches_path, [existing[key] for key in sorted(existing)])
    state_overlaps = set()
    for record in existing.values():
        board = GomokuBoard(9)
        for move in record["moves"]:
            if move.get("model") is not None:
                key, _ = canonicalize_position(board, str(move["player"]))
                if key in training_states:
                    state_overlaps.add(key)
            board.place_stone(str(move["player"]), (int(move["row"]), int(move["col"])))
    summary = {
        "schedule_games": 6,
        "completed_games": len(existing),
        "new_match_ids": sorted(existing),
        "training_state_overlap_count": len(state_overlaps),
        "training_unseen_opening_groups": len(openings),
        "opening_seeds": [int(row["opening_seed"]) for row in openings],
    }
    _atomic_json(results_dir / "match_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
