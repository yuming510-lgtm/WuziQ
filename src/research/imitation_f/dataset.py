"""Validated, symmetry-aware data preparation for the F V1 policy/value model.

This module intentionally consumes only the project's recorded 9x9 games and
never changes any historical game or search engine.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.gomoku import GomokuBoard


HISTORICAL_9X9_SOURCES = (
    "research/results/9x9_games.jsonl",
    "research/v2/results/9x9_extreme-9x9-d2-1s-vs-c-50ms_games.jsonl",
    "research/v2/results/9x9_extreme-9x9-d2-200ms-vs-c-50ms_games.jsonl",
    "research/v2/results/9x9_extreme-9x9-d2-5s-vs-c-50ms_games.jsonl",
    "research/v2/results/9x9_fair-9x9-1s_games.jsonl",
    "research/v2/results/9x9_fair-9x9-200ms_games.jsonl",
    "research/v2/results/9x9_fair-9x9-50ms_games.jsonl",
    "research/v2/results/9x9_fair-9x9-5s_games.jsonl",
    "research/v3/results/9x9_d3-10s-vs-a_games.jsonl",
    "research/v3/results/9x9_d3-10s-vs-b_games.jsonl",
    "research/v3/results/9x9_d3-10s-vs-c_games.jsonl",
    "research/v3/results/9x9_d3-20s-vs-a_games.jsonl",
    "research/v3/results/9x9_d3-20s-vs-b_games.jsonl",
    "research/v3/results/9x9_d3-20s-vs-c_games.jsonl",
    "research/v3/results/9x9_d3-50s-vs-a_games.jsonl",
    "research/v3/results/9x9_d3-50s-vs-b_games.jsonl",
    "research/v3/results/9x9_d3-50s-vs-c_games.jsonl",
    "research/v4b/results/9x9_games.jsonl",
    "research/v4c/results/9x9_games.jsonl",
    "research/v5/results/9x9_games.jsonl",
    "research/v5_1/results/9x9_games.jsonl",
    "research/v6/results/9x9_games.jsonl",
    "research/v6_2/results/9x9_games.jsonl",
    "research/v6_3/results/9x9_games.jsonl",
)
_NEXT = {"B": "W", "W": "B"}


def _transform_coord(row: int, col: int, size: int, transform: int) -> tuple[int, int]:
    """Apply one of the eight square symmetries; rotations precede reflection."""
    if transform >= 4:
        col = size - 1 - col
    for _ in range(transform % 4):
        row, col = col, size - 1 - row
    return row, col


def _position_text(board: GomokuBoard, to_move: str, transform: int) -> str:
    cells = ["."] * (board.size * board.size)
    for row in range(board.size):
        for col in range(board.size):
            player = board.grid[row][col]
            if player is not None:
                target_row, target_col = _transform_coord(row, col, board.size, transform)
                cells[target_row * board.size + target_col] = "1" if player == to_move else "2"
    # The model sees own/opponent planes, not absolute B/W labels. Canonicalize
    # color-swapped positions too, so paired games cannot leak across splits.
    return "P:" + "".join(cells)


def canonicalize_position(
    board: GomokuBoard,
    to_move: str,
    action: tuple[int, int] | None = None,
) -> tuple[str, int | None]:
    """Return a D4-canonical state key and action index in the same orientation."""
    if to_move not in _NEXT:
        raise ValueError("to_move must be 'B' or 'W'")
    candidates = [(_position_text(board, to_move, t), t) for t in range(8)]
    state_key, transform = min(candidates, key=lambda item: item[0])
    if action is None:
        return state_key, None
    row, col = action
    if not board.in_bounds(row, col):
        raise ValueError("action is outside the board")
    if board.get(action) is not None:
        raise ValueError("action must point to an empty cell")
    canonical_row, canonical_col = _transform_coord(row, col, board.size, transform)
    return state_key, canonical_row * board.size + canonical_col


def _read_move(move: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    try:
        player = str(move["player"])
        row, col = int(move["row"]), int(move["col"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("record has a malformed move") from exc
    if player not in _NEXT:
        raise ValueError("record has an invalid player")
    return player, (row, col)


def validate_game_record(record: dict[str, Any]) -> None:
    """Replay and validate a complete, legal 9x9 game record."""
    if not isinstance(record, dict) or int(record.get("board_size", -1)) != 9:
        raise ValueError("record must describe a 9x9 game")
    moves = record.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("record is missing complete moves")
    if record.get("move_count", len(moves)) != len(moves):
        raise ValueError("move_count does not match recorded moves")

    board = GomokuBoard(9)
    expected = "B"
    winner: str | None = None
    for index, move in enumerate(moves):
        if not isinstance(move, dict):
            raise ValueError("record has a malformed move")
        if "ply" in move and int(move["ply"]) != index:
            raise ValueError("record ply sequence is malformed")
        player, coord = _read_move(move)
        if player != expected:
            raise ValueError("record has an out-of-turn move")
        if winner is not None or board.is_full():
            raise ValueError("record contains a move after terminal state")
        try:
            board.place_stone(player, coord)
        except (ValueError, IndexError) as exc:
            raise ValueError("record contains an illegal move") from exc
        expected = _NEXT[player]
        if board.check_victory(coord, player):
            winner = player
            if index != len(moves) - 1:
                raise ValueError("record contains a move after terminal victory")

    termination = record.get("termination")
    winner_color = record.get("winner_color")
    if winner is not None:
        if winner_color != winner or termination not in (None, "five_in_row"):
            raise ValueError("record winner/termination does not match replay")
        outcome = "b_win" if winner == "B" else "w_win"
        if record.get("outcome", outcome) != outcome:
            raise ValueError("record outcome does not match replay")
        last_model = moves[-1].get("model")
        if record.get("winner_model") not in (None, last_model):
            raise ValueError("record winner_model does not match final move")
    else:
        if not board.is_full():
            raise ValueError("record is not a complete terminal game")
        if winner_color is not None or termination not in (None, "full_board"):
            raise ValueError("record draw/termination does not match replay")
        if record.get("outcome", "draw") != "draw":
            raise ValueError("record draw outcome does not match replay")

    opening = record.get("opening")
    if isinstance(opening, list):
        if len(opening) > len(moves):
            raise ValueError("record opening is longer than its game")
        for index, entry in enumerate(opening):
            expected_move = moves[index]
            if (entry.get("player"), int(entry.get("row", -1)), int(entry.get("col", -1))) != (
                expected_move.get("player"), int(expected_move.get("row", -2)), int(expected_move.get("col", -2))
            ):
                raise ValueError("record opening does not match move prefix")


def _game_fingerprint(record: dict[str, Any]) -> str:
    serialized = ";".join(f"{m['player']}:{int(m['row'])}:{int(m['col'])}" for m in record["moves"])
    return hashlib.sha256(serialized.encode()).hexdigest()


def _opening_prefix(record: dict[str, Any]) -> list[dict[str, Any]]:
    opening = record.get("opening")
    if isinstance(opening, list):
        return opening
    return [move for move in record["moves"] if move.get("opening")]


def _opening_group_id(record: dict[str, Any]) -> str:
    board = GomokuBoard(9)
    next_player = "B"
    for move in _opening_prefix(record):
        player, coord = _read_move(move)
        if player != next_player:
            raise ValueError("opening prefix is out of turn")
        board.place_stone(player, coord)
        next_player = _NEXT[player]
    key, _ = canonicalize_position(board, next_player)
    return "opening-" + hashlib.sha256(key.encode()).hexdigest()[:20]


def records_to_positions(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay games and aggregate C/E5 decisions into canonical soft targets."""
    seen_games: set[str] = set()
    aggregates: dict[str, dict[str, Any]] = {}
    source_games = 0
    duplicate_games = 0
    teacher_moves = Counter()
    opening_groups_by_state: dict[str, set[str]] = defaultdict(set)
    for record in records:
        validate_game_record(record)
        source_games += 1
        fingerprint = _game_fingerprint(record)
        if fingerprint in seen_games:
            duplicate_games += 1
            continue
        seen_games.add(fingerprint)
        group_id = _opening_group_id(record)
        winner = record.get("winner_color")
        board = GomokuBoard(9)
        for index, move in enumerate(record["moves"]):
            player, coord = _read_move(move)
            is_teacher = move.get("model") in ("C", "E5") and not bool(move.get("opening", False))
            if is_teacher:
                state_key, action_index = canonicalize_position(board, player, coord)
                if action_index is None:
                    raise AssertionError("canonical action unexpectedly missing")
                current = aggregates.setdefault(state_key, {
                    "state_key": state_key,
                    "canonical_board": state_key.split(":", 1)[1],
                    "to_move": player,
                    "policy_counts": Counter(),
                    "value_sum": 0.0,
                    "value_count": 0,
                    "opening_group_ids": set(),
                    "source_game_ids": set(),
                    "teachers": Counter(),
                    "teacher_budgets_seconds": [],
                })
                current["policy_counts"][action_index] += 1
                current["opening_group_ids"].add(group_id)
                current["source_game_ids"].add(str(record.get("game_id", fingerprint)))
                current["teachers"][str(move["model"])] += 1
                current["teacher_budgets_seconds"].append(float(move.get("elapsed_seconds", 0.0)))
                # Labels are from the position's side-to-move perspective.
                current["value_sum"] += 0.0 if winner is None else (1.0 if winner == player else -1.0)
                current["value_count"] += 1
                teacher_moves[str(move["model"])] += 1
                opening_groups_by_state[state_key].add(group_id)
            board.place_stone(player, coord)

    rows: list[dict[str, Any]] = []
    conflicts = 0
    for key in sorted(aggregates):
        agg = aggregates[key]
        counts: Counter[int] = agg["policy_counts"]
        total = sum(counts.values())
        if len(counts) > 1:
            conflicts += 1
        policy = [0.0] * 81
        for action, count in counts.items():
            policy[action] = count / total
        rows.append({
            "state_key": key,
            "canonical_board": agg["canonical_board"],
            "to_move": agg["to_move"],
            "policy_target": policy,
            "policy_action_counts": {str(action): count for action, count in sorted(counts.items())},
            "value_target": agg["value_sum"] / max(1, agg["value_count"]),
            "value_count": agg["value_count"],
            "policy_count": total,
            "opening_group_ids": sorted(agg["opening_group_ids"]),
            "source_game_ids": sorted(agg["source_game_ids"]),
            "teacher_counts": dict(sorted(agg["teachers"].items())),
            "teacher_elapsed_seconds": agg["teacher_budgets_seconds"],
        })
    return rows, {
        "source_games": source_games,
        "duplicate_games": duplicate_games,
        "unique_games": len(seen_games),
        "positions": len(rows),
        "policy_decisions": dict(sorted(teacher_moves.items())),
        "conflicting_policy_positions": conflicts,
    }


def _components(rows: list[dict[str, Any]]) -> list[tuple[set[str], list[dict[str, Any]]]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        groups = sorted(set(row.get("opening_group_ids", [])))
        for group in groups:
            find(group)
        for group in groups[1:]:
            union(groups[0], group)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups = set(row.get("opening_group_ids", []))
        if not groups:
            # Unknown provenance is kept as a unique component rather than mixed.
            groups = {"unassigned-" + str(row.get("state_key", id(row)))}
            find(next(iter(groups)))
        root = find(next(iter(groups)))
        for group in groups:
            root = find(group)
        grouped_rows[root].append(row)
        component_groups[root].update(groups)
    # A row with multiple ids has already unioned those ids; normalize roots.
    normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized_groups: dict[str, set[str]] = defaultdict(set)
    for root, component_rows in grouped_rows.items():
        actual = find(root)
        normalized[actual].extend(component_rows)
        normalized_groups[actual].update(component_groups[root])
    return [(normalized_groups[key], normalized[key]) for key in sorted(normalized)]


def split_positions(
    rows: list[dict[str, Any]],
    seed: int,
    validation_min: int,
    test_min: int,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split whole connected opening/symmetry components."""
    if not rows:
        raise ValueError("cannot split an empty position set")
    components = _components(rows)
    if len(components) < 3:
        raise ValueError("at least three independent opening components are required")
    rng = random.Random(seed)
    rng.shuffle(components)
    total = len(rows)
    val_target = max(validation_min, round(total * 0.10))
    test_target = max(test_min, round(total * 0.10))
    chosen: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "blind_test": []}
    remaining = list(components)
    for name, target in (("validation", val_target), ("blind_test", test_target)):
        while len(chosen[name]) < target:
            if not remaining:
                raise ValueError(f"not enough independent groups to meet {name} minimum")
            # Prefer the component that least overshoots the remaining target.
            index = min(range(len(remaining)), key=lambda i: abs(target - len(chosen[name]) - len(remaining[i][1])))
            _, component_rows = remaining.pop(index)
            chosen[name].extend(component_rows)
    for _, component_rows in remaining:
        chosen["train"].extend(component_rows)
    if not chosen["train"]:
        # Keep one smallest component for training if validation/test consumed all.
        source = "blind_test" if len(chosen["blind_test"]) > len(chosen["validation"]) else "validation"
        moved = min(_components(chosen[source]), key=lambda item: len(item[1]))
        move_groups = moved[0]
        component_rows = [row for row in chosen[source] if set(row.get("opening_group_ids", [])) & move_groups]
        for row in component_rows:
            chosen[source].remove(row)
        chosen["train"].extend(component_rows)
    return {name: sorted(group, key=lambda row: row.get("state_key", "")) for name, group in chosen.items()}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temp.replace(path)


def _merge_annotations(
    rows: list[dict[str, Any]], annotations_path: Path | None, frozen_assignment: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    if annotations_path is None or not annotations_path.exists():
        return rows, 0
    by_key = {row["state_key"]: row for row in rows}
    heldout_keys = {row["state_key"] for row in rows if any(
        frozen_assignment.get(group) in ("validation", "blind_test") for group in row["opening_group_ids"]
    )}
    accepted = 0
    for ann in _load_jsonl(annotations_path):
        key = str(ann["state_key"])
        group = str(ann["opening_group_id"])
        # Generated openings are train-only by construction; an exact opening
        # canonical key that exists historically inherits that frozen split.
        destination = frozen_assignment.setdefault(group, "train")
        if destination != "train" or key in heldout_keys:
            continue
        target = by_key.get(key)
        if target is None:
            policy = [0.0] * 81
            target = {
                "state_key": key,
                "canonical_board": ann.get("canonical_board", key.split(":", 1)[1]),
                "to_move": ann["to_move"],
                "policy_target": [float(x) for x in policy],
                "policy_action_counts": {},
                "value_target": 0.0,
                "value_count": 0,
                "policy_count": 0,
                "opening_group_ids": [],
                "source_game_ids": [],
                "teacher_counts": {},
                "teacher_elapsed_seconds": [],
            }
            by_key[key] = target
            rows.append(target)
        action = ann.get("action_index")
        if action is not None:
            counts = Counter({int(index): int(count) for index, count in target.get("policy_action_counts", {}).items()})
            if not 0 <= int(action) < 81:
                raise ValueError("annotation policy target has invalid shape or action")
            counts[int(action)] += 1
            total = sum(counts.values())
            target["policy_action_counts"] = {str(index): count for index, count in sorted(counts.items())}
            policy = [0.0] * 81
            for index, count in counts.items():
                policy[index] = count / total
            target["policy_target"] = policy
            target["policy_count"] = int(target.get("policy_count", 0)) + 1
        value_count = int(target.get("value_count", 0))
        added_value_count = int(ann.get("value_count", 1))
        if added_value_count > 0:
            target["value_target"] = (
                float(target.get("value_target", 0.0)) * value_count
                + float(ann.get("value_target", 0.0)) * added_value_count
            ) / (value_count + added_value_count)
            target["value_count"] = value_count + added_value_count
        target["opening_group_ids"] = sorted(set(target["opening_group_ids"]) | {group})
        target["source_game_ids"] = sorted(set(target["source_game_ids"]) | {str(ann.get("source_game_id", "generated"))})
        target["teacher_counts"][str(ann.get("teacher", "C"))] = target["teacher_counts"].get(str(ann.get("teacher", "C")), 0) + 1
        accepted += 1
    return rows, accepted


def prepare_dataset(
    root: Path,
    output_dir: Path,
    seed: int,
    validation_min: int,
    test_min: int,
    annotations_path: Path | None = None,
) -> dict[str, Any]:
    """Read registered official games, freeze leakage-safe splits, and write JSONL."""
    root, output_dir = Path(root), Path(output_dir)
    records: list[dict[str, Any]] = []
    source_info = []
    for relative in HISTORICAL_9X9_SOURCES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered historical source missing: {path}")
        loaded = _load_jsonl(path)
        if any(int(record.get("board_size", -1)) != 9 for record in loaded):
            raise ValueError(f"non-9x9 record found in registered source: {relative}")
        records.extend(loaded)
        source_info.append({"path": relative, "records": len(loaded), "sha256": _sha256(path)})
    rows, stats = records_to_positions(records)
    frozen_path = output_dir / "historical_split_manifest.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assignment = frozen["opening_group_assignment"]
        base_splits = frozen["splits"]
        # Reconstruct a consistent historical split from saved IDs, never re-roll.
        split = {name: [] for name in ("train", "validation", "blind_test")}
        for row in rows:
            destinations = {assignment[group] for group in row["opening_group_ids"] if group in assignment}
            if len(destinations) != 1:
                raise ValueError("historical opening group assignment is incomplete or inconsistent")
            split[next(iter(destinations))].append(row)
        if set(assignment.values()) != {"train", "validation", "blind_test"}:
            raise ValueError("frozen historical split does not contain all partitions")
    else:
        split = split_positions(rows, seed, validation_min, test_min)
        assignment: dict[str, str] = {}
        for name, part in split.items():
            for row in part:
                for group in row["opening_group_ids"]:
                    previous = assignment.setdefault(group, name)
                    if previous != name:
                        raise ValueError("shared opening component leaked across splits")
        frozen = {
            "schema_version": 1,
            "seed": seed,
            "counts": {name: len(part) for name, part in split.items()},
            "opening_group_assignment": assignment,
            "splits": {name: sorted(row["state_key"] for row in part) for name, part in split.items()},
            "historical_stats": stats,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        temp = frozen_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(frozen_path)

    split_ids = {name: {row["state_key"] for row in part} for name, part in split.items()}
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "validation"), ("train", "blind_test"), ("validation", "blind_test"))):
        raise ValueError("canonical positions leaked across historical splits")
    final_rows, accepted_annotations = _merge_annotations(rows, annotations_path, assignment)
    split["train"] = [row for row in final_rows if row["state_key"] in split_ids["train"] or (
        row.get("opening_group_ids") and all(assignment.get(group) == "train" for group in row["opening_group_ids"])
    )]
    split["validation"] = [row for row in final_rows if row["state_key"] in split_ids["validation"]]
    split["blind_test"] = [row for row in final_rows if row["state_key"] in split_ids["blind_test"]]
    for name, part in split.items():
        _write_jsonl(output_dir / f"{name}.jsonl", part)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "sources": source_info,
        "historical": stats,
        "accepted_training_annotations": accepted_annotations,
        "counts": {name: len(part) for name, part in split.items()},
        "files": {name: {"path": f"{name}.jsonl", "sha256": _sha256(output_dir / f"{name}.jsonl")} for name in split},
        "split_manifest": "historical_split_manifest.json",
        "split_hash": _sha256(frozen_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest
