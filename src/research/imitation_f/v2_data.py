"""F V2 DAgger collection and train-only exact tactical balancing."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable

from ...gomoku import GomokuBoard
from ..runner import derive_seed
from .dataset import canonicalize_position, validate_game_record
from .evaluation import _board_from_canonical, find_immediate_wins, play_match
from .generation import annotate_position
from .model import load_f_checkpoint, predict_policy_value


def verify_network_actions(model, records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Replay records and prove each F move is exactly masked-policy argmax."""
    games = 0
    network_actions = 0
    for record in records:
        board = GomokuBoard(9)
        for index, move in enumerate(record.get("moves", [])):
            player = str(move["player"])
            action = int(move["row"]) * 9 + int(move["col"])
            if move.get("model") in ("F", "F1", "F2"):
                probabilities, _ = predict_policy_value(model, board, player, device="cpu")
                expected = int(probabilities.argmax().item())
                if action != expected:
                    raise ValueError(
                        f"{move.get('model')} action at game {record.get('game_id')} ply {index} "
                        f"does not match masked-policy argmax ({action} != {expected})"
                    )
                network_actions += 1
            board.place_stone(player, divmod(action, 9))
            if board.check_victory(divmod(action, 9), player) and index != len(record["moves"]) - 1:
                raise ValueError("network-action audit found moves after terminal victory")
        validate_game_record(record)
        games += 1
    return {"games": games, "network_actions": network_actions, "mismatches": 0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _policy_for_actions(actions: Iterable[int]) -> list[float]:
    unique = sorted(set(int(action) for action in actions))
    if not unique or any(action < 0 or action >= 81 for action in unique):
        raise ValueError("exact tactical label must contain legal board actions")
    policy = [0.0] * 81
    for action in unique:
        policy[action] = 1.0 / len(unique)
    return policy


def exact_tactical_targets(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only proven one-move wins and unique mandatory blocks."""
    targets = []
    for source in rows:
        row = dict(source)
        to_move = str(row.get("to_move", "B"))
        board, _ = _board_from_canonical(str(row["canonical_board"]), to_move)
        own_wins = find_immediate_wins(board, to_move)
        if own_wins:
            kind, actions = "win", own_wins
        else:
            opponent = "W" if to_move == "B" else "B"
            threats = find_immediate_wins(board, opponent)
            if len(threats) != 1:
                continue
            kind, actions = "block", threats
        row["tactical_kind"] = kind
        row["correct_actions"] = [r * 9 + c for r, c in actions]
        row["policy_target"] = _policy_for_actions(row["correct_actions"])
        row["label_source"] = "exact-five-rule"
        targets.append(row)
    return targets


def _merge_dagger_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    source_rows = list(rows)
    aggregates: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        key = str(source["state_key"])
        target = [float(value) for value in source["policy_target"]]
        if len(target) != 81 or not math.isclose(sum(target), 1.0, abs_tol=1e-5):
            raise ValueError("DAgger policy target must be a normalized 81-action distribution")
        if key not in aggregates:
            aggregates[key] = {**dict(source), "_policy_sum": [0.0] * 81, "_policy_count": 0,
                               "_value_sum": 0.0, "_value_count": 0, "_groups": set(), "_sources": set()}
        current = aggregates[key]
        if current["canonical_board"] != source["canonical_board"] or current["to_move"] != source["to_move"]:
            raise ValueError("same DAgger state key maps to different positions")
        for index, value in enumerate(target):
            current["_policy_sum"][index] += value
        current["_policy_count"] += 1
        if int(source.get("value_count", 1)) > 0:
            current["_value_sum"] += float(source["value_target"])
            current["_value_count"] += 1
        groups = source.get("opening_group_ids", [source.get("opening_group_id")])
        current["_groups"].update(str(value) for value in groups if value)
        current["_sources"].add(str(source.get("source_game_id", "")))

    merged = []
    for key, current in sorted(aggregates.items()):
        policy_count = current.pop("_policy_count")
        policy_sum = current.pop("_policy_sum")
        value_count = current.pop("_value_count")
        value_sum = current.pop("_value_sum")
        groups = current.pop("_groups")
        sources = current.pop("_sources")
        for internal in tuple(current):
            if internal.startswith("_"):
                current.pop(internal)
        current["state_key"] = key
        current["policy_target"] = [value / policy_count for value in policy_sum]
        current["value_target"] = value_sum / value_count if value_count else 0.0
        current["value_count"] = value_count
        current["opening_group_ids"] = sorted(groups)
        current["source_game_ids"] = sorted(value for value in sources if value)
        merged.append(current)
    return merged, len(source_rows) - len(merged)


def build_weighted_training_rows(
    v1_train: list[dict[str, Any]],
    dagger_rows: list[dict[str, Any]],
    validation_keys: set[str],
    blind_keys: set[str],
    tactical_share: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all ordinary train rows, override exact tactics, and balance attack/defense."""
    if not 0.0 < float(tactical_share) < 0.8:
        raise ValueError("tactical_share must be between zero and 0.8")
    dagger_rows = list(dagger_rows)
    heldout = set(validation_keys) | set(blind_keys)
    base = {str(row["state_key"]): dict(row) for row in v1_train}
    if len(base) != len(v1_train):
        raise ValueError("V1 train partition contains duplicate canonical states")
    if heldout.intersection(base):
        raise ValueError("V1 train partition overlaps a held-out canonical state")
    merged_dagger, duplicate_dagger_rows = _merge_dagger_rows(dagger_rows)
    dagger_keys = {str(row["state_key"]) for row in merged_dagger}
    if heldout.intersection(dagger_keys):
        raise ValueError("DAgger row contains a held-out canonical state")
    dagger_collisions = len(set(base).intersection(dagger_keys))
    combined = dict(base)
    for row in merged_dagger:
        key = str(row["state_key"])
        if key not in combined:
            combined[key] = row

    exact_rows = exact_tactical_targets(combined.values())
    exact_by_key = {str(row["state_key"]): row for row in exact_rows}
    ordinary: list[dict[str, Any]] = []
    wins: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for key, source in sorted(combined.items()):
        row = dict(source)
        exact = exact_by_key.get(key)
        if exact is None:
            row["tactical_kind"] = "ordinary"
            ordinary.append(row)
        else:
            row = exact
            if row["tactical_kind"] == "win":
                wins.append(row)
            else:
                blocks.append(row)

    target_rows = math.ceil(len(ordinary) * float(tactical_share) / (1.0 - float(tactical_share)))
    if wins and blocks:
        target_per_kind = max(len(wins), len(blocks), math.ceil(target_rows / 2))
        target_wins = target_blocks = target_per_kind
    else:
        target_wins, target_blocks = len(wins), len(blocks)
        available = wins or blocks
        if available:
            if wins:
                target_wins = max(len(wins), target_rows)
            else:
                target_blocks = max(len(blocks), target_rows)

    rng = random.Random(int(seed))
    def sample_to(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
        result = [dict(row) for row in rows]
        while len(result) < target:
            result.append(dict(rng.choice(rows)))
        return result

    balanced = [dict(row) for row in ordinary]
    balanced.extend(sample_to(wins, target_wins))
    balanced.extend(sample_to(blocks, target_blocks))
    for index, row in enumerate(balanced):
        row["training_row_id"] = hashlib.sha256(
            f"{seed}:{index}:{row['state_key']}:{row['tactical_kind']}".encode()
        ).hexdigest()
    tactical_rows = target_wins + target_blocks
    manifest = {
        "v1_train_unique": len(base),
        "dagger_annotation_rows": len(dagger_rows),
        "dagger_unique_states": len(merged_dagger),
        "dagger_duplicate_rows_aggregated": duplicate_dagger_rows,
        "dagger_train_state_collisions_skipped": dagger_collisions,
        "unique_combined_states": len(combined),
        "unique_ordinary_states": len(ordinary),
        "unique_tactical_win_states": len(wins),
        "unique_tactical_block_states": len(blocks),
        "training_rows": len(balanced),
        "tactical_share_target": float(tactical_share),
        "tactical_win_rows": target_wins,
        "tactical_block_rows": target_blocks,
        "actual_tactical_share": tactical_rows / len(balanced) if balanced else 0.0,
        "validation_state_overlap": len(heldout.intersection(combined)),
        "heldout_canonical_states_rejected": 0,
    }
    return balanced, manifest


def _rename_f1_record(record: dict[str, Any], game_id: str, opening: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["game_id"] = game_id
    result["group_id"] = str(opening["opening_group_id"])
    result["experiment_id"] = "wuziq-imitation-learning-f-v2"
    result["stage"] = "dagger-training"
    result["pair"] = ["F1", "A"]
    for name in ("black_model", "white_model", "winner_model"):
        if result.get(name) == "F":
            result[name] = "F1"
    if "F" in result.get("budget_seconds_by_model", {}):
        result["budget_seconds_by_model"] = dict(result["budget_seconds_by_model"])
        result["budget_seconds_by_model"]["F1"] = result["budget_seconds_by_model"].pop("F")
    for move in result["moves"]:
        if move.get("model") == "F":
            move["model"] = "F1"
    validate_game_record(result)
    return result


def _game_annotations(
    record: dict[str, Any],
    teacher_budget_seconds: float,
    seed: int,
    heldout_state_keys: set[str],
    persisted_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    board = GomokuBoard(9)
    annotations = []
    counts = Counter()
    winner_color = record["winner_color"]
    group_id = str(record["group_id"])
    heldout_skipped = 0
    for move in record["moves"]:
        player = str(move["player"])
        coord = int(move["row"]), int(move["col"])
        if move.get("model") == "F1" and not move.get("opening", False):
            state_key, _ = canonicalize_position(board, player)
            annotation_id = hashlib.sha256(f"{record['game_id']}:{int(move['ply'])}".encode()).hexdigest()
            if annotation_id not in persisted_ids and state_key in heldout_state_keys:
                heldout_skipped += 1
            if annotation_id not in persisted_ids and state_key not in heldout_state_keys:
                own_wins = find_immediate_wins(board, player)
                if own_wins:
                    label_kind, correct_actions = "win", own_wins
                    teacher, elapsed, nodes, depth, timed_out = "exact-rule", 0.0, 0, 0, False
                else:
                    opponent = "W" if player == "B" else "B"
                    threats = find_immediate_wins(board, opponent)
                    if len(threats) == 1:
                        label_kind, correct_actions = "block", threats
                        teacher, elapsed, nodes, depth, timed_out = "exact-rule", 0.0, 0, 0, False
                    else:
                        label_kind, correct_actions = "ordinary", []
                        teacher = "C" if (int(seed) + int(move["ply"])) % 2 == 0 else "E5"
                        teacher_result = annotate_position(
                            board, player, teacher, teacher_budget_seconds,
                            derive_seed(int(seed), "f-v2-dagger-teacher", record["game_id"], int(move["ply"]), teacher),
                        )
                        # annotate_position returns a flattened canonical index.
                        correct_actions = [int(teacher_result["action_index"])]
                        elapsed = float(teacher_result["elapsed_seconds"])
                        nodes = int(teacher_result["nodes"])
                        depth = int(teacher_result["completed_depth"])
                        timed_out = bool(teacher_result["timed_out"])
                if label_kind in ("win", "block"):
                    action_indices = [
                        int(canonicalize_position(board, player, (int(r), int(c)))[1])
                        for r, c in correct_actions
                    ]
                else:
                    action_indices = correct_actions
                target = _policy_for_actions(action_indices)
                annotations.append({
                    "annotation_id": annotation_id,
                    "state_key": state_key,
                    "canonical_board": state_key.split(":", 1)[1],
                    "to_move": player,
                    "policy_target": target,
                    "value_target": 0.0 if winner_color is None else (1.0 if winner_color == player else -1.0),
                    "value_count": 1,
                    "opening_group_ids": [group_id],
                    "opening_group_id": group_id,
                    "source_game_id": record["game_id"],
                    "source_ply": int(move["ply"]),
                    "teacher": teacher,
                    "teacher_budget_seconds": 0.0 if teacher == "exact-rule" else teacher_budget_seconds,
                    "elapsed_seconds": elapsed,
                    "nodes": nodes,
                    "completed_depth": depth,
                    "timed_out": timed_out,
                    "tactical_kind": label_kind,
                })
                counts[teacher] += 1
        board.place_stone(player, coord)
    return annotations, dict(counts), heldout_skipped


def collect_dagger_data(
    f1_checkpoint: str | Path,
    openings: list[dict[str, Any]],
    output_dir: str | Path,
    teacher_budget_seconds: float,
    seed: int,
    heldout_state_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Play paired F1-vs-A training games and annotate F1 turns idempotently."""
    if teacher_budget_seconds < 0 or not openings:
        raise ValueError("a schedule and nonnegative teacher budget are required")
    output_dir = Path(output_dir)
    games_path = output_dir / "dagger_games.jsonl"
    annotations_path = output_dir / "dagger_annotations.jsonl"
    progress_path = output_dir / "dagger_progress.json"
    model = load_f_checkpoint(f1_checkpoint, device="auto")
    existing_games = {}
    for row in _read_jsonl(games_path):
        validate_game_record(row)
        game_id = str(row["game_id"])
        if game_id in existing_games:
            raise ValueError(f"duplicate DAgger game id: {game_id}")
        existing_games[game_id] = row
    existing_annotations = {}
    for row in _read_jsonl(annotations_path):
        annotation_id = str(row["annotation_id"])
        if annotation_id in existing_annotations:
            raise ValueError(f"duplicate DAgger annotation id: {annotation_id}")
        existing_annotations[annotation_id] = row

    expected = {}
    for opening in openings:
        for color in ("B", "W"):
            game_id = f"f-v2-dagger-f1-vs-a-s{int(opening['opening_seed'])}-f{color.lower()}"
            expected[game_id] = (opening, color)
    if set(existing_games) - set(expected):
        raise ValueError("persisted DAgger games do not belong to the frozen schedule")

    teacher_counts = Counter()
    heldout = set(heldout_state_keys or ())
    heldout_fingerprint = hashlib.sha256("\n".join(sorted(heldout)).encode()).hexdigest()
    collection_config = {
        "teacher_budget_seconds": float(teacher_budget_seconds),
        "seed": int(seed),
        "schedule": [
            {"game_id": game_id, "group_id": opening["opening_group_id"], "opening": opening["opening"], "color": color}
            for game_id, (opening, color) in sorted(expected.items())
        ],
        "heldout_fingerprint": heldout_fingerprint,
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(collection_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    existing_progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    if existing_progress and existing_progress.get("config_fingerprint") != config_fingerprint:
        raise ValueError("persisted DAgger progress does not match frozen collection configuration")
    if any(row["state_key"] in heldout for row in existing_annotations.values()):
        raise ValueError("persisted DAgger annotations overlap a held-out canonical state")
    # Recompute the count from the frozen schedule so re-running collection does
    # not inflate it after a restart.
    heldout_skipped_total = 0
    for game_id, (opening, color) in expected.items():
        record = existing_games.get(game_id)
        if record is None:
            record = play_match(
                model, "A", opening["opening"], color,
                int(opening["opening_seed"]), opponent_budget_seconds=0.05,
            )
            record = _rename_f1_record(record, game_id, opening)
            existing_games[game_id] = record
            _atomic_jsonl(games_path, [existing_games[key] for key in sorted(existing_games)])
        elif record.get("opening") != opening["opening"] or record.get("group_id") != opening["opening_group_id"]:
            raise ValueError(f"persisted DAgger game differs from frozen opening: {game_id}")
        annotations, _, heldout_skipped = _game_annotations(
            record, teacher_budget_seconds, seed, heldout, set(existing_annotations),
        )
        for row in annotations:
            annotation_id = str(row["annotation_id"])
            if annotation_id not in existing_annotations:
                existing_annotations[annotation_id] = row
        heldout_skipped_total += heldout_skipped
        teacher_counts = Counter(row["teacher"] for row in existing_annotations.values())
        _atomic_jsonl(annotations_path, [existing_annotations[key] for key in sorted(existing_annotations)])
        _atomic_json(progress_path, {
            "schema_version": 1,
            "seed": int(seed),
            "config_fingerprint": config_fingerprint,
            "heldout_state_fingerprint": heldout_fingerprint,
            "schedule_games": len(expected),
            "completed_games": len(existing_games),
            "completed_game_ids": sorted(existing_games),
            "annotations": len(existing_annotations),
            "remaining_games": len(expected) - len(existing_games),
            "heldout_state_keys_skipped": heldout_skipped_total,
            "last_completed_game_id": game_id,
        })
    return {
        "schedule_games": len(expected),
        "completed_games": len(existing_games),
        "annotation_rows": len(existing_annotations),
        "teacher_counts": dict(sorted(teacher_counts.items())),
        "heldout_state_keys": len(heldout),
        "schedule_game_ids": sorted(expected),
    }
