"""Deterministic mixed-engine games and C/E5 annotations for F V1."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from ...ai import SearchConfig, choose_move
from ...gomoku import GomokuBoard
from ..runner import derive_seed
from ..v6.engine import V6Agent
from .dataset import canonicalize_position, validate_game_record


PAIRINGS = (("A", "B"), ("A", "C"), ("A", "E5"), ("B", "C"), ("B", "E5"), ("C", "E5"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _safe_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        value = asdict(result)
    else:
        value = {}
        for key in ("move", "root_value", "nodes", "completed_depth", "cutoffs", "simulations",
                    "elapsed_seconds", "timed_out", "stats", "tree_depth", "proof_status"):
            if hasattr(result, key):
                value[key] = getattr(result, key)
    return json.loads(json.dumps(value, default=lambda obj: list(obj) if isinstance(obj, tuple) else str(obj)))


def _opening_group_id(opening: list[dict[str, Any]]) -> str:
    board = GomokuBoard(9)
    for move in opening:
        board.place_stone(str(move["player"]), (int(move["row"]), int(move["col"])))
    side = "B" if len(opening) % 2 == 0 else "W"
    key, _ = canonicalize_position(board, side)
    return "opening-" + hashlib.sha256(key.encode()).hexdigest()[:20]


def _make_opening(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    center = (4, 4)
    candidates = [(row, col) for row in range(2, 7) for col in range(2, 7)]
    opening = [{"player": "B", "row": center[0], "col": center[1]}]
    board = GomokuBoard(9)
    board.place_stone("B", center)
    for player in ("W", "B", "W"):
        available = [coord for coord in candidates if board.get(coord) is None]
        coord = rng.choice(available)
        board.place_stone(player, coord)
        if board.check_victory(coord, player):
            raise AssertionError("four-ply opening unexpectedly ended the game")
        opening.append({"player": player, "row": coord[0], "col": coord[1]})
    return opening


def build_evaluation_openings(
    seed: int, count: int, excluded_opening_group_ids: list[str] | tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Freeze new legal openings whose canonical four-ply groups are unseen."""
    if count < 1:
        raise ValueError("count must be positive")
    used = set(excluded_opening_group_ids)
    openings = []
    candidate = 0
    while len(openings) < count:
        opening_seed = derive_seed(seed, "f-v1-evaluation-opening", len(openings), candidate)
        candidate += 1
        opening = _make_opening(opening_seed)
        group_id = _opening_group_id(opening)
        if group_id in used:
            continue
        used.add(group_id)
        openings.append({
            "opening_seed": opening_seed,
            "opening_group_id": group_id,
            "opening": opening,
        })
    return openings


def build_generation_config(
    seed: int = 731401,
    openings_per_pair: int = 5,
    budget_seconds: float = 0.05,
    excluded_opening_group_ids: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Freeze 5 opening components per matchup and paired color swaps."""
    if openings_per_pair < 1 or budget_seconds < 0:
        raise ValueError("opening count must be positive and budget nonnegative")
    excluded = set(excluded_opening_group_ids)
    used: set[str] = set(excluded)
    games: list[dict[str, Any]] = []
    opening_seeds_by_pair: dict[str, list[int]] = {}
    candidate_index = 0
    for pair in PAIRINGS:
        pair_key = "-".join(pair)
        opening_seeds: list[int] = []
        for opening_index in range(openings_per_pair):
            while True:
                opening_seed = derive_seed(seed, "f-v1-opening", pair, opening_index, candidate_index)
                candidate_index += 1
                opening = _make_opening(opening_seed)
                group_id = _opening_group_id(opening)
                if group_id not in used:
                    used.add(group_id)
                    break
            opening_seeds.append(opening_seed)
            for color_swap in (False, True):
                black_model, white_model = pair
                if color_swap:
                    black_model, white_model = white_model, black_model
                game_id = f"f-v1-9x9-{pair[0].lower()}-{pair[1].lower()}-o{opening_seed}-s{int(color_swap)}"
                games.append({
                    "game_id": game_id,
                    "pair": list(pair),
                    "board_size": 9,
                    "opening_seed": int(opening_seed),
                    "opening_group_id": group_id,
                    "opening": opening,
                    "color_swap": color_swap,
                    "black_model": black_model,
                    "white_model": white_model,
                    "game_seed": derive_seed(seed, "f-v1-game", game_id),
                })
        opening_seeds_by_pair[pair_key] = opening_seeds
    return {
        "schema_version": 1,
        "experiment_id": "wuziq-imitation-learning-f-v1",
        "seed": int(seed),
        "board_size": 9,
        "budget_seconds": float(budget_seconds),
        "teacher_models": ["C", "E5"],
        "teacher_budget_seconds": 0.05,
        "pairings": [list(pair) for pair in PAIRINGS],
        "openings_per_pair": openings_per_pair,
        "opening_seeds_by_pair": opening_seeds_by_pair,
        "games": games,
        "maximum_games": 2 * openings_per_pair * len(PAIRINGS),
        "excluded_opening_group_ids": sorted(excluded),
    }


def _run_engine(board: GomokuBoard, player: str, model: str, budget: float, seed: int) -> Any:
    if model == "E5":
        return V6Agent("E5").choose_move(
            board, player, time_limit_seconds=budget, seed=seed, max_depth=16
        )
    return choose_move(board, player, model, SearchConfig(time_limit_seconds=budget), seed)


def generate_game(spec: dict[str, Any], seed: int, budget_seconds: float) -> dict[str, Any]:
    """Play and return one fully replayable game using the frozen engines."""
    if int(spec.get("board_size", 0)) != 9:
        raise ValueError("F V1 generated games are frozen to 9x9")
    board = GomokuBoard(9)
    opening = [dict(move) for move in spec["opening"]]
    moves: list[dict[str, Any]] = []
    current = "B"
    for ply, move in enumerate(opening):
        player = str(move["player"])
        coord = int(move["row"]), int(move["col"])
        if player != current:
            raise ValueError("opening is not alternating B-W")
        board.place_stone(player, coord)
        if board.check_victory(coord, player):
            raise ValueError("opening is terminal")
        moves.append({
            "ply": ply, "player": player, "model": None, "row": coord[0], "col": coord[1],
            "elapsed_seconds": 0.0, "nodes": 0, "completed_depth": 0, "cutoffs": 0,
            "simulations": 0, "timed_out": False, "opening": True,
        })
        current = "W" if player == "B" else "B"
    if current != "B":
        raise ValueError("opening must contain an even number of plies")

    black_model, white_model = str(spec["black_model"]), str(spec["white_model"])
    winner_color = winner_model = None
    termination = "full_board"
    while not board.is_full():
        model = black_model if current == "B" else white_model
        move_seed = derive_seed(int(seed), "move", len(moves), model, current)
        result = _run_engine(board, current, model, budget_seconds, move_seed)
        details = _safe_result(result)
        coordinate = getattr(result, "move", None)
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
            raise ValueError(f"{model} returned a malformed move at ply {len(moves)}")
        coord = int(coordinate[0]), int(coordinate[1])
        if not board.in_bounds(*coord) or board.get(coord) is not None:
            raise ValueError(f"{model} returned an illegal move at ply {len(moves)}")
        elapsed = float(getattr(result, "elapsed_seconds", 0.0))
        timed_out = bool(getattr(result, "timed_out", False) or elapsed > budget_seconds)
        board.place_stone(current, coord)
        moves.append({
            "ply": len(moves), "player": current, "model": model,
            "row": coord[0], "col": coord[1], "elapsed_seconds": elapsed,
            "root_value": int(getattr(result, "root_value", 0)),
            "nodes": int(getattr(result, "nodes", 0)),
            "completed_depth": int(getattr(result, "completed_depth", 0)),
            "tree_depth": int(getattr(result, "tree_depth", getattr(result, "completed_depth", 0))),
            "cutoffs": int(getattr(result, "cutoffs", 0)),
            "simulations": int(getattr(result, "simulations", 0)),
            "timed_out": timed_out, "opening": False, "stats": details.get("stats", {}),
        })
        if board.check_victory(coord, current):
            winner_color, winner_model, termination = current, model, "five_in_row"
            break
        current = "W" if current == "B" else "B"

    outcome = "draw" if winner_color is None else ("b_win" if winner_color == "B" else "w_win")
    record = {
        "game_id": str(spec["game_id"]),
        "group_id": str(spec.get("opening_group_id", _opening_group_id(opening))),
        "experiment_id": "wuziq-imitation-learning-f-v1",
        "stage": "mixed_game_generation",
        "board_size": 9,
        "pair": list(spec.get("pair", sorted((black_model, white_model)))),
        "black_model": black_model, "white_model": white_model,
        "opening_index": int(spec.get("opening_seed", 0)),
        "opening_seed": int(spec.get("opening_seed", 0)),
        "opening": opening, "color_swap": bool(spec.get("color_swap", False)),
        "game_seed": int(seed), "budget_seconds": float(budget_seconds),
        "teacher_budget_seconds": 0.05, "moves": moves, "move_count": len(moves),
        "winner_color": winner_color, "winner_model": winner_model,
        "outcome": outcome, "termination": termination,
        "thinking_seconds_total": sum(float(move.get("elapsed_seconds", 0.0)) for move in moves),
        "search_nodes_total": sum(int(move.get("nodes", 0)) for move in moves),
        "simulations_total": sum(int(move.get("simulations", 0)) for move in moves),
        "timeout_count": sum(bool(move.get("timed_out")) for move in moves),
    }
    validate_game_record(record)
    return record


def annotate_position(
    board: GomokuBoard, to_move: str, teacher: str, budget_seconds: float, seed: int
) -> dict[str, Any]:
    """Use frozen C or E5 to label one nonterminal position."""
    if teacher not in ("C", "E5"):
        raise ValueError("teacher must be C or E5")
    if board.is_full() or budget_seconds < 0:
        raise ValueError("position must have a legal move and nonnegative budget")
    result = _run_engine(board, to_move, teacher, budget_seconds, seed)
    move = getattr(result, "move", None)
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        raise ValueError("teacher returned a malformed action")
    row, col = int(move[0]), int(move[1])
    if not board.in_bounds(row, col) or board.get((row, col)) is not None:
        raise ValueError("teacher returned an illegal action")
    state_key, action_index = canonicalize_position(board, to_move, (row, col))
    return {
        "teacher": teacher, "state_key": state_key,
        "canonical_board": state_key.split(":", 1)[1],
        "to_move": to_move, "action_index": action_index,
        "nodes": int(getattr(result, "nodes", 0)),
        "completed_depth": int(getattr(result, "completed_depth", 0)),
        "cutoffs": int(getattr(result, "cutoffs", 0)),
        "simulations": int(getattr(result, "simulations", 0)),
        "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
        "timed_out": bool(getattr(result, "timed_out", False)),
    }


def _load_frozen_split(value: dict[str, Any] | Path | str) -> tuple[dict[str, str], set[str]]:
    frozen = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (Path, str)) else value
    assignment = {str(key): str(split) for key, split in frozen.get("opening_group_assignment", {}).items()}
    splits = frozen.get("splits", {})
    heldout = set(str(key) for name in ("validation", "blind_test") for key in splits.get(name, []))
    return assignment, heldout


def _collect_annotations(
    games: list[dict[str, Any]], assignment: dict[str, str], heldout: set[str], teacher_budget: float,
    existing: dict[str, dict[str, Any]] | None = None, persist: Any = None,
) -> tuple[list[dict[str, Any]], int]:
    annotations = dict(existing or {})
    skipped = 0
    new_since_save = 0
    for game in games:
        group_id = str(game["group_id"])
        if assignment.setdefault(group_id, "train") != "train":
            skipped += sum(not move.get("opening", False) for move in game["moves"])
            continue
        board = GomokuBoard(9)
        terminal = game["winner_color"]
        for move in game["moves"]:
            player = str(move["player"])
            coord = int(move["row"]), int(move["col"])
            if not move.get("opening", False):
                annotation_id = hashlib.sha256(f"{game['game_id']}:{move['ply']}".encode()).hexdigest()
                if annotation_id in annotations:
                    board.place_stone(player, coord)
                    continue
                if move.get("model") in ("C", "E5"):
                    teacher = str(move["model"])
                    state_key, action_index = canonicalize_position(board, player, coord)
                    label = {
                        "teacher": teacher, "state_key": state_key,
                        "canonical_board": state_key.split(":", 1)[1], "to_move": player,
                        "action_index": action_index, "nodes": move.get("nodes", 0),
                        "completed_depth": move.get("completed_depth", 0),
                        "cutoffs": move.get("cutoffs", 0), "simulations": move.get("simulations", 0),
                        "elapsed_seconds": move.get("elapsed_seconds", 0.0),
                        "timed_out": move.get("timed_out", False),
                    }
                else:
                    teacher = "C" if (int(game["game_seed"]) + int(move["ply"])) % 2 == 0 else "E5"
                    label = annotate_position(
                        board, player, teacher, teacher_budget,
                        derive_seed(int(game["game_seed"]), "teacher", int(move["ply"]), teacher),
                    )
                if label["state_key"] in heldout:
                    skipped += 1
                else:
                    annotations[annotation_id] = {
                        "annotation_id": annotation_id,
                        "state_key": label["state_key"],
                        "canonical_board": label["canonical_board"],
                        "to_move": label["to_move"],
                        "action_index": int(label["action_index"]),
                        "value_target": 0.0 if terminal is None else (1.0 if terminal == player else -1.0),
                        "value_count": 1,
                        "opening_group_id": group_id,
                        "source_game_id": game["game_id"],
                        "teacher": label["teacher"],
                        "teacher_budget_seconds": teacher_budget,
                        "elapsed_seconds": float(label.get("elapsed_seconds", 0.0)),
                        "nodes": int(label.get("nodes", 0)),
                        "completed_depth": int(label.get("completed_depth", 0)),
                        "cutoffs": int(label.get("cutoffs", 0)),
                        "simulations": int(label.get("simulations", 0)),
                        "timed_out": bool(label.get("timed_out", False)),
                    }
                    new_since_save += 1
                    if persist is not None and new_since_save >= 10:
                        persist([annotations[key] for key in sorted(annotations)], skipped)
                        new_since_save = 0
            board.place_stone(player, coord)
    if persist is not None:
        persist([annotations[key] for key in sorted(annotations)], skipped)
    return [annotations[key] for key in sorted(annotations)], skipped


def run_generation(
    config: dict[str, Any], output_dir: Path, frozen_split_manifest: dict[str, Any] | Path | str
) -> dict[str, Any]:
    """Resume a frozen schedule by game id and atomically persist every result."""
    output_dir = Path(output_dir)
    games_path = output_dir / "generated_games.jsonl"
    annotations_path = output_dir / "generated_annotations.jsonl"
    progress_path = output_dir / "generation_progress.json"
    game_by_id: dict[str, dict[str, Any]] = {}
    for game in _read_jsonl(games_path):
        validate_game_record(game)
        game_id = str(game["game_id"])
        if game_id in game_by_id:
            raise ValueError(f"duplicate persisted game id: {game_id}")
        game_by_id[game_id] = game
    expected: dict[str, dict[str, Any]] = {}
    for spec in config.get("games", []):
        game_id = str(spec["game_id"])
        if game_id in expected:
            raise ValueError(f"duplicate schedule game id: {game_id}")
        expected[game_id] = spec
    for game_id, game in game_by_id.items():
        spec = expected.get(game_id)
        if spec is None or game.get("opening") != spec.get("opening") or game.get("black_model") != spec.get("black_model") or game.get("white_model") != spec.get("white_model"):
            raise ValueError(f"persisted game does not match frozen schedule: {game_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for game_id, spec in expected.items():
        if game_id not in game_by_id:
            game = generate_game(spec, int(spec.get("game_seed", config.get("seed", 0))), float(config.get("budget_seconds", 0.05)))
            game_by_id[game_id] = game
            _atomic_jsonl(games_path, [game_by_id[key] for key in sorted(game_by_id)])
            _atomic_json(progress_path, {
                "schema_version": 1, "seed": config.get("seed"), "schedule_games": len(expected),
                "completed_games": len(game_by_id), "last_completed_game_id": game_id,
                "remaining_games": len(expected) - len(game_by_id),
            })

    assignment, heldout = _load_frozen_split(frozen_split_manifest)
    persisted_annotations: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(annotations_path):
        annotation_id = str(row["annotation_id"])
        if annotation_id in persisted_annotations:
            raise ValueError(f"duplicate persisted annotation id: {annotation_id}")
        persisted_annotations[annotation_id] = row

    def persist_annotations(rows: list[dict[str, Any]], skipped_count: int) -> None:
        _atomic_jsonl(annotations_path, rows)
        _atomic_json(progress_path, {
            "schema_version": 1, "seed": config.get("seed"), "schedule_games": len(expected),
            "completed_games": len(game_by_id), "remaining_games": len(expected) - len(game_by_id),
            "annotations": len(rows), "skipped_heldout_annotations": skipped_count,
            "last_annotation_id": rows[-1]["annotation_id"] if rows else None,
        })

    annotations, skipped = _collect_annotations(
        [game_by_id[key] for key in sorted(game_by_id)], assignment, heldout,
        float(config.get("teacher_budget_seconds", 0.05)), persisted_annotations,
        persist_annotations,
    )
    final_progress = {
        "schema_version": 1, "seed": config.get("seed"), "schedule_games": len(expected),
        "completed_games": len(game_by_id), "remaining_games": 0,
        "annotations": len(annotations), "skipped_heldout_annotations": skipped,
        "completed_game_ids": sorted(game_by_id),
    }
    _atomic_json(progress_path, final_progress)
    return final_progress
