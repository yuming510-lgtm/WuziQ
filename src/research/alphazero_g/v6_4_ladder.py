"""Frozen G5000 ladder evaluation against A and three Alpha-Beta budgets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch

from ...ai import SearchConfig
from ...ai import choose_move
from .v5_evaluation import _IndependentRuleBoard, _project_board, other, summarize_paired_games
from .v5_reference import ReferencePolicyValueNet, instrumented_player, make_board
from .v6_3_training import ARTIFACT as V63_ARTIFACT, ARM as V63_ARM, sha_file
from .v5_evaluation import generate_opening_pool, opening_key


EXPERIMENT_ID = "wuziq-g5000-ladder-v6-4"
BASE_COMMIT = "66b88d2270c799870aa81651fe3c07be18479cf0"
DEFAULT_OPENING_SEED = 640614
OPENING_GROUPS = 3
OPENING_PLIES = 6
GAME_COUNT = 24
MODEL_NAME = "G5000"
MODEL_SHA256 = "cc383f0667cfe02b8e8ccbaecf51eccc2c4f088d56c836a4e91e016cfae3ee97"
TRAINING_STATE_SHA256 = "0fb9886068a014d000227ae7b275ff9043322a678d7f259e9730d44ac95398d2"
OPPONENTS = (
    {"name": "A", "engine": "rules-A", "time_limit_seconds": None},
    {"name": "C@200ms", "engine": "alpha-beta-C", "time_limit_seconds": 0.2},
    {"name": "C@1s", "engine": "alpha-beta-C", "time_limit_seconds": 1.0},
    {"name": "C@5s", "engine": "alpha-beta-C", "time_limit_seconds": 5.0},
)
ALPHA_BETA_SETTINGS = {"max_depth": 69, "candidate_radius": 2}
ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_4")
CHECKPOINT_RELATIVE = V63_ARTIFACT / "checkpoints" / V63_ARM
SNAPSHOT_SOURCE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1u3556ink-VFf9a7AbS701flr0V0spSzQ"
SNAPSHOT_SOURCE_SEQUENCE = 12
SNAPSHOT_SOURCE_SHA256 = "c7a615c112324d955e32122f66f6930254510d111ff23606f7805e58f61af592"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _six_ply_sequence(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or len(value) < OPENING_PLIES:
        return None
    result = []
    for move in value[:OPENING_PLIES]:
        if not isinstance(move, dict) or not {"row", "col"} <= set(move):
            return None
        player = move.get("player", move.get("color", move.get("stone")))
        if player not in {"B", "W"}:
            return None
        result.append({"player": str(player), "row": int(move["row"]), "col": int(move["col"])})
    if [move["player"] for move in result] != ["B", "W", "B", "W", "B", "W"]:
        return None
    return result


def _collect_sequences(value: Any, found: dict[str, list[dict[str, Any]]]) -> None:
    if isinstance(value, dict):
        for key in ("moves", "opening", "opening_moves"):
            sequence = _six_ply_sequence(value.get(key))
            if sequence is not None:
                found[opening_key(sequence, 9)] = sequence
        for child in value.values():
            _collect_sequences(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_sequences(child, found)


def collect_historical_openings(root: str | Path) -> list[list[dict[str, Any]]]:
    """Collect D4-normalized six-ply starts from historical JSON/JSONL artifacts."""
    root = Path(root).resolve()
    research_root = root / "research" if (root / "research").is_dir() else root
    found: dict[str, list[dict[str, Any]]] = {}
    paths = sorted({*research_root.rglob("*.json"), *research_root.rglob("*.jsonl")})
    for path in paths:
        if "g_v6_4" in path.parts:
            continue
        try:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            _collect_sequences(json.loads(line), found)
            else:
                _collect_sequences(json.loads(path.read_text(encoding="utf-8")), found)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return [found[key] for key in sorted(found)]


def build_schedule(
    *, seed: int = DEFAULT_OPENING_SEED,
    historical_openings: Iterable[list[dict[str, Any]]] = (),
) -> dict[str, Any]:
    historical = {opening_key(moves, 9) for moves in historical_openings}
    used_seed = int(seed)
    for _ in range(1001):
        openings = generate_opening_pool(
            size=9, n_in_row=5, groups=OPENING_GROUPS,
            plies=OPENING_PLIES, seed=used_seed,
        )
        keys = [opening_key(row["moves"], 9) for row in openings]
        if len(set(keys)) == OPENING_GROUPS and not (set(keys) & historical):
            break
        used_seed += 1
    else:
        raise RuntimeError("could not produce three fresh V6.4 opening groups")

    for index, opening in enumerate(openings):
        opening["opening_group_id"] = f"v6-4-{used_seed}-{index:03d}"
        opening["seed"] = used_seed + index
    opponents = [dict(row) for row in OPPONENTS]
    schedule = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "base_commit": BASE_COMMIT,
        "board_size": 9,
        "n_in_row": 5,
        "candidate": {
            "name": MODEL_NAME,
            "model_sha256": MODEL_SHA256,
            "training_state_sha256": TRAINING_STATE_SHA256,
            "search": {"type": "frozen AlphaZero PUCT", "simulations": 400, "configured_playouts": 401, "c_puct": 5.0},
        },
        "opponents": opponents,
        "alpha_beta_settings": dict(ALPHA_BETA_SETTINGS),
        "opening_seed": used_seed,
        "opening_plies": OPENING_PLIES,
        "openings": openings,
        "games_per_opponent": OPENING_GROUPS * 2,
        "planned_games": GAME_COUNT,
        "score_rule": {"win": 1.0, "draw": 0.5, "loss": 0.0},
        "validation": {
            "all_openings_legal_and_nonterminal": True,
            "unique_d4_openings": len(set(keys)),
            "historical_d4_overlap": len(set(keys) & historical),
            "same_openings_for_all_opponents": True,
            "color_pairs_per_opponent": OPENING_GROUPS,
        },
    }
    schedule["schedule_sha256"] = hashlib.sha256(canonical_json(schedule).encode("utf-8")).hexdigest()
    return schedule


def schedule_games(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    if schedule.get("experiment_id") != EXPERIMENT_ID or len(schedule.get("openings", [])) != OPENING_GROUPS:
        raise ValueError("invalid V6.4 schedule")
    games = []
    for opponent in schedule["opponents"]:
        for opening in schedule["openings"]:
            for color in ("B", "W"):
                key = f"{opponent['name']}|{opening['opening_group_id']}|{color}"
                game_id = "v64-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
                seed = int.from_bytes(hashlib.sha256((EXPERIMENT_ID + "|" + key).encode("utf-8")).digest()[:4], "big")
                games.append({
                    "game_id": game_id,
                    "experiment_id": EXPERIMENT_ID,
                    "opponent": opponent["name"],
                    "candidate": MODEL_NAME,
                    "candidate_color": color,
                    "opening_group_id": opening["opening_group_id"],
                    "opening_seed": int(opening["seed"]),
                    "opening": opening["moves"],
                    "seed": seed,
                })
    if len(games) != GAME_COUNT or len({row["game_id"] for row in games}) != GAME_COUNT:
        raise ValueError("V6.4 schedule must contain 24 unique games")
    return games


def validate_schedule(schedule: dict[str, Any], *,
                      historical_openings: Iterable[list[dict[str, Any]]] = ()) -> dict[str, Any]:
    digest_payload = dict(schedule)
    expected_digest = digest_payload.pop("schedule_sha256", None)
    observed_digest = hashlib.sha256(canonical_json(digest_payload).encode("utf-8")).hexdigest()
    if expected_digest != observed_digest:
        raise ValueError("V6.4 schedule SHA-256 does not match its frozen contents")
    if (schedule.get("experiment_id") != EXPERIMENT_ID
            or schedule.get("base_commit") != BASE_COMMIT
            or schedule.get("board_size") != 9 or schedule.get("n_in_row") != 5):
        raise ValueError("V6.4 schedule identity or Gomoku rules changed")
    candidate = schedule.get("candidate", {})
    if (candidate.get("name") != MODEL_NAME
            or candidate.get("model_sha256") != MODEL_SHA256
            or candidate.get("training_state_sha256") != TRAINING_STATE_SHA256
            or candidate.get("search", {}).get("simulations") != 400
            or candidate.get("search", {}).get("c_puct") != 5.0):
        raise ValueError("V6.4 candidate model or PUCT configuration changed")
    if schedule.get("opponents") != [dict(row) for row in OPPONENTS]:
        raise ValueError("V6.4 opponent identities or budgets changed")
    if schedule.get("alpha_beta_settings") != ALPHA_BETA_SETTINGS:
        raise ValueError("V6.4 C search settings changed")
    openings = schedule.get("openings")
    if not isinstance(openings, list) or len(openings) != OPENING_GROUPS:
        raise ValueError("V6.4 must retain its three registered openings")
    keys = []
    for opening in openings:
        moves = opening.get("moves")
        if not isinstance(moves, list) or len(moves) != OPENING_PLIES:
            raise ValueError("V6.4 opening must contain six plies")
        board = make_board(9, 5, 0)
        oracle = _IndependentRuleBoard(9, 5)
        for index, move in enumerate(moves):
            color = str(move.get("player"))
            expected_color = "B" if index % 2 == 0 else "W"
            row, col = int(move.get("row", -1)), int(move.get("col", -1))
            if color != expected_color or not (0 <= row < 9 and 0 <= col < 9):
                raise ValueError("V6.4 opening contains invalid color order or coordinates")
            flat = row * 9 + col
            if flat not in board.availables or oracle.get((row, col)) is not None:
                raise ValueError("V6.4 opening contains duplicate or illegal coordinates")
            board.do_move(flat)
            oracle.place_stone(color, (row, col))
            if board.game_end()[0] or oracle.check_victory((row, col), color):
                raise ValueError("V6.4 opening must be legal and nonterminal")
        keys.append(opening_key(moves, 9))
    if len(set(keys)) != OPENING_GROUPS:
        raise ValueError("V6.4 openings are not unique under board symmetries")
    history_keys = {opening_key(row, 9) for row in historical_openings}
    if set(keys) & history_keys:
        raise ValueError("V6.4 opening overlaps a historical D4 opening")
    if len(schedule_games(schedule)) != GAME_COUNT:
        raise ValueError("V6.4 schedule does not expand to 24 unique games")
    return {"valid": True, "opening_count": len(openings), "game_count": GAME_COUNT,
            "schedule_sha256": expected_digest, "historical_d4_overlap": len(set(keys) & history_keys)}


def estimate_ladder_cost(*, candidate_cpu_per_move: float,
                         opponent_cpu_per_move: dict[str, float],
                         side_turns: tuple[int, ...] = (10, 20, 30)) -> dict[str, Any]:
    """Extrapolate measured per-move CPU costs across registered game counts."""
    levels = ("optimistic", "typical", "conservative")
    if len(side_turns) != len(levels) or any(int(value) <= 0 for value in side_turns):
        raise ValueError("side_turns must contain positive low/typical/high values")
    required = {row["name"] for row in OPPONENTS}
    if set(opponent_cpu_per_move) != required:
        raise ValueError("preflight must measure all four registered opponents")
    candidate, opponent = {}, {}
    for level, count in zip(levels, side_turns, strict=True):
        candidate[level] = GAME_COUNT * int(count) * float(candidate_cpu_per_move)
        opponent[level] = OPENING_GROUPS * 2 * int(count) * sum(
            float(opponent_cpu_per_move[name]) for name in required
        )
    total = {level: candidate[level] + opponent[level] for level in levels}
    return {
        "games": GAME_COUNT,
        "side_turns_per_game": list(map(int, side_turns)),
        "candidate_cpu_per_move_seconds": float(candidate_cpu_per_move),
        "opponent_cpu_per_move_seconds": dict(opponent_cpu_per_move),
        "candidate_cpu_seconds": candidate,
        "opponent_cpu_seconds": opponent,
        "subtotal_cpu_seconds": total,
        "estimated_cpu_seconds_with_20pct_overhead": {key: value * 1.2 for key, value in total.items()},
        "assumption": "12/20/30 decisions per side after the six-ply opening; endpoints are planning estimates, not measured game lengths",
    }


def complete_opening_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Return only two-game color swaps; report how many pairs remain incomplete."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["opening_group_id"]), []).append(row)
    paired: list[dict[str, Any]] = []
    incomplete = 0
    for group, members in sorted(groups.items()):
        colors = [str(row["candidate_color"]) for row in members]
        if len(colors) != len(set(colors)):
            raise ValueError(f"opening {group} contains a duplicate candidate color")
        if len(members) == 2 and set(colors) == {"B", "W"}:
            paired.extend(members)
        else:
            incomplete += 1
    return paired, incomplete


def summarize_condition(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    paired, incomplete = complete_opening_pairs(rows)
    if paired:
        summary = summarize_paired_games(paired, seed=int(seed))
        paired_games = int(summary["games"])
    else:
        summary = {
            "games": 0, "paired_openings": 0, "wins": 0, "draws": 0, "losses": 0,
            "points": 0.0, "score_rate": None, "paired_opening_score_rate": None,
            "paired_opening_bootstrap_ci95": None, "black_score_rate": None,
            "white_score_rate": None, "mean_moves": None,
            "mean_candidate_cpu_seconds": None, "mean_opponent_cpu_seconds": None,
        }
        paired_games = 0
    summary.update({"raw_completed_games": len(rows), "paired_games": paired_games,
                    "planned_games": OPENING_GROUPS * 2, "incomplete_pairs": incomplete,
                    "complete": len(rows) == OPENING_GROUPS * 2 and incomplete == 0})
    return summary


def search_config_for_opponent(name: str) -> SearchConfig:
    if name == "A":
        return SearchConfig(max_depth=69, time_limit_seconds=1e9, candidate_radius=2)
    opponent = next((row for row in OPPONENTS if row["name"] == name), None)
    if opponent is None or opponent["engine"] != "alpha-beta-C":
        raise ValueError(f"unknown V6.4 opponent: {name}")
    return SearchConfig(
        max_depth=ALPHA_BETA_SETTINGS["max_depth"],
        time_limit_seconds=float(opponent["time_limit_seconds"]),
        candidate_radius=ALPHA_BETA_SETTINGS["candidate_radius"],
    )


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_match_record(record: dict[str, Any]) -> dict[str, Any]:
    """Replay a completed game using both the frozen board and independent rules oracle."""
    if int(record.get("board_size", -1)) != 9 or int(record.get("n_in_row", -1)) != 5:
        raise ValueError("V6.4 game uses an unexpected board or win rule")
    candidate = str(record.get("candidate", ""))
    opponent = str(record.get("opponent", ""))
    candidate_color = str(record.get("candidate_color", ""))
    if candidate != MODEL_NAME or opponent not in {row["name"] for row in OPPONENTS}:
        raise ValueError("V6.4 game identity is invalid")
    if candidate_color not in {"B", "W"}:
        raise ValueError("V6.4 game candidate color is invalid")
    moves = record.get("moves")
    if not isinstance(moves, list) or len(moves) != int(record.get("move_count", -1)):
        raise ValueError("V6.4 game move list is incomplete")
    opening = record.get("opening")
    if not isinstance(opening, list) or len(opening) != OPENING_PLIES:
        raise ValueError("V6.4 game has an invalid registered opening")
    board = make_board(9, 5, 0)
    independent = _IndependentRuleBoard(9, 5)
    winner_color: str | None = None
    termination = ""
    for index, move in enumerate(moves):
        if winner_color is not None or termination == "full_board":
            raise ValueError("V6.4 game contains moves after terminal")
        color = str(move.get("player", ""))
        expected_color = "B" if index % 2 == 0 else "W"
        if color != expected_color:
            raise ValueError("V6.4 game move colors do not alternate")
        row, col = int(move.get("row", -1)), int(move.get("col", -1))
        if not (0 <= row < 9 and 0 <= col < 9):
            raise ValueError("V6.4 game contains an out of bounds move")
        coord = (row, col)
        flat = row * 9 + col
        if flat not in board.availables or independent.get(coord) is not None:
            raise ValueError("V6.4 game contains an illegal or repeated move")
        is_opening = index < OPENING_PLIES
        if bool(move.get("opening")) != is_opening:
            raise ValueError("V6.4 game opening flag does not match the registered prefix")
        if is_opening:
            expected = opening[index]
            if (color, row, col) != (str(expected["player"]), int(expected["row"]), int(expected["col"])):
                raise ValueError("V6.4 game opening differs from the frozen schedule")
        else:
            expected_actor = candidate if color == candidate_color else opponent
            if move.get("actor") != expected_actor:
                raise ValueError("V6.4 game actor does not match the assigned side")
            for metric in ("elapsed_seconds", "cpu_seconds", "search_depth", "search_nodes", "simulations"):
                if metric not in move:
                    raise ValueError(f"V6.4 game move is missing {metric}")
            if float(move["elapsed_seconds"]) < 0 or float(move["cpu_seconds"]) < 0:
                raise ValueError("V6.4 game contains negative timing")
            if expected_actor == MODEL_NAME and int(move["simulations"]) != 400:
                raise ValueError("G5000 did not complete exactly 400 MCTS simulations")
        board.do_move(flat)
        independent.place_stone(color, coord)
        oracle_wins = independent.check_victory(coord, color)
        engine_end, engine_winner = board.game_end()
        if bool(engine_end) != bool(oracle_wins or independent.is_full()):
            raise ValueError("frozen board and independent rule oracle disagree")
        if oracle_wins:
            winner_color, termination = color, "five_in_row"
            if engine_winner != board.current_player ^ 3:
                raise ValueError("board-engine winner disagrees with independent replay")
        elif independent.is_full():
            termination = "full_board"
    if not termination:
        raise ValueError("V6.4 game has no terminal result")
    if record.get("termination") != termination or record.get("winner_color") != winner_color:
        raise ValueError("V6.4 stored result differs from replayed winner/termination")
    points = 0.5 if winner_color is None else 1.0 if winner_color == candidate_color else 0.0
    if abs(float(record.get("points", -1)) - points) > 1e-9:
        raise ValueError("V6.4 stored score differs from replayed result")
    return {"legal": True, "winner_color": winner_color, "termination": termination,
            "replayed_plies": len(moves)}


def _make_agent(name: str, model: ReferencePolicyValueNet | None) -> dict[str, Any]:
    if name == MODEL_NAME:
        if model is None:
            raise ValueError("the frozen G5000 model is required")
        player = instrumented_player(model.policy_value_fn, c_puct=5.0, n_playout=401, is_selfplay=False)
        return {"kind": "G5000", "model": model, "player": player}
    if name in {row["name"] for row in OPPONENTS}:
        return {"kind": name}
    raise ValueError(f"unknown V6.4 agent: {name}")


def _agent_choice(agent: dict[str, Any], board, color: str, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
    kind = agent["kind"]
    started_cpu = time.process_time()
    if kind == "G5000":
        move, _policy = agent["player"].get_action(board, temp=1e-3, return_prob=1)
        row, col = divmod(int(move), int(board.width))
        stats = dict(agent["player"].last_search_stats)
        simulations = int(agent["player"].mcts._n_playout - 1)
        return (row, col), {
            **stats, "simulations": simulations,
            "root_action_visits": int(stats.get("root_visits", 0)),
            "search_depth": int(stats.get("tree_depth", 0)),
            "search_nodes": int(stats.get("tree_nodes", 0)),
            "cpu_seconds": time.process_time() - started_cpu,
            "timed_out": False,
        }
    projected = _project_board(board)
    result = choose_move(projected, color, "A" if kind == "A" else "C",
                         search_config_for_opponent(kind), int(seed))
    return result.move, {
        "simulations": int(result.simulations), "root_action_visits": 0,
        "search_depth": int(result.completed_depth), "search_nodes": int(result.nodes),
        "cutoffs": int(result.cutoffs), "cpu_seconds": time.process_time() - started_cpu,
        "timed_out": bool(result.timed_out), "completed_depth": int(result.completed_depth),
        "elapsed_search_seconds": float(result.elapsed_seconds),
    }


def play_match(*, game: dict[str, Any], opening: dict[str, Any],
               candidate_model: ReferencePolicyValueNet) -> dict[str, Any]:
    candidate_color = str(game["candidate_color"])
    opponent_name = str(game["opponent"])
    if candidate_color not in {"B", "W"}:
        raise ValueError("V6.4 candidate color must be B or W")
    candidate = _make_agent(MODEL_NAME, candidate_model)
    opponent = _make_agent(opponent_name, None)
    board = make_board(9, 5, 0)
    independent = _IndependentRuleBoard(9, 5)
    stored_moves: list[dict[str, Any]] = []
    for item in opening["moves"]:
        color, row, col = str(item["player"]), int(item["row"]), int(item["col"])
        flat = row * 9 + col
        if color != ("B" if len(stored_moves) % 2 == 0 else "W"):
            raise ValueError("V6.4 opening colors do not alternate")
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError("V6.4 opening contains an illegal move")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        if board.game_end()[0] or independent.check_victory((row, col), color):
            raise ValueError("V6.4 opening must be nonterminal")
        stored_moves.append({"player": color, "row": row, "col": col, "opening": True})

    np.random.seed(int(game["seed"]) & 0xFFFFFFFF)
    random.seed(int(game["seed"]))
    rng = np.random.default_rng(int(game["seed"]))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    candidate_cpu = opponent_cpu = 0.0
    winner_color: str | None = None
    termination = ""
    while True:
        color = "B" if board.current_player == 1 else "W"
        is_candidate = color == candidate_color
        actor_name = MODEL_NAME if is_candidate else opponent_name
        agent = candidate if is_candidate else opponent
        move_seed = int(rng.integers(0, 2**31 - 1))
        move_started = time.perf_counter()
        coord, stats = _agent_choice(agent, board, color, move_seed)
        move_wall = time.perf_counter() - move_started
        if is_candidate:
            candidate_cpu += float(stats["cpu_seconds"])
        else:
            opponent_cpu += float(stats["cpu_seconds"])
        row, col = map(int, coord)
        flat = row * 9 + col
        if flat not in board.availables or independent.get((row, col)) is not None:
            raise ValueError(f"{actor_name} selected illegal V6.4 move {(row, col)}")
        board.do_move(flat)
        independent.place_stone(color, (row, col))
        stored_moves.append({"player": color, "row": row, "col": col, "opening": False,
                             "actor": actor_name, "elapsed_seconds": move_wall, **stats})
        independent_wins = independent.check_victory((row, col), color)
        board_end, board_winner = board.game_end()
        if bool(board_end) != bool(independent_wins or independent.is_full()):
            raise AssertionError("V6.4 board and independent rule oracle disagree")
        if independent_wins:
            winner_color, termination = color, "five_in_row"
            if board_winner != board.current_player ^ 3:
                raise AssertionError("V6.4 engine and independent oracle winner disagree")
            break
        if independent.is_full():
            termination = "full_board"
            break
    points = 0.5 if winner_color is None else 1.0 if winner_color == candidate_color else 0.0
    record = {
        "format_version": 1, "experiment_id": EXPERIMENT_ID,
        "game_id": game["game_id"], "candidate": MODEL_NAME, "opponent": opponent_name,
        "board_size": 9, "n_in_row": 5, "candidate_color": candidate_color,
        "winner_color": winner_color, "points": points, "termination": termination,
        "opening_group_id": game["opening_group_id"], "opening_seed": int(game["opening_seed"]),
        "opening": opening["moves"], "seed": int(game["seed"]),
        "candidate_search": "frozen V6.3 AlphaZero PUCT-400 c_puct=5.0",
        "candidate_model_sha256": MODEL_SHA256,
        "training_state_sha256": TRAINING_STATE_SHA256,
        "candidate_configured_playouts_per_turn": 401,
        "simulations_per_candidate_turn": 400,
        "opponent_config": {
            "engine": "rules-A" if opponent_name == "A" else "Minimax + Alpha-Beta",
            "time_limit_seconds": None if opponent_name == "A" else search_config_for_opponent(opponent_name).time_limit_seconds,
            "max_depth": None if opponent_name == "A" else ALPHA_BETA_SETTINGS["max_depth"],
            "candidate_radius": None if opponent_name == "A" else ALPHA_BETA_SETTINGS["candidate_radius"],
        },
        "moves": stored_moves, "move_count": len(stored_moves),
        "candidate_cpu_seconds": candidate_cpu, "opponent_cpu_seconds": opponent_cpu,
        "total_cpu_seconds": time.process_time() - started_cpu,
        "total_wall_seconds": time.perf_counter() - started_wall,
    }
    validate_match_record(record)
    return record


def load_g5000_model(root: str | Path = ROOT, *,
                     checkpoint_dir: str | Path | None = None) -> tuple[ReferencePolicyValueNet, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path(root) / CHECKPOINT_RELATIVE
    model_path = checkpoint_dir / "node_games_5000.pt"
    state_path = checkpoint_dir / "training_state.pt"
    if not model_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"verified V6.3 G5000 checkpoint pair missing under {checkpoint_dir}")
    model_sha, state_sha = sha_file(model_path), sha_file(state_path)
    if model_sha != MODEL_SHA256 or state_sha != TRAINING_STATE_SHA256:
        raise ValueError("G5000 model or full training state SHA-256 does not match V6.3 registration")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (int(state.get("game_count", -1)), int(state.get("environment_states", -1)),
            int(state.get("update_count", -1))) != (5000, 87999, 4999):
        raise ValueError("G5000 full training state counts are invalid")
    node = torch.load(model_path, map_location="cpu", weights_only=False)
    if int(node.get("game_count", -1)) != 5000:
        raise ValueError("G5000 node checkpoint has a mismatched game count")
    model = ReferencePolicyValueNet(9, seed=int(node.get("base_seed", 590502)))
    model.load_policy_param(node["model_state"])
    model.network.eval()
    return model, {
        "model_path": str(model_path), "training_state_path": str(state_path),
        "model_sha256": model_sha, "training_state_sha256": state_sha,
        "game_count": int(state["game_count"]), "state_count": int(state["environment_states"]),
        "update_count": int(state["update_count"]), "replay_buffer_size": len(state["replay_buffer"]),
    }


def run_preflight(root: str | Path = ROOT, *,
                  checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    validate_schedule(schedule)
    model, checkpoint = load_g5000_model(root, checkpoint_dir=checkpoint_dir)
    opening = schedule["openings"][0]
    board = make_board(9, 5, 0)
    for move in opening["moves"]:
        board.do_move(int(move["row"]) * 9 + int(move["col"]))
    candidate_agent = _make_agent(MODEL_NAME, model)
    move_samples: dict[str, Any] = {}
    candidate_move, candidate_stats = _agent_choice(candidate_agent, board, "B", 640614)
    move_samples[MODEL_NAME] = {"move": list(candidate_move), **candidate_stats}
    opponent_costs: dict[str, float] = {}
    for opponent in OPPONENTS:
        agent = _make_agent(str(opponent["name"]), None)
        color = "B"
        move, stats = _agent_choice(agent, board, color, 640614)
        move_samples[str(opponent["name"])] = {"move": list(move), **stats}
        opponent_costs[str(opponent["name"])] = float(stats["cpu_seconds"])
    if int(candidate_stats.get("simulations", -1)) != 400:
        raise ValueError("preflight did not execute exactly 400 candidate simulations")
    estimate = estimate_ladder_cost(candidate_cpu_per_move=float(candidate_stats["cpu_seconds"]),
                                    opponent_cpu_per_move=opponent_costs)
    try:
        with Path("/proc/meminfo").open() as handle:
            meminfo = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                       for line in handle if ":" in line}
    except OSError:
        meminfo = {}
    result = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resources": {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                      "memory": {key: meminfo.get(key) for key in ("MemTotal", "MemAvailable")},
                      "disk_free_bytes": __import__("shutil").disk_usage(root).free,
                      "accelerator": "CPU-only PyTorch"},
        "checkpoint": checkpoint,
        "opening_group_id": opening["opening_group_id"],
        "move_samples": move_samples,
        "cost_estimate": estimate,
    }
    atomic_json(artifact / "results" / "preflight.json", result)
    return result


def validate_existing_match(record: dict[str, Any], game: dict[str, Any], opening: dict[str, Any]) -> None:
    expected = {"game_id": game["game_id"], "candidate": MODEL_NAME,
                "opponent": game["opponent"], "candidate_color": game["candidate_color"],
                "opening_group_id": game["opening_group_id"], "seed": game["seed"],
                "candidate_model_sha256": MODEL_SHA256,
                "training_state_sha256": TRAINING_STATE_SHA256}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"V6.4 unique ID/config collision: {key}")
    if record.get("opening") != opening["moves"]:
        raise ValueError("V6.4 stored game has a different opening")
    validate_match_record(record)


def run_matches(root: str | Path = ROOT, *, limit: int | None = None,
                checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    validate_schedule(schedule)
    games = schedule_games(schedule)
    model, checkpoint = load_g5000_model(root, checkpoint_dir=checkpoint_dir)
    path_by_group = {row["opening_group_id"]: row for row in schedule["openings"]}
    result_dir = artifact / "results" / "matches"
    result_dir.mkdir(parents=True, exist_ok=True)
    progress_path = artifact / "results" / "progress.json"
    progress = {"experiment_id": EXPERIMENT_ID, "planned_games": GAME_COUNT,
                "completed_game_ids": [], "cumulative_cpu_seconds": 0.0,
                "checkpoint": checkpoint, "schedule_sha256": schedule["schedule_sha256"]}
    if progress_path.exists():
        old = json.loads(progress_path.read_text(encoding="utf-8"))
        if old.get("experiment_id") != EXPERIMENT_ID or old.get("schedule_sha256") != schedule["schedule_sha256"]:
            raise ValueError("V6.4 progress file belongs to a different registration")
    newly_run = 0
    for game in games:
        path = result_dir / f"{game['game_id']}.json"
        opening = path_by_group[game["opening_group_id"]]
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_existing_match(record, game, opening)
            progress["completed_game_ids"].append(game["game_id"])
            progress["cumulative_cpu_seconds"] += float(record["total_cpu_seconds"])
            continue
        if limit is not None and newly_run >= int(limit):
            break
        record = play_match(game=game, opening=opening, candidate_model=model)
        atomic_json(path, record)
        progress["completed_game_ids"].append(game["game_id"])
        progress["cumulative_cpu_seconds"] += float(record["total_cpu_seconds"])
        newly_run += 1
        atomic_json(progress_path, progress)
    summaries = {}
    for opponent in (row["name"] for row in OPPONENTS):
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(result_dir.glob("*.json"))]
        rows = [row for row in rows if row.get("opponent") == opponent]
        rows.sort(key=lambda row: row["game_id"])
        if rows:
            summaries[opponent] = summarize_condition(
                rows, seed=int.from_bytes(hashlib.sha256(f"{EXPERIMENT_ID}|{opponent}".encode()).digest()[:4], "big"))
    result = {
        "experiment_id": EXPERIMENT_ID, "planned_games": GAME_COUNT,
        "completed_games": len(progress["completed_game_ids"]),
        "newly_run_games_this_call": newly_run,
        "complete": len(progress["completed_game_ids"]) == GAME_COUNT,
        "cumulative_cpu_seconds": progress["cumulative_cpu_seconds"],
        "candidate_checkpoint": checkpoint, "schedule_sha256": schedule["schedule_sha256"],
        "conditions": summaries,
    }
    atomic_json(artifact / "results" / "summary.json", result)
    return result


def audit_results(root: str | Path = ROOT, *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact = root / ARTIFACT
    schedule = json.loads((artifact / "config/opening_schedule.json").read_text(encoding="utf-8"))
    schedule_audit = validate_schedule(schedule)
    scheduled_games = schedule_games(schedule)
    game_dir = artifact / "results" / "matches"
    files = sorted(game_dir.glob("*.json")) if game_dir.is_dir() else []
    records_by_id: dict[str, dict[str, Any]] = {}
    extras = []
    schedule_by_id = {row["game_id"]: row for row in scheduled_games}
    openings = {row["opening_group_id"]: row for row in schedule["openings"]}
    replayed = 0
    budget_overruns = []
    opponent_decisions: dict[str, list[dict[str, Any]]] = {row["name"]: [] for row in OPPONENTS}
    candidate_decisions: list[dict[str, Any]] = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(record.get("game_id", ""))
        if game_id in records_by_id:
            raise ValueError(f"duplicate unique game ID: {game_id}")
        records_by_id[game_id] = record
        game = schedule_by_id.get(game_id)
        if game is None:
            extras.append(game_id)
            continue
        validate_existing_match(record, game, openings[game["opening_group_id"]])
        replayed += 1
        for move in record["moves"]:
            if move.get("opening"):
                continue
            if move.get("actor") == MODEL_NAME:
                candidate_decisions.append(move)
                if int(move.get("simulations", -1)) != 400 or int(move.get("root_action_visits", -1)) != 400:
                    raise ValueError(f"G5000 move did not receive 400 root simulations in {game_id}")
            else:
                opponent_decisions[str(record["opponent"])].append(move)
                budget = next(row["time_limit_seconds"] for row in OPPONENTS
                              if row["name"] == record["opponent"])
                elapsed = move.get("elapsed_search_seconds")
                if budget is not None and elapsed is not None and float(elapsed) > float(budget) + 0.25:
                    budget_overruns.append({"game_id": game_id, "elapsed": elapsed, "budget": budget})
    missing = sorted(set(schedule_by_id) - set(records_by_id))
    if extras:
        raise ValueError(f"unexpected unscheduled match IDs: {extras}")
    progress_path = artifact / "results" / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    progress_ids = [str(value) for value in progress.get("completed_game_ids", [])]
    if (progress.get("schedule_sha256") != schedule["schedule_sha256"]
            or len(progress_ids) != len(set(progress_ids))
            or set(progress_ids) != set(records_by_id)):
        raise ValueError("V6.4 persisted progress does not match the validated result IDs")
    if budget_overruns:
        raise ValueError(f"opponent search exceeded the registered deadline: {budget_overruns[:3]}")
    if require_complete and (missing or len(records_by_id) != GAME_COUNT or replayed != GAME_COUNT):
        raise ValueError(f"V6.4 result set incomplete: {len(records_by_id)}/24 games, missing={missing}")
    conditions: dict[str, Any] = {}
    game_table = []
    for opponent in (row["name"] for row in OPPONENTS):
        rows = [record for record in records_by_id.values() if record.get("opponent") == opponent]
        seed = int.from_bytes(hashlib.sha256(f"{EXPERIMENT_ID}|{opponent}".encode()).digest()[:4], "big")
        summary = summarize_condition(rows, seed=seed)
        moves = opponent_decisions[opponent]
        summary["opponent_move_telemetry"] = {
            "decisions": len(moves),
            "mean_cpu_seconds": float(np.mean([row["cpu_seconds"] for row in moves])) if moves else None,
            "mean_elapsed_seconds": float(np.mean([row["elapsed_search_seconds"] for row in moves])) if moves else None,
            "mean_completed_depth": float(np.mean([row["completed_depth"] for row in moves])) if moves else None,
            "mean_nodes": float(np.mean([row["search_nodes"] for row in moves])) if moves else None,
            "mean_cutoffs": float(np.mean([row.get("cutoffs", 0) for row in moves])) if moves else None,
            "timed_out_decisions": sum(bool(row["timed_out"]) for row in moves),
            "budget_seconds": next(row["time_limit_seconds"] for row in OPPONENTS if row["name"] == opponent),
        }
        conditions[opponent] = summary
        for record in sorted(rows, key=lambda row: (row["opening_group_id"], row["candidate_color"])):
            game_table.append({key: record[key] for key in (
                "game_id", "opponent", "opening_group_id", "opening_seed", "candidate_color",
                "winner_color", "points", "move_count", "candidate_cpu_seconds",
                "opponent_cpu_seconds", "total_cpu_seconds", "total_wall_seconds")})
    total_cpu = float(sum(float(row["total_cpu_seconds"]) for row in records_by_id.values()))
    total_wall = float(sum(float(row["total_wall_seconds"]) for row in records_by_id.values()))
    candidate_telemetry = {
        "decisions": len(candidate_decisions),
        "simulations_total": sum(int(row["simulations"]) for row in candidate_decisions),
        "simulations_per_move_min": min((int(row["simulations"]) for row in candidate_decisions), default=None),
        "simulations_per_move_max": max((int(row["simulations"]) for row in candidate_decisions), default=None),
        "mean_cpu_seconds": float(np.mean([row["cpu_seconds"] for row in candidate_decisions])
                                   if candidate_decisions else float("nan")),
        "mean_tree_depth": float(np.mean([row["search_depth"] for row in candidate_decisions])
                                 if candidate_decisions else float("nan")),
        "mean_tree_nodes": float(np.mean([row["search_nodes"] for row in candidate_decisions])
                                 if candidate_decisions else float("nan")),
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "schedule_audit": schedule_audit,
        "planned_games": GAME_COUNT,
        "completed_games": len(records_by_id),
        "replayed_legal_games": replayed,
        "missing_game_ids": missing,
        "unique_game_ids": len(records_by_id),
        "progress_completed_ids": len(progress_ids),
        "progress_matches_results": set(progress_ids) == set(records_by_id),
        "unregistered_game_ids": extras,
        "all_colors_paired": all(value["complete"] for value in conditions.values()),
        "total_match_cpu_seconds": total_cpu,
        "total_match_cpu_hours": total_cpu / 3600.0,
        "sum_of_game_wall_seconds": total_wall,
        "candidate_move_telemetry": candidate_telemetry,
        "conditions": conditions,
        "game_table": game_table,
        "complete": len(records_by_id) == GAME_COUNT and replayed == GAME_COUNT
                    and all(value["complete"] for value in conditions.values()),
    }
    if require_complete and not result["complete"]:
        raise ValueError("V6.4 audit rejected the incomplete or unpaired result set")
    atomic_json(artifact / "results" / "audit.json", result)
    return result
