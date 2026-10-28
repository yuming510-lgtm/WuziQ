"""V6.2 settings and source checks for the frozen C, D, and E5 players."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...gomoku import GomokuBoard
from ..checkpoint import load_game_records
from ..v6.config import CONFIG_PATH as V6_CONFIG_PATH
from ..v6.config import load_config as load_v6_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/v6_2/championship.json"
BASELINE_COMMIT = "0c7e0e5881b8e0bcd2bdd8c07c247912ad86783d"
EXPECTED_OPENING = (
    ("B", 4, 4),
    ("W", 3, 5),
    ("B", 5, 3),
    ("W", 2, 6),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _opening_tuple(opening: list[dict[str, Any]]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(move["player"]), int(move["row"]), int(move["col"]))
        for move in opening
    )


def _require_fresh_opening(config: dict[str, Any], v6: dict[str, Any]) -> None:
    opening = _opening_tuple(config["opening"])
    if opening != EXPECTED_OPENING:
        raise ValueError("V6.2 common opening differs from its preregistered coordinates")
    catalog = {
        tuple((str(stone[0]), int(stone[1]), int(stone[2])) for stone in stones)
        for stones in v6["opening_catalog"].values()
    }
    if opening in catalog:
        raise ValueError("V6.2 opening was already in the frozen V6 opening catalog")

    game_path = ROOT / "research/v6/results/9x9_games.jsonl"
    if not game_path.is_file():
        raise FileNotFoundError("Cannot verify that the V6.2 opening is absent from V6 games")
    used: set[tuple[tuple[str, int, int], ...]] = set()
    for line_number, line in enumerate(game_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            game = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid V6 game JSON at line {line_number}") from exc
        recorded = game.get("opening")
        if isinstance(recorded, list):
            used.add(_opening_tuple(recorded[:4]))
    if opening in used:
        raise ValueError("V6.2 opening sequence was already used by a V6 game")


def load_v6_2_config(source: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    """Read and validate the immutable championship configuration and V6 engine pins."""
    if source is None:
        path = CONFIG_PATH
        config = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        path = CONFIG_PATH
        config = json.loads(json.dumps(source))
    else:
        path = Path(source)
        config = json.loads(path.read_text(encoding="utf-8"))

    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported V6.2 config schema")
    if config.get("experiment_id") != "wuziq-champion-200s-v6-2-2026-10-28":
        raise ValueError("Unexpected V6.2 experiment ID")
    if config.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("V6.2 config does not reference the verified V6 baseline")
    if int(config.get("board_size", 0)) != 9:
        raise ValueError("V6.2 is frozen to a 9x9 board")
    if tuple(config.get("models", ())) != ("C", "D", "E5"):
        raise ValueError("V6.2 must use exactly the frozen C, D, and E5 models")
    if float(config.get("time_limit_seconds", -1)) != 200.0:
        raise ValueError("Every V6.2 search move must have a 200-second cap")
    if int(config.get("games_per_matchup", 0)) != 2 or int(config.get("expected_games", 0)) != 6:
        raise ValueError("V6.2 requires exactly two games for each of three pairs")
    expected_pairs = [["C", "D"], ["C", "E5"], ["D", "E5"]]
    if config.get("pairs") != expected_pairs:
        raise ValueError("V6.2 matchup list differs from the preregistered three pairs")
    if str(config.get("opening_id")) != "731201":
        raise ValueError("V6.2 common opening ID must remain 731201")
    if config.get("score_rule") != {
        "win": 1.0,
        "draw": 0.5,
        "loss": 0.0,
        "tie": "joint_champion",
    }:
        raise ValueError("V6.2 point or tie rule changed after preregistration")

    v6 = load_v6_config()
    if config.get("engine_sha256") != v6["engine_sha256"]:
        raise ValueError("V6.2 engine hashes do not match the frozen V6 decision code")
    if config.get("v6_config_sha256") != _sha(V6_CONFIG_PATH.read_bytes()):
        raise ValueError("V6.2 V6 configuration hash mismatch")
    engine_config_path = ROOT / str(v6["engine_config_path"])
    if config.get("engine_config_sha256") != _sha(engine_config_path.read_bytes()):
        raise ValueError("V6.2 source engine configuration hash mismatch")

    opening = config.get("opening")
    if not isinstance(opening, list) or len(opening) != 4:
        raise ValueError("V6.2 requires a four-ply common opening")
    board = GomokuBoard(9)
    for index, move in enumerate(opening):
        player = "B" if index % 2 == 0 else "W"
        if move.get("player") != player:
            raise ValueError("V6.2 opening has invalid alternating colors")
        coord = (int(move["row"]), int(move["col"]))
        board.place_stone(player, coord)
        if board.check_victory(coord, player):
            raise ValueError("V6.2 opening is terminal")
    _require_fresh_opening(config, v6)
    return config
