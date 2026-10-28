"""Frozen V6.3 openings and source-code identity checks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from ...gomoku import GomokuBoard
from ..v6.config import CONFIG_PATH as V6_CONFIG_PATH
from ..v6.config import load_config as load_v6_config
from ..v6.config import sha256


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/v6_3/championship.json"
SOURCE_V6_2_COMMIT = "f2af15017573383115246242e8b820d1a56c1e1e"
V6_BASELINE_COMMIT = "0c7e0e5881b8e0bcd2bdd8c07c247912ad86783d"
V6_2_CONFIG_PATH = ROOT / "research/v6_2/results/config.json"
V6_2_SCHEDULE_PATH = ROOT / "research/v6_2/results/schedule.json"
V6_2_RECORDS_PATH = ROOT / "research/v6_2/results/9x9_games.jsonl"
V6_RECORDS_PATH = ROOT / "research/v6/results/9x9_games.jsonl"


def _opening_tuple(opening: list[dict[str, Any]]) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(move["player"]), int(move["row"]), int(move["col"])) for move in opening)


def _historic_openings(v6: dict[str, Any]) -> set[tuple[tuple[str, int, int], ...]]:
    used = {
        tuple((str(stone[0]), int(stone[1]), int(stone[2])) for stone in opening)
        for opening in v6["opening_catalog"].values()
    }
    for path in (V6_RECORDS_PATH, V6_2_RECORDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot verify opening freshness without {path}")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                game = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid historical game JSON in {path}:{number}") from exc
            opening = game.get("opening")
            if isinstance(opening, list) and len(opening) >= 4:
                used.add(_opening_tuple(opening[:4]))
    if not V6_2_CONFIG_PATH.is_file() or not V6_2_SCHEDULE_PATH.is_file():
        raise FileNotFoundError("V6.2 config and schedule are required to verify V6.3 openings")
    v6_2_config = json.loads(V6_2_CONFIG_PATH.read_text(encoding="utf-8"))
    used.add(_opening_tuple(v6_2_config["opening"]))
    for spec in json.loads(V6_2_SCHEDULE_PATH.read_text(encoding="utf-8")):
        used.add(_opening_tuple(spec["opening"]))
    return used


def _validate_openings(config: dict[str, Any], v6: dict[str, Any]) -> None:
    ids = [*config["initial_opening_ids"], str(config["tiebreak_opening_id"])]
    if len(ids) != 3 or len(set(ids)) != 3 or set(ids) != set(config["openings"]):
        raise ValueError("V6.3 must preregister exactly three unique opening IDs")
    historic = _historic_openings(v6)
    seen: set[tuple[tuple[str, int, int], ...]] = set()
    for opening_id in ids:
        opening = config["openings"][opening_id]
        if not isinstance(opening, list) or len(opening) != 4:
            raise ValueError(f"V6.3 opening {opening_id} must contain exactly four plies")
        board = GomokuBoard(9)
        for index, move in enumerate(opening):
            player = "B" if index % 2 == 0 else "W"
            if str(move.get("player")) != player:
                raise ValueError(f"V6.3 opening {opening_id} has invalid alternating colors")
            coord = int(move["row"]), int(move["col"])
            if not board.in_bounds(*coord) or board.get(coord) is not None:
                raise ValueError(f"V6.3 opening {opening_id} has an illegal square")
            board.place_stone(player, coord)
            if board.check_victory(coord, player):
                raise ValueError(f"V6.3 opening {opening_id} is terminal")
        normalized = _opening_tuple(opening)
        if normalized in historic:
            raise ValueError(f"V6.3 opening {opening_id} repeats a V6 or V6.2 opening")
        if normalized in seen:
            raise ValueError("V6.3 openings must be distinct")
        seen.add(normalized)


def load_v6_3_config(source: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    if source is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        config = json.loads(json.dumps(source))
    else:
        config = json.loads(Path(source).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported V6.3 config schema")
    if config.get("experiment_id") != "wuziq-c-e5-champion-tiebreak-v6-3-2026-10-28":
        raise ValueError("Unexpected V6.3 experiment ID")
    if config.get("branch") != "research/champion-tiebreak-v6-3":
        raise ValueError("V6.3 config is registered to a different branch")
    if config.get("source_v6_2_commit") != SOURCE_V6_2_COMMIT:
        raise ValueError("V6.3 must use the specified V6.2 source commit")
    if config.get("v6_algorithm_baseline_commit") != V6_BASELINE_COMMIT:
        raise ValueError("V6.3 V6 algorithm baseline differs from V6")
    if int(config.get("board_size", 0)) != 9 or config.get("models") != ["C", "E5"]:
        raise ValueError("V6.3 is restricted to C and E5 on a 9x9 board")
    if config.get("budget_seconds_by_model") != {"C": 200.0, "E5": 200.0}:
        raise ValueError("V6.3 requires the same fixed 200-second budget for C and E5")
    if float(config.get("time_limit_seconds", -1)) != 200.0:
        raise ValueError("V6.3 per-move time limit changed")
    if (config.get("initial_games"), config.get("games_per_opening"), config.get("max_games")) != (4, 2, 6):
        raise ValueError("V6.3 must schedule four initial and two conditional games")
    if config.get("score_rule") != {
        "win": 1.0, "draw": 0.5, "loss": 0.0,
        "first_stage_tie": "play_conditional_opening", "final_tie": "joint_champion",
    }:
        raise ValueError("V6.3 score and tiebreak rules changed")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if branch != config["branch"]:
        raise ValueError("V6.3 config must be used on its independent research branch")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_V6_2_COMMIT, "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("V6.3 branch is not descended from the registered V6.2 commit")

    v6 = load_v6_config()
    if config.get("engine_sha256") != v6["engine_sha256"]:
        raise ValueError("V6.3 C/E5 engine pins differ from the verified V6 code")
    if config.get("v6_config_sha256") != sha256(V6_CONFIG_PATH.read_bytes()):
        raise ValueError("V6.3 V6 experiment config hash mismatch")
    engine_config_path = ROOT / str(v6["engine_config_path"])
    if config.get("engine_config_sha256") != sha256(engine_config_path.read_bytes()):
        raise ValueError("V6.3 frozen V5.1 search config hash mismatch")
    _validate_openings(config, v6)
    return config

