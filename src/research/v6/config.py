"""Frozen V6 model families, budgets, openings, and source-code pins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import __file__ as _package_file


ROOT = Path(_package_file).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/v6/experiment.json"
FAMILIES = ("A", "B", "C", "D", "E1", "E2", "E3", "E4", "E5")
SEARCH_FAMILIES = FAMILIES[1:]
BUDGET_LABELS = ("50ms", "200ms", "1s", "5s", "10s", "20s", "50s")
BUDGET_SECONDS = (0.05, 0.2, 1.0, 5.0, 10.0, 20.0, 50.0)
ABCD_FAMILIES = ("A", "B", "C", "D")
ABCD_ENGINE_FILES = ("src/ai.py", "src/gomoku.py", "src/research/mcts_d2.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def load_config(source: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    if source is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        config = json.loads(json.dumps(source))
    else:
        config = json.loads(Path(source).read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported V6 config schema")
    if config.get("experiment_id") != "wuziq-advanced-search-v6-2026-10-22":
        raise ValueError("Unexpected V6 experiment ID")
    if int(config.get("board_size", 0)) != 9 or tuple(config.get("families", ())) != FAMILIES:
        raise ValueError("V6 is frozen to nine families on a 9x9 board")
    if tuple(config.get("search_families", ())) != SEARCH_FAMILIES:
        raise ValueError("V6 search family order changed")
    ladder = config.get("budget_ladder", ())
    if tuple(item.get("label") for item in ladder) != BUDGET_LABELS:
        raise ValueError("V6 must use the seven frozen budget rungs")
    if tuple(float(item.get("seconds", -1)) for item in ladder) != BUDGET_SECONDS:
        raise ValueError("V6 budget seconds do not match labels")
    if (int(config.get("maximum_routes", 0)), int(config.get("maximum_groups", 0)),
            int(config.get("maximum_games", 0))) != (456, 1428, 2856):
        raise ValueError("V6 theoretical maximum schedule changed")
    if int(config.get("opening_plies", 0)) != 4:
        raise ValueError("V6 uses four-ply paired openings")
    seeds = tuple(map(int, config.get("opening_seeds", ())))
    if len(seeds) != 12 or len(set(seeds)) != 12:
        raise ValueError("V6 requires twelve frozen opening seeds")
    engine_files = tuple(config.get("engine_files", ()))
    expected = set(ABCD_ENGINE_FILES) | {
        "src/research/v6/__init__.py",
        "src/research/v6/engine.py",
        "src/research/v6/e5.py",
        "src/research/v6/pns.py",
        "src/research/v6/tactics.py",
        "src/research/v6/types.py",
    }
    if set(engine_files) != expected or set(config.get("engine_sha256", {})) != expected:
        raise ValueError("V6 engine hash list does not match the registered source files")
    for relative in engine_files:
        path = ROOT / relative
        if not path.is_file() or sha256(path.read_bytes()) != config["engine_sha256"][relative]:
            raise ValueError(f"V6 engine source hash mismatch: {relative}")
    puzzle_path = ROOT / str(config.get("tactical_positions_path", ""))
    if (not puzzle_path.is_file()
            or sha256(puzzle_path.read_bytes()) != config.get("tactical_positions_sha256")):
        raise ValueError("V6 tactical puzzle fixture hash mismatch")
    source_config_path = ROOT / str(config.get("engine_config_path", ""))
    if (not source_config_path.is_file()
            or sha256(source_config_path.read_bytes()) != config.get("engine_config_sha256")):
        raise ValueError("Pinned V5.1 engine configuration changed")
    from ..abcd_ladder_v5_1 import engine_config as v51_engine_config, load_config as load_v51_config
    v51 = load_v51_config(source_config_path)
    if tuple(v51.get("families", ())) != ABCD_FAMILIES:
        raise ValueError("V6 source engine config is not the frozen V5.1 ABCD config")
    if config.get("v5_1_engine_sha256") != v51.get("engine_sha256"):
        raise ValueError("V6 V5.1 engine pins differ from the source config")
    for seed in seeds:
        stones = config["opening_catalog"].get(str(seed))
        if not isinstance(stones, list) or len(stones) != 4:
            raise ValueError(f"V6 opening {seed} must have exactly four plies")
        from ...gomoku import GomokuBoard
        board = GomokuBoard(9)
        for index, stone in enumerate(stones):
            if len(stone) != 3 or str(stone[0]) != ("B" if index % 2 == 0 else "W"):
                raise ValueError(f"V6 opening {seed} has invalid color order")
            coord = int(stone[1]), int(stone[2])
            board.place_stone(str(stone[0]), coord)
            if board.check_victory(coord, str(stone[0])):
                raise ValueError(f"V6 opening {seed} is terminal")
    if config.get("opening_catalog") != v51.get("opening_catalog") or seeds != tuple(v51["opening_seeds"]):
        raise ValueError("V6 opening pool differs from V5.1's frozen legal catalog")
    return config


def v5_1_engine_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    frozen = load_config(config)
    source_v51 = json.loads((ROOT / str(frozen["engine_config_path"])).read_text(encoding="utf-8"))
    from ..abcd_ladder_v5_1 import engine_config
    return engine_config(source_v51)


def budget_seconds(label: str) -> float | None:
    if label == "natural":
        return None
    try:
        return BUDGET_SECONDS[BUDGET_LABELS.index(label)]
    except ValueError as exc:
        raise ValueError(f"Unknown V6 budget label: {label}") from exc
