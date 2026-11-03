"""Freeze V5 source, opening schedules, historical hashes, and compute plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from .v5_evaluation import build_historical_midgame_pool, generate_opening_pool, generate_six_by_six_tactics, opening_key


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "research/g_v5/config/experiment.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temp.replace(path)


def register_v5() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != config["branch"]:
        raise ValueError("V5 registration must run on research/alphazero-g-v5")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise ValueError("commit implementation before freezing the V5 experiment config")
    preflight_path = ROOT / "research/g_v5/results/preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if float(preflight["estimates"]["conservative_cpu_hours"]) > float(config["compute"]["cpu_hour_hard_cap"]):
        raise ValueError("conservative preflight estimate exceeds the registered CPU cap")

    for item in config["nine_by_nine"]["historical_checkpoints"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        item["sha256"] = _sha(path)

    schedule = {
        "format_version": 1,
        "six_by_six": {
            "development": generate_opening_pool(size=6, n_in_row=4, groups=20, plies=6, seed=560603),
            "blind": generate_opening_pool(size=6, n_in_row=4, groups=20, plies=6, seed=560604),
        },
        "nine_by_nine": {
            "development": generate_opening_pool(size=9, n_in_row=5, groups=8, plies=6, seed=590503),
            "historical": generate_opening_pool(size=9, n_in_row=5, groups=16, plies=6, seed=590505),
            "blind": generate_opening_pool(size=9, n_in_row=5, groups=20, plies=6, seed=590504),
        },
    }
    opening_keys: dict[str, set[str]] = {}
    for board_name, pools in schedule.items():
        if board_name == "format_version":
            continue
        seen = set()
        for pool_name, rows in pools.items():
            size = 6 if board_name == "six_by_six" else 9
            keys = {opening_key(row["moves"], size) for row in rows}
            if len(keys) != len(rows) or seen.intersection(keys):
                raise ValueError(f"opening pool collision for {board_name}/{pool_name}")
            seen.update(keys)
            opening_keys[f"{board_name}/{pool_name}"] = keys
    schedule_path = ROOT / "research/g_v5/config/opening_schedule.json"
    _atomic_json(schedule_path, schedule)

    puzzles = generate_six_by_six_tactics(seed=560605, per_category=32)
    puzzle_path = ROOT / "research/g_v5/config/six_by_six_tactics.json"
    _atomic_json(puzzle_path, puzzles)
    if len(puzzles) != 64 or len({row["position_key"] for row in puzzles}) != len(puzzles):
        raise ValueError("registered 6x6 tactic set must have 64 distinct mother positions")

    midgame = build_historical_midgame_pool(ROOT, maximum=96)
    if len(midgame) < 24:
        raise ValueError("fewer than 24 unique legal historical G0 middle-game prefixes were found")
    midgame_path = ROOT / "research/g_v5/config/historical_midgame_prefixes.json"
    _atomic_json(midgame_path, midgame)

    source_manifest_path = ROOT / "research/g_v5/reference/source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for row in source_manifest["files"]:
        if _sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"vendored source changed after its manifest was frozen: {row['path']}")
    config["registered_code_commit"] = commit
    config["preflight"] = {
        "path": "research/g_v5/results/preflight.json",
        "sha256": _sha(preflight_path),
        "estimates_cpu_hours": preflight["estimates"],
    }
    config["opening_schedule_sha256"] = _sha(schedule_path)
    config["six_by_six_tactics_sha256"] = _sha(puzzle_path)
    config["historical_midgame_prefix_pool_sha256"] = _sha(midgame_path)
    config["source_manifest_sha256"] = _sha(source_manifest_path)
    config.pop("config_sha256", None)
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    config["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json(CONFIG_PATH, config)
    registration = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "branch": branch,
        "registered_code_commit": commit,
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": config["config_sha256"],
        "opening_schedule_sha256": config["opening_schedule_sha256"],
        "six_by_six_tactics_sha256": config["six_by_six_tactics_sha256"],
        "historical_midgame_prefix_pool_sha256": config["historical_midgame_prefix_pool_sha256"],
        "six_by_six_puzzles": len(puzzles),
        "historical_midgame_prefixes": len(midgame),
        "preflight_estimates_cpu_hours": preflight["estimates"],
    }
    _atomic_json(ROOT / "research/g_v5/config/registration.json", registration)
    return registration


if __name__ == "__main__":
    print(json.dumps(register_v5(), ensure_ascii=False, indent=2))
