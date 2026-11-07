"""Frozen registration and paired-opening protocol for G V6.2."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from .v5_evaluation import generate_opening_pool, opening_key
from .v5_reference import make_board
from .v6_2_training import ROOT, V6_NODE, V6_STATE


ARTIFACT = Path("research/g_v6_2")
DEV_SEED = 861_201
BLIND_SEED = 861_202
DEV_PAIRS = 80
BLIND_PAIRS = 40
EVAL_BUDGETS = (64, 400)
TARGET_GAMES = (1_250, 1_500, 1_750, 2_000, 2_500, 3_000)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _make_pool(groups: int, seed: int, prefix: str) -> list[dict[str, Any]]:
    rows = generate_opening_pool(size=9, n_in_row=5, groups=groups, plies=6, seed=seed)
    for index, row in enumerate(rows):
        row["opening_group_id"] = f"{prefix}-{seed}-{index:03d}"
        row["seed"] = int(seed + index)
    return rows


def _historical_keys(root: Path) -> set[str]:
    paths = [
        root / "research/g_v5/config/opening_schedule.json",
        root / "research/g_v6/config/opening_schedule.json",
        root / ARTIFACT / "config/v6_1_openings_reference.json",
    ]
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        schedule = json.loads(path.read_text(encoding="utf-8"))
        def visit(value):
            if isinstance(value, dict):
                if isinstance(value.get("moves"), list) and value.get("moves"):
                    try:
                        if all("row" in move and "col" in move and "player" in move
                               for move in value["moves"]):
                            keys.add(opening_key(value["moves"], 9))
                    except (KeyError, TypeError, ValueError):
                        pass
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(schedule)
    return keys


def validate_schedule(schedule: dict[str, Any], *, root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    development = schedule.get("development", [])
    blind = schedule.get("blind", [])
    if (int(schedule.get("development_pairs", -1)) != DEV_PAIRS
            or int(schedule.get("blind_pairs", -1)) != BLIND_PAIRS
            or len(development) != DEV_PAIRS or len(blind) != BLIND_PAIRS):
        raise ValueError("V6.2 opening schedule count differs from the frozen registration")
    seen: set[str] = set()
    game_ids: set[str] = set()
    for label, rows in (("development", development), ("blind", blind)):
        for opening in rows:
            moves = opening.get("moves", [])
            if len(moves) != 6 or not str(opening.get("opening_group_id", "")).startswith("v6-2-"):
                raise ValueError(f"malformed V6.2 {label} opening")
            board = make_board(9, 5, 0)
            expected = "B"
            for move in moves:
                if move["player"] != expected:
                    raise ValueError("V6.2 opening colors do not alternate")
                flat = int(move["row"]) * 9 + int(move["col"])
                if flat not in board.availables:
                    raise ValueError("V6.2 opening has an illegal or repeated action")
                board.do_move(flat)
                if board.game_end()[0]:
                    raise ValueError("V6.2 opening is terminal")
                expected = "W" if expected == "B" else "B"
            key = opening_key(moves, 9)
            if key in seen:
                raise ValueError("V6.2 development/blind openings are not D4-distinct")
            seen.add(key)
            if str(opening["opening_group_id"]) in game_ids:
                raise ValueError("duplicate V6.2 opening ID")
            game_ids.add(str(opening["opening_group_id"]))
    overlap = seen & _historical_keys(root)
    if overlap:
        raise ValueError(f"V6.2 opening pool overlaps historical V5/V6/V6.1: {len(overlap)}")
    return {
        "development_openings": len(development),
        "blind_openings": len(blind),
        "unique_d4_positions": len(seen),
        "historical_d4_overlap": 0,
        "all_openings_legal_and_nonterminal": True,
    }


def build_registration(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    v5_config = json.loads((root / "research/g_v5/config/experiment.json").read_text(encoding="utf-8"))
    v6_registration = json.loads((root / "research/g_v6/config/registration.json").read_text(encoding="utf-8"))
    source_state = torch.load(root / V6_STATE, map_location="cpu", weights_only=False)
    if (int(source_state["game_count"]) != 1_012
            or int(source_state["environment_states"]) != 20_001
            or int(source_state["update_count"]) != 1_011):
        raise ValueError("frozen V6 source state counters differ from the registered G20K baseline")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    registration = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-2-9x9-selfplay-scale",
        "branch": "research/alphazero-g-v6-2-selfplay-scale",
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registration_code_commit": commit,
        "base_v6_commit": "0405b288d44e0cc35cfefb08a17f0714a1b3683e",
        "board_size": 9,
        "n_in_row": 5,
        "resume_source": str(V6_STATE),
        "resume_state_sha256": _sha(root / V6_STATE),
        "resume_model_sha256": _sha(root / V6_NODE),
        "resume_initial_games": 1_012,
        "resume_initial_states": 20_001,
        "resume_initial_updates": 1_011,
        "resume_seed": 590502,
        "resume_cumulative_cpu_seconds": float(source_state["cumulative_cpu_seconds"]),
        "resume_validation_commit": v6_registration["registration_sha256"],
        "reference_implementation": "frozen V5 vendored junxiaosong/AlphaZero_Gomoku reference code used by V6",
        "reference_config_sha256": v5_config["config_sha256"],
        "training_parameters": {
            "simulations_per_selfplay_move": 400,
            "c_puct": 5.0,
            **v5_config["training"],
            "selfplay_opening": "empty board, unchanged from V6",
            "selfplay_players": "same current model plays both colors",
        },
        "target_cumulative_games": [1_250, 1_500, 1_750, 2_000, 2_500, 3_000],
        "conditional_expansion": {
            "2k_primary_opponent": "frozen G20K",
            "2k_primary_budget": 400,
            "2k_primary_pairs": 80,
            "advance_to_2500_and_3000_if": "score rate > 0.55 and paired bootstrap CI95 lower bound > 0.50",
            "otherwise": "stop training at 2,000 games and report; no metric or opening changes",
        },
        "evaluation": {
            "budgets_root_action_visits": [64, 400],
            "development_seed": DEV_SEED,
            "development_pairs_per_stage": 20,
            "development_openings": DEV_PAIRS,
            "blind_seed": BLIND_SEED,
            "blind_pairs_per_candidate_opponent_budget": BLIND_PAIRS,
            "blind_openings": BLIND_PAIRS,
            "opponents": ["G20K", "A-Minimax", "B-AB-50ms"],
            "blind_selection": "development only; maximize paired PUCT-400 score against G20K; exact ties choose the lower cumulative game count",
            "final_blind_must_not_affect_selection": True,
        },
        "tactics": {
            "historical_64_path": "research/g_v6_2/config/tactics_historical_v6_64.json",
            "historical_64_sha256": _sha(root / ARTIFACT / "config/tactics_historical_v6_64.json"),
            "new_128_path": "research/g_v6_2/config/tactics_new_128.json",
            "new_128_sha256": _sha(root / ARTIFACT / "config/tactics_new_128.json"),
            "rule": "evaluation only; never added to self-play or replay buffer",
        },
        "forbidden_training_inputs": [
            "A/B or any historical G0 action labels or games",
            "teacher models, external games, or F-series data",
            "handcrafted tactical rewards or inference rules",
            "held-out tactical puzzles or evaluation openings",
        ],
        "compute": {
            "device": "cpu",
            "cpu_hard_cap_each_run_hours": 10,
            "cpu_hard_cap_total_new_hours": 30,
            "torch_threads": int(v5_config["compute"]["torch_threads"]),
            "drive_folder_name": "WuziQ G V6.2 Recovery",
            "drive_folder_id": "1csQOmFn1Bz62hHO22wIrdETp6mB415Wl",
            "snapshot_policy": "new upload, readback, SHA-256, empty restore before deleting previous V6.2 snapshot",
        },
        "resource_snapshot": {
            "logical_cpus": 3,
            "cpu_cgroup_quota": "200000 100000",
            "effective_cpu_cores": 2,
            "memory_limit_bytes": 8_589_934_592,
            "disk_free_bytes_at_registration": shutil_disk_free(root),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    registration["registration_sha256"] = hashlib.sha256(json.dumps(
        registration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return registration


def shutil_disk_free(path: Path) -> int:
    return int(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize)


def create_opening_schedule(root: str | Path = ROOT) -> dict[str, Any]:
    schedule = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-2-9x9-selfplay-scale",
        "development_seed": DEV_SEED,
        "blind_seed": BLIND_SEED,
        "development_pairs": DEV_PAIRS,
        "blind_pairs": BLIND_PAIRS,
        "plies": 6,
        "development": _make_pool(DEV_PAIRS, DEV_SEED, "v6-2-development"),
        "blind": _make_pool(BLIND_PAIRS, BLIND_SEED, "v6-2-blind"),
    }
    report = validate_schedule(schedule, root=root)
    schedule["validation"] = report
    return schedule


def write_frozen_registration(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    config_dir = root / ARTIFACT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    schedule = create_opening_schedule(root)
    _atomic_json(config_dir / "opening_schedule.json", schedule)
    registration = build_registration(root)
    registration["opening_schedule_sha256"] = _sha(config_dir / "opening_schedule.json")
    registration["opening_validation"] = schedule["validation"]
    registration.pop("registration_sha256", None)
    registration["registration_sha256"] = hashlib.sha256(json.dumps(
        registration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    _atomic_json(config_dir / "registration.json", registration)
    return registration
