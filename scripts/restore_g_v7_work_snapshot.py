#!/usr/bin/env python3
"""Safely restore and validate a downloaded V7 WORK P6 recovery bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.research.alphazero_g import v6_5_training, v6_7_training
from src.research.alphazero_g.v7_work_p6_50k import BASE_GAMES, BASE_STATE_SHA256, ROOT


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(archive_path: Path, destination: Path, *, game_chunks: list[Path] | None = None,
            install_run_dir: Path | None = None) -> dict:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("restore destination must be an empty directory")
    with zipfile.ZipFile(archive_path) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"ZIP integrity failed at {bad}")
        manifest = json.loads(zf.read("V7_WORK_P6_SNAPSHOT_MANIFEST.json"))
        if manifest.get("experiment_id") != "wuziq-alphazero-g-v7-work-p6-50k":
            raise ValueError("this is not the registered V7 WORK P6 bundle")
        for member_name in zf.namelist():
            p = PurePosixPath(member_name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("snapshot contains an unsafe archive path")
        zf.extractall(destination)
    for relative, expected in manifest["file_hashes"].items():
        path = destination / relative
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f"restored file hash mismatch: {relative}")
    checkpoint = destination / "checkpoints/training_state.pt"
    validation = v6_7_training.validate_training_state(checkpoint, root=ROOT)
    if validation["sha256"] != manifest["training_state_sha256"]:
        raise ValueError("restored full-state hash differs from snapshot manifest")
    if not all(validation.get(k) for k in (
        "valid", "model_restored", "adam_restored", "replay_restored",
        "python_rng_restored", "numpy_rng_restored", "torch_rng_restored")):
        raise ValueError("checkpoint failed full-state restore validation")
    if (validation["game_count"] != manifest["cumulative_games"]
            or validation["optimizer_updates"] != manifest["optimizer_updates"]):
        raise ValueError("restored counters differ from snapshot manifest")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    node_candidates = sorted((destination / "checkpoints").glob("node_G*.pt"))
    if not node_candidates:
        raise ValueError("snapshot omitted its model node")
    node = torch.load(node_candidates[-1], map_location="cpu", weights_only=False)
    if not v6_5_training._state_equal(state["model_state"], node["model_state"]):
        raise ValueError("restored model node differs from the full-state checkpoint")
    rows = [json.loads(line) for line in (destination / "per_game.jsonl").read_text().splitlines() if line.strip()]
    task_state = json.loads((destination / "task_state.json").read_text())
    added = int(validation["game_count"]) - BASE_GAMES
    if len(rows) != added or task_state["committed_game_count"] != added:
        raise ValueError("restored ledger and committed task list do not match the checkpoint")
    ids = [r["game_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != {r["game_id"] for r in task_state["committed_games"]}:
        raise ValueError("restored committed game IDs are missing or duplicated")
    if {int(r["ordinal"]) for r in rows} != set(range(added)):
        raise ValueError("restored committed ordinals are not a complete prefix")
    if len(task_state["pending_games"]) != 30_000 - added or task_state["in_flight_games"]:
        raise ValueError("restored pending/in-flight task manifest is inconsistent")
    restored_raw: dict[str, Path] = {}
    chunk_ranges = []
    for chunk_path in game_chunks or []:
        with zipfile.ZipFile(chunk_path) as zf:
            if zf.testzip():
                raise ValueError(f"game chunk ZIP integrity failed: {chunk_path}")
            chunk_manifest = json.loads(zf.read("V7_WORK_P6_GAME_CHUNK_MANIFEST.json"))
            if chunk_manifest.get("experiment_id") != manifest["experiment_id"]:
                raise ValueError(f"game chunk has a different experiment ID: {chunk_path}")
            if int(chunk_manifest.get("unique_game_ids", -1)) != int(chunk_manifest.get("game_count", -2)):
                raise ValueError(f"game chunk IDs are not unique: {chunk_path}")
            chunk_root = destination / "_chunk_restore" / Path(chunk_path).stem
            chunk_root.mkdir(parents=True, exist_ok=True)
            for name in zf.namelist():
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError("game chunk contains an unsafe archive path")
            zf.extractall(chunk_root)
        for relative, expected_hash in chunk_manifest["file_hashes"].items():
            member = chunk_root / relative
            if not member.is_file() or sha_file(member) != expected_hash:
                raise ValueError(f"game chunk file hash mismatch: {chunk_path.name}:{relative}")
        start, stop = map(int, chunk_manifest["range_added_games"])
        if stop <= start or stop > added or int(chunk_manifest["game_count"]) != stop - start:
            raise ValueError(f"game chunk range does not reconcile with the checkpoint: {chunk_path.name}")
        chunk_ranges.append((start, stop))
        for gid in chunk_manifest["game_ids"]:
            for suffix in (".json", ".npz"):
                path = chunk_root / "games" / f"{gid}{suffix}"
                if not path.is_file():
                    raise ValueError(f"game chunk omitted a game artifact: {path.name}")
                key = f"{gid}{suffix}"
                if key in restored_raw:
                    raise ValueError(f"game artifact appears in more than one chunk: {key}")
                restored_raw[key] = path
    if game_chunks:
        ordered_ranges = sorted(chunk_ranges)
        cursor = 0
        for start, stop in ordered_ranges:
            if start != cursor:
                raise ValueError("provided game chunks are not a contiguous prefix")
            cursor = stop
        if cursor != added or len(restored_raw) != 2 * added:
            raise ValueError("provided game chunks do not cover all committed raw data")
    if install_run_dir is not None:
        install_run_dir = Path(install_run_dir).resolve()
        install_run_dir.mkdir(parents=True, exist_ok=True)
        if any(install_run_dir.iterdir()):
            raise ValueError("run installation destination must be empty")
        path_map = {
            "checkpoints/training_state.pt": "checkpoints/training_state.pt",
            "progress.json": "results/progress.json",
            "run_summary.json": "results/run_summary.json",
            "per_game.jsonl": "results/per_game.jsonl",
            "baseline_provenance.json": "results/baseline_provenance.json",
        }
        for relative, target_relative in path_map.items():
            target = install_run_dir / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination / relative, target)
        for node_path in (destination / "checkpoints").glob("node_G*.pt"):
            shutil.copy2(node_path, install_run_dir / "checkpoints" / node_path.name)
        if game_chunks:
            raw_dir = install_run_dir / "raw/selfplay/reference400_empty"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for name, source in restored_raw.items():
                shutil.copy2(source, raw_dir / name)
    report = {
        "experiment_id": manifest["experiment_id"], "archive_sha256": sha_file(archive_path),
        "empty_directory_restore": True, "file_count": manifest["file_count"],
        "cumulative_games": validation["game_count"], "added_games": added,
        "optimizer_updates": validation["optimizer_updates"],
        "training_state_sha256": validation["sha256"],
        "checkpoint_validation": validation, "model_node_matches_full_state": True,
        "committed_game_ids": len(ids), "pending_tasks": len(task_state["pending_games"]),
        "restored_game_chunks": len(game_chunks or []),
        "raw_game_artifacts_restored": len(restored_raw),
        "installed_resumable_run": install_run_dir is not None,
        "valid": True,
    }
    (destination / "RESTORE_VALIDATION.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--game-chunk", type=Path, action="append", default=[])
    parser.add_argument("--install-run-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.destination, game_chunks=args.game_chunk,
                             install_run_dir=args.install_run_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
