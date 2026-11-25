#!/usr/bin/env python3
"""Create a recovery snapshot and immutable game-data chunk for one V7 segment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.research.alphazero_g import v6_7_training
from src.research.alphazero_g.v7_work_p6_50k import (
    ARTIFACT, BASE_ARCHIVE_SHA256, BASE_CHECKPOINT_COMMIT, BASE_COMMIT, BASE_GAMES, BASE_STATE_SHA256,
    EXPERIMENT_ID, ROOT, _sha_file,
)

RUN_ID = "V7_WORK_P6_50K"
TARGET_ADDED = 30_000
RUN_DIR = ROOT / ARTIFACT / "runs" / RUN_ID
CODE_PATHS = (
    "src/research/alphazero_g/v7_work_p6_50k.py",
    "src/research/alphazero_g/v7_work_p6_50k_evaluation.py",
    "src/research/alphazero_g/v6_9_parallel.py",
    "src/research/alphazero_g/v6_9_snapshot.py",
    "src/research/alphazero_g/v6_7_training.py",
    "src/research/alphazero_g/v6_5_training.py",
    "src/research/alphazero_g/v6_5_audit.py",
    "src/research/alphazero_g/v6_2_training.py",
    "src/research/alphazero_g/v5_training.py",
    "src/research/alphazero_g/v5_reference.py",
    "src/research/alphazero_g/v5_evaluation.py",
    "src/research/alphazero_g/v7_work_p6_50k_evaluation.py",
    "src/ai.py",
    "src/gomoku.py",
    "third_party/AlphaZero_Gomoku/__init__.py",
    "third_party/AlphaZero_Gomoku/game.py",
    "third_party/AlphaZero_Gomoku/mcts_alphaZero.py",
    "third_party/AlphaZero_Gomoku/policy_value_net_pytorch.py",
    "scripts/run_g_v7_work_p6_50k.py",
    "scripts/package_g_v7_work_snapshot.py",
    "scripts/restore_g_v7_work_snapshot.py",
    "scripts/run_g_v7_work_p6_50k_evaluation.py",
    "scripts/run_g_v7_work_pairwise.py",
    "research/g_v6_7/config/reference_training_config.json",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_zip(path: Path, members: dict[str, Path], metadata: dict, metadata_name: str) -> dict:
    hashes = {name: _sha_file(src) for name, src in sorted(members.items())}
    manifest = {**metadata, "file_hashes": hashes, "file_count": len(hashes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, src in sorted(members.items()):
            zf.write(src, name)
        zf.writestr(metadata_name, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.replace(temp, path)
    return {"path": str(path), "sha256": _sha_file(path), "size_bytes": path.stat().st_size,
            "manifest": manifest}


def package(*, chunk_start: int, output_dir: Path) -> dict:
    checkpoint = RUN_DIR / "checkpoints/training_state.pt"
    metrics_path = RUN_DIR / "results/per_game.jsonl"
    progress_path = RUN_DIR / "results/progress.json"
    summary_path = RUN_DIR / "results/run_summary.json"
    node_files = sorted((RUN_DIR / "checkpoints").glob("node_G*.pt"))
    if not all(p.is_file() for p in (checkpoint, metrics_path, progress_path, summary_path)) or not node_files:
        raise FileNotFoundError("the V7 segment did not finish with a complete checkpoint, node, metrics, and progress")
    progress = _json(progress_path)
    if progress.get("status") not in {"checkpointed", "complete"}:
        raise ValueError("recovery snapshots are only packaged at clean committed-game boundaries")
    state_check = v6_7_training.validate_training_state(checkpoint, root=ROOT)
    added = int(state_check["game_count"]) - BASE_GAMES
    if added <= chunk_start or added - chunk_start > 250:
        raise ValueError("snapshot must close exactly one non-empty 250-game or shorter final segment")
    if state_check["sha256"] != progress["checkpoint_sha256"]:
        raise ValueError("progress and current checkpoint SHA-256 do not match")
    if int(state_check["optimizer_updates"]) != 19_999 + added:
        raise ValueError("optimizer update counter does not match committed V7 games")
    if int(state_check["game_count"]) != BASE_GAMES + added:
        raise ValueError("checkpoint game counter does not match committed V7 games")
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != added or len({r["game_id"] for r in rows}) != added:
        raise ValueError("committed-game ledger does not reconcile with checkpoint counters")
    if {int(r["ordinal"]) for r in rows} != set(range(added)):
        raise ValueError("committed ordinals are not a contiguous, duplicate-free prefix")
    if any(int(r["training_update_calls"]) != 1 or int(r["updates_after"]) != int(r["updates_before"]) + 1
           for r in rows):
        raise ValueError("a committed game does not have exactly one central optimizer update")

    code_members = {}
    for relative in CODE_PATHS:
        src = ROOT / relative
        if not src.is_file():
            raise FileNotFoundError(f"recovery source is missing: {relative}")
        code_members[f"code/{relative}"] = src
    code_manifest = {"experiment_id": EXPERIMENT_ID,
                     "base_commit": BASE_COMMIT,
                     "files": {relative: _sha_file(ROOT / relative) for relative in CODE_PATHS}}
    code_manifest_path = ROOT / ARTIFACT / "config/code_manifest.json"
    code_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    code_manifest_path.write_text(json.dumps(code_manifest, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    node = node_files[-1]
    members = {
        "checkpoints/training_state.pt": checkpoint,
        f"checkpoints/{node.name}": node,
        "progress.json": progress_path,
        "run_summary.json": summary_path,
        "per_game.jsonl": metrics_path,
        "baseline_provenance.json": RUN_DIR / "results/baseline_provenance.json",
        "training_registration.json": ROOT / ARTIFACT / "config/training_registration.json",
        "preflight.json": ROOT / ARTIFACT / "config/preflight.json",
        "drive_locations.json": ROOT / ARTIFACT / "config/drive_locations.json",
        "code_manifest.json": ROOT / ARTIFACT / "config/code_manifest.json",
        **code_members,
    }
    if any(not src.is_file() for src in members.values()):
        missing = [name for name, src in members.items() if not src.is_file()]
        raise FileNotFoundError(f"snapshot source files are missing: {missing}")
    committed = [{"ordinal": int(r["ordinal"]), "game_id": str(r["game_id"]),
                  "seed": int(r["seed"]), "model_version_used": int(r["model_version_used"]),
                  "updates_after": int(r["updates_after"])} for r in rows]
    pending = [{"ordinal": i, "game_id": f"{RUN_ID}-game-{BASE_GAMES+i:06d}",
                "seed": 590_502 + (BASE_GAMES+i) * 10_007,
                "status": "not_submitted_after_clean_segment_boundary"}
               for i in range(added, TARGET_ADDED)]
    task_state = {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
        "training_game_count": int(state_check["game_count"]),
        "optimizer_updates": int(state_check["optimizer_updates"]),
        "committed_game_count": added, "committed_games": committed,
        "in_flight_games": [], "pending_games": pending,
        "current_model_version": int(state_check["optimizer_updates"]),
        "central_update_calls": added,
        "resume_rule": "skip every committed ordinal; submit only pending ordinals; each commit applies one update",
    }
    task_file = output_dir / "staging_task_state.json"
    task_file.parent.mkdir(parents=True, exist_ok=True)
    task_file.write_text(json.dumps(task_state, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    members["task_state.json"] = task_file
    snapshot_metadata = {
        "format_version": 1, "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
        "snapshot_kind": "full_recovery_checkpoint", "games_added": added,
        "cumulative_games": int(state_check["game_count"]), "optimizer_updates": int(state_check["optimizer_updates"]),
        "training_state_sha256": state_check["sha256"], "baseline_state_sha256": BASE_STATE_SHA256,
        "baseline_archive_sha256": BASE_ARCHIVE_SHA256, "base_commit": BASE_COMMIT,
        "checkpoint_source_commit": BASE_CHECKPOINT_COMMIT,
        "checkpoint_validation": state_check,
    }
    snap = _write_zip(output_dir / f"WuziQ_V7_WORK_P6_50K_rolling_G{state_check['game_count']}.zip",
                      members, snapshot_metadata, "V7_WORK_P6_SNAPSHOT_MANIFEST.json")

    raw_dir = RUN_DIR / "raw/selfplay/reference400_empty"
    chunk_members: dict[str, Path] = {}
    chunk_rows = [r for r in rows if chunk_start <= int(r["ordinal"]) < added]
    if len(chunk_rows) != added - chunk_start:
        raise ValueError("segment ledger rows do not match the requested chunk boundary")
    row_file = output_dir / "staging_chunk_rows.jsonl"
    row_file.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in chunk_rows), encoding="utf-8")
    chunk_members["per_game.jsonl"] = row_file
    game_ids = []
    for r in chunk_rows:
        gid = str(r["game_id"])
        game_ids.append(gid)
        for suffix in (".json", ".npz"):
            file_path = raw_dir / f"{gid}{suffix}"
            if not file_path.is_file():
                raise FileNotFoundError(f"raw game artifact missing: {file_path}")
            chunk_members[f"games/{gid}{suffix}"] = file_path
    chunk_metadata = {
        "format_version": 1, "experiment_id": EXPERIMENT_ID,
        "range_added_games": [chunk_start, added], "cumulative_games": int(state_check["game_count"]),
        "game_count": len(game_ids), "unique_game_ids": len(set(game_ids)),
        "game_ids": game_ids,
    }
    chunk = _write_zip(output_dir / f"WuziQ_V7_WORK_P6_50K_games_{chunk_start+1:05d}_{added:05d}.zip",
                       chunk_members, chunk_metadata, "V7_WORK_P6_GAME_CHUNK_MANIFEST.json")
    return {"snapshot": snap, "game_chunk": chunk, "checkpoint_validation": state_check,
            "chunk_start": chunk_start, "chunk_stop": added, "rows": len(rows),
            "committed_ids": added, "pending_tasks": len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-start", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ARTIFACT / "backup_staging")
    args = parser.parse_args()
    result = package(chunk_start=args.chunk_start, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
