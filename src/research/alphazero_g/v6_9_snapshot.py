"""Compact, content-addressed full-state recovery bundles for G V6.9."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

from . import v6_9_training as training


def _source_files(root: Path) -> list[Path]:
    files = [
        root / "requirements-research.txt", root / "src/ai.py", root / "src/gomoku.py",
        root / "src/research/alphazero_g/v5_reference.py",
        root / "src/research/alphazero_g/v5_training.py",
        root / "src/research/alphazero_g/v6_9_training.py",
        root / "src/research/alphazero_g/v6_9_snapshot.py",
        root / "scripts/prepare_g_v6_9.py", root / "scripts/run_g_v6_9_batch.py",
        root / "scripts/create_g_v6_9_snapshot.py", root / "scripts/restore_g_v6_9_snapshot.py",
        root / "scripts/register_g_v6_9_snapshot.py", root / "research/g_v6_7/config/reference_training_config.json",
    ]
    third_party = root / "third_party/AlphaZero_Gomoku"
    files.extend(third_party.rglob("*.py"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V6.9 recovery source is incomplete: {missing[0]}")
    return sorted(set(files))


def _artifact_files(root: Path) -> list[Path]:
    artifact = training.artifact_path(root)
    relative = [
        "config/reference_training_config.json", "config/registration.json",
        "results/baseline.json", "results/progress.json", "results/training_progress.json",
        "results/stage_metrics.json", "results/cpu_accounting.json",
        f"results/training/{training.ARM}.jsonl",
        f"results/training/{training.ARM}_metrics.jsonl",
        f"checkpoints/{training.ARM}/training_state.pt",
        f"checkpoints/{training.ARM}/training_state_games_{training.BASE_GAMES}.pt",
        f"checkpoints/{training.ARM}/node_games_{training.BASE_GAMES}.pt",
        "models/G20000.pt", "results/drive_backup_status.json",
    ]
    paths = [artifact / item for item in relative]
    paths = [path for path in paths if path.is_file()]
    # The currently selected milestone is already represented by the live full
    # state; include its small inference node for direct model export. Previous
    # milestones live in permanent Drive milestone snapshots, so they are not
    # recopied into every rolling archive.
    current = training.state_path(root)
    games = int(training.validate_training_state(current, root=root)["game_count"])
    current_node = training.artifact_path(root) / "checkpoints" / training.ARM / f"node_games_{games}.pt"
    if current_node.is_file():
        paths.extend((current_node, training.artifact_path(root) / "models" / f"G{games}.pt"))
    snapshots_dir = training.artifact_path(root) / "results/snapshots"
    if snapshots_dir.is_dir():
        paths.extend(path for path in snapshots_dir.glob("*.json") if path.is_file())
    return sorted(set(paths))


def create_snapshot(root: str | Path, *, label: str, output: str | Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if not label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in label):
        raise ValueError("snapshot label must use ASCII letters, digits, '-' or '_'")
    validation = training.validate_training_state(root=root)
    games = int(validation["game_count"])
    paths = sorted(set(_source_files(root) + _artifact_files(root)))
    hashes = {path.relative_to(root).as_posix(): training.sha_file(path) for path in paths}
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    progress = json.loads(training.progress_path(root).read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-9-selfplay-50k",
        "snapshot_label": label, "branch": branch, "code_commit": commit,
        "source_commit": training.SOURCE_COMMIT, "games_completed": games,
        "environment_states": int(validation["environment_states"]),
        "optimizer_updates": int(validation["optimizer_updates"]),
        "training_state_sha256": validation["sha256"],
        "model_sha256": training.SOURCE_MODEL_SHA256 if games == training.BASE_GAMES else None,
        "replay_rows": int(validation["replay_rows"]), "replay_sha256": validation["replay_sha256"],
        "cumulative_training_cpu_seconds": validation["cumulative_training_cpu_seconds"],
        "new_training_cpu_seconds": float(progress.get("new_training_cpu_seconds", 0.0)),
        "progress": progress, "file_hashes": hashes, "file_count": len(hashes),
        "snapshot_contents_note": "full training state already contains Adam, replay buffer, RNG and counters; raw self-play tensors are retained in the workspace and are not duplicated in every rolling recovery bundle",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temp_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("V6_9_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temp_output.replace(output)
    return {"path": str(output), "sha256": training.sha_file(output), "size_bytes": output.stat().st_size,
            "games_completed": games, "training_state_sha256": validation["sha256"],
            "file_count": len(paths), "label": label, "branch": branch, "code_commit": commit}


def restore_snapshot(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    archive_sha = training.sha_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"zip CRC failed for {bad}")
        manifest = json.loads(archive.read("V6_9_RECOVERY_MANIFEST.json"))
        if manifest.get("experiment_id") != "wuziq-alphazero-g-v6-9-selfplay-50k":
            raise ValueError("archive is not a V6.9 recovery package")
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("unsafe recovery archive member path")
        archive.extractall(destination)
    for relative, expected in manifest["file_hashes"].items():
        path = destination / relative
        if not path.is_file() or training.sha_file(path) != expected:
            raise ValueError(f"snapshot file hash mismatch: {relative}")
    validation = training.validate_training_state(
        destination / training.ARTIFACT / "checkpoints" / training.ARM / "training_state.pt",
        root=destination, expected_games=int(manifest["games_completed"]))
    if validation["sha256"] != manifest["training_state_sha256"]:
        raise ValueError("restored full training state differs from archive manifest")
    progress_path = training.progress_path(destination)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update({
        "games_completed": int(manifest["games_completed"]),
        "training_states_cumulative": int(manifest["environment_states"]),
        "optimizer_updates_cumulative": int(manifest["optimizer_updates"]),
        "training_state_sha256": manifest["training_state_sha256"],
        "last_snapshot_games": int(manifest["games_completed"]), "snapshot_due": False,
        "phase": "training_complete" if int(manifest["games_completed"]) == training.TARGET_GAMES else "self_play_training",
        "completed": int(manifest["games_completed"]) == training.TARGET_GAMES,
        "recovered_from_snapshot": manifest["snapshot_label"],
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    training.atomic_json(progress_path, progress)
    return {"empty_directory_restore": True, "archive_sha256": archive_sha,
            "snapshot_label": manifest["snapshot_label"], "games_completed": manifest["games_completed"],
            "file_count": manifest["file_count"], "training_state": validation,
            "progress_reconciled_to_snapshot": True}


def register_snapshot(root: str | Path, row: dict) -> dict:
    if row.get("sha256") != row.get("downloaded_sha256") or not row.get("empty_directory_restore"):
        raise ValueError("snapshot upload/download/hash/restore verification must pass before registration")
    root = Path(root).resolve()
    status_path = training.artifact_path(root) / "results/drive_backup_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
        "experiment_id": "wuziq-alphazero-g-v6-9-selfplay-50k",
        "folder_id": training.DRIVE_FOLDER_ID, "snapshots": []}
    if row.get("folder_id") != status.get("folder_id"):
        raise ValueError("snapshot is not in the dedicated V6.9 Drive folder")
    if any(item.get("snapshot_id") == row.get("snapshot_id") for item in status["snapshots"]):
        raise ValueError("duplicate Drive snapshot registration")
    state = training.validate_training_state(root=root, expected_games=int(row["games_completed"]))
    if state["sha256"] != row["training_state_sha256"]:
        raise ValueError("Drive snapshot checkpoint differs from local current state")
    progress = training.mark_snapshot_verified(root, games=int(row["games_completed"]), snapshot_record=row)
    status["snapshots"].append(row)
    status["latest_verified_snapshot"] = row["snapshot_id"]
    status["latest_games"] = int(row["games_completed"])
    status["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    training.atomic_json(status_path, status)
    return {"status": status, "progress": progress}
