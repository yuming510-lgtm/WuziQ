"""Create, restore, and register V6.7 full-state Google Drive recovery packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

from . import v6_7_training as training


def sha_file(path: str | Path) -> str:
    return training.sha_file(path)


def _source_files(root: Path) -> list[Path]:
    exact = [
        root / "requirements-research.txt",
        root / "src/ai.py", root / "src/gomoku.py",
        root / "src/research/alphazero_g/v5_reference.py",
        root / "src/research/alphazero_g/v5_training.py",
        root / "src/research/alphazero_g/v5_evaluation.py",
        root / "src/research/alphazero_g/v6_2_training.py",
        root / "src/research/alphazero_g/v6_5_training.py",
        root / "src/research/alphazero_g/v6_7_training.py",
        root / "src/research/alphazero_g/v6_7_evaluation.py",
        root / "src/research/alphazero_g/v6_7_snapshot.py",
        root / "src/research/alphazero_g/v6_7_audit.py",
        root / "src/research/alphazero_g/v6_7_preflight.py",
        root / "src/research/alphazero_g/v6_7_report.py",
        root / "scripts/prepare_g_v6_7.py",
        root / "scripts/run_g_v6_7_batch.py",
        root / "scripts/evaluate_g_v6_7.py",
        root / "scripts/create_g_v6_7_snapshot.py",
        root / "scripts/restore_g_v6_7_snapshot.py",
        root / "scripts/register_g_v6_7_snapshot.py",
        root / "scripts/audit_g_v6_7.py",
        root / "scripts/preflight_g_v6_7.py",
        root / "scripts/report_g_v6_7.py",
        root / "tests/test_alphazero_g_v6_7.py",
        root / "research/g_v6_5/config/reference_training_config.json",
    ]
    third_party = root / "third_party/AlphaZero_Gomoku"
    exact.extend(third_party.rglob("*.py"))
    missing = [path for path in exact if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V6.7 recovery source is incomplete: {missing[0]}")
    return sorted(set(exact))


def _artifact_files(root: Path, games: int) -> list[Path]:
    artifact = training.artifact_path(root)
    wanted = []
    for subdir in ("config", "models", "results", "raw/selfplay"):
        base = artifact / subdir
        if base.exists():
            wanted.extend(path for path in base.rglob("*") if path.is_file()
                          and not path.name.endswith((".tmp", ".zip", ".part")))
    current = training.state_path(root)
    if not current.is_file():
        raise FileNotFoundError(current)
    wanted.append(current)
    # Keep only the newly generated 8K..N raw self-play records. The independently
    # verified G8000 archive remains a permanent sibling in the V6.7 Drive folder.
    raw_dir = artifact / "raw/selfplay" / training.ARM
    wanted = [path for path in wanted if not (raw_dir in path.parents and path.suffix in {".json", ".npz"}
                                                  and int(path.stem.rsplit("-", 1)[-1]) < training.BASE_GAMES)]
    return sorted(set(wanted))


def create_snapshot(root: str | Path, *, label: str, output: str | Path) -> dict:
    started_cpu = time.process_time()
    root, output = Path(root).resolve(), Path(output).resolve()
    if not label or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in label):
        raise ValueError("snapshot label must use ASCII letters, digits, '-' and '_'")
    validation = training.validate_training_state(root=root)
    games = int(validation["game_count"])
    sources = _source_files(root)
    artifacts = _artifact_files(root, games)
    files = sorted(set(sources + artifacts))
    file_hashes = {path.relative_to(root).as_posix(): sha_file(path) for path in files}
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    progress = json.loads(training.progress_path(root).read_text(encoding="utf-8"))
    baseline = json.loads((training.artifact_path(root) / "results/baseline.json").read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
        "snapshot_label": label,
        "branch": branch,
        "code_commit": commit,
        "source_base_commit": training.SOURCE_COMMIT,
        "source_g8000_model_sha256": training.SOURCE_MODEL_SHA256,
        "source_g8000_training_state_sha256": training.SOURCE_STATE_SHA256,
        "source_v6_5_archive_sha256": training.SOURCE_ARCHIVE_SHA256,
        "games_completed": games,
        "environment_states": int(validation["environment_states"]),
        "optimizer_updates": int(validation["optimizer_updates"]),
        "training_state_sha256": validation["sha256"],
        "model_node_sha256": sha_file(training.artifact_path(root) / "models" / f"G{games}.pt")
            if (training.artifact_path(root) / "models" / f"G{games}.pt").is_file()
            else None,
        "replay_rows": int(validation["replay_rows"]),
        "replay_sha256": validation["replay_sha256"],
        "cumulative_training_cpu_seconds": float(validation["cumulative_training_cpu_seconds"]),
        "new_experiment_cpu_seconds": float(training.total_new_cpu_seconds(root)),
        "progress": progress,
        "baseline_cumulative_training_cpu_seconds": float(baseline["cumulative_training_cpu_seconds"]),
        "file_hashes": file_hashes,
        "file_count": len(file_hashes),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("V6_7_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    packaging_cpu = time.process_time() - started_cpu
    training.record_aux_cpu(root, "snapshot_packaging_cpu_seconds", packaging_cpu)
    return {"path": str(output), "sha256": sha_file(output), "size_bytes": output.stat().st_size,
            "games_completed": games, "training_state_sha256": validation["sha256"],
            "file_count": len(files), "label": label, "branch": branch,
            "code_commit": commit, "packaging_cpu_seconds": packaging_cpu}


def restore_snapshot(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    archive_sha = sha_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"zip CRC failed for {bad}")
        manifest = json.loads(archive.read("V6_7_RECOVERY_MANIFEST.json"))
        if manifest.get("experiment_id") != "wuziq-alphazero-g-v6-7-selfplay-20k":
            raise ValueError("archive is not a V6.7 recovery package")
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("unsafe archive member path")
        archive.extractall(destination)
    for relative, expected in manifest["file_hashes"].items():
        path = destination / relative
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f"snapshot file hash mismatch: {relative}")
    current = destination / training.ARTIFACT / "checkpoints" / training.ARM / "training_state.pt"
    validation = training.validate_training_state(current, root=destination,
                                                   expected_games=int(manifest["games_completed"]))
    if validation["sha256"] != manifest["training_state_sha256"]:
        raise ValueError("restored checkpoint SHA-256 does not match the manifest")
    current_node = destination / training.ARTIFACT / "models" / f"G{manifest['games_completed']}.pt"
    if current_node.is_file():
        node = __import__("torch").load(current_node, map_location="cpu", weights_only=False)
        payload = __import__("torch").load(current, map_location="cpu", weights_only=False)
        if not training.v65._state_equal(node["model_state"], payload["model_state"]):
            raise ValueError("restored model node differs from the full training-state weights")
        if manifest.get("model_node_sha256") and sha_file(current_node) != manifest["model_node_sha256"]:
            raise ValueError("restored model node SHA-256 differs from the manifest")
    return {"empty_directory_restore": True, "archive_sha256": archive_sha,
            "snapshot_label": manifest["snapshot_label"], "games_completed": manifest["games_completed"],
            "file_count": manifest["file_count"], "training_state": validation,
            "source_v6_5_archive_sha256": manifest["source_v6_5_archive_sha256"]}


def register_snapshot(root: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    if row.get("sha256") != row.get("downloaded_sha256") or not row.get("empty_directory_restore"):
        raise ValueError("cannot register a Drive snapshot before download and empty-directory restore pass")
    root = Path(root).resolve()
    path = training.artifact_path(root) / "results/drive_backup_status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "experiment_id": "wuziq-alphazero-g-v6-7-selfplay-20k",
        "folder_id": training.V67_DRIVE_FOLDER_ID, "snapshots": []}
    if row.get("folder_id") not in (None, status.get("folder_id")):
        raise ValueError("snapshot was uploaded to a different Drive folder")
    entries = status.setdefault("snapshots", [])
    if any(item.get("snapshot_id") == row.get("snapshot_id") for item in entries):
        raise ValueError("duplicate Drive snapshot registration")
    entries.append(row)
    status["latest_verified_snapshot"] = row["snapshot_id"]
    status["latest_games"] = int(row["games_completed"])
    status["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    training.atomic_json(path, status)
    progress = json.loads(training.progress_path(root).read_text(encoding="utf-8"))
    training.mark_snapshot_verified(root, games=int(row["games_completed"]),
                                    total_cpu_seconds=float(row["total_new_cpu_seconds"]),
                                    snapshot_record=row)
    return status

