"""Integrity-checked, empty-directory recovery packages for G V6.5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

import torch

from . import v6_5_training as training

ARTIFACT = training.ARTIFACT
ARM = training.ARM


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> list[Path]:
    artifact = root / ARTIFACT
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    return sorted(path for path in artifact.rglob("*") if path.is_file()
                  and not path.name.endswith((".tmp", ".zip", ".sha256.json"))
                  and "recovery" not in path.relative_to(artifact).parts)


def _source_files(root: Path) -> list[Path]:
    wanted = [
        root / "requirements-research.txt",
        root / "src/ai.py", root / "src/gomoku.py",
        root / "src/research/alphazero_g/v5_reference.py",
        root / "src/research/alphazero_g/v5_training.py",
        root / "src/research/alphazero_g/v6_2_training.py",
        root / "src/research/alphazero_g/v6_3_training.py",
        root / "src/research/alphazero_g/v6_5_training.py",
        root / "src/research/alphazero_g/v6_5_snapshot.py",
        root / "src/research/alphazero_g/v6_5_audit.py",
        root / "scripts/run_g_v6_5.py",
        root / "scripts/create_g_v6_5_snapshot.py",
        root / "scripts/restore_g_v6_5_snapshot.py",
        root / "scripts/register_g_v6_5_snapshot.py",
        root / "scripts/audit_g_v6_5.py",
        root / "scripts/report_g_v6_5.py",
        root / "tests/test_alphazero_g_v6_5.py",
    ]
    third_party = root / "third_party/AlphaZero_Gomoku"
    wanted.extend(third_party.rglob("*.py"))
    missing = [path for path in wanted if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"recovery code file missing: {missing[0]}")
    return sorted(set(wanted))


def create_snapshot(root: str | Path, *, label: str, output: str | Path) -> dict:
    started_cpu = time.process_time()
    root, output = Path(root).resolve(), Path(output).resolve()
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in label):
        raise ValueError("snapshot label must contain only ASCII letters, digits, '-' or '_'")
    current = root / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
    validation = training.validate_training_state(current, root=root)
    files = sorted(set(_artifact_files(root) + _source_files(root)))
    relative_hashes = {path.relative_to(root).as_posix(): sha_file(path) for path in files}
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                 capture_output=True, text=True).stdout.strip()
    payload = torch.load(current, map_location="cpu", weights_only=False)
    manifest = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-5-selfplay-scale",
        "snapshot_label": label,
        "branch": branch,
        "code_commit": code_commit,
        "source_v6_3_commit": training.SOURCE_COMMIT,
        "source_archive_code_commit": training.SOURCE_ARCHIVE_CODE_COMMIT,
        "source_v6_3_archive_sha256": training.SOURCE_ARCHIVE_SHA256,
        "source_g5000_training_state_sha256": training.SOURCE_STATE_SHA256,
        "source_g5000_model_sha256": training.SOURCE_MODEL_SHA256,
        "games_completed": int(payload["game_count"]),
        "environment_states": int(payload["environment_states"]),
        "optimizer_updates": int(payload["update_count"]),
        "new_cpu_seconds": float(payload["cumulative_cpu_seconds"] - training._baseline_cpu_seconds(root)),
        "training_state_sha256": sha_file(current),
        "files": relative_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("V6_5_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {
        "path": str(output), "sha256": sha_file(output), "size_bytes": output.stat().st_size,
        "file_count": len(files), "code_commit": code_commit, "branch": branch,
        "snapshot_label": label, "games_completed": manifest["games_completed"],
        "training_state_sha256": manifest["training_state_sha256"],
        "packaging_cpu_seconds": time.process_time() - started_cpu,
    }


def restore_snapshot(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("recovery destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    archive_sha = sha_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"zip CRC failed for {bad}")
        manifest = json.loads(archive.read("V6_5_RECOVERY_MANIFEST.json"))
        if manifest.get("experiment_id") != "wuziq-alphazero-g-v6-5-selfplay-scale":
            raise ValueError("archive is not a WuziQ V6.5 recovery package")
        for member in archive.namelist():
            candidate = PurePosixPath(member)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("unsafe path in V6.5 recovery archive")
        archive.extractall(destination)
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f"snapshot file SHA-256 mismatch: {relative}")
    state_path = destination / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
    restored = training.validate_training_state(
        state_path, root=destination, expected_games=int(manifest["games_completed"]),
    )
    if restored["sha256"] != manifest["training_state_sha256"]:
        raise ValueError("restored training-state hash differs from archive manifest")
    return {
        "empty_directory_restore": True,
        "archive_sha256": archive_sha,
        "manifest_file_count": len(manifest["files"]),
        "snapshot_label": manifest["snapshot_label"],
        "code_commit": manifest["code_commit"], "branch": manifest["branch"],
        "training_state": restored,
    }


def register_verified_snapshot(root: str | Path, row: dict) -> dict:
    """Persist only a Drive archive that has passed remote SHA and restore checks."""
    if row.get("sha256") != row.get("downloaded_sha256") or not row.get("empty_directory_restore"):
        raise ValueError("unverified V6.5 snapshot cannot be registered")
    root = Path(root).resolve()
    status_path = root / ARTIFACT / "results/drive_backup_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
        "experiment_id": "wuziq-alphazero-g-v6-5-selfplay-scale", "folder_id": None,
        "folder_url": None, "verified_snapshots": [],
    }
    if row.get("folder_id"):
        if status.get("folder_id") not in (None, row["folder_id"]):
            raise ValueError("snapshot Drive folder differs from the registered V6.5 folder")
        status["folder_id"] = row["folder_id"]
        status["folder_url"] = row.get("folder_url") or status.get("folder_url")
    for previous in status["verified_snapshots"]:
        if previous["name"] == row["name"] or previous["sha256"] == row["sha256"]:
            raise ValueError("duplicate V6.5 Drive snapshot registration")
    status["verified_snapshots"].append(row)
    status["latest_verified_snapshot"] = row["name"]
    status["latest_games"] = int(row["games_completed"])
    status["latest_training_state_sha256"] = row["training_state_sha256"]
    training.atomic_json(status_path, status)
    return status
