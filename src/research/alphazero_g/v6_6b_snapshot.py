"""Integrity checked recovery snapshots for the V6.6B match run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

from . import v6_6b_evaluation as evaluation


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> list[Path]:
    artifact = root / evaluation.ARTIFACT
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    files = [path for path in artifact.rglob("*") if path.is_file()
             and not path.name.endswith((".tmp", ".zip"))]
    source_names = [
        "requirements-research.txt",
        "src/ai.py", "src/gomoku.py",
        "src/research/alphazero_g/v5_reference.py",
        "src/research/alphazero_g/v5_evaluation.py",
        "src/research/alphazero_g/v6_6b_evaluation.py",
        "src/research/alphazero_g/v6_6b_snapshot.py",
        "scripts/run_g_v6_6b.py", "scripts/snapshot_g_v6_6b.py",
        "scripts/record_g_v6_6b_snapshot.py",
        "tests/test_alphazero_g_v6_6b.py",
    ]
    files.extend(root / relative for relative in source_names)
    third_party = root / "third_party/AlphaZero_Gomoku"
    files.extend(third_party.rglob("*.py"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"recovery dependency is missing: {missing[0]}")
    return sorted(set(files))


def create_snapshot(root: str | Path, *, label: str, output: str | Path) -> dict:
    started = time.process_time()
    root, output = Path(root).resolve(), Path(output).resolve()
    if not label or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in label):
        raise ValueError("snapshot label must contain only ASCII letters, digits, '-' and '_'")
    files = _snapshot_files(root)
    hashes = {path.relative_to(root).as_posix(): _sha(path) for path in files}
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    progress_path = root / evaluation.ARTIFACT / "results/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {
        "completed_games": 0, "planned_games": 200, "completed_game_ids": [], "complete": False,
    }
    manifest = {
        "format_version": 1, "experiment_id": evaluation.EXPERIMENT_ID,
        "snapshot_label": label, "branch": branch, "code_commit": commit,
        "completed_games": int(progress["completed_games"]),
        "planned_games": int(progress["planned_games"]),
        "opening_schedule_sha256": json.loads((root / evaluation.ARTIFACT / "config/registration.json").read_text(encoding="utf-8"))["opening_schedule_sha256"],
        "match_plan_sha256": json.loads((root / evaluation.ARTIFACT / "config/registration.json").read_text(encoding="utf-8"))["match_plan_sha256"],
        "models": evaluation.MODEL_SHA256,
        "files": hashes,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("V6_6B_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {
        "path": str(output), "sha256": _sha(output), "size_bytes": output.stat().st_size,
        "file_count": len(files), "completed_games": manifest["completed_games"],
        "branch": branch, "code_commit": commit,
        "packaging_cpu_seconds": time.process_time() - started,
    }


def restore_snapshot(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    archive_sha = _sha(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"zip CRC failed for {bad}")
        manifest = json.loads(archive.read("V6_6B_RECOVERY_MANIFEST.json"))
        if manifest.get("experiment_id") != evaluation.EXPERIMENT_ID:
            raise ValueError("archive is not a V6.6B recovery snapshot")
        for member in archive.namelist():
            candidate = PurePosixPath(member)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("unsafe path in recovery archive")
        archive.extractall(destination)
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"recovery file SHA-256 mismatch: {relative}")
    for model, expected in evaluation.MODEL_SHA256.items():
        path = destination / evaluation.ARTIFACT / "checkpoints" / f"{model}.pt"
        if _sha(path) != expected:
            raise ValueError(f"restored {model} checkpoint hash mismatch")
        evaluation.load_frozen_model(path, model)
    replay = evaluation.audit_all(destination, require_complete=False)
    return {
        "empty_directory_restore": True, "archive_sha256": archive_sha,
        "manifest_file_count": len(manifest["files"]),
        "completed_games": manifest["completed_games"],
        "restored_games_audited": replay["audited_games"],
        "all_restored_games_valid": replay["audited_games"] == manifest["completed_games"],
        "manifest": manifest,
    }
