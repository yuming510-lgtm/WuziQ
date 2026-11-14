from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


ARTIFACT = Path("research/g_v6_4")
MANIFEST_NAME = "V6_4_RECOVERY_MANIFEST.json"
RECOVERY_CODE = (
    "src/research/alphazero_g/v6_4_ladder.py",
    "src/research/alphazero_g/v6_4_snapshot.py",
    "scripts/prepare_g_v6_4.py",
    "scripts/preflight_g_v6_4.py",
    "scripts/run_g_v6_4.py",
    "scripts/audit_g_v6_4.py",
    "scripts/report_g_v6_4.py",
    "scripts/create_g_v6_4_snapshot.py",
    "scripts/restore_g_v6_4_snapshot.py",
    "tests/test_alphazero_g_v6_4.py",
    "tests/test_alphazero_g_v6_4_snapshot.py",
)


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> list[Path]:
    artifact = root / ARTIFACT
    files = []
    for subdir in ("config", "results"):
        source = artifact / subdir
        if source.is_dir():
            files.extend(path for path in source.rglob("*") if path.is_file()
                         and not path.name.endswith(".tmp")
                         and not path.name.endswith(".zip"))
    report = artifact / "REPORT.md"
    if report.is_file():
        files.append(report)
    charts = artifact / "charts"
    if charts.is_dir():
        files.extend(path for path in charts.rglob("*") if path.is_file())
    for relative in RECOVERY_CODE:
        source = root / relative
        if source.is_file():
            files.append(source)
    recovery_note = artifact / "RECOVERY.md"
    if recovery_note.is_file():
        files.append(recovery_note)
    return sorted(set(files))


def create_snapshot_archive(root: str | Path, output: str | Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    files = _snapshot_files(root)
    if not files:
        raise ValueError("V6.4 recovery package has no configuration or progress data")
    try:
        code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                     capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = "unknown"
    manifest = {
        "format_version": 1,
        "experiment_id": "wuziq-g5000-ladder-v6-4",
        "code_commit": code_commit,
        "files": {path.relative_to(root).as_posix(): sha_file(path) for path in files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as package:
        for path in files:
            package.write(path, path.relative_to(root).as_posix())
        package.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return {"path": str(output), "sha256": sha_file(output), "size_bytes": output.stat().st_size,
            "file_count": len(files), "code_commit": code_commit, "manifest": manifest}


def restore_snapshot_archive(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path).resolve(), Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("V6.4 recovery restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as package:
        manifest = json.loads(package.read(MANIFEST_NAME))
        for name in package.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path in V6.4 recovery package")
        package.extractall(destination)
    for relative, expected_sha in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or sha_file(path) != expected_sha:
            raise ValueError(f"V6.4 recovery file hash mismatch: {relative}")
    return {
        "empty_directory_restore": True,
        "sha256": sha_file(archive_path),
        "manifest_files_verified": len(manifest["files"]),
        "code_commit": manifest["code_commit"],
        "experiment_id": manifest["experiment_id"],
    }
