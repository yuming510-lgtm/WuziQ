"""Minimal, checksummed F V1 recovery snapshots and restore verification."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


SNAPSHOT_INTERVAL_HOURS = (1, 2, 4, 6, 10, 16, 26)
SNAPSHOT_SCHEMA = 1
_DEFAULT_PATHS = (
    "src/research/imitation_f",
    "scripts/prepare_f_v1_dataset.py",
    "scripts/generate_f_v1_data.py",
    "scripts/preflight_f_v1.py",
    "scripts/train_f_v1.py",
    "scripts/create_f_v1_snapshot.py",
    "scripts/restore_f_v1_snapshot.py",
    "scripts/play_f_v1_matches.py",
    "scripts/evaluate_f_v1.py",
    "scripts/build_f_v1_report.py",
    "requirements-f.txt",
    "research/f_v1/config",
    "research/f_v1/data",
    "research/f_v1/results/preflight.json",
    "research/f_v1/results/generated_games.jsonl",
    "research/f_v1/results/generated_annotations.jsonl",
    "research/f_v1/results/generation_progress.json",
    "research/f_v1/results/recovery_state.json",
    "research/f_v1/checkpoints",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _expand_paths(root: Path, requested: list[str] | tuple[str, ...]) -> list[Path]:
    result: set[Path] = set()
    base = root.resolve()
    for item in requested:
        candidate = (base / item).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"snapshot path escapes root: {item}")
        if candidate.is_dir():
            for child in candidate.rglob("*"):
                if child.is_file():
                    resolved = child.resolve()
                    if base not in resolved.parents:
                        raise ValueError(f"snapshot symlink escapes root: {child}")
                    relative = resolved.relative_to(base).as_posix()
                    if (
                        relative.startswith(".git/")
                        or relative.endswith(".tmp")
                        or "__pycache__" in PurePosixPath(relative).parts
                        or relative.endswith(".pyc")
                    ):
                        continue
                    if relative.startswith("research/f_v1/results/") and relative.split("/")[-1] not in {
                        "preflight.json", "generated_games.jsonl", "generated_annotations.jsonl",
                        "generation_progress.json", "recovery_state.json", "matches.jsonl",
                        "evaluation_openings.json", "match_summary.json",
                    }:
                        continue
                    result.add(resolved)
        elif candidate.is_file():
            result.add(candidate)
    return sorted(result, key=lambda path: path.relative_to(base).as_posix())


def create_snapshot(
    state_root: str | Path,
    output_path: str | Path,
    paths: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Write a small zip with only task code, data, progress and checkpoints."""
    root = Path(state_root).resolve()
    output = Path(output_path)
    selected = _expand_paths(root, paths or _DEFAULT_PATHS)
    files: dict[str, dict[str, object]] = {}
    for path in selected:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": _sha(data), "size_bytes": len(data)}
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": _git_commit(root),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("snapshot_manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        for path in selected:
            archive.write(path, path.relative_to(root).as_posix())
    temp.replace(output)
    digest = _sha(output.read_bytes())
    return {
        "path": str(output),
        "sha256": digest,
        "size_bytes": output.stat().st_size,
        "files": files,
        "base_commit": manifest["base_commit"],
    }


def restore_snapshot(snapshot_path: str | Path, empty_dir: str | Path) -> dict:
    """Verify every member hash, then restore only into a genuinely empty folder."""
    snapshot_path, empty_dir = Path(snapshot_path), Path(empty_dir)
    empty_dir.mkdir(parents=True, exist_ok=True)
    if any(empty_dir.iterdir()):
        raise ValueError("restore destination must be empty")
    archive_sha = _sha(snapshot_path.read_bytes())
    with zipfile.ZipFile(snapshot_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("snapshot contains duplicate members; integrity check failed")
        if "snapshot_manifest.json" not in names:
            raise ValueError("snapshot manifest is missing")
        manifest = json.loads(archive.read("snapshot_manifest.json"))
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported snapshot schema")
        expected = manifest.get("files")
        if not isinstance(expected, dict):
            raise ValueError("snapshot file manifest is malformed")
        if set(names) != set(expected) | {"snapshot_manifest.json"}:
            raise ValueError("snapshot members do not match manifest")
        payloads: dict[str, bytes] = {}
        for relative, metadata in expected.items():
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("snapshot contains an unsafe file path")
            payload = archive.read(relative)
            if _sha(payload) != metadata.get("sha256") or len(payload) != metadata.get("size_bytes"):
                raise ValueError(f"snapshot SHA-256/hash integrity mismatch: {relative}")
            payloads[relative] = payload
    for relative, payload in payloads.items():
        target = empty_dir / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".restore-tmp")
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(target)
    return {
        "sha256": archive_sha,
        "restored_files": sorted(payloads),
        "file_count": len(payloads),
        "base_commit": manifest.get("base_commit"),
    }


def next_snapshot_deadline(effective_hours: float) -> float:
    """Return cumulative hour deadline for the requested 1,2,4,6,10,16,26… cadence."""
    if effective_hours < 0:
        raise ValueError("effective_hours cannot be negative")
    deadlines = []
    total = 0
    for interval in SNAPSHOT_INTERVAL_HOURS:
        total += interval
        deadlines.append(total)
    while deadlines[-1] <= effective_hours:
        deadlines.append(deadlines[-1] + 26)
    return next(deadline for deadline in deadlines if deadline > effective_hours)
