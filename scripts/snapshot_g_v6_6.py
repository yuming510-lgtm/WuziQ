#!/usr/bin/env python3
"""Create or validate a self-contained, hash-manifested V6.6 recovery archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "research/g_v6_6"
SOURCES = [
    Path("src/research/alphazero_g/v6_6_evaluation.py"),
    Path("scripts/run_g_v6_6.py"),
    Path("scripts/snapshot_g_v6_6.py"),
    Path("scripts/build_g_v6_6_report.py"),
    Path("scripts/record_g_v6_6_snapshot.py"),
    Path("tests/test_alphazero_g_v6_6.py"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tracked_files() -> list[Path]:
    paths = []
    for base in (ARTIFACT / "config", ARTIFACT / "checkpoints", ARTIFACT / "results", ARTIFACT / "figures"):
        if base.exists():
            paths.extend(p for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    report = ARTIFACT / "REPORT.md"
    if report.is_file():
        paths.append(report)
    for rel in SOURCES:
        path = ROOT / rel
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def create(output: Path, *, snapshot_id: str) -> dict:
    files = tracked_files()
    if not files:
        raise RuntimeError("no V6.6 recovery files found")
    rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "size_bytes": p.stat().st_size}
            for p in files]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "format_version": 1, "experiment_id": "wuziq-alphazero-g-v6-6-acceptance",
        "snapshot_id": snapshot_id, "git_commit": commit, "file_count": len(rows), "files": rows,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("snapshot_manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        for path in files:
            archive.write(path, "files/" + path.relative_to(ROOT).as_posix())
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, output)
    manifest["archive_sha256"] = digest(output)
    manifest["archive_size_bytes"] = output.stat().st_size
    return manifest


def restore(archive_path: Path, destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("recovery destination must be an existing empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if "snapshot_manifest.json" not in names:
            raise ValueError("snapshot manifest missing")
        manifest = json.loads(archive.read("snapshot_manifest.json"))
        expected = {f"files/{row['path']}": row for row in manifest["files"]}
        if set(names) != set(expected) | {"snapshot_manifest.json"}:
            raise ValueError("archive contains missing or unregistered files")
        for archive_name, row in expected.items():
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path in snapshot manifest")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(archive_name))
            if target.stat().st_size != int(row["size_bytes"]) or digest(target) != row["sha256"]:
                raise ValueError(f"restored file checksum mismatch: {row['path']}")
    return {"snapshot_id": manifest["snapshot_id"], "git_commit": manifest["git_commit"],
            "restored_files": len(manifest["files"]), "archive_sha256": digest(archive_path),
            "all_files_verified": True, "destination": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--snapshot-id", required=True)
    check = sub.add_parser("restore")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = create(args.output, snapshot_id=args.snapshot_id) if args.command == "create" else restore(args.archive, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
