#!/usr/bin/env python3
"""Restore V4.2 mutable recovery files to an empty directory and verify hashes."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/restore_g_v4_2_snapshot.py ARCHIVE EMPTY_DIRECTORY")
    archive_path, destination = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SystemExit("restore destination must be empty")
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        manifest_name = "V4_2_RECOVERY_MANIFEST.json"
        if names.count(manifest_name) != 1:
            raise SystemExit("snapshot must contain exactly one V4.2 recovery manifest")
        manifest = json.loads(archive.read(manifest_name))
        for relative, expected_hash in manifest["files"].items():
            member = PurePosixPath(relative)
            if member.is_absolute() or ".." in member.parts:
                raise SystemExit("snapshot contains an unsafe relative path")
            target = destination.joinpath(*member.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(relative)
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != expected_hash:
                raise SystemExit(f"snapshot member hash mismatch: {relative}")
            target.write_bytes(data)
        if set(names) != set(manifest["files"]) | {manifest_name}:
            raise SystemExit("snapshot archive and manifest file lists differ")
    expected_files = set(manifest["files"])
    restored = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if restored != expected_files:
        raise SystemExit("empty-directory restore produced a different file set")
    receipt = {"restore_succeeded": True, "files_verified": len(restored),
               "code_commit": manifest["code_commit"], "config_sha256": manifest["config_sha256"],
               "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    (destination / "RESTORE_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
