from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile
from typing import Any


EXPERIMENT_ID = "wuziq-alphazero-g-v6-9x9-400-equalized-training"


def restore_snapshot(archive_path: str | Path, target: str | Path) -> dict[str, Any]:
    archive_path, target = Path(archive_path), Path(target)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("V6 snapshot restore target must be an empty directory")
    target.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.v6-restore.tmp"
    if staging.exists():
        raise ValueError("V6 snapshot restore staging directory already exists")
    staging.mkdir(parents=True)
    try:
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("V6 snapshot contains duplicate archive paths")
            manifest_name = "v6_snapshot_manifest.json"
            if manifest_name not in names:
                raise ValueError("V6 snapshot manifest is missing")
            manifest = json.loads(archive.read(manifest_name))
            if (manifest.get("format_version") != 1
                    or manifest.get("experiment_id") != EXPERIMENT_ID):
                raise ValueError("V6 snapshot identity or version mismatch")
            records = manifest.get("files", [])
            if int(manifest.get("file_count", -1)) != len(records):
                raise ValueError("V6 snapshot manifest count mismatch")
            seen: set[str] = set()
            for row in records:
                name = str(row["path"])
                relative = PurePosixPath(name)
                if (name in seen or relative.is_absolute() or ".." in relative.parts
                        or "\\" in name or relative.parts[:2] != ("research", "g_v6")):
                    raise ValueError("V6 snapshot contains an unsafe or duplicate member path")
                seen.add(name)
                if name not in names:
                    raise ValueError(f"V6 snapshot member is missing: {name}")
                payload = archive.read(name)
                if (hashlib.sha256(payload).hexdigest() != row["sha256"]
                        or len(payload) != int(row["size_bytes"])):
                    raise ValueError(f"V6 snapshot member hash or size mismatch: {name}")
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if set(names) != seen | {manifest_name}:
                raise ValueError("V6 snapshot contains unregistered archive members")
        (target / "research").mkdir(parents=True, exist_ok=True)
        os.replace(staging / "research/g_v6", target / "research/g_v6")
        shutil.rmtree(staging)
        return {
            "restore_succeeded": True,
            "files_verified": len(records),
            "code_commit": manifest["code_commit"],
            "registration_sha256": manifest.get("registration_sha256"),
            "experiment_id": manifest["experiment_id"],
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/restore_g_v6_snapshot.py INPUT.zip EMPTY_DIRECTORY")
    print(json.dumps(restore_snapshot(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
