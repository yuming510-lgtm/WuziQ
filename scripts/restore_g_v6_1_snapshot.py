from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile


EXPERIMENT_ID = "wuziq-alphazero-g-v6-1-checkpoint-tournament"


def restore_snapshot(archive_path: str | Path, target: str | Path) -> dict:
    archive_path, target = Path(archive_path), Path(target)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("V6.1 recovery must restore into an empty directory")
    target.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.v6_1_restore.tmp"
    if staging.exists():
        raise ValueError("V6.1 restore staging path already exists")
    staging.mkdir(parents=True)
    try:
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "v6_1_snapshot_manifest.json" not in names:
                raise ValueError("V6.1 snapshot manifest or member names are invalid")
            manifest = json.loads(archive.read("v6_1_snapshot_manifest.json"))
            rows = manifest.get("files", [])
            if (manifest.get("format_version") != 1 or manifest.get("experiment_id") != EXPERIMENT_ID
                    or int(manifest.get("file_count", -1)) != len(rows)):
                raise ValueError("V6.1 snapshot identity/count mismatch")
            seen = set()
            for row in rows:
                name = str(row["path"])
                parts = PurePosixPath(name)
                if (name in seen or parts.is_absolute() or ".." in parts.parts
                        or "\\" in name or parts.parts[:2] != ("research", "g_v6_1")
                        or name not in names):
                    raise ValueError("V6.1 snapshot has unsafe or missing members")
                seen.add(name)
                payload = archive.read(name)
                if (hashlib.sha256(payload).hexdigest() != row["sha256"]
                        or len(payload) != int(row["size_bytes"])):
                    raise ValueError(f"V6.1 snapshot member integrity failed: {name}")
                destination = staging.joinpath(*parts.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if set(names) != seen | {"v6_1_snapshot_manifest.json"}:
                raise ValueError("V6.1 archive contains unregistered files")
        (target / "research").mkdir(parents=True, exist_ok=True)
        os.replace(staging / "research/g_v6_1", target / "research/g_v6_1")
        shutil.rmtree(staging)
        return {"restore_succeeded": True, "files_verified": len(rows),
                "branch": manifest["branch"], "code_commit": manifest["code_commit"],
                "experiment_id": manifest["experiment_id"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/restore_g_v6_1_snapshot.py INPUT.zip EMPTY_DIRECTORY")
    print(json.dumps(restore_snapshot(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
