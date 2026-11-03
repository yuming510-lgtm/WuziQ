from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def restore(archive_path: Path, target: Path) -> dict:
    if target.exists() and any(target.iterdir()):
        raise ValueError("restore target must be an empty directory")
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("snapshot contains duplicate paths")
        if "v5_snapshot_manifest.json" not in names:
            raise ValueError("snapshot manifest is missing")
        manifest = json.loads(archive.read("v5_snapshot_manifest.json"))
        if manifest.get("format_version") != 1 or manifest.get("experiment_id") != "wuziq-alphazero-g-v5-reference-and-closed-loop":
            raise ValueError("unexpected V5 snapshot identity")
        records = manifest.get("files", [])
        if int(manifest.get("file_count", -1)) != len(records):
            raise ValueError("V5 snapshot manifest count mismatch")
        for row in records:
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts[:2] == ("research", "g_v5"):
                raise ValueError("snapshot contains an unsafe path")
            payload = archive.read(row["path"])
            if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != int(row["size_bytes"]):
                raise ValueError(f"snapshot member hash mismatch: {row['path']}")
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    return {"restore_succeeded": True, "files_verified": len(records),
            "code_commit": manifest["code_commit"], "experiment_id": manifest["experiment_id"]}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/restore_g_v5_snapshot.py INPUT.zip EMPTY_DIRECTORY")
    print(json.dumps(restore(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False))


if __name__ == "__main__":
    main()
