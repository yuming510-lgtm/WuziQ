from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile


EXPERIMENT_ID = "wuziq-alphazero-g-v6-1-checkpoint-tournament"


def create_snapshot(root: str | Path, output_path: str | Path) -> dict:
    root = Path(root).resolve()
    base = root / "research/g_v6_1"
    output = Path(output_path).resolve()
    if not base.is_dir() or output == base or base in output.parents:
        raise ValueError("snapshot source/target is invalid")
    paths = [p for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
             and "__pycache__" not in p.parts]
    if not paths:
        raise ValueError("V6.1 recovery snapshot cannot be empty")
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root,
                            check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            check=True, capture_output=True, text=True).stdout.strip()
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            archive.writestr(relative, payload)
            records.append({"path": relative, "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
        manifest = {
            "format_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "branch": branch,
            "code_commit": commit,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_count": len(records),
            "files": records,
        }
        archive.writestr("v6_1_snapshot_manifest.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    temp.replace(output)
    return {"path": str(output), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), **manifest}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/create_g_v6_1_snapshot.py OUTPUT.zip")
    print(json.dumps(create_snapshot(Path(__file__).resolve().parents[1], sys.argv[1]),
                     ensure_ascii=False, indent=2))
