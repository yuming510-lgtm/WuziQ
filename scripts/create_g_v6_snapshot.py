from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile
from typing import Any


EXPERIMENT_ID = "wuziq-alphazero-g-v6-9x9-400-equalized-training"


def create_snapshot(root: str | Path, output_path: str | Path,
                    code_commit: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output_path).resolve()
    base = root / "research/g_v6"
    if not base.is_dir():
        raise FileNotFoundError(base)
    if output == base or base in output.parents:
        raise ValueError("V6 snapshot output cannot be inside the archived V6 directory")
    if code_commit is None:
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        branch = "snapshot-test"
    files = [path for path in base.rglob("*") if path.is_file()
             and not path.name.endswith(".tmp") and "__pycache__" not in path.parts]
    if not files:
        raise ValueError("V6 snapshot has no recovery data")
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            archive.writestr(relative, payload)
            records.append({
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            })
        registration_path = base / "config/registration.json"
        registration_hash = None
        if registration_path.is_file():
            registration_hash = json.loads(registration_path.read_text(encoding="utf-8")).get("registration_sha256")
        manifest = {
            "format_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "branch": branch,
            "code_commit": code_commit,
            "registration_sha256": registration_hash,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_count": len(records),
            "files": records,
        }
        archive.writestr("v6_snapshot_manifest.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    temp.replace(output)
    archive_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"path": str(output), "bytes": output.stat().st_size,
            "sha256": archive_hash, **manifest}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/create_g_v6_snapshot.py OUTPUT.zip")
    print(json.dumps(create_snapshot(Path(__file__).resolve().parents[1], sys.argv[1]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
