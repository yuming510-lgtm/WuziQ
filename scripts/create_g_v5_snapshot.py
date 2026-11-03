from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/create_g_v5_snapshot.py OUTPUT.zip")
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]).resolve()
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                 check=True, capture_output=True, text=True).stdout.strip()
    base = root / "research/g_v5"
    records = []
    files = [p for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            archive.writestr(relative, payload)
            records.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload)})
        manifest = {
            "format_version": 1,
            "experiment_id": "wuziq-alphazero-g-v5-reference-and-closed-loop",
            "branch": "research/alphazero-g-v5",
            "code_commit": code_commit,
            "file_count": len(records),
            "files": records,
        }
        archive.writestr("v5_snapshot_manifest.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    data = output.read_bytes()
    print(json.dumps({"path": str(output), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "files": len(records), "code_commit": code_commit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
