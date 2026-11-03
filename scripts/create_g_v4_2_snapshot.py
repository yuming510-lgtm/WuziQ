#!/usr/bin/env python3
"""Build a minimal V4.2 recovery archive, using committed code plus mutable state."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_2_config import load_v4_2_config


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect_files() -> list[Path]:
    relative_roots = [
        "research/g_v4_2/config", "research/g_v4_2/results", "research/g_v4_2/checkpoints",
    ]
    files = []
    for relative in relative_roots:
        base = ROOT / relative
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*") if path.is_file() and not path.name.endswith(".tmp"))
    return sorted(set(files))


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("/tmp/wuziq-g-v4-2-recovery.zip")
    config = load_v4_2_config(ROOT)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    files = _collect_files()
    payloads = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {
        "format_version": 1, "experiment_id": config["experiment_id"],
        "branch": branch, "code_commit": commit,
        "config_sha256": config["config_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {relative: _sha(data) for relative, data in payloads.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, data in payloads.items():
            archive.writestr(relative, data)
        archive.writestr("V4_2_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size,
                      "sha256": _sha(output.read_bytes()), "files": len(payloads),
                      "code_commit": commit, "config_sha256": config["config_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
