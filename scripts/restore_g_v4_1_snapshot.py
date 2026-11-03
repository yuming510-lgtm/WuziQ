#!/usr/bin/env python3
"""Verify and restore a G V4.1 recovery archive into an empty directory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_1_snapshot import restore_snapshot_archive


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: restore_g_v4_1_snapshot.py ARCHIVE EMPTY_DIRECTORY")
    manifest = restore_snapshot_archive(sys.argv[1], sys.argv[2])
    print(json.dumps({"files_count": manifest["files_count"],
                      "total_file_bytes": manifest["total_file_bytes"],
                      "branch": manifest["metadata"].get("branch"),
                      "config_sha256": manifest["metadata"].get("config_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
