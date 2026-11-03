#!/usr/bin/env python3
"""Create a minimal self-contained G V4.1 recovery archive."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_1_snapshot import create_v4_1_snapshot


def main() -> None:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / "g_v4_1_recovery.zip"
    manifest = create_v4_1_snapshot(ROOT, target)
    print(json.dumps({"archive": str(target), "files_count": manifest["files_count"],
                      "total_file_bytes": manifest["total_file_bytes"],
                      "metadata": manifest["metadata"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
