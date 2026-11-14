from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_4_snapshot import restore_snapshot_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and restore an empty-directory V6.4 snapshot")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore_snapshot_archive(args.archive, args.destination),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
