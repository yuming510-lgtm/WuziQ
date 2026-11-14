from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_5_snapshot import restore_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and restore a V6.5 recovery archive")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(restore_snapshot(args.archive, args.destination),
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
