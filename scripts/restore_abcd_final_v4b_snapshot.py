from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.abcd_final_v4b_snapshot import restore_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and restore a V4B recovery ZIP")
    parser.add_argument("snapshot")
    parser.add_argument("destination")
    args = parser.parse_args()
    result = restore_snapshot(args.snapshot, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
