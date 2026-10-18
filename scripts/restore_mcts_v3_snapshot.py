"""Verify and restore a V3 recovery archive into a results directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.mcts_v3_snapshot import restore_v3_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "research/v3/results")
    args = parser.parse_args()
    print(
        json.dumps(
            restore_v3_snapshot(args.snapshot, args.destination), ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
