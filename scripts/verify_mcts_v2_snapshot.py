"""Run the local snapshot checksum and restore drill."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.mcts_v2_snapshot import run_local_restore_drill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="research/v2/pilot")
    args = parser.parse_args()
    result = run_local_restore_drill(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
