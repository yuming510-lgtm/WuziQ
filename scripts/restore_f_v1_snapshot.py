#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.snapshot import restore_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--empty-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore_snapshot(args.snapshot, args.empty_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
