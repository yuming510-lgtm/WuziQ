#!/usr/bin/env python3
"""Create or empty-directory-restore a V6.6B recovery archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_6b_snapshot import create_snapshot, restore_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--label", required=True)
    create.add_argument("--output", type=Path, required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_snapshot(ROOT, label=args.label, output=args.output)
    else:
        result = restore_snapshot(args.archive, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
