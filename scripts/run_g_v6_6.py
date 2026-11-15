#!/usr/bin/env python3
"""Prepare and run the frozen-model V6.6 acceptance tournament."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_6_evaluation import (  # noqa: E402
    audit_all,
    prepare_v6_6,
    run_batch,
    run_preflight,
    summarize_v6_6,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--restored-root", type=Path, required=True,
                         help="directory with G5000/ and G8000/ restored Drive packages")
    prepare.add_argument("--drive-folder-id", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("run-batch")
    run.add_argument("--batch-games", type=int, default=10)
    audit = sub.add_parser("audit")
    audit.add_argument("--partial", action="store_true")
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_v6_6(ROOT, restored_root=args.restored_root,
                              drive_folder_id=args.drive_folder_id)
    elif args.command == "preflight":
        result = run_preflight(ROOT)
    elif args.command == "run-batch":
        result = run_batch(ROOT, batch_games=args.batch_games)
    elif args.command == "audit":
        result = audit_all(ROOT, require_complete=not args.partial)
    else:
        result = summarize_v6_6(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
