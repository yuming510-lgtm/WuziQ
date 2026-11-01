#!/usr/bin/env python3
"""Restore a verified G V1 recovery ZIP into an empty directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.snapshot import restore_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    metadata = restore_snapshot(args.archive, args.destination)
    print(json.dumps({
        "destination": str(args.destination),
        "commit": metadata["commit"],
        "active_seconds": metadata["active_seconds"],
        "files": len(metadata["files"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
