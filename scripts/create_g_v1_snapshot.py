#!/usr/bin/env python3
"""Create a hash-indexed G V1 recovery ZIP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from src.research.alphazero_g.snapshot import create_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-seconds", type=float, required=True)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    commit = args.commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
    ).strip()
    digest = create_snapshot(args.root, args.output, commit, args.active_seconds)
    print(json.dumps({"archive": str(args.output), "sha256": digest, "commit": commit}, sort_keys=True))


if __name__ == "__main__":
    main()
