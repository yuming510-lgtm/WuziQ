from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_9_snapshot import create_bundle, restore_bundle
from src.research.alphazero_g.v6_9_parallel import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Package or restore a complete V6.9 run")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restore-archive", type=Path)
    parser.add_argument("--restore-to", type=Path)
    args = parser.parse_args()
    if args.restore_archive:
        if args.restore_to is None:
            parser.error("--restore-to is required with --restore-archive")
        result = restore_bundle(args.restore_archive, args.restore_to, root=ROOT)
    else:
        if args.run_dir is None or args.output is None:
            parser.error("--run-dir and --output are required to create a bundle")
        result = create_bundle(args.run_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
