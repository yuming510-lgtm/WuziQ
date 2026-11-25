from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_9_snapshot import create_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact full-state V6.9 recovery bundle")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(create_snapshot(Path.cwd(), label=args.label, output=args.output),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
