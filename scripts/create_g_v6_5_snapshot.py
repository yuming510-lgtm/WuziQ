from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_5_snapshot import create_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Package one full V6.5 Drive recovery point")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(create_snapshot(ROOT, label=args.label, output=args.output),
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
