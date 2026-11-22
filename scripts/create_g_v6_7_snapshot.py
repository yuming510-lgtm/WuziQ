from __future__ import annotations

import argparse
import json

from src.research.alphazero_g.v6_7_snapshot import create_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a complete V6.7 recovery state")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(create_snapshot(".", label=args.label, output=args.output),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
