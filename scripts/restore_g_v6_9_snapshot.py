from __future__ import annotations

import argparse
import json

from src.research.alphazero_g.v6_9_snapshot import restore_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a V6.9 snapshot in an empty directory")
    parser.add_argument("archive")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(restore_snapshot(args.archive, args.destination),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
