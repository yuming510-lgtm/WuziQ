from __future__ import annotations

import argparse
import json

from src.research.alphazero_g.v6_8_tournament import restore_snapshot_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a V6.8 snapshot into an empty directory and verify every file SHA")
    parser.add_argument("archive")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(restore_snapshot_archive(args.archive, args.destination),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
