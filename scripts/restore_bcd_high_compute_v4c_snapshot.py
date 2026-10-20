from __future__ import annotations

import argparse
import json

from src.research.bcd_high_compute_v4c_snapshot import restore_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and restore a V4C recovery archive into an empty directory")
    parser.add_argument("snapshot")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(restore_snapshot(args.snapshot, args.destination), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
