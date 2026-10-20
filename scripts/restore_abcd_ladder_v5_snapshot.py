import argparse
import json

from src.research.abcd_ladder_v5_snapshot import restore_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and restore a V5 recovery snapshot.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--empty-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(restore_snapshot(args.snapshot, args.empty_directory), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
