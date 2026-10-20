import argparse
import json

from src.research.abcd_ladder_v5_snapshot import create_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a minimal V5 recovery snapshot.")
    parser.add_argument("--results-dir", default="research/v5/results")
    parser.add_argument("--output", default="research/v5/v5_recovery_snapshot.json")
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(create_snapshot(args.results_dir, args.output, args.implementation_commit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
