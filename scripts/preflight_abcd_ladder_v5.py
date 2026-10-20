import argparse
import json

from src.research.abcd_ladder_v5_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure V5 budgets and estimate wall time.")
    parser.add_argument("--config", default="configs/abcd_ladder_v5.json")
    parser.add_argument("--output", default="research/v5/results/preflight.json")
    args = parser.parse_args()
    result = run_preflight(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
