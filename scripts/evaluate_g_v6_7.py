from __future__ import annotations

import argparse
import json

from src.research.alphazero_g.v6_7_evaluation import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a registered V6.7 paired evaluation stage")
    parser.add_argument("--games", type=int, required=True, choices=(10_000, 12_000, 16_000, 20_000))
    parser.add_argument("--pool", choices=("development", "blind"), default="development")
    args = parser.parse_args()
    print(json.dumps(run_stage(".", candidate_games=args.games, pool=args.pool),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
