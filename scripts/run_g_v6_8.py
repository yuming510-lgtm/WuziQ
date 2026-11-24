from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_8_tournament import (
    audit, create_snapshot, prepare_experiment, report, run_batch, run_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/recover the frozen G V6.8 six-stage tournament")
    parser.add_argument("--prepare", action="store_true", help="verify models and freeze schedule/protocol")
    parser.add_argument("--preflight", action="store_true", help="run four registered pilot games")
    parser.add_argument("--max-new-games", type=int, default=40, help="maximum new games in this idempotent batch")
    parser.add_argument("--audit", action="store_true", help="replay and audit all scheduled games")
    parser.add_argument("--report", action="store_true", help="generate final Chinese report and charts")
    parser.add_argument("--snapshot", type=str, help="create a recovery archive with this label")
    args = parser.parse_args()
    if args.prepare:
        result = prepare_experiment()
    elif args.preflight:
        result = run_preflight()
    elif args.audit:
        result = audit()
    elif args.report:
        result = report()
    elif args.snapshot:
        result = create_snapshot(label=args.snapshot)
    else:
        result = run_batch(max_new_games=args.max_new_games)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
