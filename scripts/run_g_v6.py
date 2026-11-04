from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_experiment import run_blind_session, run_to_stage


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume the registered WuziQ G V6 experiment")
    parser.add_argument("--through-stage", type=int, choices=(2_000, 5_000, 10_000, 15_000, 20_000))
    parser.add_argument("--finalize-blind", action="store_true")
    parser.add_argument("--session-cpu-seconds", type=float, default=None,
                        help="yield at a complete game or color-swapped pair boundary after this CPU budget")
    args = parser.parse_args()
    if (args.through_stage is None) == (not args.finalize_blind):
        parser.error("choose exactly one of --through-stage or --finalize-blind")
    if args.session_cpu_seconds is not None and args.session_cpu_seconds <= 0:
        parser.error("--session-cpu-seconds must be positive")
    if args.finalize_blind:
        result = run_blind_session(ROOT, session_cpu_seconds=args.session_cpu_seconds)
    else:
        result = run_to_stage(ROOT, args.through_stage, session_cpu_seconds=args.session_cpu_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
