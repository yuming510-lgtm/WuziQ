from __future__ import annotations

import argparse
import json
import os

from src.research.alphazero_g.v5_experiment import run_v5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume registered WuziQ G V5 experiment")
    parser.add_argument("--session-cpu-seconds", type=float, default=None,
                        help="stop at the next game/pair boundary after this CPU time; global cap remains enforced")
    args = parser.parse_args()
    if args.session_cpu_seconds is not None:
        if args.session_cpu_seconds <= 0:
            parser.error("--session-cpu-seconds must be positive")
        os.environ["WUZIQ_V5_SESSION_CPU_SECONDS"] = str(args.session_cpu_seconds)
    print(json.dumps(run_v5(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
