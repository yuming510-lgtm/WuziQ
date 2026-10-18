"""Measure V3 D3 compute budgets and estimate the registered 54-game run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.mcts_v3_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mcts_high_compute_v3.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "research/v3/pilot")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/v3/pilot/preflight.json"
    )
    args = parser.parse_args()
    result = run_preflight(args.config, args.results_dir, args.output)
    print(
        json.dumps(
            {
                "benchmarks": result["benchmarks"],
                "resources": result["resources"],
                "estimate": result["estimate"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
