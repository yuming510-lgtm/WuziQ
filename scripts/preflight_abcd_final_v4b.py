from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.abcd_final_v4b_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="V4B real move preflight and runtime estimate")
    parser.add_argument("--config", default="configs/abcd_final_v4b.json")
    parser.add_argument("--output", default="research/v4b/results/preflight.json")
    args = parser.parse_args()
    result = run_preflight(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
