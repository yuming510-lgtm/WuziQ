from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.abcd_final_v4b_snapshot import create_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a minimal V4B recovery ZIP")
    parser.add_argument("--results", default="research/v4b/results")
    parser.add_argument("--output", default="research/v4b/v4b-recovery-latest.zip")
    parser.add_argument("--code-version")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = args.code_version or subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    result = create_snapshot(args.results, args.output, code_version=version)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
