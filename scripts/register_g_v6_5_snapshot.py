from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_5_snapshot import register_verified_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a downloaded and restore-verified V6.5 Drive snapshot")
    parser.add_argument("--row-json", required=True)
    args = parser.parse_args()
    row = json.loads(Path(args.row_json).read_text(encoding="utf-8"))
    print(json.dumps(register_verified_snapshot(ROOT, row), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
