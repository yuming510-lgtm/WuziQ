from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_9_snapshot import register_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a Drive snapshot only after full verification")
    parser.add_argument("snapshot_record", type=Path)
    args = parser.parse_args()
    row = json.loads(args.snapshot_record.read_text(encoding="utf-8"))
    print(json.dumps(register_snapshot(Path.cwd(), row), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
