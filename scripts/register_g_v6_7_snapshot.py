from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_7_snapshot import register_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a fully verified V6.7 Drive snapshot")
    parser.add_argument("record_file", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record_file.read_text(encoding="utf-8"))
    print(json.dumps(register_snapshot(".", record), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
