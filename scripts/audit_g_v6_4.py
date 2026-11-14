from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_4_ladder import audit_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay and audit G5000 V6.4 ladder games")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_results(ROOT, require_complete=not args.allow_partial),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
