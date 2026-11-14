from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_5_audit import audit_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-audit G V6.5 self-play records")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_experiment(ROOT, require_complete=not args.allow_partial),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
