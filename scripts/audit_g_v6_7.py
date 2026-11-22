from __future__ import annotations

import argparse
import json

from src.research.alphazero_g.v6_7_audit import audit_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V6.7 training records, checkpoints, and paired matches")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_experiment(".", require_complete=not args.allow_incomplete),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
