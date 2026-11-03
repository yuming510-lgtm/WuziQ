#!/usr/bin/env python3
"""Verify every G V4.1 training and evaluation record and write an audit receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_1_audit import audit_experiment


def main() -> None:
    result = audit_experiment(ROOT)
    target = ROOT / "research/g_v4_1/results/audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({"audit": str(target.relative_to(ROOT)), **result}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
