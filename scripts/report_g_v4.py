#!/usr/bin/env python3
"""Audit all completed G V4 records and generate the Chinese report/figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_report import generate_v4_report


def main() -> None:
    print(json.dumps(generate_v4_report(ROOT), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
