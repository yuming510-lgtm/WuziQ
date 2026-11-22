from __future__ import annotations

import json
from pathlib import Path

from src.research.alphazero_g.v6_7_report import build_report


if __name__ == "__main__":
    print(json.dumps(build_report(Path(".")), ensure_ascii=False, sort_keys=True, indent=2))
