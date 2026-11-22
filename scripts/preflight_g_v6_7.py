from __future__ import annotations

import json
from src.research.alphazero_g.v6_7_preflight import run_preflight


if __name__ == "__main__":
    print(json.dumps(run_preflight(), ensure_ascii=False, sort_keys=True, indent=2))
