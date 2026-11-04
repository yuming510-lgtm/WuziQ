from __future__ import annotations

import json

from src.research.alphazero_g.v6_1_experiment import run_v61


if __name__ == "__main__":
    print(json.dumps(run_v61(), ensure_ascii=False, indent=2))
