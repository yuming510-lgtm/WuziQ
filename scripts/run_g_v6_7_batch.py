from __future__ import annotations

import json
from src.research.alphazero_g.v6_7_training import run_training_batch


if __name__ == "__main__":
    print(json.dumps(run_training_batch(), ensure_ascii=False, sort_keys=True, indent=2))
