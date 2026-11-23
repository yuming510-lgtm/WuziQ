from __future__ import annotations

import json
from pathlib import Path
from src.research.alphazero_g.v6_7_training import exclusive_training_lock, run_training_batch


if __name__ == "__main__":
    lock_path = Path("/tmp/wuziq-alphazero-g-v6-7-training.lock")
    with exclusive_training_lock(lock_path):
        print(json.dumps(run_training_batch(), ensure_ascii=False, sort_keys=True, indent=2))
