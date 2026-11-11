from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.research.alphazero_g.v6_3_training import ARTIFACT, prepare_experiment, preflight


def main() -> None:
    torch.set_num_threads(1)
    prepared = prepare_experiment(ROOT)
    preflight_result = preflight(ROOT)
    (ROOT / ARTIFACT / "results/resume_validation.json").write_text(
        json.dumps(prepared, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(json.dumps({"prepared": prepared, "preflight": preflight_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
