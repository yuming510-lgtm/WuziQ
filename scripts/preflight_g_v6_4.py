from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v6_4_ladder import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure G5000 and opponent decision costs")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="restored V6.3 reference400_empty checkpoint directory")
    args = parser.parse_args()
    print(json.dumps(run_preflight(ROOT, checkpoint_dir=args.checkpoint_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
