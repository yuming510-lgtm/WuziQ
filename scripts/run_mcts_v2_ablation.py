"""Run the configured fixed-position MCTS V2 ablations."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.mcts_v2_ablation import main


if __name__ == "__main__":
    main()
