"""Convenience entry point for the Minimax/Alpha-Beta efficiency study."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.efficiency import main


if __name__ == "__main__":
    raise SystemExit(main())
